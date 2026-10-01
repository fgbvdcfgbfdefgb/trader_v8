"""
Main training orchestrator.

Each epoch:
  1. Picks ONE random day (common across BTC/ETH/LTC) from the offline dataset.
  2. Resets a $20 portfolio and simulates the full trading day minute-by-minute:
       - PricePredictor forecasts next-minute return per asset (causal).
       - MarketAnalyser encodes a regime embedding per asset (causal).
       - TradeMaker (PPO actor-critic) consumes both (detached) + raw features
         + portfolio state, and chooses a target allocation.
  3. After the day completes, all three networks are updated **in parallel**
     (separate threads -> separate GPUs when available):
       - TradeMaker via PPO (clipped surrogate objective).
       - PricePredictor via supervised MSE on realized next-minute returns.
       - MarketAnalyser via supervised cross-entropy on realized trend/
         volatility buckets over the following 30 minutes.
  4. A PNG is auto-saved with the price action, portfolio curve, predictor
     accuracy, and the Market Analyser's text advisory for that day.

No network access is used anywhere in this file -> fully offline, safe for
Snowflake / an air-gapped training box.
"""
import argparse
import os
import random
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import get_hw_and_sizes
from .data import MarketData, SYMBOLS, MINUTES_PER_DAY, PRED_LOOKBACK
from .models import PricePredictor, MarketAnalyser, TradeMaker
from .ppo import PPOTrainer, RolloutBuffer
from .env import TradingEnv, START_CAPITAL, TARGET_CAPITAL
from .advisor import summarize_day, build_advisor_report
from .plotting import save_epoch_plot, save_training_curves

PORTFOLIO_DIM = 1 + len(SYMBOLS) + 1   # cash_frac + 3 asset fracs + norm net worth


def trend_vol_labels(day_arrays, sym):
    fwd = day_arrays[sym]["future_return_30"]
    trend = np.ones(len(fwd), dtype=np.int64)       # 1 = flat
    trend[fwd > 0.0012] = 2                            # up
    trend[fwd < -0.0012] = 0                           # down

    vol = day_arrays[sym]["volatility"]
    q1, q2 = np.quantile(vol, [0.33, 0.66])
    vol_label = np.zeros(len(vol), dtype=np.int64)
    vol_label[vol > q1] = 1
    vol_label[vol > q2] = 2
    return trend, vol_label


def build_trader_features(day_arrays, sym, t, pred_return, embed):
    """Compact per-asset feature row fed into the TradeMaker, on top of the
    detached predictor/analyser outputs."""
    raw = np.array([
        day_arrays[sym]["ret1"][t],
        day_arrays[sym]["rsi"][t] / 100.0,
        day_arrays[sym]["bb"][t],
        day_arrays[sym]["volatility"][t],
        day_arrays[sym]["momentum"][t],
        day_arrays[sym]["macd"][t],
    ], dtype=np.float32)
    return np.concatenate([raw, [pred_return], embed])


def run_epoch(epoch, market: MarketData, models, opts, devices, rng, minutes, save_plot=True):
    predictor, analyser, trader = models["predictor"], models["analyser"], models["trader"]
    predictor_opt, analyser_opt, ppo = opts["predictor"], opts["analyser"], opts["ppo"]
    dev_p, dev_a, dev_t = devices["predictor"], devices["analyser"], devices["trader"]

    date = market.sample_day(rng)
    day_arrays = market.get_day_arrays(date)
    labels = {sym: trend_vol_labels(day_arrays, sym) for sym in SYMBOLS}

    env = TradingEnv()
    obs_portfolio = env.reset(day_arrays)

    buf = RolloutBuffer()
    pred_samples = {sym: [] for sym in SYMBOLS}     # (window, label)
    an_samples = {sym: [] for sym in SYMBOLS}       # (window, trend_label, vol_label)
    pred_vs_actual = {sym: ([], []) for sym in SYMBOLS}

    n_steps = min(minutes, MINUTES_PER_DAY - 1)
    episode_reward = 0.0

    predictor.eval()
    analyser.eval()
    with torch.no_grad():
        for t in range(n_steps):
            per_asset_feats = []
            for sym in SYMBOLS:
                window = day_arrays[sym]["windows"][t]
                w_pred = torch.as_tensor(window, dtype=torch.float32, device=dev_p).unsqueeze(0)
                pred_return, _ = predictor(w_pred)
                pred_return = float(pred_return.item())

                w_an = torch.as_tensor(window, dtype=torch.float32, device=dev_a).unsqueeze(0)
                embed, trend_logits, vol_logits = analyser(w_an)
                embed_np = embed.squeeze(0).cpu().numpy()

                feats = build_trader_features(day_arrays, sym, t, pred_return, embed_np)
                per_asset_feats.append(feats)

                pred_samples[sym].append((window.copy(), day_arrays[sym]["ret1"][min(t + 1, n_steps)]))
                an_samples[sym].append((window.copy(), labels[sym][0][t], labels[sym][1][t]))

                actual_next_ret = day_arrays[sym]["ret1"][min(t + 1, n_steps)]
                pred_vs_actual[sym][0].append(pred_return)
                pred_vs_actual[sym][1].append(actual_next_ret)

            trader_obs = np.concatenate(per_asset_feats + [obs_portfolio])
            weights, raw_action, logprob, value = ppo.act(trader_obs)

            obs_portfolio, reward, done, info = env.step(weights)
            episode_reward += reward
            buf.add(trader_obs, raw_action, logprob, value, reward, done)

            if done:
                break

    # bootstrap value for truncated episodes
    last_value = 0.0
    if not buf.dones[-1]:
        with torch.no_grad():
            last_obs = torch.as_tensor(buf.obs[-1], dtype=torch.float32, device=dev_t).unsqueeze(0)
            _, _, v = trader(last_obs)
            last_value = float(v.item())

    results = {}

    def update_ppo():
        trader.train()
        results["ppo"] = ppo.update(buf, last_value=last_value)
        trader.eval()

    def update_predictor():
        predictor.train()
        losses = []
        for sym in SYMBOLS:
            windows = np.array([w for w, _ in pred_samples[sym]], dtype=np.float32)
            labels_arr = np.array([l for _, l in pred_samples[sym]], dtype=np.float32)
            wt = torch.as_tensor(windows, device=dev_p)
            lt = torch.as_tensor(labels_arr, device=dev_p)
            bs = 256
            for start in range(0, len(wt), bs):
                xb, yb = wt[start:start + bs], lt[start:start + bs]
                pred, _ = predictor(xb)
                loss = F.mse_loss(pred, yb)
                predictor_opt.zero_grad()
                loss.backward()
                predictor_opt.step()
                losses.append(loss.item())
        results["predictor_loss"] = float(np.mean(losses)) if losses else 0.0
        predictor.eval()

    def update_analyser():
        analyser.train()
        losses = []
        for sym in SYMBOLS:
            windows = np.array([w for w, _, _ in an_samples[sym]], dtype=np.float32)
            trend_l = np.array([t_ for _, t_, _ in an_samples[sym]], dtype=np.int64)
            vol_l = np.array([v_ for _, _, v_ in an_samples[sym]], dtype=np.int64)
            wt = torch.as_tensor(windows, device=dev_a)
            tt = torch.as_tensor(trend_l, device=dev_a)
            vt = torch.as_tensor(vol_l, device=dev_a)
            bs = 256
            for start in range(0, len(wt), bs):
                xb = wt[start:start + bs]
                tb, vb = tt[start:start + bs], vt[start:start + bs]
                _, trend_logits, vol_logits = analyser(xb)
                loss = F.cross_entropy(trend_logits, tb) + F.cross_entropy(vol_logits, vb)
                analyser_opt.zero_grad()
                loss.backward()
                analyser_opt.step()
                losses.append(loss.item())
        results["analyser_loss"] = float(np.mean(losses)) if losses else 0.0
        analyser.eval()

    threads = [threading.Thread(target=fn) for fn in (update_ppo, update_predictor, update_analyser)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    final_value = env.history["net_worth"][-1] if env.history["net_worth"] else START_CAPITAL

    summaries = {}
    with torch.no_grad():
        for sym in SYMBOLS:
            window = day_arrays[sym]["windows"][n_steps - 1]
            w_an = torch.as_tensor(window, dtype=torch.float32, device=dev_a).unsqueeze(0)
            embed, trend_logits, vol_logits = analyser(w_an)
            trend_probs = F.softmax(trend_logits, dim=-1).squeeze(0).cpu().numpy()
            vol_probs = F.softmax(vol_logits, dim=-1).squeeze(0).cpu().numpy()
            day_ret_pct = (day_arrays[sym]["close"][n_steps - 1] / day_arrays[sym]["close"][0] - 1) * 100
            summaries[sym] = summarize_day(
                sym, trend_probs, vol_probs,
                day_arrays[sym]["rsi"][n_steps - 1], day_arrays[sym]["macd"][n_steps - 1],
                day_arrays[sym]["bb"][n_steps - 1], day_ret_pct,
            )
    advisor_text = build_advisor_report(summaries)
    advisor_text += f"\nEpoch {epoch} | Day {date} | Start=${START_CAPITAL:.2f} -> Final=${final_value:.2f} " \
                     f"({(final_value/START_CAPITAL-1)*100:+.1f}%) | Target ${TARGET_CAPITAL:.0f} " \
                     f"{'HIT' if final_value >= TARGET_CAPITAL else 'missed'}"

    plot_path = None
    if save_plot:
        plot_path = save_epoch_plot(
            epoch, date, day_arrays, env.history["net_worth"], pred_vs_actual, advisor_text, final_value,
        )

    return {
        "date": date,
        "final_value": final_value,
        "episode_reward": episode_reward,
        "predictor_loss": results.get("predictor_loss", 0.0),
        "analyser_loss": results.get("analyser_loss", 0.0),
        "policy_loss": results["ppo"]["policy_loss"],
        "value_loss": results["ppo"]["value_loss"],
        "entropy": results["ppo"]["entropy"],
        "advisor_text": advisor_text,
        "plot_path": plot_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--minutes", type=int, default=MINUTES_PER_DAY - 1,
                     help="steps per simulated day (use a small number for a fast smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--plot-every", type=int, default=1)
    ap.add_argument("--ckpt-dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints"))
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    hw, sizes = get_hw_and_sizes()
    print(hw.summary())
    print("Model sizes:", sizes)
    devices = hw.devices()
    print("Device map:", {k: (v if not isinstance(v, list) else [str(x) for x in v]) for k, v in devices.items()})

    market = MarketData(SYMBOLS)
    print(f"Loaded {len(market.available_days)} usable common trading days across {SYMBOLS}")

    per_asset_in_dim = 6 + 1 + sizes["embed"]   # raw feats + pred_return + analyser embed
    predictor = PricePredictor(hidden=sizes["hidden"], layers=sizes["layers"], pred_lookback=PRED_LOOKBACK).to(devices["predictor"])
    analyser = MarketAnalyser(hidden=sizes["hidden"], layers=sizes["layers"], embed=sizes["embed"]).to(devices["analyser"])
    trader = TradeMaker(per_asset_in_dim, portfolio_dim=PORTFOLIO_DIM, hidden=sizes["ppo_hidden"]).to(devices["trader"])

    predictor_opt = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    analyser_opt = torch.optim.Adam(analyser.parameters(), lr=1e-3)
    ppo = PPOTrainer(trader, devices["trader"], lr=3e-4, batch_size=sizes["hidden"])

    models = {"predictor": predictor, "analyser": analyser, "trader": trader}
    opts = {"predictor": predictor_opt, "analyser": analyser_opt, "ppo": ppo}

    history = {k: [] for k in [
        "epoch", "final_value", "episode_reward", "predictor_loss", "analyser_loss",
        "policy_loss", "value_loss",
    ]}

    hits = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        out = run_epoch(epoch, market, models, opts, devices, rng, args.minutes,
                         save_plot=(epoch % args.plot_every == 0))
        hits += int(out["final_value"] >= TARGET_CAPITAL)
        for k in history:
            history[k].append(out[k] if k != "epoch" else epoch)

        print(f"[epoch {epoch:4d}] day={out['date']} final=${out['final_value']:.2f} "
              f"reward={out['episode_reward']:.3f} pred_loss={out['predictor_loss']:.5f} "
              f"an_loss={out['analyser_loss']:.4f} pol_loss={out['policy_loss']:.4f} "
              f"hit_rate={hits/epoch:.2%} elapsed={time.time()-t0:.1f}s")

        if epoch % 5 == 0:
            save_training_curves(history)

        if epoch % args.checkpoint_every == 0:
            torch.save({
                "predictor": predictor.state_dict(),
                "analyser": analyser.state_dict(),
                "trader": trader.state_dict(),
                "epoch": epoch,
            }, os.path.join(args.ckpt_dir, f"ckpt_epoch_{epoch}.pt"))

    save_training_curves(history)
    print("Training complete.")


if __name__ == "__main__":
    main()
