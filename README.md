# trader_v8

A fully-offline-trainable multi-agent reinforcement learning system that trades
**BTC, ETH and LTC** (USDT pairs) on simulated 1-minute bars. Three networks
train jointly every epoch and lean on each other's outputs:

| Agent | Role | Training signal |
|---|---|---|
| **PricePredictor** | LSTM that forecasts the next 1-minute return per asset | Supervised MSE vs. the realized next-minute return |
| **MarketAnalyser** | LSTM encoder that produces a "regime" embedding per asset + trend/volatility classification | Supervised cross-entropy vs. realized 30-min-forward trend & volatility buckets |
| **TradeMaker** | PPO actor-critic that decides a target portfolio allocation over `{cash, BTC, ETH, LTC}` every minute, using the (detached) outputs of the two agents above plus raw technical features and current portfolio state | PPO clipped surrogate objective on portfolio P&L |

All three have **separate optimizers** and are updated in parallel threads
after every simulated trading day, so on a multi-GPU box each agent can sit on
its own GPU and genuinely train concurrently (see "Hardware scaling" below).

## Why this is fully offline-trainable

Everything the training loop reads (`data/*.parquet`, `data/manifest.json`)
is already committed to this repo. `src/train.py` makes **zero network
calls**. That means:

- On Snowflake (or any sandbox that can only `pip install` + clone this repo
  once, with no further git pull/push and no outbound internet during
  training), you just: clone → `pip install -r requirements.txt` → `python -m
  src.train`. Nothing else is ever fetched.
- The only script that needs internet is `scripts/download_data.py`, and it is
  **not** part of the training path — it's a separate, one-time utility you'd
  run locally if you ever want to refresh/extend the dataset before pushing a
  new commit.

## Data

- Source: Binance's free public historical archive (`data.binance.vision`),
  official 1-minute kline files, no API key required.
- Symbols: `BTCUSDT`, `ETHUSDT`, `LTCUSDT`.
- Range: 12 months of 1-minute bars (~525,600 rows/symbol), stored as
  Zstandard-compressed Parquet (~40MB total for all three symbols).
- `data/manifest.json` lists, per symbol, every calendar day that has a
  (near-)complete 1440-minute bar count. `MarketData` only samples days that
  are complete for **all three** symbols simultaneously, so every episode can
  trade all three coins on the same calendar day.
- To regenerate/extend the dataset later: `python scripts/download_data.py
  --months 12 && python scripts/process_data.py`.

## Environment / reward design

- Each **epoch = one randomly-sampled trading day** (re-sampled every epoch,
  per the spec), simulated minute-by-minute (1440 steps).
- Portfolio starts at **$20** cash.
- Action: a target allocation vector over `[cash, BTC, ETH, LTC]` (softmax of
  the policy's sampled logits), rebalanced every minute with an 8bps
  transaction-fee drag to discourage pointless churn.
- Reward = log return of portfolio value each step, plus a terminal
  shaping bonus/penalty based on whether the **$30 (1.5x) target** was hit by
  day's end, plus an early-termination penalty if the portfolio collapses
  below $3 (bankruptcy guard).

> **Reality check:** a reliable +50% intraday return is an extremely
> aggressive target — no real trading system can guarantee it, and this
> reward is a training objective/game for the RL agent, not a promise of real
> profit. Treat any live deployment of the resulting policy as experimental
> and high-risk; backtest thoroughly and never trade money you can't afford
> to lose.

## Hardware auto-scaling (`src/config.py`)

`HW.tier` inspects `torch.cuda.device_count()` / per-GPU memory and picks a
model-size preset + device map automatically, no manual config needed:

| Detected hardware | Tier | Behavior |
|---|---|---|
| No GPU | `tiny` | Small nets, CPU only — good for smoke tests |
| 1 GPU (e.g. a single T4) | `small` | All three agents share the one GPU, sized to fit |
| 2+ GPUs (e.g. your 4x T4 Clore box) | `medium` | PricePredictor → GPU0, MarketAnalyser → GPU1, TradeMaker → GPU2, remaining GPU(s) reserved for parallel day-rollouts to speed up data collection |

## Repo layout

```
data/                    1y of 1m BTC/ETH/LTC parquet + manifest.json (offline dataset)
scripts/
  download_data.py       (internet-requiring, run locally, NOT used in training)
  process_data.py        (raw zip -> parquet + manifest, run locally)
src/
  config.py              hardware detection + model sizing + device map
  data.py                offline data loader / day sampler / feature windows
  indicators.py          technical indicators (SMA/EMA/RSI/MACD/Bollinger/vol/momentum)
  models.py              PricePredictor, MarketAnalyser, TradeMaker nn.Modules
  env.py                 TradingEnv ($20 start, $30 target, fees, bankruptcy guard)
  ppo.py                 PPO trainer (GAE, clipped surrogate) for TradeMaker
  advisor.py             rule-based text commentary from the analyser's outputs
  plotting.py            per-epoch PNG + running training_curves.png
  train.py               main orchestrator / entrypoint
outputs/
  epoch_plots/           auto-saved PNG per epoch (price action, portfolio curve,
                          predictor accuracy, advisor's text commentary)
  checkpoints/           periodic .pt checkpoints of all 3 networks
  training_curves.png    running loss/reward curves across all epochs so far
```

## Running it

```bash
pip install -r requirements.txt

# full run: 1440-minute days, e.g. 500 epochs
python -m src.train --epochs 500

# fast smoke test (short days, just to verify the pipeline runs)
python -m src.train --epochs 3 --minutes 60
```

Useful flags: `--epochs`, `--minutes` (steps per simulated day, default full
1440), `--seed`, `--checkpoint-every`, `--plot-every`, `--ckpt-dir`.

Every epoch prints a one-line summary (sampled day, final portfolio value,
reward, all 3 agents' losses, running hit-rate of reaching $30) and writes
`outputs/epoch_plots/epoch_XXXXX_<date>.png`. `outputs/training_curves.png` is
refreshed every 5 epochs with the full history.

## Setting this up on Snowflake (offline training)

1. In Snowsight, create a **Git Repository** integration pointing at
   `https://github.com/<you>/trader_v8` (public, so no secret/token needed)
   and create a Notebook / Container Runtime workspace from it. This is the
   one-time clone — no further `git pull`/`push` is required or used.
2. In that workspace: `pip install -r requirements.txt`.
3. Run `python -m src.train --epochs <N>`. Everything it touches
   (`data/*.parquet`, code) is already inside the cloned workspace — no
   outbound network calls happen during training.
4. Periodically download `outputs/` (checkpoints + PNGs) from the Snowflake
   workspace through its UI/stage, since there's no git push path back out.

## Setting this up on your Clore.ai 4x T4 box

```bash
ssh root@n1.us.clorecloud.net -p 2502
git clone https://github.com/<you>/trader_v8.git
cd trader_v8
pip install -r requirements.txt
python -m src.train --epochs 1000
```

With 4 GPUs detected, `src/config.py` automatically places PricePredictor,
MarketAnalyser and TradeMaker on 3 separate GPUs (4th GPU reserved for
rollout parallelism), and upsizes the networks to the `medium` preset. This
does not touch or interrupt any other process (e.g. mining) already running
on the box — it only allocates its own CUDA contexts on the GPUs it finds.

## Notes / next steps

- The current rollout loop processes one minute at a time per day (true
  sequential decision process); for higher GPU utilization at scale you can
  batch several simulated days together (`batch_days_parallel` in
  `config.MODEL_SIZES`) by running N environments in a vectorized step — the
  single-env version here is the correctness-first baseline.
- `TARGET_CAPITAL` / `START_CAPITAL` / fee / bankruptcy floor are constants in
  `src/env.py` if you want to tune the reward shaping.
