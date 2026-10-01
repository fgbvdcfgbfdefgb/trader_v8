"""
Auto-saves a PNG per epoch with:
  1. Normalized price paths for BTC/ETH/LTC over the sampled day
  2. Portfolio value curve ($20 start, $30 target line)
  3. Predicted vs. realized next-minute return (predictor accuracy) per asset
  4. The Market Analyser's text advisory for that day + epoch summary stats

Also maintains a running `training_curves.png` with loss/reward history across
all epochs so far.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import SYMBOLS
from .env import START_CAPITAL, TARGET_CAPITAL

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "epoch_plots")
os.makedirs(OUT_DIR, exist_ok=True)


def save_epoch_plot(epoch, date, day_arrays, net_worth_history, pred_vs_actual, advisor_text, final_value):
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), gridspec_kw={"height_ratios": [3, 2, 2]})

    ax = axes[0]
    for sym in SYMBOLS:
        closes = day_arrays[sym]["close"]
        norm = closes / closes[0] * 100
        ax.plot(norm, label=sym)
    ax.set_title(f"Epoch {epoch} | Day {date} | Normalized price (start=100)")
    ax.legend()
    ax.set_xlabel("minute of day")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(net_worth_history, color="green", label="Portfolio value ($)")
    ax2.axhline(START_CAPITAL, color="gray", linestyle="--", label=f"Start ${START_CAPITAL:.0f}")
    ax2.axhline(TARGET_CAPITAL, color="red", linestyle="--", label=f"Target ${TARGET_CAPITAL:.0f}")
    hit = "YES" if final_value >= TARGET_CAPITAL else "no"
    ax2.set_title(f"Portfolio value over the day | Final=${final_value:.2f} | Target hit: {hit}")
    ax2.set_xlabel("minute of day")
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    for sym in SYMBOLS:
        preds, actual = pred_vs_actual[sym]
        if len(preds) == 0:
            continue
        ax3.plot(actual, alpha=0.6, label=f"{sym} actual ret")
        ax3.plot(preds, alpha=0.6, linestyle="--", label=f"{sym} predicted ret")
    ax3.set_title("Price Predictor: predicted vs realized next-minute return")
    ax3.set_xlabel("minute of day")
    ax3.legend(fontsize=7, ncol=3)
    ax3.grid(alpha=0.3)

    fig.text(0.02, 0.01, advisor_text, fontsize=8, family="monospace", va="bottom",
              bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    path = os.path.join(OUT_DIR, f"epoch_{epoch:05d}_{date}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save_training_curves(history: dict):
    """history: dict of lists, e.g. {'epoch':[...], 'final_value':[...], 'reward':[...],
    'predictor_loss':[...], 'analyser_loss':[...], 'policy_loss':[...], 'value_loss':[...]}"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(history["epoch"], history["final_value"], marker=".", ms=2)
    axes[0, 0].axhline(TARGET_CAPITAL, color="red", linestyle="--")
    axes[0, 0].axhline(START_CAPITAL, color="gray", linestyle="--")
    axes[0, 0].set_title("Final portfolio value per epoch")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(history["epoch"], history["episode_reward"], marker=".", ms=2, color="purple")
    axes[0, 1].set_title("TradeMaker episode reward")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(history["epoch"], history["predictor_loss"], label="predictor", color="orange")
    axes[1, 0].plot(history["epoch"], history["analyser_loss"], label="analyser", color="teal")
    axes[1, 0].set_title("Predictor / Analyser training loss")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(history["epoch"], history["policy_loss"], label="policy_loss", color="red")
    axes[1, 1].plot(history["epoch"], history["value_loss"], label="value_loss", color="blue")
    axes[1, 1].set_title("PPO TradeMaker losses")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(OUT_DIR), "training_curves.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
