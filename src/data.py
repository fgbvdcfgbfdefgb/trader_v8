"""
Offline data access layer.

Everything here reads only from local files in data/*.parquet + data/manifest.json
that are committed to the repo. No network calls -> safe for an air-gapped
Snowflake / training box that only did a single `git clone` at setup time.
"""
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .indicators import add_indicators, FEATURE_COLUMNS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "LTCUSDT"]

LOOKBACK = 120          # minutes of history fed to the Market Analyser
PRED_LOOKBACK = 30       # minutes of history fed to the Price Predictor
MINUTES_PER_DAY = 1440


class MarketData:
    """Loads all symbols once, keeps them resident in memory (they're tiny)."""

    def __init__(self, symbols: List[str] = None, data_dir: str = DATA_DIR):
        self.symbols = symbols or SYMBOLS
        self.data_dir = data_dir
        self.frames: Dict[str, pd.DataFrame] = {}
        with open(os.path.join(data_dir, "manifest.json")) as f:
            manifest = json.load(f)

        for sym in self.symbols:
            path = os.path.join(data_dir, f"{sym}_1m.parquet")
            df = pd.read_parquet(path)
            df = add_indicators(df)
            df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
            self.frames[sym] = df.set_index("timestamp")

        # Only use days that are "complete" for *every* symbol so a single
        # episode can trade all three coins simultaneously.
        common_days = set(manifest[self.symbols[0]])
        for sym in self.symbols[1:]:
            common_days &= set(manifest[sym])
        self.available_days = sorted(common_days)
        if not self.available_days:
            raise RuntimeError("No common complete trading day across all symbols.")

    def sample_day(self, rng: random.Random = None) -> str:
        rng = rng or random
        return rng.choice(self.available_days)

    def get_day_arrays(self, date: str):
        """
        Returns, for each symbol:
          - raw_window_feed: (MINUTES_PER_DAY, LOOKBACK, F) float32 array where
            row t is the feature window ending at minute t (causal, no lookahead)
          - close_prices: (MINUTES_PER_DAY,) float32 raw close price path
          - future_return_30: (MINUTES_PER_DAY,) realized forward 30-min return,
            used ONLY after the fact for the analyser/predictor auxiliary losses.
        """
        out = {}
        for sym in self.symbols:
            df = self.frames[sym]
            day_df = df[df["date"] == date]
            idx = day_df.index[0]
            loc = df.index.get_loc(idx)
            start = max(0, loc - LOOKBACK)
            end = loc + MINUTES_PER_DAY
            ctx = df.iloc[start:end]
            feats = ctx[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

            # pad front if not enough lookback history available
            pad = LOOKBACK - (loc - start)
            if pad > 0:
                feats = np.vstack([np.repeat(feats[:1], pad, axis=0), feats])

            windows = np.zeros((MINUTES_PER_DAY, LOOKBACK, feats.shape[1]), dtype=np.float32)
            for t in range(MINUTES_PER_DAY):
                windows[t] = feats[t: t + LOOKBACK]

            close = ctx["close"].to_numpy(dtype=np.float32)[-MINUTES_PER_DAY:]
            ret1 = ctx["ret_1"].to_numpy(dtype=np.float32)[-MINUTES_PER_DAY:]
            volatility = ctx["volatility_20"].to_numpy(dtype=np.float32)[-MINUTES_PER_DAY:]
            rsi = ctx["rsi_14"].to_numpy(dtype=np.float32)[-MINUTES_PER_DAY:]
            macd = ctx["macd"].to_numpy(dtype=np.float32)[-MINUTES_PER_DAY:]
            bb = ctx["bb_pct"].to_numpy(dtype=np.float32)[-MINUTES_PER_DAY:]
            momentum = ctx["momentum_10"].to_numpy(dtype=np.float32)[-MINUTES_PER_DAY:]

            fwd = np.zeros(MINUTES_PER_DAY, dtype=np.float32)
            closes_full = ctx["close"].to_numpy(dtype=np.float32)
            base_loc = len(closes_full) - MINUTES_PER_DAY
            for t in range(MINUTES_PER_DAY):
                j = base_loc + t
                j2 = min(j + 30, len(closes_full) - 1)
                fwd[t] = (closes_full[j2] - closes_full[j]) / (closes_full[j] + 1e-9)

            out[sym] = dict(
                windows=windows, close=close, ret1=ret1, future_return_30=fwd,
                volatility=volatility, rsi=rsi, macd=macd, bb=bb, momentum=momentum,
            )
        return out
