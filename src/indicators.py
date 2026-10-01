"""Vectorized technical indicators used by the Market Analyser & the Advisor."""
import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    df["ret_1"] = close.pct_change().fillna(0.0)
    df["sma_5"] = close.rolling(5, min_periods=1).mean()
    df["sma_20"] = close.rolling(20, min_periods=1).mean()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    mid = close.rolling(20, min_periods=1).mean()
    std = close.rolling(20, min_periods=1).std().fillna(0.0)
    upper = mid + 2 * std
    lower = mid - 2 * std
    df["bb_pct"] = ((close - lower) / (upper - lower + 1e-9)).clip(0, 1)

    df["volatility_20"] = df["ret_1"].rolling(20, min_periods=1).std().fillna(0.0)
    df["momentum_10"] = close.pct_change(10).fillna(0.0)

    df = df.fillna(0.0)
    return df


FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "ret_1", "sma_5", "sma_20", "ema_12", "ema_26",
    "macd", "macd_signal", "rsi_14", "bb_pct", "volatility_20", "momentum_10",
]


def normalize_window(window: np.ndarray) -> np.ndarray:
    """Z-score normalize each feature column within the lookback window."""
    mu = window.mean(axis=0, keepdims=True)
    sd = window.std(axis=0, keepdims=True) + 1e-6
    return (window - mu) / sd
