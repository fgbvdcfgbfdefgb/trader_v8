"""
Processes raw Binance monthly kline zip files into compact per-symbol Parquet
files, ready for offline RL training (no internet needed after this step).

Run ONCE locally (or anywhere with internet) before committing to GitHub.
On Snowflake / the training box, only the resulting data/*.parquet files are
ever read -> fully offline.
"""
import glob
import os
import zipfile
import io
import sys

import pandas as pd

COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw_dl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def detect_unit(sample_ts: int) -> str:
    digits = len(str(int(sample_ts)))
    if digits >= 19:
        return "ns"
    if digits >= 16:
        return "us"
    if digits >= 13:
        return "ms"
    return "s"


def load_symbol(symbol: str) -> pd.DataFrame:
    frames = []
    pattern = os.path.join(RAW_DIR, symbol, f"{symbol}-1m-*.zip")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No raw zip files found for {symbol} at {pattern}")
    for fp in files:
        with zipfile.ZipFile(fp) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as f:
                raw = f.read()
        # Some newer Binance files ship a header row, some don't.
        first_line = raw.split(b"\n", 1)[0].decode()
        has_header = not first_line.split(",")[0].strip().isdigit()
        df = pd.read_csv(
            io.BytesIO(raw),
            names=None if has_header else COLS,
            header=0 if has_header else None,
        )
        df.columns = [c.lower() for c in df.columns]
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    unit = detect_unit(full["open_time"].iloc[0])
    full["timestamp"] = pd.to_datetime(full["open_time"], unit=unit, utc=True)
    full = full[["timestamp", "open", "high", "low", "close", "volume", "num_trades"]]
    for c in ["open", "high", "low", "close", "volume"]:
        full[c] = full[c].astype("float32")
    full["num_trades"] = full["num_trades"].astype("int32")
    full = full.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return full


def main():
    symbols = sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "LTCUSDT"]
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {}
    for sym in symbols:
        print(f"Processing {sym} ...")
        df = load_symbol(sym)
        out_path = os.path.join(OUT_DIR, f"{sym}_1m.parquet")
        df.to_parquet(out_path, index=False, compression="zstd")
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  -> {out_path} ({len(df):,} rows, {size_mb:.2f} MB)")

        # Build list of "complete" trading days (>= 1430 of 1440 minutes present)
        days = df["timestamp"].dt.date
        counts = days.value_counts()
        good_days = sorted(str(d) for d, c in counts.items() if c >= 1430)
        manifest[sym] = good_days
        print(f"  usable complete days: {len(good_days)}")

    import json
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote data/manifest.json")


if __name__ == "__main__":
    main()
