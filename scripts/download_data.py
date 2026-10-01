"""
ONE-TIME, INTERNET-REQUIRING step. Run this locally (NOT on your offline
Snowflake / training box) whenever you want to refresh/extend the dataset.

Downloads official Binance monthly 1-minute kline archives (free, public,
no API key needed) for BTCUSDT / ETHUSDT / LTCUSDT, then call
scripts/process_data.py to turn them into the compact Parquet files that
actually get committed to the repo and used for offline training.

Usage:
    python scripts/download_data.py --months 12
    python scripts/process_data.py
"""
import argparse
import datetime
import os
import urllib.request

SYMBOLS = ["BTCUSDT", "ETHUSDT", "LTCUSDT"]
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw_dl")


def month_range(n_months, end=None):
    end = end or datetime.date.today().replace(day=1) - datetime.timedelta(days=1)
    y, m = end.year, end.month
    months = []
    for _ in range(n_months):
        months.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return list(reversed(months))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    args = ap.parse_args()

    months = month_range(args.months)
    print("Fetching months:", months)

    for sym in args.symbols:
        out_dir = os.path.join(RAW_DIR, sym)
        os.makedirs(out_dir, exist_ok=True)
        for mo in months:
            fname = f"{sym}-1m-{mo}.zip"
            out_path = os.path.join(out_dir, fname)
            if os.path.exists(out_path):
                continue
            url = f"{BASE_URL}/{sym}/1m/{fname}"
            try:
                urllib.request.urlretrieve(url, out_path)
                print("downloaded", url)
            except Exception as e:
                print("skip (not available yet):", url, e)


if __name__ == "__main__":
    main()
