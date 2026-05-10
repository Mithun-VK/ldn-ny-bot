"""
data_loader.py — Download all historical data from MT5 and save locally
=======================================================================
Downloads M1, M15, H1, H4, D1 bars for all backtest symbols.
Saves to data/ folder as CSV files for offline backtesting.

Run: python data_loader.py
     python data_loader.py --symbols EURUSD GBPUSD XAUUSD
     python data_loader.py --start 2022-01-01 --end 2024-12-31
"""

import os
import sys
import argparse
import datetime
import logging

import MetaTrader5 as mt5
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("data_loader")

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "XAUUSD"]
DEFAULT_START   = "2022-01-01"
DEFAULT_END     = "2024-12-31"
DATA_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TIMEFRAMES = {
    "M1" : mt5.TIMEFRAME_M1,
    "M5" : mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1" : mt5.TIMEFRAME_H1,
    "H4" : mt5.TIMEFRAME_H4,
    "D1" : mt5.TIMEFRAME_D1,
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    log.info("Data directory: %s", DATA_DIR)


def download(symbol: str, tf_name: str, tf_const,
             start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame | None:
    """
    Smart fetch strategy per timeframe:
    - M1: use copy_rates_from_pos (many brokers block range API for M1)
    - All others: use copy_rates_range (exact date range)
    """
    start_naive = start.replace(tzinfo=None) if start.tzinfo else start
    end_naive   = end.replace(tzinfo=None)   if end.tzinfo   else end

    log.info("  Downloading %s %s (%s → %s)...",
             symbol, tf_name,
             start_naive.strftime("%Y-%m-%d"),
             end_naive.strftime("%Y-%m-%d"))

    rates = None

    # M1/M5: brokers often block copy_rates_range for short TFs — use position-based fetch
    if tf_name in ("M1", "M5"):
        days  = (end_naive - start_naive).days + 30
        count = min(days * 24 * 60 + 1000, 2_000_000)
        log.info("  [M1] Using copy_rates_from_pos (count=%d)...", count)
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, count)
        if rates is None or len(rates) == 0:
            # Last resort: try copy_rates_from with the end date
            log.info("  [M1] Trying copy_rates_from with end date...")
            rates = mt5.copy_rates_from(symbol, tf_const, end_naive, count)
    else:
        rates = mt5.copy_rates_range(symbol, tf_const, start_naive, end_naive)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            log.warning("  copy_rates_range failed (err=%s). Trying fallback...", err)
            tf_minutes = {"M5":5,"M15":15,"H1":60,"H4":240,"D1":1440}
            days  = (end_naive - start_naive).days + 30
            count = min(int(days * 24 * 60 / tf_minutes.get(tf_name, 60)) + 500, 500_000)
            rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, count)

    if rates is None or len(rates) == 0:
        log.error("  ❌ No data for %s %s — broker may not provide this history depth.", symbol, tf_name)
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]

    # Filter strictly to requested date range
    start_ts = pd.Timestamp(start_naive, tz="UTC")
    end_ts   = pd.Timestamp(end_naive,   tz="UTC")
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]

    if len(df) == 0:
        log.warning("  ⚠️  %s %s: data fetched but 0 bars in requested range.", symbol, tf_name)
        log.warning("  Broker may only keep limited M1 history. Try a shorter date range.")
        return None

    log.info("  ✅ %s %s: %d bars", symbol, tf_name, len(df))
    return df


def save(df: pd.DataFrame, symbol: str, tf_name: str):
    fname = os.path.join(DATA_DIR, f"{symbol}_{tf_name}.csv")
    df.to_csv(fname)
    size_kb = os.path.getsize(fname) / 1024
    log.info("  💾 Saved: %s (%.1f KB)", fname, size_kb)


def verify_symbol(symbol: str) -> bool:
    info = mt5.symbol_info(symbol)
    if info is None:
        log.error("Symbol %s not found. Add it to Market Watch first.", symbol)
        return False
    if not info.visible:
        log.info("Making %s visible in Market Watch...", symbol)
        mt5.symbol_select(symbol, True)
    return True


def check_existing(symbol: str, tf_name: str) -> bool:
    fname = os.path.join(DATA_DIR, f"{symbol}_{tf_name}.csv")
    if os.path.exists(fname):
        size_kb = os.path.getsize(fname) / 1024
        log.info("  ⏭️  %s %s already exists (%.1f KB) — skipping. Use --force to re-download.",
                 symbol, tf_name, size_kb)
        return True
    return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MT5 Historical Data Downloader")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help=f"Symbols to download (default: {DEFAULT_SYMBOLS})")
    parser.add_argument("--start",   default=DEFAULT_START,
                        help="Start date YYYY-MM-DD (default: 2022-01-01)")
    parser.add_argument("--end",     default=DEFAULT_END,
                        help="End date YYYY-MM-DD (default: 2024-12-31)")
    parser.add_argument("--tf",      nargs="+", default=list(TIMEFRAMES.keys()),
                        help="Timeframes to download (default: M1 M15 H1 H4 D1)")
    parser.add_argument("--force",   action="store_true",
                        help="Re-download even if file already exists")
    args = parser.parse_args()

    start_dt = datetime.datetime.strptime(args.start, "%Y-%m-%d")
    end_dt   = datetime.datetime.strptime(args.end,   "%Y-%m-%d")

    # ── Connect to MT5
    if not mt5.initialize():
        log.critical("MT5 initialization failed. Is MetaTrader 5 open and logged in?")
        sys.exit(1)

    info = mt5.terminal_info()
    log.info("Connected to MT5: %s | Build %s", info.name, info.build)

    ensure_dirs()

    total_ok, total_fail = 0, 0

    for symbol in args.symbols:
        log.info("")
        log.info("=" * 55)
        log.info("  Symbol: %s", symbol)
        log.info("=" * 55)

        if not verify_symbol(symbol):
            total_fail += len(args.tf)
            continue

        for tf_name in args.tf:
            if tf_name not in TIMEFRAMES:
                log.warning("Unknown timeframe: %s — skipping", tf_name)
                continue

            if not args.force and check_existing(symbol, tf_name):
                total_ok += 1
                continue

            df = download(symbol, tf_name, TIMEFRAMES[tf_name], start_dt, end_dt)
            if df is not None and len(df) > 0:
                save(df, symbol, tf_name)
                total_ok += 1
            else:
                total_fail += 1

    mt5.shutdown()

    log.info("")
    log.info("=" * 55)
    log.info("  Download complete: %d OK | %d Failed", total_ok, total_fail)
    log.info("  Data saved to: %s", DATA_DIR)
    log.info("=" * 55)

    if total_fail > 0:
        log.warning("Some downloads failed. Check that:")
        log.warning("  1. Symbol is visible in MT5 Market Watch")
        log.warning("  2. MT5 is connected to broker (not offline)")
        log.warning("  3. Broker provides history for requested date range")


if __name__ == "__main__":
    main()
