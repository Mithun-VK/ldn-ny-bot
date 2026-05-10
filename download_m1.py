import requests, struct, lzma, pandas as pd
from datetime import datetime, timedelta, timezone
import os, sys

SYMBOL = "EURUSD"
START  = datetime(2022, 1, 1, tzinfo=timezone.utc)
END    = datetime(2024, 12, 31, tzinfo=timezone.utc)
OUT    = r"D:\ldn-ny-bot\data\EURUSD_M1.csv"

def fetch_hour(symbol, dt):
    url = (f"https://datafeed.dukascopy.com/datafeed/{symbol}/"
           f"{dt.year}/{dt.month-1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5")
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200 or len(r.content) < 10:
            return []
        data = lzma.decompress(r.content)
        ticks = []
        for i in range(0, len(data), 20):
            chunk = data[i:i+20]
            if len(chunk) < 20: break
            ms, ask, bid, avol, bvol = struct.unpack(">IIIff", chunk)
            ts = dt + timedelta(milliseconds=ms)
            ticks.append((ts, (ask + bid) / 200000))
        return ticks
    except Exception:
        return []

rows, cur = [], START
total_hours = int((END - START).total_seconds() / 3600)
done = 0

while cur <= END:
    if cur.weekday() < 5:
        ticks = fetch_hour(SYMBOL, cur)
        if ticks:
            df = pd.DataFrame(ticks, columns=["time", "mid"])
            df = df.set_index("time").resample("1min")["mid"].ohlc()
            rows.append(df)
    cur += timedelta(hours=1)
    done += 1
    if done % 720 == 0:
        pct = done / total_hours * 100
        print(f"  {pct:.0f}% — {done}/{total_hours} hrs | bars: {sum(len(r) for r in rows):,}", flush=True)

final = pd.concat(rows).sort_index()
final = final[~final.index.duplicated()]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
final.to_csv(OUT)
print(f"\nDone! Saved {len(final):,} M1 bars → {OUT}")