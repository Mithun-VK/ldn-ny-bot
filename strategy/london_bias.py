"""
strategy/london_bias.py — Step 1: London Session Directional Bias
==================================================================
Enhanced bias detection using 5 independent confluence signals:

  S1. London open vs close direction          (primary direction)
  S2. Price position vs H1 EMA-20             (trend filter)
  S3. NY open price vs London session midpoint (strongest directional predictor)
  S4. Asian session liquidity sweep direction  (smart money sweep filter)
  S5. H4 higher timeframe trend alignment      (macro bias filter)

Each signal contributes a weighted score → final strength 0.0–1.0.
Bias is only returned if strength >= MIN_BIAS_STRENGTH (default 0.40).

Reference: 2,839-day NQ backtest showed NY Open vs London Midpoint
alone achieves 80–87% directional accuracy when used as primary filter.
"""

import logging
import datetime
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

import mt5_client

log = logging.getLogger("london_bias")

UTC          = datetime.timezone.utc
LONDON_OPEN  = "07:00"
LONDON_CLOSE = "13:14"
ASIA_OPEN    = "00:00"
ASIA_CLOSE   = "06:59"

# Minimum composite strength to return a bias (below this → None)
MIN_BIAS_STRENGTH = 0.40

# Signal weights (must sum to 1.0)
W_OPEN_CLOSE   = 0.20   # S1: London direction
W_EMA          = 0.15   # S2: EMA position
W_NY_MIDPOINT  = 0.35   # S3: NY open vs London mid (strongest signal)
W_ASIA_SWEEP   = 0.20   # S4: Asian liquidity sweep direction
W_HTF_TREND    = 0.10   # S5: H4 alignment


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL FUNCTIONS
# Each returns (direction: "BULL"|"BEAR"|None, confidence: 0.0–1.0)
# ─────────────────────────────────────────────────────────────────────────────

def _s1_open_close(london: pd.DataFrame) -> tuple[Optional[str], float]:
    """
    S1: Compare London session open price vs last close before NY.
    Confidence = normalized magnitude of the move vs session range.
    """
    if len(london) < 3:
        return None, 0.0

    open_p  = london["open"].iloc[0]
    close_p = london["close"].iloc[-1]
    session_range = london["high"].max() - london["low"].min()

    move = close_p - open_p
    if session_range == 0:
        return None, 0.0

    confidence = min(abs(move) / session_range, 1.0)

    if close_p > open_p:
        return "BULL", confidence
    elif close_p < open_p:
        return "BEAR", confidence
    return None, 0.0


def _s2_ema_position(df: pd.DataFrame, symbol: str) -> tuple[Optional[str], float]:
    """
    S2: Is the last London close above or below the H1 EMA-20?
    Confidence = normalized distance from EMA as fraction of ATR.
    """
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    london = df.between_time(LONDON_OPEN, LONDON_CLOSE)
    if len(london) == 0:
        return None, 0.0

    close_p  = london["close"].iloc[-1]
    ema_val  = ema20.iloc[-1]
    atr      = df["atr"].dropna().iloc[-1] if "atr" in df.columns else 0.0001

    dist = close_p - ema_val
    confidence = min(abs(dist) / (atr * 2), 1.0) if atr > 0 else 0.0

    if close_p > ema_val:
        return "BULL", confidence
    elif close_p < ema_val:
        return "BEAR", confidence
    return None, 0.0


def _s3_ny_open_vs_london_mid(df: pd.DataFrame) -> tuple[Optional[str], float]:
    """
    S3: NY Open (13:30 UTC) price position relative to London session midpoint.
    This is the highest-accuracy single filter — 80-87% directional accuracy
    on 2,839 days of backtested NQ data.

    Above midpoint → BULL continuation expected.
    Below midpoint → BEAR continuation expected.
    Confidence = normalized distance from midpoint as fraction of London range.
    """
    london = df.between_time(LONDON_OPEN, LONDON_CLOSE)
    if len(london) < 3:
        return None, 0.0

    london_high = london["high"].max()
    london_low  = london["low"].min()
    london_mid  = (london_high + london_low) / 2
    london_range = london_high - london_low

    # NY open price: use the open of the first bar at or after 13:30
    post_london = df[df.index.time >= datetime.time(13, 30)]
    if len(post_london) == 0:
        return None, 0.0

    ny_open_price = post_london["open"].iloc[0]

    dist = ny_open_price - london_mid
    confidence = min(abs(dist) / (london_range / 2), 1.0) if london_range > 0 else 0.0

    if ny_open_price > london_mid:
        return "BULL", confidence
    elif ny_open_price < london_mid:
        return "BEAR", confidence
    return None, 0.0


def _s4_asian_sweep(df: pd.DataFrame) -> tuple[Optional[str], float]:
    """
    S4: Which side of the Asian range did London sweep?

    If London sweeps ONLY the Asian LOW  → expect BULL move into NY.
    If London sweeps ONLY the Asian HIGH → expect BEAR move into NY.
    If London sweeps BOTH or NEITHER     → ambiguous, return None.

    This is the ICT/SMC liquidity sweep logic: London manipulates one
    side to collect stops, then the real move goes the other direction.
    Confidence is 1.0 (binary clean sweep) or 0.5 (partial/both).
    """
    asian  = df.between_time(ASIA_OPEN, ASIA_CLOSE)
    london = df.between_time(LONDON_OPEN, LONDON_CLOSE)

    if len(asian) < 2 or len(london) < 2:
        return None, 0.0

    asia_high = asian["high"].max()
    asia_low  = asian["low"].min()

    london_high = london["high"].max()
    london_low  = london["low"].min()

    swept_high = london_high > asia_high
    swept_low  = london_low  < asia_low

    if swept_low and not swept_high:
        # London swept Asia LOW → real direction is BULL
        return "BULL", 1.0
    elif swept_high and not swept_low:
        # London swept Asia HIGH → real direction is BEAR
        return "BEAR", 1.0
    elif swept_high and swept_low:
        # Both swept → choppy, no clean bias
        return None, 0.0
    else:
        # Neither swept → London stayed inside Asia range (very low volatility)
        return None, 0.3


def _s5_htf_trend(symbol: str) -> tuple[Optional[str], float]:
    """
    S5: H4 higher timeframe trend alignment.
    Uses the relationship of H4 EMA-20 vs EMA-50.
    EMA-20 > EMA-50 → macro BULL alignment.
    EMA-20 < EMA-50 → macro BEAR alignment.
    Confidence = normalized distance between EMAs.
    """
    df_h4 = mt5_client.get_bars(symbol, mt5.TIMEFRAME_H4, 60)
    if df_h4 is None or len(df_h4) < 55:
        return None, 0.0

    ema20 = df_h4["close"].ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = df_h4["close"].ewm(span=50, adjust=False).mean().iloc[-1]
    atr   = df_h4["atr"].dropna().iloc[-1] if "atr" in df_h4.columns else 0.001

    dist = ema20 - ema50
    confidence = min(abs(dist) / (atr * 3), 1.0) if atr > 0 else 0.0

    if ema20 > ema50:
        return "BULL", confidence
    elif ema20 < ema50:
        return "BEAR", confidence
    return None, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORER
# ─────────────────────────────────────────────────────────────────────────────
def _composite_bias(symbol: str) -> tuple[Optional[str], float, dict]:
    """
    Runs all 5 signals, computes a weighted composite score,
    returns (bias, strength, breakdown_dict).

    Breakdown dict is useful for logging and future agent memory feeds.
    """
    df = mt5_client.get_bars(symbol, mt5.TIMEFRAME_H1, 50)
    if df is None:
        return None, 0.0, {}

    london = df.between_time(LONDON_OPEN, LONDON_CLOSE)

    # Run all signals
    d1, c1 = _s1_open_close(london)
    d2, c2 = _s2_ema_position(df, symbol)
    d3, c3 = _s3_ny_open_vs_london_mid(df)
    d4, c4 = _s4_asian_sweep(df)
    d5, c5 = _s5_htf_trend(symbol)

    signals = [
        ("S1_OPEN_CLOSE",  d1, c1, W_OPEN_CLOSE),
        ("S2_EMA",         d2, c2, W_EMA),
        ("S3_NY_MID",      d3, c3, W_NY_MIDPOINT),
        ("S4_ASIA_SWEEP",  d4, c4, W_ASIA_SWEEP),
        ("S5_HTF",         d5, c5, W_HTF_TREND),
    ]

    bull_score = 0.0
    bear_score = 0.0
    breakdown  = {}

    for name, direction, confidence, weight in signals:
        breakdown[name] = {"direction": direction, "confidence": round(confidence, 3)}
        if direction == "BULL":
            bull_score += weight * confidence
        elif direction == "BEAR":
            bear_score += weight * confidence

    total = bull_score + bear_score
    if total == 0:
        return None, 0.0, breakdown

    if bull_score > bear_score:
        strength = bull_score / (bull_score + bear_score)
        return "BULL", round(strength, 3), breakdown
    elif bear_score > bull_score:
        strength = bear_score / (bull_score + bear_score)
        return "BEAR", round(strength, 3), breakdown

    return None, 0.0, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def get_bias(symbol: str) -> Optional[str]:
    """
    Returns "BULL", "BEAR", or None.
    None means no clear consensus across the 5 signals.
    """
    bias, strength, _ = _composite_bias(symbol)
    if bias is None or strength < MIN_BIAS_STRENGTH:
        log.debug("[%s] No bias: strength=%.3f < min=%.2f", symbol, strength, MIN_BIAS_STRENGTH)
        return None
    log.debug("[%s] Bias=%s strength=%.3f", symbol, bias, strength)
    return bias


def get_bias_strength(symbol: str) -> float:
    """
    Returns 0.0–1.0 composite strength score.
    Used by entry_signal.py F1 filter.
    """
    _, strength, _ = _composite_bias(symbol)
    return strength


def get_bias_full(symbol: str) -> dict:
    """
    Returns full diagnostic dict — useful for monitoring, dashboards,
    agent memory, or manual review.

    Example output:
    {
      "symbol": "EURUSD",
      "bias": "BULL",
      "strength": 0.74,
      "approved": True,
      "breakdown": {
        "S1_OPEN_CLOSE": {"direction": "BULL", "confidence": 0.62},
        "S2_EMA":        {"direction": "BULL", "confidence": 0.45},
        "S3_NY_MID":     {"direction": "BULL", "confidence": 0.88},
        "S4_ASIA_SWEEP": {"direction": "BULL", "confidence": 1.0},
        "S5_HTF":        {"direction": "BULL", "confidence": 0.31},
      }
    }
    """
    bias, strength, breakdown = _composite_bias(symbol)
    approved = bias is not None and strength >= MIN_BIAS_STRENGTH

    if approved:
        log.info(
            "📐 BIAS [%s] %s | Strength=%.3f | "
            "S1=%s(%.2f) S2=%s(%.2f) S3=%s(%.2f) S4=%s(%.2f) S5=%s(%.2f)",
            symbol, bias, strength,
            breakdown.get("S1_OPEN_CLOSE", {}).get("direction", "-"),
            breakdown.get("S1_OPEN_CLOSE", {}).get("confidence", 0),
            breakdown.get("S2_EMA",        {}).get("direction", "-"),
            breakdown.get("S2_EMA",        {}).get("confidence", 0),
            breakdown.get("S3_NY_MID",     {}).get("direction", "-"),
            breakdown.get("S3_NY_MID",     {}).get("confidence", 0),
            breakdown.get("S4_ASIA_SWEEP", {}).get("direction", "-"),
            breakdown.get("S4_ASIA_SWEEP", {}).get("confidence", 0),
            breakdown.get("S5_HTF",        {}).get("direction", "-"),
            breakdown.get("S5_HTF",        {}).get("confidence", 0),
        )

    return {
        "symbol"   : symbol,
        "bias"     : bias,
        "strength" : strength,
        "approved" : approved,
        "breakdown": breakdown,
    }