"""
strategy/entry_signal.py — Strategy 2: London-NY Overlap Momentum
==================================================================
Pipeline:
  Step 1 → london_bias.py    : Was London session directionally clear? (BULL/BEAR/None)
  Step 2 → ny_open_range.py  : What is the 15-min consolidation box at 13:30 UTC?
  Step 3 → THIS FILE         : Did price break the box in the bias direction?
                               Are all confluence filters satisfied?
                               Build and return a validated signal dict.

Entry filters (ALL must pass before signal is generated):
  F1. London bias confirmed (EMA-20 + open-close direction)
  F2. NY range box defined and non-trivial (> 0.3× ATR)
  F3. Price has broken the box in the bias direction (close beyond level)
  F4. Breakout candle body ≥ 50% of candle range (no wick-heavy indecision)
  F5. ATR is within normal range (not a news spike — < 1.8× ATR average)
  F6. Spread is acceptable (< 3× normal spread for this symbol)
  F7. Volume confirmation — breakout bar volume > 1.2× 20-bar average
  F8. No opposing position already open on this symbol
  F9. Signal is within the valid entry window (13:45–16:00 UTC)
  F10. Price has not already moved > 1.5× ATR from box edge (late entry guard)

Signal dict structure:
  {
    symbol      : str,
    direction   : "BUY" | "SELL",
    entry       : float,        # current ask/bid
    sl          : float,        # ATR-based stop
    tp          : float,        # RR-based target
    atr         : float,        # current ATR(14) on 15M
    atr_avg     : float,        # 20-period ATR average (for volatility gate)
    rr          : float,        # computed R:R
    bias_strength: float,       # London session directional strength (0.0–1.0)
    box_high    : float,
    box_low     : float,
    comment     : str,
    reason      : str,          # human-readable signal rationale
    filters_passed: list[str],  # list of filter codes that passed
  }
"""

import logging
import datetime
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

import config
import mt5_client
from strategy.london_bias import get_bias, get_bias_strength
from strategy.ny_open_range import get_range

log = logging.getLogger("entry_signal")

UTC = datetime.timezone.utc

# ── Entry window: only generate signals between 13:45 and 16:00 UTC
ENTRY_WINDOW_START = datetime.time(13, 45)
ENTRY_WINDOW_END   = datetime.time(16, 0)

# ── Minimum box size as a fraction of ATR
MIN_BOX_ATR_RATIO  = 0.3

# ── Late entry guard: if price already ran more than this × ATR from box, skip
MAX_CHASE_ATR_MULT = 1.5

# ── Breakout candle body must be ≥ this fraction of total candle range
MIN_BODY_RATIO     = 0.50

# ── Volume: breakout bar must exceed this × 20-bar avg volume
MIN_VOL_RATIO      = 1.2

# ── Spread: must be ≤ this × symbol's average spread
MAX_SPREAD_MULT    = 3.0

# ── ATR spike filter: current ATR must be ≤ this × ATR average
MAX_ATR_SPIKE_MULT = 1.8


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL FILTER FUNCTIONS
# Each returns (passed: bool, tag: str, detail: str)
# ─────────────────────────────────────────────────────────────────────────────

def _f1_london_bias(symbol: str) -> tuple[bool, str, str, Optional[str], float]:
    """F1: London session bias must be BULL or BEAR with sufficient strength."""
    bias     = get_bias(symbol)
    strength = get_bias_strength(symbol)
    if bias is None:
        return False, "F1_BIAS", "No clear London bias", None, 0.0
    if strength < 0.3:
        return False, "F1_BIAS_WEAK", f"Bias={bias} but strength={strength:.2f}<0.3", None, strength
    return True, "F1_BIAS", f"Bias={bias} strength={strength:.2f}", bias, strength


def _f2_box_valid(r_high: Optional[float], r_low: Optional[float],
                  atr: float) -> tuple[bool, str, str]:
    """F2: NY range box must exist and be large enough to be meaningful."""
    if r_high is None or r_low is None:
        return False, "F2_BOX_MISSING", "NY range not yet available"
    box_size = r_high - r_low
    if box_size <= 0:
        return False, "F2_BOX_ZERO", "Box size is zero"
    min_size = MIN_BOX_ATR_RATIO * atr
    if box_size < min_size:
        return False, "F2_BOX_TINY", f"Box={box_size:.5f} < {MIN_BOX_ATR_RATIO}×ATR={min_size:.5f}"
    return True, "F2_BOX", f"Box={box_size:.5f} ATR={atr:.5f} ratio={box_size/atr:.2f}"


def _f3_price_breakout(bias: str, price: float,
                        r_high: float, r_low: float) -> tuple[bool, str, str]:
    """F3: Current price must have broken the box in the bias direction."""
    if bias == "BULL":
        if price > r_high:
            return True, "F3_BREAKOUT", f"BULL break above {r_high:.5f} price={price:.5f}"
        return False, "F3_NO_BREAKOUT", f"price={price:.5f} <= box_high={r_high:.5f}"
    else:
        if price < r_low:
            return True, "F3_BREAKOUT", f"BEAR break below {r_low:.5f} price={price:.5f}"
        return False, "F3_NO_BREAKOUT", f"price={price:.5f} >= box_low={r_low:.5f}"


def _f4_body_quality(df: pd.DataFrame) -> tuple[bool, str, str]:
    """F4: The breakout candle body must be ≥ 50% of its total range (conviction candle)."""
    bar = df.iloc[-2]   # last closed bar
    candle_range = bar["high"] - bar["low"]
    if candle_range == 0:
        return False, "F4_ZERO_RANGE", "Zero-range candle"
    body = abs(bar["close"] - bar["open"])
    ratio = body / candle_range
    if ratio < MIN_BODY_RATIO:
        return False, "F4_WEAK_BODY", f"Body ratio={ratio:.2f} < {MIN_BODY_RATIO}"
    return True, "F4_BODY", f"Body ratio={ratio:.2f}"


def _f5_atr_normal(atr: float, atr_avg: float) -> tuple[bool, str, str]:
    """F5: ATR must not be spiking abnormally (indicates news event in progress)."""
    if atr_avg == 0:
        return True, "F5_ATR", "ATR avg unavailable — skipping check"
    ratio = atr / atr_avg
    if ratio > MAX_ATR_SPIKE_MULT:
        return False, "F5_ATR_SPIKE", f"ATR={atr:.5f} is {ratio:.1f}× avg={atr_avg:.5f}"
    return True, "F5_ATR", f"ATR ratio={ratio:.2f}× avg"


def _f6_spread_ok(symbol: str, tick) -> tuple[bool, str, str]:
    """F6: Live spread must be within acceptable limits."""
    info = mt5.symbol_info(symbol)
    if info is None or tick is None:
        return True, "F6_SPREAD", "Cannot check spread — allowing"
    live_spread = tick.ask - tick.bid
    # Approximate normal spread as the symbol's average spread in points
    avg_spread  = info.spread * info.point
    if avg_spread <= 0:
        return True, "F6_SPREAD", "Avg spread unavailable"
    ratio = live_spread / avg_spread
    if ratio > MAX_SPREAD_MULT:
        return False, "F6_SPREAD_WIDE", f"Spread={live_spread:.5f} is {ratio:.1f}× avg"
    return True, "F6_SPREAD", f"Spread ratio={ratio:.2f}×"


def _f7_volume_confirm(df: pd.DataFrame) -> tuple[bool, str, str]:
    """F7: Breakout bar volume must exceed 1.2× 20-bar average volume."""
    if "volume" not in df.columns:
        return True, "F7_VOLUME", "Volume data unavailable"
    bar   = df.iloc[-2]
    vol   = bar["volume"]
    avg_vol = df["volume"].iloc[-22:-2].mean()
    if avg_vol == 0:
        return True, "F7_VOLUME", "Zero avg volume"
    ratio = vol / avg_vol
    if ratio < MIN_VOL_RATIO:
        return False, "F7_LOW_VOL", f"Vol={vol:.0f} is {ratio:.2f}× avg={avg_vol:.0f}"
    return True, "F7_VOLUME", f"Vol ratio={ratio:.2f}×"


def _f8_no_open_position(symbol: str) -> tuple[bool, str, str]:
    """F8: No existing position open on this symbol."""
    pos = mt5.positions_get(symbol=symbol)
    if pos and len(pos) > 0:
        return False, "F8_DUPE_POS", f"Already {len(pos)} open position(s) on {symbol}"
    return True, "F8_NO_POS", "No open position"


def _f9_entry_window() -> tuple[bool, str, str]:
    """F9: Current UTC time must be within the valid entry window."""
    now = datetime.datetime.now(UTC).time()
    if ENTRY_WINDOW_START <= now <= ENTRY_WINDOW_END:
        return True, "F9_WINDOW", f"In window ({now.strftime('%H:%M')} UTC)"
    return False, "F9_OUTSIDE_WINDOW", f"Outside entry window ({now.strftime('%H:%M')} UTC)"


def _f10_not_late_entry(bias: str, price: float,
                         r_high: float, r_low: float,
                         atr: float) -> tuple[bool, str, str]:
    """F10: Price must not have already run too far from the box edge (chasing guard)."""
    if bias == "BULL":
        run = price - r_high
    else:
        run = r_low - price
    max_run = MAX_CHASE_ATR_MULT * atr
    if run > max_run:
        return False, "F10_LATE_ENTRY", f"Price ran {run:.5f} > {MAX_CHASE_ATR_MULT}×ATR={max_run:.5f}"
    return True, "F10_IN_RANGE", f"Run={run:.5f} within {MAX_CHASE_ATR_MULT}×ATR={max_run:.5f}"


# ─────────────────────────────────────────────────────────────────────────────
# SL / TP CALCULATION
# ─────────────────────────────────────────────────────────────────────────────
def _compute_sl_tp(bias: str, price: float, atr: float,
                   r_high: float, r_low: float) -> tuple[float, float]:
    """
    SL: config.ATR_MULT_SL × ATR beyond entry.
    TP: config.RR_RATIO × SL distance.

    For BUY:  SL is placed below the box low (tighter) or ATR-based, whichever is wider.
    For SELL: SL is placed above the box high (tighter) or ATR-based, whichever is wider.
    This ensures the stop is beyond the breakout structure, not arbitrarily in the air.
    """
    atr_stop = config.ATR_MULT_SL * atr

    if bias == "BULL":
        structure_stop = price - (price - r_low) * 1.1    # 10% beyond box low
        sl = min(price - atr_stop, structure_stop)         # wider stop wins
        tp = price + config.RR_RATIO * abs(price - sl)
    else:
        structure_stop = price + (r_high - price) * 1.1
        sl = max(price + atr_stop, structure_stop)
        tp = price - config.RR_RATIO * abs(sl - price)

    return round(sl, 5), round(tp, 5)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def get_signal(symbol: str) -> Optional[dict]:
    """
    Run all 10 filters for the given symbol.
    Returns a fully validated signal dict or None.

    All filters must pass. The function logs the first failure reason
    at DEBUG level so the log is not noisy during non-window hours.
    """

    # ── fetch data once, share across filters ──
    df_15m = mt5_client.get_bars(symbol, mt5.TIMEFRAME_M15, 60)
    if df_15m is None or len(df_15m) < 25:
        log.debug("[%s] Insufficient 15M data", symbol)
        return None

    atr     = df_15m["atr"].dropna().iloc[-1]
    atr_avg = df_15m["atr"].dropna().tail(20).mean()

    r_high, r_low = get_range(symbol)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.debug("[%s] No tick data", symbol)
        return None

    # ── run filters in order ──
    filters_passed = []
    bias    = None
    strength= 0.0

    # F1
    ok, tag, detail, bias, strength = _f1_london_bias(symbol)
    if not ok:
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return None
    filters_passed.append(tag)

    # price for direction checks
    price = tick.ask if bias == "BULL" else tick.bid

    # F2
    ok, tag, detail = _f2_box_valid(r_high, r_low, atr)
    if not ok:
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return None
    filters_passed.append(tag)

    # F3
    ok, tag, detail = _f3_price_breakout(bias, price, r_high, r_low)
    if not ok:
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return None
    filters_passed.append(tag)

    # F4
    ok, tag, detail = _f4_body_quality(df_15m)
    if not ok:
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return None
    filters_passed.append(tag)

    # F5
    ok, tag, detail = _f5_atr_normal(atr, atr_avg)
    if not ok:
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return None
    filters_passed.append(tag)

    # F6
    ok, tag, detail = _f6_spread_ok(symbol, tick)
    if not ok:
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return None
    filters_passed.append(tag)

    # F7
    ok, tag, detail = _f7_volume_confirm(df_15m)
    if not ok:
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return None
    filters_passed.append(tag)

    # F8
    ok, tag, detail = _f8_no_open_position(symbol)
    if not ok:
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return None
    filters_passed.append(tag)

    # F9
    ok, tag, detail = _f9_entry_window()
    if not ok:
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return None
    filters_passed.append(tag)

    # F10
    ok, tag, detail = _f10_not_late_entry(bias, price, r_high, r_low, atr)
    if not ok:
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return None
    filters_passed.append(tag)

    # ── all 10 filters passed → build signal ──
    sl, tp     = _compute_sl_tp(bias, price, atr, r_high, r_low)
    stop_dist  = abs(price - sl)
    target_dist= abs(price - tp)
    rr         = round(target_dist / stop_dist, 2) if stop_dist > 0 else 0

    direction = "BUY" if bias == "BULL" else "SELL"

    reason = (
        f"LDN {bias}(str={strength:.2f}) | "
        f"NY box [{r_low:.5f}-{r_high:.5f}] | "
        f"Break @ {price:.5f} | "
        f"ATR={atr:.5f} | RR={rr}"
    )

    signal = {
        "symbol"        : symbol,
        "direction"     : direction,
        "entry"         : round(price, 5),
        "sl"            : sl,
        "tp"            : tp,
        "atr"           : round(atr, 5),
        "atr_avg"       : round(atr_avg, 5),
        "rr"            : rr,
        "bias_strength" : round(strength, 3),
        "box_high"      : round(r_high, 5),
        "box_low"       : round(r_low, 5),
        "comment"       : f"LDN-NY {direction} {symbol}",
        "reason"        : reason,
        "filters_passed": filters_passed,
    }

    log.info(
        "✅ SIGNAL [%s] %s | %s | Filters: %s",
        direction, symbol, reason, ",".join(filters_passed)
    )

    return signal
