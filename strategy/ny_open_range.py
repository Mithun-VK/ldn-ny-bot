"""
strategy/ny_open_range.py — Step 2: NY Open Range Detection
============================================================
Captures and validates the 15-minute consolidation box formed at the
New York session open (13:30–13:45 UTC / 19:00–19:15 IST).

This box is the trigger structure for Strategy 2. A breakout above or
below this box — in the direction of the London bias — is the entry signal.

Public API:
  get_range(symbol)         → (high, low) or (None, None)
  get_range_quality(symbol) → RangeQuality dataclass with full diagnostics

Range quality checks:
  Q1. Range window is complete (current time > 13:45 UTC)
  Q2. Sufficient bars exist in the range window
  Q3. Range size is non-trivial (> MIN_PIPS minimum)
  Q4. Range is compact relative to ADR (not an already-extended range)
  Q5. No major wicks protruding beyond 1.5× body (not a spiked range)
  Q6. Range has not been breached already in both directions (not a fake box)
  Q7. Range is fresh — detected within the valid usage window (< 2.5 hours old)
"""

import logging
import datetime
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import pandas as pd

import mt5_client

log = logging.getLogger("ny_range")

UTC = datetime.timezone.utc
IST = ZoneInfo("Asia/Kolkata")

# ── NY session range window (UTC)
RANGE_START_UTC = datetime.time(13, 30)
RANGE_END_UTC   = datetime.time(13, 45)

# ── Range is valid for entry only within this window after close
RANGE_VALID_UNTIL_UTC = datetime.time(16, 0)

# ── Minimum range size in pips (symbol-dependent — see _min_pips)
_MIN_PIPS_DEFAULT = 5    # default for major FX pairs
_MIN_PIPS_GOLD    = 50   # XAUUSD has larger pip values
_MIN_PIPS_INDICES = 10   # US30, NAS100 etc.

# ── Max range as fraction of 20-day ADR — blocks extended ranges
MAX_RANGE_ADR_RATIO = 0.35

# ── Wick quality: total wick height must be < this × body height per bar
MAX_WICK_BODY_RATIO = 1.5


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RangeQuality:
    """Full diagnostic report for the NY open range."""
    symbol       : str
    high         : Optional[float] = None
    low          : Optional[float] = None
    size         : float = 0.0          # high - low in price
    size_pips    : float = 0.0          # size in pips
    adr_ratio    : float = 0.0          # range size / 20-day ADR
    bar_count    : int   = 0            # number of 1M bars in range window
    wick_ratio   : float = 0.0          # avg wick/body ratio in range bars
    is_valid     : bool  = False
    failure_reason: str  = ""
    checks_passed : list = field(default_factory=list)
    computed_at   : str  = ""


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _point(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    return info.point if info else 0.00001


def _min_pips(symbol: str) -> float:
    s = symbol.upper()
    if "XAU" in s or "GOLD" in s:
        return _MIN_PIPS_GOLD
    if any(x in s for x in ["US30", "NAS", "SPX", "GER", "UK100"]):
        return _MIN_PIPS_INDICES
    return _MIN_PIPS_DEFAULT


def _pips(symbol: str, price_diff: float) -> float:
    """Convert a price difference to pip value for the given symbol."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return price_diff / 0.0001
    # For JPY pairs: 1 pip = 0.01; for most others: 1 pip = 0.0001
    pip_size = info.point * (10 if "JPY" in symbol.upper() else 1)
    return price_diff / pip_size if pip_size > 0 else 0.0


def _compute_adr(symbol: str, days: int = 20) -> float:
    """Average Daily Range over last N days (using D1 bars)."""
    df = mt5_client.get_bars(symbol, mt5.TIMEFRAME_D1, days + 2)
    if df is None or len(df) < 5:
        return 0.0
    return (df["high"] - df["low"]).tail(days).mean()


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(UTC)


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY CHECKS
# Each returns (passed: bool, tag: str, detail: str)
# ─────────────────────────────────────────────────────────────────────────────
def _q1_window_complete() -> tuple[bool, str, str]:
    """Q1: Range collection window (13:30–13:45 UTC) must be complete."""
    now = _now_utc().time()
    if now < RANGE_END_UTC:
        remaining = (
            datetime.datetime.combine(datetime.date.today(), RANGE_END_UTC, tzinfo=UTC)
            - _now_utc()
        ).seconds
        return False, "Q1_WINDOW_INCOMPLETE", f"Range window ends in {remaining}s"
    return True, "Q1_WINDOW_COMPLETE", f"Window closed at 13:45 UTC"


def _q2_bars_available(df_range: pd.DataFrame) -> tuple[bool, str, str]:
    """Q2: Must have at least 3 bars in the range window."""
    count = len(df_range)
    if count < 3:
        return False, "Q2_INSUFFICIENT_BARS", f"Only {count} bars in range window"
    return True, "Q2_BARS_OK", f"{count} bars in range"


def _q3_minimum_size(symbol: str, size_pips: float) -> tuple[bool, str, str]:
    """Q3: Range must be at least MIN_PIPS wide to be a meaningful structure."""
    min_p = _min_pips(symbol)
    if size_pips < min_p:
        return False, "Q3_TOO_TIGHT", f"Range={size_pips:.1f} pips < min={min_p} pips"
    return True, "Q3_SIZE_OK", f"Range={size_pips:.1f} pips (min={min_p})"


def _q4_compact_vs_adr(symbol: str, size: float) -> tuple[bool, str, str]:
    """Q4: Range must be compact — not already an extended move vs daily range."""
    adr = _compute_adr(symbol)
    if adr <= 0:
        return True, "Q4_ADR_UNAVAIL", "ADR unavailable — skipping"
    ratio = size / adr
    if ratio > MAX_RANGE_ADR_RATIO:
        return (False, "Q4_RANGE_EXTENDED",
                f"Range={size:.5f} is {ratio:.1%} of ADR={adr:.5f} (max {MAX_RANGE_ADR_RATIO:.0%})")
    return True, "Q4_COMPACT", f"Range is {ratio:.1%} of ADR"


def _q5_wick_quality(df_range: pd.DataFrame) -> tuple[bool, str, str]:
    """
    Q5: Bars in the range window should not have extreme wicks.
    Extreme wicks indicate a spike already happened inside the range — not a clean box.
    """
    bodies = (df_range["close"] - df_range["open"]).abs()
    upper_wicks = df_range["high"] - df_range[["close", "open"]].max(axis=1)
    lower_wicks = df_range[["close", "open"]].min(axis=1) - df_range["low"]
    total_wicks = upper_wicks + lower_wicks

    # Avoid division by zero
    valid = bodies[bodies > 0]
    if len(valid) == 0:
        return True, "Q5_WICK_OK", "No body — skipping wick check"

    wick_body_ratios = total_wicks[bodies > 0] / bodies[bodies > 0]
    avg_ratio = wick_body_ratios.mean()

    if avg_ratio > MAX_WICK_BODY_RATIO:
        return (False, "Q5_SPIKED_RANGE",
                f"Avg wick/body={avg_ratio:.2f} > {MAX_WICK_BODY_RATIO} — spiked range")
    return True, "Q5_CLEAN_RANGE", f"Wick/body ratio={avg_ratio:.2f}"


def _q6_not_already_breached(symbol: str,
                               r_high: float, r_low: float) -> tuple[bool, str, str]:
    """
    Q6: Check if price has already breached BOTH sides of the box.
    A two-sided breach means the box is invalidated (price chopped through it).
    One-sided breach is fine — that is the actual signal.
    """
    df = mt5_client.get_bars(symbol, mt5.TIMEFRAME_M5, 10)
    if df is None:
        return True, "Q6_DATA_MISSING", "Cannot verify breach — allowing"

    # Look at bars after the range window
    post_range = df[df.index.time > RANGE_END_UTC]
    if len(post_range) == 0:
        return True, "Q6_NO_POST_BARS", "No post-range bars yet"

    broke_high = (post_range["high"] > r_high).any()
    broke_low  = (post_range["low"]  < r_low).any()

    if broke_high and broke_low:
        return (False, "Q6_BOTH_BREACHED",
                f"Price breached both box_high={r_high:.5f} AND box_low={r_low:.5f}")

    return True, "Q6_ONE_SIDE_INTACT", (
        f"high_breached={broke_high} low_breached={broke_low}"
    )


def _q7_range_still_fresh() -> tuple[bool, str, str]:
    """Q7: Range must be used within the valid entry window (before 16:00 UTC)."""
    now = _now_utc().time()
    if now > RANGE_VALID_UNTIL_UTC:
        return (False, "Q7_RANGE_STALE",
                f"Current time {now.strftime('%H:%M')} UTC > valid until {RANGE_VALID_UNTIL_UTC}")
    return True, "Q7_RANGE_FRESH", f"Range valid until {RANGE_VALID_UNTIL_UTC} UTC"


# ─────────────────────────────────────────────────────────────────────────────
# CORE RANGE FETCH
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_range_bars(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch 1M bars covering the 13:30–13:44 UTC range window."""
    df = mt5_client.get_bars(symbol, mt5.TIMEFRAME_M1, 30)
    if df is None:
        return None
    return df.between_time(
        RANGE_START_UTC.strftime("%H:%M"),
        "13:44"   # inclusive end of range window
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def get_range(symbol: str) -> tuple[Optional[float], Optional[float]]:
    """
    Fast path — returns (range_high, range_low) or (None, None).
    Runs all quality checks internally; returns None on any failure.
    Use get_range_quality() for diagnostic detail.
    """
    quality = get_range_quality(symbol)
    if quality.is_valid:
        return quality.high, quality.low
    return None, None


def get_range_quality(symbol: str) -> RangeQuality:
    """
    Full diagnostic path — runs all 7 quality checks and returns
    a RangeQuality object with complete metadata.

    Use this in monitoring/dashboard integrations to understand
    why the range was accepted or rejected each session.
    """
    result = RangeQuality(
        symbol      = symbol,
        computed_at = _now_utc().isoformat(),
    )

    # Q1 — window complete
    ok, tag, detail = _q1_window_complete()
    if not ok:
        result.failure_reason = f"{tag}: {detail}"
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return result
    result.checks_passed.append(tag)

    # Fetch range bars
    df_range = _fetch_range_bars(symbol)
    if df_range is None or len(df_range) == 0:
        result.failure_reason = "NO_RANGE_BARS: No 1M bars in 13:30–13:44 UTC"
        log.debug("[%s] No range bars found", symbol)
        return result

    r_high = df_range["high"].max()
    r_low  = df_range["low"].min()
    size   = r_high - r_low

    # Pre-compute pip metrics
    pt         = _point(symbol)
    size_pips  = _pips(symbol, size)
    adr        = _compute_adr(symbol)
    adr_ratio  = size / adr if adr > 0 else 0.0

    # Wick ratio for diagnostics
    bodies = (df_range["close"] - df_range["open"]).abs()
    upper_wicks = df_range["high"] - df_range[["close","open"]].max(axis=1)
    lower_wicks = df_range[["close","open"]].min(axis=1) - df_range["low"]
    valid_b = bodies[bodies > 0]
    wick_ratio = ((upper_wicks + lower_wicks)[bodies > 0] / valid_b).mean() if len(valid_b) > 0 else 0.0

    # Populate metrics before checks (useful for logging even on failure)
    result.high       = round(r_high, 5)
    result.low        = round(r_low, 5)
    result.size       = round(size, 5)
    result.size_pips  = round(size_pips, 1)
    result.adr_ratio  = round(adr_ratio, 4)
    result.bar_count  = len(df_range)
    result.wick_ratio = round(float(wick_ratio), 3)

    # Q2
    ok, tag, detail = _q2_bars_available(df_range)
    if not ok:
        result.failure_reason = f"{tag}: {detail}"
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return result
    result.checks_passed.append(tag)

    # Q3
    ok, tag, detail = _q3_minimum_size(symbol, size_pips)
    if not ok:
        result.failure_reason = f"{tag}: {detail}"
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return result
    result.checks_passed.append(tag)

    # Q4
    ok, tag, detail = _q4_compact_vs_adr(symbol, size)
    if not ok:
        result.failure_reason = f"{tag}: {detail}"
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return result
    result.checks_passed.append(tag)

    # Q5
    ok, tag, detail = _q5_wick_quality(df_range)
    if not ok:
        result.failure_reason = f"{tag}: {detail}"
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return result
    result.checks_passed.append(tag)

    # Q6
    ok, tag, detail = _q6_not_already_breached(symbol, r_high, r_low)
    if not ok:
        result.failure_reason = f"{tag}: {detail}"
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return result
    result.checks_passed.append(tag)

    # Q7
    ok, tag, detail = _q7_range_still_fresh()
    if not ok:
        result.failure_reason = f"{tag}: {detail}"
        log.debug("[%s] %s: %s", symbol, tag, detail)
        return result
    result.checks_passed.append(tag)

    # All checks passed
    result.is_valid = True
    log.info(
        "📦 NY RANGE VALID [%s] | High=%.5f Low=%.5f | Size=%.1f pips "
        "| ADR ratio=%.1f%% | Wicks=%.2f | Bars=%d | Checks=%s",
        symbol, r_high, r_low, size_pips,
        adr_ratio * 100, wick_ratio, len(df_range),
        ",".join(result.checks_passed),
    )

    return result
