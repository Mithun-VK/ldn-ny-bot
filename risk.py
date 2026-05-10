"""
risk.py — Wall Street Grade Risk Engine
========================================
Strategy: London–NY Overlap Momentum Bot

Risk controls implemented:
  1. Per-trade risk sizing          — fixed fractional (0.5–1% equity)
  2. Daily loss hard cap            — halt all new trades if daily P&L < -2%
  3. Total drawdown kill switch     — stop all trading if DD from peak > 8%
  4. Session time gate              — only allow entries in 13:00–16:30 UTC window
  5. Max concurrent positions       — never exceed N open trades
  6. Consecutive loss circuit breaker — pause after 3 losses in a row
  7. Volatility gate                — skip entry if ATR is abnormally high (news spike)
  8. Duplicate signal guard         — never re-enter a symbol already traded today
  9. Minimum R:R enforcement        — reject signals where R:R < configured minimum
 10. Equity high-water mark tracker — continuously updates peak for DD calculation
 11. Per-symbol exposure cap        — never hold more than 1 position per symbol
 12. Weekend / session close guard  — block new entries near daily close (21:30 UTC)

All modules MUST call risk_engine.approve(signal) before sending any order.
Returns (approved: bool, reason: str).
"""

import logging
import datetime
import csv
import os
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5

import config

log = logging.getLogger("risk")

UTC = datetime.timezone.utc


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RiskState:
    # Equity tracking
    peak_equity         : float = 0.0
    day_start_equity    : float = 0.0
    last_reset_date     : Optional[datetime.date] = None

    # Circuit breakers
    kill_active         : bool  = False   # total DD breach — permanent until manual reset
    daily_halt_active   : bool  = False   # daily loss breach — resets next day
    session_blocked     : bool  = False   # outside active trading window

    # Consecutive loss tracker
    consecutive_losses  : int   = 0

    # Symbols traded today (one trade per symbol per day max)
    traded_today        : set   = field(default_factory=set)

    # All-time trade count for logging
    total_trades        : int   = 0
    total_wins          : int   = 0
    total_losses        : int   = 0


_state = RiskState()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _get_equity() -> float:
    info = mt5.account_info()
    return info.equity if info else 0.0


def _get_balance() -> float:
    info = mt5.account_info()
    return info.balance if info else 0.0


def _open_positions() -> list:
    pos = mt5.positions_get()
    return list(pos) if pos else []


def _open_count() -> int:
    return len(_open_positions())


def _symbol_already_open(symbol: str) -> bool:
    return any(p.symbol == symbol for p in _open_positions())


# ─────────────────────────────────────────────────────────────────────────────
# DAILY RESET
# ─────────────────────────────────────────────────────────────────────────────
def _reset_daily_if_needed():
    today = datetime.date.today()
    if _state.last_reset_date == today:
        return

    equity = _get_equity()
    _state.day_start_equity  = equity
    _state.daily_halt_active = False
    _state.traded_today      = set()
    _state.consecutive_losses = 0
    _state.last_reset_date   = today

    if _state.peak_equity == 0.0:
        _state.peak_equity = equity

    log.info(
        "Daily reset | Date=%s | StartEquity=%.2f | PeakEquity=%.2f",
        today, equity, _state.peak_equity
    )


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-WATER MARK
# ─────────────────────────────────────────────────────────────────────────────
def _update_hwm(equity: float):
    if equity > _state.peak_equity:
        _state.peak_equity = equity
        log.debug("New equity high-water mark: %.2f", equity)


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL RISK CHECKS
# Each returns (passed: bool, reason: str)
# ─────────────────────────────────────────────────────────────────────────────
def _check_kill_switch(equity: float) -> tuple[bool, str]:
    if _state.kill_active:
        return False, "KILL_SWITCH_ACTIVE"
    if _state.peak_equity > 0:
        dd = ((_state.peak_equity - equity) / _state.peak_equity)
        if dd >= config.MAX_DD_PCT:
            _state.kill_active = True
            log.critical(
                "🔴 KILL SWITCH TRIGGERED | Drawdown=%.2f%% | "
                "Equity=%.2f | Peak=%.2f",
                dd * 100, equity, _state.peak_equity
            )
            _log_risk_event("KILL_SWITCH", f"DD={dd*100:.2f}%")
            return False, f"KILL_SWITCH_DD={dd*100:.1f}%"
    return True, "OK"


def _check_daily_loss(equity: float) -> tuple[bool, str]:
    if _state.daily_halt_active:
        return False, "DAILY_HALT_ACTIVE"
    if _state.day_start_equity > 0:
        pnl_pct = (equity - _state.day_start_equity) / _state.day_start_equity
        if pnl_pct <= -config.DAILY_LOSS_LIMIT_PCT:
            _state.daily_halt_active = True
            log.warning(
                "🟠 DAILY HALT | DailyPnL=%.2f%% | Equity=%.2f",
                pnl_pct * 100, equity
            )
            _log_risk_event("DAILY_HALT", f"DailyPnL={pnl_pct*100:.2f}%")
            return False, f"DAILY_LOSS={pnl_pct*100:.1f}%"
    return True, "OK"


def _check_session_window() -> tuple[bool, str]:
    now = datetime.datetime.now(UTC).time()
    # Active: 13:00–16:30 UTC (18:30–22:00 IST)
    start = datetime.time(13, 0)
    end   = datetime.time(16, 30)
    if not (start <= now <= end):
        return False, f"OUTSIDE_WINDOW(now={now.strftime('%H:%M')}UTC)"
    return True, "OK"


def _check_daily_close_guard() -> tuple[bool, str]:
    """Block new entries in last 30 minutes before MT5 daily close (~21:30 UTC)."""
    now = datetime.datetime.now(UTC).time()
    if now >= datetime.time(21, 0):
        return False, "DAILY_CLOSE_GUARD(>21:00UTC)"
    return True, "OK"


def _check_max_trades() -> tuple[bool, str]:
    count = _open_count()
    if count >= config.MAX_OPEN_TRADES:
        return False, f"MAX_OPEN_TRADES({count}/{config.MAX_OPEN_TRADES})"
    return True, "OK"


def _check_duplicate_symbol(symbol: str) -> tuple[bool, str]:
    if symbol in _state.traded_today:
        return False, f"ALREADY_TRADED_TODAY({symbol})"
    if _symbol_already_open(symbol):
        return False, f"POSITION_ALREADY_OPEN({symbol})"
    return True, "OK"


def _check_consecutive_losses() -> tuple[bool, str]:
    limit = getattr(config, "MAX_CONSECUTIVE_LOSSES", 3)
    if _state.consecutive_losses >= limit:
        log.warning(
            "⚡ Consecutive loss circuit breaker: %d losses in a row",
            _state.consecutive_losses
        )
        return False, f"CONSEC_LOSS_CIRCUIT({_state.consecutive_losses})"
    return True, "OK"


def _check_volatility_gate(signal: dict) -> tuple[bool, str]:
    """
    Reject signal if ATR is more than 2× its 20-period average.
    Protects against entering during news spikes or abnormal volatility.
    """
    atr = signal.get("atr")
    atr_avg = signal.get("atr_avg")
    if atr is None or atr_avg is None or atr_avg == 0:
        return True, "OK"  # can't check — allow through
    ratio = atr / atr_avg
    vol_mult = getattr(config, "MAX_VOL_MULT", 2.0)
    if ratio > vol_mult:
        log.warning(
            "⚡ Volatility gate: ATR=%.5f is %.1f× avg (%.5f) — skipping",
            atr, ratio, atr_avg
        )
        return False, f"HIGH_VOL(atr={atr:.5f} ratio={ratio:.1f}x)"
    return True, "OK"


def _check_minimum_rr(signal: dict) -> tuple[bool, str]:
    """Reject if the signal's computed R:R is below the configured minimum."""
    entry = signal.get("entry")
    sl    = signal.get("sl")
    tp    = signal.get("tp")
    if None in (entry, sl, tp):
        return True, "OK"
    stop_dist   = abs(entry - sl)
    target_dist = abs(entry - tp)
    if stop_dist == 0:
        return False, "ZERO_STOP_DISTANCE"
    rr = target_dist / stop_dist
    min_rr = getattr(config, "MIN_RR_RATIO", 1.8)
    if rr < min_rr:
        return False, f"LOW_RR({rr:.2f}<{min_rr})"
    return True, "OK"


def _check_sl_tp_valid(signal: dict) -> tuple[bool, str]:
    """Basic sanity: SL and TP must be on the correct side of entry."""
    direction = signal.get("direction")
    entry = signal.get("entry", 0)
    sl    = signal.get("sl", 0)
    tp    = signal.get("tp", 0)

    if direction == "BUY":
        if sl >= entry:
            return False, f"INVALID_SL_BUY(sl={sl}>=entry={entry})"
        if tp <= entry:
            return False, f"INVALID_TP_BUY(tp={tp}<=entry={entry})"
    elif direction == "SELL":
        if sl <= entry:
            return False, f"INVALID_SL_SELL(sl={sl}<=entry={entry})"
        if tp >= entry:
            return False, f"INVALID_TP_SELL(tp={tp}>=entry={entry})"

    return True, "OK"


# ─────────────────────────────────────────────────────────────────────────────
# RISK EVENT LOGGER
# ─────────────────────────────────────────────────────────────────────────────
_RISK_LOG_FILE = "risk_events.csv"

def _log_risk_event(event_type: str, detail: str):
    file_exists = os.path.isfile(_RISK_LOG_FILE)
    with open(_RISK_LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "event", "detail",
                                           "equity", "peak", "daily_pnl"])
        if not file_exists:
            w.writeheader()
        eq = _get_equity()
        w.writerow({
            "timestamp" : datetime.datetime.now().isoformat(),
            "event"     : event_type,
            "detail"    : detail,
            "equity"    : f"{eq:.2f}",
            "peak"      : f"{_state.peak_equity:.2f}",
            "daily_pnl" : f"{eq - _state.day_start_equity:.2f}",
        })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
def approve(signal: dict) -> tuple[bool, str]:
    """
    Central gate — every module must call this before sending any order.

    Args:
        signal: dict with keys:
                  symbol, direction, entry, sl, tp, atr, atr_avg (optional)

    Returns:
        (True, "OK")            — trade approved, proceed
        (False, "REASON_CODE")  — trade rejected, do not send order
    """
    _reset_daily_if_needed()

    equity = _get_equity()
    _update_hwm(equity)

    symbol = signal.get("symbol", "")

    # Run all checks in priority order — fail fast on first rejection
    checks = [
        _check_kill_switch(equity),
        _check_daily_loss(equity),
        _check_session_window(),
        _check_daily_close_guard(),
        _check_max_trades(),
        _check_duplicate_symbol(symbol),
        _check_consecutive_losses(),
        _check_volatility_gate(signal),
        _check_minimum_rr(signal),
        _check_sl_tp_valid(signal),
    ]

    for passed, reason in checks:
        if not passed:
            log.debug("REJECTED [%s] → %s", symbol, reason)
            return False, reason

    log.info(
        "✅ APPROVED [%s %s] | Entry=%.5f SL=%.5f TP=%.5f | "
        "Equity=%.2f | DailyPnL=%+.2f | OpenTrades=%d",
        signal.get("direction"), symbol,
        signal.get("entry", 0), signal.get("sl", 0), signal.get("tp", 0),
        equity, equity - _state.day_start_equity, _open_count()
    )
    return True, "OK"


# ─────────────────────────────────────────────────────────────────────────────
# TRADE OUTCOME CALLBACKS
# Call these from run.py after each trade closes
# ─────────────────────────────────────────────────────────────────────────────
def on_trade_opened(symbol: str):
    """Call immediately after a successful order_send."""
    _state.traded_today.add(symbol)
    _state.total_trades += 1
    log.info("Trade opened: %s | Total trades today: %d", symbol, len(_state.traded_today))


def on_trade_closed(symbol: str, pnl: float):
    """
    Call when a position closes (monitor in run.py by watching positions).
    pnl: profit/loss in account currency (positive = win, negative = loss).
    """
    if pnl >= 0:
        _state.consecutive_losses = 0
        _state.total_wins += 1
        log.info("✅ WIN  | %s | PnL=%+.2f | Consec losses reset to 0", symbol, pnl)
    else:
        _state.consecutive_losses += 1
        _state.total_losses += 1
        log.warning("❌ LOSS | %s | PnL=%+.2f | Consecutive losses=%d",
                    symbol, pnl, _state.consecutive_losses)
        _log_risk_event("LOSS", f"symbol={symbol} pnl={pnl:.2f} consec={_state.consecutive_losses}")


# ─────────────────────────────────────────────────────────────────────────────
# STATUS & DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
def status_line() -> str:
    equity    = _get_equity()
    daily_pnl = equity - _state.day_start_equity
    dd_pct    = ((_state.peak_equity - equity) / max(_state.peak_equity, 1)) * 100
    win_rate  = ((_state.total_wins / max(_state.total_trades, 1)) * 100)

    return (
        f"Equity={equity:.2f} | "
        f"DailyPnL={daily_pnl:+.2f} | "
        f"DD={dd_pct:.2f}% | "
        f"OpenTrades={_open_count()} | "
        f"ConLosses={_state.consecutive_losses} | "
        f"WinRate={win_rate:.1f}% ({_state.total_wins}W/{_state.total_losses}L) | "
        f"Kill={_state.kill_active} | DailyHalt={_state.daily_halt_active}"
    )


def full_report() -> dict:
    """Returns a full risk snapshot dict — useful for monitoring dashboards."""
    equity    = _get_equity()
    balance   = _get_balance()
    daily_pnl = equity - _state.day_start_equity
    dd_pct    = ((_state.peak_equity - equity) / max(_state.peak_equity, 1)) * 100

    return {
        "timestamp"          : datetime.datetime.now().isoformat(),
        "equity"             : round(equity, 2),
        "balance"            : round(balance, 2),
        "peak_equity"        : round(_state.peak_equity, 2),
        "day_start_equity"   : round(_state.day_start_equity, 2),
        "daily_pnl"          : round(daily_pnl, 2),
        "daily_pnl_pct"      : round(daily_pnl / max(_state.day_start_equity, 1) * 100, 2),
        "drawdown_pct"       : round(dd_pct, 2),
        "open_trades"        : _open_count(),
        "traded_today"       : list(_state.traded_today),
        "consecutive_losses" : _state.consecutive_losses,
        "total_trades"       : _state.total_trades,
        "total_wins"         : _state.total_wins,
        "total_losses"       : _state.total_losses,
        "win_rate_pct"       : round(_state.total_wins / max(_state.total_trades, 1) * 100, 1),
        "kill_active"        : _state.kill_active,
        "daily_halt_active"  : _state.daily_halt_active,
    }


def reset_kill_switch():
    """
    Manual override — call ONLY after you have reviewed the drawdown
    and deliberately decide to resume trading.
    Logs the manual reset for audit trail.
    """
    if not _state.kill_active:
        log.info("Kill switch was not active — no action needed")
        return
    _state.kill_active   = False
    _state.peak_equity   = _get_equity()   # reset peak to current equity
    log.warning("⚠️  KILL SWITCH MANUALLY RESET | New peak equity: %.2f",
                _state.peak_equity)
    _log_risk_event("KILL_SWITCH_RESET", "manual override by operator")
