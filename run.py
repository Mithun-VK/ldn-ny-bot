"""
run.py — London-NY Overlap Momentum Bot | Production Entry Point
================================================================
Strategy   : London session bias → NY open range breakout → momentum entry
Active IST : 18:30 – 22:00 (13:00 – 16:30 UTC)
Run        : python run.py
Stop       : Ctrl+C

Architecture note:
  This file is intentionally structured as an agent-ready orchestrator.
  Each phase (sense → think → act → observe → report) is a discrete method
  so the entire class can be wrapped by an LLM agent, AutoGen, or CrewAI
  agent in a future iteration without refactoring core logic.

  BotOrchestrator
  ├── sense()       → fetch live market state
  ├── think()       → run strategy pipeline, generate signals
  ├── act()         → pass signals through risk gate → execute orders
  ├── observe()     → monitor open positions, detect closes, update risk
  └── report()      → emit structured status log + equity snapshot
"""

import csv
import datetime
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import MetaTrader5 as mt5
import schedule

import config
import mt5_client
import risk
from strategy.entry_signal import get_signal

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def _setup_logging():
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

log = logging.getLogger("orchestrator")


# ─────────────────────────────────────────────────────────────────────────────
# JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
_JOURNAL_FIELDS = [
    "timestamp", "symbol", "direction", "entry",
    "sl", "tp", "atr", "rr", "lot", "comment",
    "equity_before", "daily_pnl_before",
]

def _write_journal(row: dict):
    exists = os.path.isfile(config.JOURNAL_FILE)
    with open(config.JOURNAL_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_JOURNAL_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# POSITION TRACKER  (for detecting closed trades and firing risk callbacks)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrackedPosition:
    ticket      : int
    symbol      : str
    direction   : str
    entry_price : float
    sl          : float
    tp          : float
    open_time   : datetime.datetime


class PositionTracker:
    """
    Monitors MT5 open positions each cycle.
    Detects newly opened and newly closed positions.
    Fires risk.on_trade_opened / on_trade_closed callbacks automatically.
    """

    def __init__(self):
        self._positions: dict[int, TrackedPosition] = {}

    def sync(self):
        current_raw = mt5_client.open_positions()
        current_map = {p.ticket: p for p in current_raw}

        # ── detect closes
        for ticket, tracked in list(self._positions.items()):
            if ticket not in current_map:
                self._handle_close(ticket, tracked)

        # ── detect new opens
        for ticket, pos in current_map.items():
            if ticket not in self._positions:
                self._handle_open(ticket, pos)

    def _handle_open(self, ticket: int, pos):
        direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
        tracked = TrackedPosition(
            ticket      = ticket,
            symbol      = pos.symbol,
            direction   = direction,
            entry_price = pos.price_open,
            sl          = pos.sl,
            tp          = pos.tp,
            open_time   = datetime.datetime.fromtimestamp(pos.time),
        )
        self._positions[ticket] = tracked
        risk.on_trade_opened(pos.symbol)
        log.info(
            "📂 POSITION OPENED | %s %s @ %.5f | SL=%.5f TP=%.5f | Ticket=%d",
            direction, pos.symbol, pos.price_open, pos.sl, pos.tp, ticket,
        )

    def _handle_close(self, ticket: int, tracked: TrackedPosition):
        # Fetch realized P&L from MT5 deal history
        pnl = 0.0
        deals = mt5.history_deals_get(position=ticket)
        if deals:
            pnl = sum(d.profit for d in deals)

        duration = datetime.datetime.now() - tracked.open_time
        outcome  = "WIN ✅" if pnl >= 0 else "LOSS ❌"

        log.info(
            "📁 POSITION CLOSED | %s %s | PnL=%+.2f | Duration=%s | Ticket=%d",
            outcome, tracked.symbol, pnl,
            str(duration).split(".")[0], ticket,
        )

        risk.on_trade_closed(tracked.symbol, pnl)
        _write_closed_trade_log(tracked, pnl)
        del self._positions[ticket]

    @property
    def open_count(self) -> int:
        return len(self._positions)

    def summary(self) -> list[dict]:
        return [
            {
                "ticket"   : t.ticket,
                "symbol"   : t.symbol,
                "direction": t.direction,
                "entry"    : t.entry_price,
                "sl"       : t.sl,
                "tp"       : t.tp,
                "open_time": t.open_time.isoformat(),
            }
            for t in self._positions.values()
        ]


_CLOSED_FIELDS = ["timestamp", "symbol", "direction", "entry", "sl", "tp", "pnl", "duration_min"]

def _write_closed_trade_log(tracked: TrackedPosition, pnl: float):
    duration = (datetime.datetime.now() - tracked.open_time).total_seconds() / 60
    exists   = os.path.isfile("closed_trades.csv")
    with open("closed_trades.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CLOSED_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({
            "timestamp"    : datetime.datetime.now().isoformat(),
            "symbol"       : tracked.symbol,
            "direction"    : tracked.direction,
            "entry"        : tracked.entry_price,
            "sl"           : tracked.sl,
            "tp"           : tracked.tp,
            "pnl"          : round(pnl, 2),
            "duration_min" : round(duration, 1),
        })


# ─────────────────────────────────────────────────────────────────────────────
# BOT ORCHESTRATOR  — Agent-ready design
# ─────────────────────────────────────────────────────────────────────────────
class BotOrchestrator:
    """
    Main orchestrator for the London-NY Overlap Momentum strategy.

    Designed as a sense → think → act → observe → report loop.
    Each phase is a discrete, independently testable method.

    Future agentic upgrade path:
      - Wrap think() with an LLM tool call for adaptive signal filtering.
      - Replace act() dispatch with an AutoGen / CrewAI agent executor.
      - Feed report() output into a RAG memory store for strategy reflection.
      - Add a meta-agent that monitors performance and adjusts config params.
    """

    def __init__(self):
        self.tracker   = PositionTracker()
        self.cycle     = 0
        self.started_at = datetime.datetime.now()
        self._last_signal_time: dict[str, datetime.datetime] = {}

    # ── PHASE 1: SENSE ───────────────────────────────────────────────────────
    def sense(self) -> dict:
        """
        Collect all relevant market and account state.
        Returns a structured context dict passed to think().

        Future: this becomes the "perception" layer of the agent,
        feeding into an LLM context window or vector memory.
        """
        equity     = mt5_client.get_equity()
        open_pos   = self.tracker.summary()
        risk_snap  = risk.full_report()
        now_utc    = datetime.datetime.now(datetime.timezone.utc)

        in_window  = (
            datetime.time(13, 0) <= now_utc.time() <= datetime.time(16, 30)
        )

        context = {
            "timestamp"    : now_utc.isoformat(),
            "equity"       : equity,
            "in_window"    : in_window,
            "open_positions": open_pos,
            "risk"         : risk_snap,
            "symbols"      : config.SYMBOLS,
        }
        return context

    # ── PHASE 2: THINK ───────────────────────────────────────────────────────
    def think(self, context: dict) -> list[dict]:
        """
        Run the strategy pipeline for each symbol and collect candidate signals.
        Returns list of signal dicts (may be empty).

        Future: an LLM agent can wrap this method to:
          - Add fundamental/news context before signal approval.
          - Score signals using a trained ML classifier.
          - Provide chain-of-thought reasoning for each decision.
        """
        if not context["in_window"]:
            return []

        if context["risk"]["kill_active"] or context["risk"]["daily_halt_active"]:
            return []

        signals = []
        for symbol in context["symbols"]:
            # Skip if already traded today
            if symbol in context["risk"]["traded_today"]:
                continue

            # Skip if last signal for this symbol was within last 5 minutes
            # (prevents re-evaluation of same breakout multiple times)
            last = self._last_signal_time.get(symbol)
            if last and (datetime.datetime.now() - last).seconds < 300:
                continue

            try:
                signal = get_signal(symbol)
                if signal:
                    signals.append(signal)
                    self._last_signal_time[symbol] = datetime.datetime.now()
                    log.info(
                        "💡 SIGNAL | %s %s | entry=%.5f sl=%.5f tp=%.5f",
                        signal["direction"], symbol,
                        signal["entry"], signal["sl"], signal["tp"],
                    )
            except Exception as e:
                log.error("think() error for %s: %s", symbol, e, exc_info=True)

        return signals

    # ── PHASE 3: ACT ─────────────────────────────────────────────────────────
    def act(self, signals: list[dict]) -> list[dict]:
        """
        Pass each signal through the risk gate.
        Execute approved signals via MT5.
        Returns list of executed trade records.

        Future: an execution agent can:
          - Choose optimal order type (limit vs market) based on spread/slippage.
          - Split large orders to minimize market impact.
          - Use smart routing across multiple broker adapters.
        """
        executed = []

        for signal in signals:
            symbol = signal["symbol"]

            # Risk gate — mandatory, non-bypassable
            approved, reason = risk.approve(signal)
            if not approved:
                log.info("🚫 BLOCKED [%s] → %s", symbol, reason)
                continue

            # Compute lot size from current equity
            equity    = mt5_client.get_equity()
            stop_dist = abs(signal["entry"] - signal["sl"])
            lot       = mt5_client.calc_lot(symbol, stop_dist, equity)

            # Execute
            ok = mt5_client.send_order(
                symbol    = symbol,
                direction = signal["direction"],
                sl        = signal["sl"],
                tp        = signal["tp"],
                comment   = signal["comment"],
            )

            if ok:
                # Compute realized R:R for journal
                target_dist = abs(signal["entry"] - signal["tp"])
                rr = round(target_dist / stop_dist, 2) if stop_dist > 0 else 0

                trade_record = {
                    "timestamp"      : datetime.datetime.now().isoformat(),
                    "symbol"         : symbol,
                    "direction"      : signal["direction"],
                    "entry"          : round(signal["entry"], 5),
                    "sl"             : round(signal["sl"], 5),
                    "tp"             : round(signal["tp"], 5),
                    "atr"            : round(signal.get("atr", 0), 5),
                    "rr"             : rr,
                    "lot"            : lot,
                    "comment"        : signal["comment"],
                    "equity_before"  : round(equity, 2),
                    "daily_pnl_before": round(equity - risk._state.day_start_equity, 2),
                }
                _write_journal(trade_record)
                executed.append(trade_record)

        return executed

    # ── PHASE 4: OBSERVE ─────────────────────────────────────────────────────
    def observe(self):
        """
        Sync position tracker — detects new opens and closes,
        fires risk callbacks (on_trade_opened, on_trade_closed).

        Future: an observer agent can:
          - Adjust trailing stops based on momentum continuation.
          - Partially close positions at 1R to lock in profits.
          - Escalate anomaly alerts to a supervisor agent.
        """
        try:
            self.tracker.sync()
        except Exception as e:
            log.error("observe() error: %s", e, exc_info=True)

    # ── PHASE 5: REPORT ──────────────────────────────────────────────────────
    def report(self):
        """
        Emit a structured status snapshot every 15 minutes.
        Writes to log and returns a dict for future dashboard/agent integration.

        Future: feed this dict into:
          - A Grafana metrics endpoint.
          - An LLM supervisor agent that flags anomalies.
          - A Telegram alert dispatcher.
        """
        snap     = risk.full_report()
        uptime   = str(datetime.datetime.now() - self.started_at).split(".")[0]

        log.info(
            "📊 STATUS | Equity=%.2f | DailyPnL=%+.2f (%.2f%%) | "
            "DD=%.2f%% | OpenTrades=%d | WinRate=%.1f%% (%dW/%dL) | "
            "ConLosses=%d | Kill=%s | Halt=%s | Uptime=%s",
            snap["equity"],
            snap["daily_pnl"],
            snap["daily_pnl_pct"],
            snap["drawdown_pct"],
            snap["open_trades"],
            snap["win_rate_pct"],
            snap["total_wins"],
            snap["total_losses"],
            snap["consecutive_losses"],
            snap["kill_active"],
            snap["daily_halt_active"],
            uptime,
        )

        # Open positions detail
        for pos in self.tracker.summary():
            log.info(
                "  📌 %s %s | entry=%.5f | sl=%.5f | tp=%.5f",
                pos["direction"], pos["symbol"],
                pos["entry"], pos["sl"], pos["tp"],
            )

        return snap

    # ── MAIN LOOP ─────────────────────────────────────────────────────────────
    def run_cycle(self):
        """
        One full sense→think→act→observe cycle.
        Called every 60 seconds by the scheduler.

        This is the atomic unit of bot operation.
        In an agentic architecture, this becomes one agent "turn".
        """
        self.cycle += 1

        try:
            context = self.sense()
            signals = self.think(context)
            if signals:
                executed = self.act(signals)
                if executed:
                    log.info("⚡ Cycle %d | %d signal(s) → %d executed",
                             self.cycle, len(signals), len(executed))
            self.observe()
        except Exception as e:
            log.error("run_cycle error (cycle=%d): %s", self.cycle, e, exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP BANNER
# ─────────────────────────────────────────────────────────────────────────────
def _print_banner():
    log.info("=" * 65)
    log.info("  London-NY Overlap Momentum Bot")
    log.info("  Strategy : LDN bias + NY open range breakout")
    log.info("  Window   : 18:30 – 22:00 IST  (13:00 – 16:30 UTC)")
    log.info("  Symbols  : %s", ", ".join(config.SYMBOLS))
    log.info("  Risk/Trade : %.1f%% | Daily halt : %.1f%% | Max DD : %.1f%%",
             config.RISK_PCT * 100,
             config.DAILY_LOSS_LIMIT_PCT * 100,
             config.MAX_DD_PCT * 100)
    log.info("  RR target : %.1f | Min RR : %.1f | Max trades : %d",
             config.RR_RATIO, config.MIN_RR_RATIO, config.MAX_OPEN_TRADES)
    log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    _setup_logging()
    _print_banner()

    if not mt5_client.connect():
        log.critical("MT5 connection failed. Is MetaTrader 5 open and logged in?")
        sys.exit(1)

    bot = BotOrchestrator()

    # Initial observe to register any pre-existing open positions
    bot.observe()

    # Schedule cycles
    schedule.every(1).minutes.do(bot.run_cycle)
    schedule.every(15).minutes.do(bot.report)

    log.info("✅ Bot running. Press Ctrl+C to stop.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutdown requested...")
    except Exception as e:
        log.critical("Fatal error: %s", e, exc_info=True)
    finally:
        mt5_client.disconnect()
        bot.report()   # final snapshot on exit
        log.info("Bot stopped cleanly.")


if __name__ == "__main__":
    main()
