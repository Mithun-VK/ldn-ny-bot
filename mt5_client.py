"""mt5_client.py — MT5 connection, data fetch, order execution, lot sizing."""
import logging
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

import config

log = logging.getLogger("mt5")


def connect() -> bool:
    if not mt5.initialize():
        log.error("MT5 init failed: %s", mt5.last_error())
        return False
    if config.MT5_LOGIN:
        ok = mt5.login(config.MT5_LOGIN,
                       password=config.MT5_PASSWORD,
                       server=config.MT5_SERVER)
        if not ok:
            log.error("MT5 login failed: %s", mt5.last_error())
            return False
    info = mt5.account_info()
    log.info("Connected | Account %s | Balance %.2f", info.login, info.balance)
    return True


def disconnect():
    mt5.shutdown()


def get_equity() -> float:
    info = mt5.account_info()
    return info.equity if info else 0.0


def get_bars(symbol: str, tf: int, count: int = 300) -> Optional[pd.DataFrame]:
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) < 20:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    # Pre-compute ATR once here so every module can use it
    _hl  = df["high"] - df["low"]
    _hc  = (df["high"] - df["close"].shift()).abs()
    _lc  = (df["low"]  - df["close"].shift()).abs()
    _tr  = pd.concat([_hl, _hc, _lc], axis=1).max(axis=1)   
    df["atr"] = _tr.ewm(alpha=1/14, adjust=False).mean()

    return df


def open_positions(magic: int = config.MAGIC) -> list:
    pos = mt5.positions_get(magic=magic)
    return list(pos) if pos else []


def all_open_count() -> int:
    pos = mt5.positions_get()
    return len(pos) if pos else 0


def calc_lot(symbol: str, stop_dist: float, equity: float) -> float:
    """Risk-based lot sizing: risk RISK_PCT of equity over stop_dist."""
    risk_amount = equity * config.RISK_PCT
    info = mt5.symbol_info(symbol)
    if info is None or stop_dist <= 0:
        return info.volume_min if info else 0.01
    pip_val = info.trade_tick_value / info.trade_tick_size * info.point
    ticks   = stop_dist / info.point
    if pip_val <= 0 or ticks <= 0:
        return info.volume_min
    lot = risk_amount / (ticks * pip_val)
    lot = max(info.volume_min,
              min(info.volume_max,
                  round(lot / info.volume_step) * info.volume_step))
    return lot


def send_order(symbol: str, direction: str,
               sl: float, tp: float, comment: str) -> bool:
    """direction: 'BUY' or 'SELL'"""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.warning("No tick data for %s", symbol)
        return False

    if direction == "BUY":
        price     = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price     = tick.bid
        order_type = mt5.ORDER_TYPE_SELL

    stop_dist = abs(price - sl)
    equity    = get_equity()
    lot       = calc_lot(symbol, stop_dist, equity)

    request = {
        "action"      : mt5.TRADE_ACTION_DEAL,
        "symbol"      : symbol,
        "volume"      : lot,
        "type"        : order_type,
        "price"       : price,
        "sl"          : round(sl, mt5.symbol_info(symbol).digits),
        "tp"          : round(tp, mt5.symbol_info(symbol).digits),
        "magic"       : config.MAGIC,
        "comment"     : comment,
        "type_time"   : mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "deviation"   : 20,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log.warning("Order FAILED [%s %s]: %s",
                    direction, symbol,
                    result.comment if result else mt5.last_error())
        return False

    log.info("ORDER OK | %s | %s | lot=%.2f | entry=%.5f | sl=%.5f | tp=%.5f",
             direction, symbol, lot, price, sl, tp)
    return True
