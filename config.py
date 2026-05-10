"""config.py — load all settings from .env"""
import os
from dotenv import load_dotenv

load_dotenv()

MT5_LOGIN    = int(os.getenv("MT5_LOGIN", 0))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER   = os.getenv("MT5_SERVER", "")

SYMBOLS      = os.getenv("SYMBOLS", "EURUSD").split(",")
RISK_PCT     = float(os.getenv("RISK_PCT", 0.01))
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", 0.02))
MAX_DD_PCT   = float(os.getenv("MAX_DD_PCT", 0.08))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 3))
RR_RATIO     = float(os.getenv("RR_RATIO", 2.5))
ATR_MULT_SL  = float(os.getenv("ATR_MULT_SL", 1.25))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", 20))
MAX_VOL_MULT           = float(os.getenv("MAX_VOL_MULT", 2.0))
MIN_RR_RATIO           = float(os.getenv("MIN_RR_RATIO", 1.8))

MAGIC        = 20001
LOG_FILE     = "bot.log"
JOURNAL_FILE = "journal.csv"