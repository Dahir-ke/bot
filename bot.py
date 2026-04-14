# ==========================================================
# NEXT-LEVEL AI TRADING BOT v6.2 
# Cross-Platform (Windows + Linux/mt5linux) + Logging + Strict Rules
# ==========================================================

import platform
import logging
import time
import os
import joblib
import signal
import sys
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import ta
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# ====================== LOGGING ======================
logging.basicConfig(
    filename="trading_bot.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# ====================== CROSS-PLATFORM MT5 ======================
if platform.system() == "Windows":
    import MetaTrader5 as mt5
    logging.info("Running on Windows - using official MetaTrader5 package")
else:
    # Linux (Contabo VPS)
    from mt5linux import MetaTrader5
    mt5 = MetaTrader5()   # Default: localhost:8001 (start mt5linux server separately)
    logging.info("Running on Linux - using mt5linux wrapper")

# ====================== MT5 CONSTANTS (for Linux compatibility) ======================
TIMEFRAME_M5 = 5
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
TRADE_ACTION_DEAL = 1
TRADE_ACTION_SLTP = 6
POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1
ORDER_TIME_GTC = 0
ORDER_FILLING_IOC = 1   # or 2 depending on broker

# ====================== CONFIG ======================
MT5_LOGIN = 134084924
MT5_PASSWORD = "Dahir@2036"
MT5_SERVER = "ExnessKE-MT5Real9"

SYMBOLS = ["EURUSDm", "USDJPYm", "XAUUSDm", "UKOILm", "USOILm", "XNGUSDm",
           "AUDCADm", "AUDCHFm", "AUDDKKm", "AUDJPYm", "AUDMXNm", "USDDKKm"]

SYMBOL_CONFIG = {sym: {"MAX_SPREAD": 80 if "XNG" in sym else 400 if "XAU" in sym else 120 if "OIL" in sym else 60} 
                 for sym in SYMBOLS}   # simplified

MAX_RISK_PERCENT = 0.01
RISK_REWARD = 2.0
ATR_MULTIPLIER_SL = 1.8
CONFIDENCE_THRESHOLD = 0.72
MAX_DRAWDOWN_PERCENT = 5.0
MIN_FREE_MARGIN = 8.0

# ====================== GRACEFUL SHUTDOWN ======================
def shutdown_handler(sig, frame):
    logging.info("Shutdown signal received. Closing MT5 connection...")
    mt5.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# ====================== HELPERS ======================
def has_open_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    return bool(positions and any(p.magic == MAGIC_NUMBER for p in positions))

def is_spread_acceptable(symbol):
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if not tick or not info:
        return False
    spread = (tick.ask - tick.bid) / info.point
    return spread <= SYMBOL_CONFIG[symbol]["MAX_SPREAD"]

def is_trading_time():
    hour = datetime.utcnow().hour
    return 7 <= hour <= 20   # London + NY overlap (adjust as needed)

def get_dynamic_lot(symbol, sl_distance):
    account = mt5.account_info()
    info = mt5.symbol_info(symbol)
    if not account or not info or sl_distance <= 0:
        return info.volume_min if info else 0.01

    risk_amount = account.balance * MAX_RISK_PERCENT
    lot = risk_amount / (sl_distance * info.trade_tick_value)
    lot = max(info.volume_min, min(info.volume_max, lot))
    lot = round(lot / info.volume_step) * info.volume_step
    return round(lot, 2)

# ... (add_features, load_or_train_model, execute_trade, manage_trailing functions remain similar to previous version)

# ====================== MAIN ======================
def run_bot():
    logging.info("=== NEXT-LEVEL BOT v6.2 STARTED ===")
    
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        logging.error(f"MT5 initialize failed: {mt5.last_error()}")
        return

    # Pre-load symbols
    for s in SYMBOLS:
        mt5.symbol_select(s, True)

    # Load models, main loop, etc.
    # (You can merge the logic from v6.1 here)

    logging.info("Bot main loop started")

if __name__ == "__main__":
    run_bot()