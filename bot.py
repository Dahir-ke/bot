# ==========================================================
# NEXT-LEVEL AI TRADING BOT v6.2 
# Cross-Platform (Windows + Linux) + Logging + One Trade Only
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
    logging.info("Running on Windows - using official MetaTrader5")
else:
    from mt5linux import MetaTrader5
    mt5 = MetaTrader5()   # For Linux - make sure mt5linux server is running
    logging.info("Running on Linux - using mt5linux")

# ====================== CONSTANTS (Linux compatibility) ======================
TIMEFRAME_M5 = 5
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
TRADE_ACTION_DEAL = 1
TRADE_ACTION_SLTP = 6
POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1
ORDER_TIME_GTC = 0
ORDER_FILLING_IOC = 1

# ====================== CONFIG ======================
MT5_LOGIN = 134084924
MT5_PASSWORD = "Dahir@2036"
MT5_SERVER = "ExnessKE-MT5Real9"

SYMBOLS = [
    "EURUSDm", "USDJPYm", "XAUUSDm", "UKOILm", "USOILm", "XNGUSDm",
    "AUDCADm", "AUDCHFm", "AUDDKKm", "AUDJPYm", "AUDMXNm", "USDDKKm"
]

SYMBOL_CONFIG = {
    sym: {"MAX_SPREAD": 400 if "XAU" in sym else 250 if "XNG" in sym else 120 if "OIL" in sym else 60}
    for sym in SYMBOLS
}

MAX_RISK_PERCENT = 0.01
RISK_REWARD = 2.0
ATR_MULTIPLIER_SL = 1.8
CONFIDENCE_THRESHOLD = 0.72
MAX_DRAWDOWN_PERCENT = 5.0
MIN_FREE_MARGIN = 8.0
BARS = 1500

MAGIC_NUMBER = 20240601

peak_equity = None

# ====================== GRACEFUL SHUTDOWN ======================
def shutdown_handler(sig, frame):
    logging.info("Shutdown signal received. Closing MT5...")
    mt5.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# ====================== HELPERS ======================
def has_open_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    return bool(positions and any(p.magic == MAGIC_NUMBER for p in positions or []))

def is_spread_acceptable(symbol):
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if not tick or not info:
        return False
    spread = (tick.ask - tick.bid) / info.point
    return spread <= SYMBOL_CONFIG[symbol]["MAX_SPREAD"]

def is_trading_time():
    hour = datetime.utcnow().hour
    return 7 <= hour <= 20

def get_dynamic_lot(symbol, sl_distance):
    account = mt5.account_info()
    info = mt5.symbol_info(symbol)
    if not account or not info or sl_distance <= 0:
        return 0.01
    risk_amount = account.balance * MAX_RISK_PERCENT
    lot = risk_amount / (sl_distance * info.trade_tick_value)
    lot = max(info.volume_min, min(info.volume_max, lot))
    lot = round(lot / info.volume_step) * info.volume_step
    return round(lot, 2)

def get_data(symbol):
    logging.info(f"Loading data for {symbol}")
    mt5.symbol_select(symbol, True)
    for _ in range(10):
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_M5, 0, BARS)
        if rates is not None and len(rates) >= 300:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            logging.info(f"✅ {symbol}: {len(df)} bars loaded")
            return df
        time.sleep(2)
    raise Exception(f"Failed to load data for {symbol}")

def add_features(df):
    df['ema20'] = ta.trend.ema_indicator(df['close'], 20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], 50)
    df['rsi'] = ta.momentum.rsi(df['close'], 14)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
    df['momentum'] = df['close'].pct_change(5)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)
    df.dropna(inplace=True)
    return df

def load_or_train_model(df, symbol):
    mf = f"{symbol.lower()}_v6.pkl"
    if os.path.exists(mf):
        try:
            model, scaler, feats = joblib.load(mf)
            logging.info(f"Model loaded for {symbol}")
            return model, scaler, feats
        except:
            pass
    logging.info(f"Training new model for {symbol}")
    df = df.copy()
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    df.dropna(inplace=True)
    feats = ['ema20', 'ema50', 'rsi', 'atr', 'momentum', 'adx']
    X = df[feats]
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_scaled, y)
    joblib.dump((model, scaler, feats), mf)
    return model, scaler, feats

def execute_trade(signal, atr, prob, symbol):
    if has_open_position(symbol):
        logging.info(f"⛔ {symbol} already has open position - skipping")
        return

    if not is_spread_acceptable(symbol):
        logging.info(f"⚠️ High spread on {symbol} - skipping")
        return

    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if not tick or not info:
        return

    if signal == "BUY":
        price = tick.ask
        sl = price - ATR_MULTIPLIER_SL * atr
        tp = price + ATR_MULTIPLIER_SL * atr * RISK_REWARD
        order_type = ORDER_TYPE_BUY
    else:
        price = tick.bid
        sl = price + ATR_MULTIPLIER_SL * atr
        tp = price - ATR_MULTIPLIER_SL * atr * RISK_REWARD
        order_type = ORDER_TYPE_SELL

    point = info.point
    sl = round(sl / point) * point
    tp = round(tp / point) * point

    lot = get_dynamic_lot(symbol, abs(price - sl))
    if lot <= 0:
        return

    request = {
        "action": TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 40,
        "magic": MAGIC_NUMBER,
        "comment": "AI_v6.2",
        "type_time": ORDER_TIME_GTC,
        "type_filling": ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == 10009:
        logging.info(f"✅ ENTRY → {signal} {symbol} | Lot {lot:.2f} | Prob {prob:.1%}")
    else:
        logging.error(f"❌ Trade failed on {symbol}: {result.comment if result else 'Unknown error'}")

# ====================== MAIN ======================
def run_bot():
    global peak_equity
    logging.info("=== BOT v6.2 STARTED ===")

    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        logging.error(f"MT5 initialize failed: {mt5.last_error()}")
        return

    # Pre-load symbols
    for s in SYMBOLS:
        mt5.symbol_select(s, True)
    time.sleep(8)

    models = {}
    for sym in SYMBOLS:
        try:
            df = get_data(sym)
            df = add_features(df)
            models[sym] = load_or_train_model(df, sym)
        except Exception as e:
            logging.error(f"Failed to prepare {sym}: {e}")

    logging.info("Bot main loop started - Monitoring symbols...\n")

    while True:
        try:
            if not is_trading_time():
                time.sleep(300)
                continue

            account = mt5.account_info()
            if peak_equity is None or account.equity > peak_equity:
                peak_equity = account.equity

            drawdown = (peak_equity - account.equity) / peak_equity * 100 if peak_equity else 0
            if drawdown > MAX_DRAWDOWN_PERCENT:
                logging.warning(f"High drawdown ({drawdown:.1f}%) - Pausing 30 min")
                time.sleep(1800)
                continue

            for symbol in list(models.keys()):
                if has_open_position(symbol):
                    continue  # One trade per symbol only

                df = get_data(symbol)
                df = add_features(df)
                model, scaler, feats = models[symbol]

                latest = scaler.transform(df[feats].iloc[-1:])
                prob = model.predict_proba(latest)[0][1]
                atr = df['atr'].iloc[-1]

                if prob >= CONFIDENCE_THRESHOLD:
                    signal = "BUY"
                elif prob <= (1 - CONFIDENCE_THRESHOLD):
                    signal = "SELL"
                else:
                    continue

                execute_trade(signal, atr, prob, symbol)

            time.sleep(60)

        except Exception as e:
            logging.error(f"Main loop error: {e}")
            time.sleep(15)

if __name__ == "__main__":
    run_bot()