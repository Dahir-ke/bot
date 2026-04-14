# ==========================================================
# STRICT SAFE BOT v6.3 - Fixed Multiple Losing Trades
# One Trade Per Symbol + Cooldown + Daily Loss Protection
# ==========================================================

import platform
import logging
import time
import os
import joblib
import signal
import sys
from datetime import datetime, timezone, timedelta
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
else:
    from mt5linux import MetaTrader5
    mt5 = MetaTrader5()

# ====================== CONSTANTS ======================
TIMEFRAME_M5 = 5
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
TRADE_ACTION_DEAL = 1
TRADE_ACTION_SLTP = 6
MAGIC_NUMBER = 20240601

# ====================== CONFIG ======================
MT5_LOGIN = 134084924
MT5_PASSWORD = "Dahir@2036"
MT5_SERVER = "ExnessKE-MT5Real9"

SYMBOLS = ["EURUSDm", "USDJPYm", "XAUUSDm", "UKOILm", "USOILm", "XNGUSDm",
           "AUDCADm", "AUDCHFm", "AUDJPYm"]

SYMBOL_CONFIG = {sym: {"MAX_SPREAD": 100} for sym in SYMBOLS}

MAX_RISK_PERCENT = 0.005      # Reduced to 0.5%
CONFIDENCE_THRESHOLD = 0.78   # Much stricter
COOLDOWN_MINUTES = 30
MAX_DAILY_LOSS_PERCENT = 3.0  # Stop if daily loss > 3%
BARS = 1200

last_trade_time = {sym: datetime.min for sym in SYMBOLS}
daily_start_equity = None

# ====================== SHUTDOWN ======================
def shutdown_handler(sig, frame):
    logging.info("Bot shutting down...")
    mt5.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)

# ====================== HELPERS ======================
def has_open_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    return bool(positions and any(p.magic == MAGIC_NUMBER for p in positions or []))

def is_on_cooldown(symbol):
    return (datetime.now(timezone.utc) - last_trade_time[symbol]).total_seconds() < COOLDOWN_MINUTES * 60

def get_daily_loss():
    global daily_start_equity
    account = mt5.account_info()
    if daily_start_equity is None:
        daily_start_equity = account.equity
        return 0.0
    loss = (daily_start_equity - account.equity) / daily_start_equity * 100
    return loss

def get_data(symbol):
    mt5.symbol_select(symbol, True)
    for _ in range(8):
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_M5, 0, BARS)
        if rates is not None and len(rates) >= 250:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        time.sleep(2)
    logging.error(f"Failed to get data for {symbol}")
    return None

def add_features(df):
    if df is None:
        return None
    df['ema20'] = ta.trend.ema_indicator(df['close'], 20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], 50)
    df['rsi'] = ta.momentum.rsi(df['close'], 14)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)
    df.dropna(inplace=True)
    return df

def load_or_train_model(df, symbol):
    mf = f"{symbol.lower()}_safe_v6.pkl"
    if os.path.exists(mf):
        try:
            return joblib.load(mf)
        except:
            pass
    # Train simple model
    df = df.copy()
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    df.dropna(inplace=True)
    feats = ['ema20','ema50','rsi','atr','adx']
    X = df[feats]
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBClassifier(n_estimators=300, max_depth=4, random_state=42)
    model.fit(X_scaled, y)
    joblib.dump((model, scaler, feats), mf)
    return model, scaler, feats

def execute_trade(signal, atr, prob, symbol):
    if has_open_position(symbol) or is_on_cooldown(symbol):
        return

    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if not tick or not info:
        return

    spread = (tick.ask - tick.bid) / info.point
    if spread > SYMBOL_CONFIG[symbol]["MAX_SPREAD"]:
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

    lot = 0.01   # Fixed small lot for safety with low balance

    request = {
        "action": TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 30,
        "magic": MAGIC_NUMBER,
        "comment": "SAFE_v6.3",
        "type_time": ORDER_TIME_GTC,
        "type_filling": ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == 10009:
        logging.info(f"✅ TRADE OPENED → {signal} {symbol} | Prob {prob:.1%}")
        last_trade_time[symbol] = datetime.now(timezone.utc)
    else:
        logging.error(f"Trade failed {symbol}: {result.comment if result else 'Unknown'}")

# ====================== MAIN ======================
def run_bot():
    logging.info("=== SAFE BOT v6.3 STARTED ===")

    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        logging.error("MT5 login failed")
        return

    for s in SYMBOLS:
        mt5.symbol_select(s, True)

    models = {}
    for sym in SYMBOLS:
        df = get_data(sym)
        if df is not None:
            df = add_features(df)
            models[sym] = load_or_train_model(df, sym)

    logging.info("Bot is now running in SAFE mode...")

    while True:
        try:
            daily_loss = get_daily_loss()
            if daily_loss > MAX_DAILY_LOSS_PERCENT:
                logging.warning(f"DAILY LOSS LIMIT HIT ({daily_loss:.1f}%) - Bot paused for 1 hour")
                time.sleep(3600)
                continue

            for symbol in SYMBOLS:
                if symbol not in models:
                    continue
                if has_open_position(symbol) or is_on_cooldown(symbol):
                    continue

                df = get_data(symbol)
                if df is None:
                    continue
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
            logging.error(f"Error in main loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run_bot()