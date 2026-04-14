# ==========================================================
# AI FOREX + COMMODITIES TRADING BOT - v5.3.7 UPDATED
# Symbols: EURUSD, USDJPY + UKOILm, USOILm, XNGUSDm, AUD* pairs
# Windows + Official MetaTrader5 (Recommended)
# ==========================================================

import MetaTrader5 as mt5   # Official package - Use this on Windows
import pandas as pd
import numpy as np
import ta
import time
import os
import joblib
import requests
import pytz
import xgboost as xgb
from datetime import datetime, timedelta, timezone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

# ==========================================================
# CONFIGURATION
# ==========================================================
MT5_LOGIN = 134084924
MT5_PASSWORD = "Dahir@2036"
MT5_SERVER = "ExnessKE-MT5Real9"   # Remove any trailing space

SYMBOLS = [
    "EURUSD", "USDJPY",
    "UKOILm", "USOILm", "XNGUSDm",
    "AUDCADm", "AUDCHFm", "AUDJPYm"
    # Add more AUD pairs here if you want: AUDCZKm, AUDDKKm, etc.
]

SYMBOL_CONFIG = {
    "EURUSD":  {"MAX_SPREAD_POINTS": 20, "MODEL_FILE": "eurusd_model.pkl"},
    "USDJPY":  {"MAX_SPREAD_POINTS": 30, "MODEL_FILE": "usdjpy_model.pkl"},
    "UKOILm":  {"MAX_SPREAD_POINTS": 80, "MODEL_FILE": "ukoilm_model.pkl"},   # Oil has wider spreads
    "USOILm":  {"MAX_SPREAD_POINTS": 80, "MODEL_FILE": "usoilm_model.pkl"},
    "XNGUSDm":{"MAX_SPREAD_POINTS": 150,"MODEL_FILE": "xngusdm_model.pkl"},  # Natural gas is very volatile
    "AUDCADm":{"MAX_SPREAD_POINTS": 40, "MODEL_FILE": "audcadm_model.pkl"},
    "AUDCHFm":{"MAX_SPREAD_POINTS": 40, "MODEL_FILE": "audchfm_model.pkl"},
    "AUDJPYm":{"MAX_SPREAD_POINTS": 50, "MODEL_FILE": "audjpym_model.pkl"},
}

TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 1500                    # Reduced for faster testing (increase later)
RISK_PERCENT = 0.005
RISK_REWARD = 2.0
ATR_MULTIPLIER = 1.5
CONFIDENCE_THRESHOLD = 0.65
COOLDOWN_MINUTES = 30
MAGIC_NUMBER = 20240601

# ==========================================================
# GLOBAL STATE (simplified)
# ==========================================================
per_symbol_state = {sym: {'LAST_TRADE_TIME': None} for sym in SYMBOLS}

def save_per_symbol_state(sym):
    pass  # You can expand this later with joblib if needed

# ==========================================================
# MT5 LOGIN
# ==========================================================
def mt5_login():
    print(f"🔐 Logging in → Account: {MT5_LOGIN} | Server: {MT5_SERVER}")
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"❌ Login failed: {mt5.last_error()}")
        return False

    account_info = mt5.account_info()
    if account_info:
        print(f"✅ Login Successful! Balance: ${account_info.balance:.2f} | Equity: ${account_info.equity:.2f}")
    return True

# ==========================================================
# IMPROVED DATA FETCHING
# ==========================================================
def get_data(symbol):
    print(f"📥 Loading data for {symbol}...")
    mt5.symbol_select(symbol, True)   # Force select + trigger download

    for attempt in range(10):
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, BARS)
        if rates is not None and len(rates) >= 200:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            print(f"✅ {symbol}: {len(df)} bars loaded")
            return df

        print(f"   Attempt {attempt+1}: waiting for data...")
        time.sleep(4)

    # Final fallback
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, 500)
    if rates is not None and len(rates) > 100:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    raise Exception(f"Failed to load enough data for {symbol}. Open the chart manually in MT5.")

def add_features(df):
    df['ema20'] = ta.trend.ema_indicator(df['close'], 20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], 50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], 200)
    df['rsi'] = ta.momentum.rsi(df['close'], 14)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
    df['momentum'] = df['close'].pct_change(5)
    df.dropna(inplace=True)
    return df

# ==========================================================
# MODEL (simple version - you can expand)
# ==========================================================
def load_or_train_model(df, symbol):
    model_file = SYMBOL_CONFIG[symbol]["MODEL_FILE"]
    if os.path.exists(model_file):
        try:
            model, scaler, features = joblib.load(model_file)
            print(f"✅ Loaded model for {symbol}")
            return model, scaler, features
        except:
            pass

    print(f"🛠️ Training new model for {symbol}...")
    df = df.copy()
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    df.dropna(inplace=True)

    feature_cols = ['ema20', 'ema50', 'rsi', 'atr', 'momentum']
    X = df[feature_cols]
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBClassifier(n_estimators=300, max_depth=5, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump((model, scaler, feature_cols), model_file)
    return model, scaler, feature_cols

# ==========================================================
# SIGNAL + TRADE EXECUTION (simplified & safe)
# ==========================================================
def generate_signal(df, model, scaler, features, symbol):
    latest = scaler.transform(df[features].iloc[-1:])
    prob = model.predict_proba(latest)[0][1]
    atr = df['atr'].iloc[-1]

    if prob >= CONFIDENCE_THRESHOLD:
        return "BUY", prob, atr
    elif prob <= (1 - CONFIDENCE_THRESHOLD):
        return "SELL", prob, atr
    return None, None, None

def execute_trade(signal, atr, prob, symbol):
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return

    config = SYMBOL_CONFIG[symbol]
    spread = (tick.ask - tick.bid) / mt5.symbol_info(symbol).point
    if spread > config["MAX_SPREAD_POINTS"]:
        print(f"⚠️ Spread too high for {symbol} ({spread:.1f})")
        return

    if signal == "BUY":
        price = tick.ask
        sl = price - ATR_MULTIPLIER * atr
        tp = price + ATR_MULTIPLIER * atr * RISK_REWARD
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        sl = price + ATR_MULTIPLIER * atr
        tp = price - ATR_MULTIPLIER * atr * RISK_REWARD
        order_type = mt5.ORDER_TYPE_SELL

    lot = 0.01   # Safe micro lot - change later with proper risk calculation

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 30,
        "magic": MAGIC_NUMBER,
        "comment": "AI_BOT_v5.3.7",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == 10009:   # TRADE_RETCODE_DONE
        print(f"✅ {signal} {symbol} | Lot {lot} | Prob {prob:.1%}")
        per_symbol_state[symbol]['LAST_TRADE_TIME'] = datetime.now(timezone.utc)
    else:
        print(f"❌ Failed {symbol}: {result.comment if result else 'No result'}")

# ==========================================================
# MAIN LOOP
# ==========================================================
def run_bot():
    if not mt5_login():
        quit()

    print("🔄 Pre-loading all symbols...")
    for sym in SYMBOLS:
        mt5.symbol_select(sym, True)
    time.sleep(8)

    print("🚀 AI Trading Bot Started with", len(SYMBOLS), "symbols")

    models = {}
    for sym in SYMBOLS:
        try:
            df = get_data(sym)
            df = add_features(df)
            models[sym] = load_or_train_model(df, sym)
        except Exception as e:
            print(f"⚠️ Skipping {sym}: {e}")

    while True:
        try:
            for symbol in SYMBOLS:
                if symbol not in models:
                    continue

                df = get_data(symbol)
                df = add_features(df)
                model, scaler, features = models[symbol]

                signal, prob, atr = generate_signal(df, model, scaler, features, symbol)

                if signal:
                    execute_trade(signal, atr, prob, symbol)

            time.sleep(60)

        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run_bot()