# ==========================================================
# AI TRADING BOT v5.3.8 - MULTI SYMBOL (Forex + Commodities)
# Updated with your latest symbols
# ==========================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta
import time
import os
import joblib
import xgboost as xgb
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# ==========================================================
# CONFIGURATION
# ==========================================================
MT5_LOGIN = 134084924
MT5_PASSWORD = "Dahir@2036"
MT5_SERVER = "ExnessKE-MT5Real9"

SYMBOLS = [
    "EURUSDm", "USDJPYm", "XAUUSDm",          # Major + Gold
    "UKOILm", "USOILm", "XNGUSDm",            # Energy
    "AUDCADm", "AUDCHFm", "AUDDKKm", "AUDJPYm", "AUDMXNm",  # AUD crosses
    "USDDKKm"                                 # USD crosses
]

# Spread limits (in points) - adjusted for each instrument
SYMBOL_CONFIG = {
    "EURUSDm":  {"MAX_SPREAD_POINTS": 20,  "MODEL_FILE": "eurusdm_model.pkl"},
    "USDJPYm":  {"MAX_SPREAD_POINTS": 30,  "MODEL_FILE": "usdj pym_model.pkl"},
    "XAUUSDm":  {"MAX_SPREAD_POINTS": 300, "MODEL_FILE": "xauusdm_model.pkl"},   # Gold wider spread
    "UKOILm":   {"MAX_SPREAD_POINTS": 100, "MODEL_FILE": "ukoilm_model.pkl"},
    "USOILm":   {"MAX_SPREAD_POINTS": 100, "MODEL_FILE": "usoilm_model.pkl"},
    "XNGUSDm":  {"MAX_SPREAD_POINTS": 200, "MODEL_FILE": "xngusdm_model.pkl"},   # Natural Gas very volatile
    "AUDCADm":  {"MAX_SPREAD_POINTS": 40,  "MODEL_FILE": "audcadm_model.pkl"},
    "AUDCHFm":  {"MAX_SPREAD_POINTS": 40,  "MODEL_FILE": "audchfm_model.pkl"},
    "AUDDKKm":  {"MAX_SPREAD_POINTS": 50,  "MODEL_FILE": "auddkkm_model.pkl"},
    "AUDJPYm":  {"MAX_SPREAD_POINTS": 50,  "MODEL_FILE": "audjpym_model.pkl"},
    "AUDMXNm":  {"MAX_SPREAD_POINTS": 60,  "MODEL_FILE": "audmxnm_model.pkl"},
    "USDDKKm":  {"MAX_SPREAD_POINTS": 40,  "MODEL_FILE": "usddkkm_model.pkl"},
}

TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 1200                    # Good balance for speed + enough data
RISK_PERCENT = 0.005           # 0.5% risk per trade (conservative)
RISK_REWARD = 2.0
ATR_MULTIPLIER = 1.5
CONFIDENCE_THRESHOLD = 0.65
COOLDOWN_MINUTES = 30
MAGIC_NUMBER = 20240601

# ==========================================================
# GLOBAL STATE
# ==========================================================
per_symbol_state = {sym: {'LAST_TRADE_TIME': None} for sym in SYMBOLS}

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
# DATA FETCHING (Improved with retries)
# ==========================================================
def get_data(symbol):
    print(f"📥 Loading data for {symbol}...")
    mt5.symbol_select(symbol, True)

    for attempt in range(10):
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, BARS)
        if rates is not None and len(rates) >= 250:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            print(f"✅ {symbol}: {len(df)} bars loaded")
            return df
        
        time.sleep(3)

    # Fallback
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, 600)
    if rates is not None and len(rates) > 100:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    raise Exception(f"Could not load enough data for {symbol}. Please open {symbol} M5 chart in MT5.")

def add_features(df):
    df['ema20'] = ta.trend.ema_indicator(df['close'], 20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], 50)
    df['rsi']   = ta.momentum.rsi(df['close'], 14)
    df['atr']   = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
    df['momentum'] = df['close'].pct_change(5)
    df.dropna(inplace=True)
    return df

# ==========================================================
# MODEL
# ==========================================================
def load_or_train_model(df, symbol):
    model_file = SYMBOL_CONFIG[symbol]["MODEL_FILE"]
    
    if os.path.exists(model_file):
        try:
            model, scaler, features = joblib.load(model_file)
            print(f"✅ Loaded existing model for {symbol}")
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
    print(f"✅ New model trained and saved for {symbol}")
    return model, scaler, feature_cols

# ==========================================================
# SIGNAL & TRADE
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

    info = mt5.symbol_info(symbol)
    spread_points = (tick.ask - tick.bid) / info.point

    if spread_points > SYMBOL_CONFIG[symbol]["MAX_SPREAD_POINTS"]:
        print(f"⚠️ High spread on {symbol}: {spread_points:.1f} points")
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

    lot = 0.01   # Start small and safe (especially with $7 balance)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "deviation": 30,
        "magic": MAGIC_NUMBER,
        "comment": "AI_BOT_5.3.8",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == 10009:   # Order executed successfully
        print(f"✅ {signal} {symbol} | Lot: {lot} | Prob: {prob:.1%} | Spread: {spread_points:.1f}")
        per_symbol_state[symbol]['LAST_TRADE_TIME'] = datetime.now(timezone.utc)
    else:
        print(f"❌ {symbol} trade failed: {result.comment if result else 'Unknown error'}")

# ==========================================================
# MAIN
# ==========================================================
def run_bot():
    if not mt5_login():
        print("❌ Could not login to MT5")
        return

    print(f"\n🚀 Starting AI Bot with {len(SYMBOLS)} symbols...")
    print("🔄 Pre-loading all symbols (this may take 10-20 seconds)...\n")

    for sym in SYMBOLS:
        mt5.symbol_select(sym, True)

    time.sleep(10)   # Give MT5 time to load history

    models = {}
    for sym in SYMBOLS:
        try:
            df = get_data(sym)
            df = add_features(df)
            models[sym] = load_or_train_model(df, sym)
        except Exception as e:
            print(f"⚠️ Could not prepare {sym}: {e}")

    print("\n✅ Bot is now running...\n")

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

            time.sleep(60)   # Check every minute

        except Exception as e:
            print(f"⚠️ Main loop error: {e}")
            time.sleep(15)

if __name__ == "__main__":
    run_bot()