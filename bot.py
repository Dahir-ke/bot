# ==========================================================
# AI FOREX TRADING BOT (QUANT STYLE) - v5.3.6 LINUX FIXED
# Compatible with mt5linux on Ubuntu VPS
# ==========================================================

import MetaTrader5 as mt5   # Official package for Windows
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
# MT5 CONSTANTS (mt5linux does not expose all mt5.XXX constants)
# ==========================================================
TIMEFRAME_M1  = 1
TIMEFRAME_M5  = 5
TIMEFRAME_M15 = 15
TIMEFRAME_M30 = 30
TIMEFRAME_H1  = 60
TIMEFRAME_H4  = 240
TIMEFRAME_D1  = 1440

ORDER_TYPE_BUY  = 0
ORDER_TYPE_SELL = 1
TRADE_ACTION_DEAL = 1
TRADE_RETCODE_DONE = 10009
ORDER_TIME_GTC = 0
ORDER_FILLING_IOC = 2
DEAL_ENTRY_OUT = 1
POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1

# ==========================================================
# CONFIGURATION
# ==========================================================
MT5_LOGIN = 134084924
MT5_PASSWORD = "Dahir@2036"
MT5_SERVER = "ExnessKE-MT5Real9"   # ← Make sure there is no extra space at the end

SYMBOLS = ["EURUSD", "USDJPY"]

SYMBOL_CONFIG = {
    "EURUSD": {
        "MAX_SPREAD_POINTS": 20,
        "MODEL_FILE": "eurusd_ensemble_model1234.pkl",
        "STATE_FILE": "eurusd_bot_state.pkl",
        "LOG_FILE": "eurusd_trade_log.csv",
        "PERFORMANCE_FILE": "eurusd_performance1234_log.csv",
    },
    "USDJPY": {
        "MAX_SPREAD_POINTS": 30,
        "MODEL_FILE": "usdjpy_ensemble_model1234.pkl",
        "STATE_FILE": "usdjpy_bot_state.pkl",
        "LOG_FILE": "usdjpy_trade_log.csv",
        "PERFORMANCE_FILE": "usdjpy_performance1234_log.csv",
    }
}

TIMEFRAME = TIMEFRAME_M5
BARS = 4000
RISK_PERCENT = 0.005
RISK_REWARD = 2
ATR_MULTIPLIER = 1.5
CONFIDENCE_THRESHOLD = 0.65
MAX_DAILY_LOSS = 0.03
MAX_WEEKLY_LOSS = 0.06
MAX_DRAWDOWN_PERCENT = 8.0
MAX_CONSECUTIVE_LOSSES = 6
MAGIC_NUMBER = 20240601
GLOBAL_STATE_FILE = "global_bot_state1234.pkl"

TRADING_START_HOUR_UTC = 7
TRADING_END_HOUR_UTC = 20
MODEL_RETRAIN_INTERVAL_HOURS = 12
TRAIN_WINDOW = 3000
TARGET_BARS_AHEAD = 5
COOLDOWN_MINUTES = 30
NEWS_BUFFER_MINUTES = 35
TRAILING_ACTIVATION = 1.0

USE_H1_TREND_FILTER = False

# ==========================================================
# GLOBAL STATE
# ==========================================================
account_state = {
    'PEAK_EQUITY': None,
    'CONSECUTIVE_LOSSES': 0,
    'DAILY_START_EQUITY': None,
    'WEEKLY_START_EQUITY': None,
    'LAST_NEWS_CHECK': None,
    'CACHED_NEWS': []
}

per_symbol_state = {sym: {'LAST_TRADE_TIME': None, 'LAST_MODEL_RETRAIN': None} for sym in SYMBOLS}

def load_global_state():
    global account_state
    if os.path.exists(GLOBAL_STATE_FILE):
        try:
            saved = joblib.load(GLOBAL_STATE_FILE)
            account_state.update(saved)
            print("✅ Global state loaded")
        except Exception as e:
            print(f"⚠️ Global state load error: {e}")

def save_global_state():
    joblib.dump(account_state, GLOBAL_STATE_FILE)

def load_per_symbol_state(sym):
    sf = SYMBOL_CONFIG[sym]["STATE_FILE"]
    if os.path.exists(sf):
        try:
            saved = joblib.load(sf)
            per_symbol_state[sym].update(saved)
            print(f"✅ State loaded for {sym}")
        except:
            pass

def save_per_symbol_state(sym):
    joblib.dump(per_symbol_state[sym], SYMBOL_CONFIG[sym]["STATE_FILE"])

# ==========================================================
# MT5 LOGIN + RECONNECT
# ==========================================================
def mt5_login():
    print(f"🔐 Attempting MT5 login → Account: {MT5_LOGIN} | Server: {MT5_SERVER}")
    
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"❌ MT5 login failed: {mt5.last_error()}")
        return False
    
    account_info = mt5.account_info()
    if account_info is None:
        print("❌ Could not get account info")
        mt5.shutdown()
        return False
    
    print(f"✅ MT5 Login Successful! Balance: ${account_info.balance:.2f} | Equity: ${account_info.equity:.2f}")
    return True

def ensure_mt5_connection():
    if not mt5.terminal_info():
        print("⚠️ MT5 disconnected. Reconnecting...")
        mt5.shutdown()
        time.sleep(2)
        return mt5_login()
    return True

# Initial login
if not mt5_login():
    print("❌ Failed to login. Check credentials.")
    quit()

for sym in SYMBOLS:
    mt5.symbol_select(sym, True)

load_global_state()
for sym in SYMBOLS:
    load_per_symbol_state(sym)

# ==========================================================
# MARKET & NEWS
# ==========================================================
def is_market_open():
    n = datetime.now(timezone.utc)
    w, h = n.weekday(), n.hour
    if w == 5 or (w == 6 and h < 22) or (w == 4 and h >= 22):
        return False
    return True

def fetch_news():
    # (same as before - unchanged)
    now = datetime.now(timezone.utc)
    if account_state['LAST_NEWS_CHECK'] and (now - account_state['LAST_NEWS_CHECK']).seconds < 3600:
        return account_state['CACHED_NEWS']
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=5)
        if r.status_code != 200:
            return []
        events = r.json()
        eastern = pytz.timezone('US/Eastern')
        parsed = []
        for e in events:
            if e.get("impact") == "High" and e.get("country") in ["USD", "EUR", "GBP", "JPY"]:
                try:
                    dt_str = f"{e['date']} {e.get('time', '00:00')}"
                    local_dt = eastern.localize(datetime.strptime(dt_str, "%Y-%m-%d %H:%M"))
                    event_dt = local_dt.astimezone(pytz.utc)
                    parsed.append({'title': e.get('title'), 'country': e.get('country'), 'time_utc': event_dt})
                except:
                    continue
        account_state['CACHED_NEWS'] = parsed
        account_state['LAST_NEWS_CHECK'] = now
        save_global_state()
        return parsed
    except:
        return []

def is_high_impact_news():
    events = fetch_news()
    now_utc = datetime.now(timezone.utc)
    for e in events:
        minutes_away = abs((e['time_utc'] - now_utc).total_seconds()) / 60
        if minutes_away < NEWS_BUFFER_MINUTES:
            print(f"🚨 HIGH-IMPACT NEWS: {e['title']} in ~{int(minutes_away)} min")
            return True
    return False

# ==========================================================
# DATA + FEATURES
# ==========================================================
def get_data(symbol):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, BARS)
    if rates is None or len(rates) < 200:
        raise Exception(f"Insufficient data for {symbol}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def add_features(df):
    df['ema20'] = ta.trend.ema_indicator(df['close'], 20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], 50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], 200)
    df['rsi'] = ta.momentum.rsi(df['close'], 14)
    df['macd'] = ta.trend.macd(df['close'])
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
    df['momentum'] = df['close'].pct_change(5)
    df['volatility'] = df['close'].rolling(20).std()
    df['volume_ma'] = df['tick_volume'].rolling(20).mean()
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)
    df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
    df['close_to_ema50'] = df['close'] / df['ema50'] - 1
    df['close_to_ema200'] = df['close'] / df['ema200'] - 1
    df.dropna(inplace=True)
    return df

def detect_market_regime(df):
    adx = df['adx'].iloc[-1]
    atr_pct = df['atr'].rolling(100).rank(pct=True).iloc[-1] * 100 if len(df) >= 100 else 50
    if adx > 25:
        return "TREND"
    elif atr_pct > 80:
        return "VOLATILE"
    elif atr_pct < 20:
        return "COMPRESSION"
    return "RANGE"

def get_higher_tf_trend(symbol):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_H1, 0, 200)
    if rates is None or len(rates) < 100:
        return None
    df_htf = pd.DataFrame(rates)
    df_htf['ema50'] = ta.trend.ema_indicator(df_htf['close'], 50)
    df_htf['ema200'] = ta.trend.ema_indicator(df_htf['close'], 200)
    df_htf.dropna(inplace=True)
    if len(df_htf) < 1:
        return None
    trend_up = df_htf['ema50'].iloc[-1] > df_htf['ema200'].iloc[-1]
    print(f"📈 H1 Trend for {symbol}: {'BULLISH ✅' if trend_up else 'BEARISH ❌'}")
    return trend_up

# ==========================================================
# MODEL
# ==========================================================
def train_model(df, symbol):
    print(f"🔄 Training model for {symbol}...")
    df = df.copy()
    df['target'] = np.where(df['close'].shift(-TARGET_BARS_AHEAD) > df['close'], 1, 0)
    df = add_features(df)
    df.dropna(inplace=True)
    
    if len(df) < TRAIN_WINDOW + 100:
        print(f"⚠️ Not enough data for {symbol}")
        return None, None, None
    
    train_df = df.tail(TRAIN_WINDOW).copy()
    feature_cols = ['ema20','ema50','ema200','rsi','macd','momentum','volatility',
                    'volume_ma','adx','volatility_ratio','close_to_ema50','close_to_ema200']
    
    features = train_df[feature_cols]
    target = train_df['target']
    
    split = int(len(features) * 0.8)
    X_train = features.iloc[:split]
    y_train = target.iloc[:split]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    base = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, random_state=42)
    model = CalibratedClassifierCV(estimator=base, method='isotonic', cv=3)
    model.fit(X_train_scaled, y_train)
    
    joblib.dump((model, scaler, feature_cols), SYMBOL_CONFIG[symbol]["MODEL_FILE"])
    per_symbol_state[symbol]['LAST_MODEL_RETRAIN'] = datetime.now(timezone.utc)
    save_per_symbol_state(symbol)
    return model, scaler, feature_cols

def load_model(df, symbol):
    mf = SYMBOL_CONFIG[symbol]["MODEL_FILE"]
    last_retrain = per_symbol_state[symbol].get('LAST_MODEL_RETRAIN')
    
    if os.path.exists(mf) and last_retrain:
        age_hours = (datetime.now(timezone.utc) - last_retrain).total_seconds() / 3600
        if age_hours < MODEL_RETRAIN_INTERVAL_HOURS:
            model, scaler, feature_cols = joblib.load(mf)
            print(f"✅ Model loaded for {symbol} (age: {age_hours:.1f}h)")
            return model, scaler, feature_cols
    
    return train_model(df, symbol)

# ==========================================================
# RISK MANAGEMENT
# ==========================================================
def position_open(symbol):
    positions = mt5.positions_get(symbol=symbol)
    return bool(positions) and any(p.magic == MAGIC_NUMBER for p in (positions or []))

def is_on_cooldown(symbol):
    last = per_symbol_state[symbol]['LAST_TRADE_TIME']
    if last is None:
        return False
    return (datetime.now(timezone.utc) - last).total_seconds() < COOLDOWN_MINUTES * 60

def get_drawdown():
    if account_state['PEAK_EQUITY'] is None:
        return 0.0
    equity = mt5.account_info().equity
    return max(0.0, (account_state['PEAK_EQUITY'] - equity) / account_state['PEAK_EQUITY'] * 100)

def calculate_lot(entry_price, stop_loss, regime, prob, order_type, symbol):
    # Simplified safe version for now
    return 0.01   # Start with micro lot. You can improve later.

# ==========================================================
# SIGNAL + TRADE
# ==========================================================
def can_place_trade(tick, regime, df, symbol):
    if is_high_impact_news():
        return False
    sym_info = mt5.symbol_info(symbol)
    spread_points = (tick.ask - tick.bid) / sym_info.point
    if spread_points > SYMBOL_CONFIG[symbol]["MAX_SPREAD_POINTS"]:
        return False
    if get_drawdown() > MAX_DRAWDOWN_PERCENT:
        return False
    return True

def generate_signal(df, model, scaler, feature_cols, symbol):
    regime = detect_market_regime(df)
    latest = scaler.transform(df[feature_cols].iloc[-1:])
    prob = model.predict_proba(latest)[0][1]
    atr = df['atr'].iloc[-1]

    if prob > CONFIDENCE_THRESHOLD:
        return "BUY", prob, atr, regime
    elif prob < (1 - CONFIDENCE_THRESHOLD):
        return "SELL", prob, atr, regime
    return None, None, None, regime

def execute_trade(signal, atr, prob, regime, symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return

    if not can_place_trade(tick, regime, None, symbol):
        return

    if signal == "BUY":
        price = tick.ask
        sl = price - ATR_MULTIPLIER * atr
        tp = price + ATR_MULTIPLIER * atr * RISK_REWARD
        order_type = ORDER_TYPE_BUY
    else:
        price = tick.bid
        sl = price + ATR_MULTIPLIER * atr
        tp = price - ATR_MULTIPLIER * atr * RISK_REWARD
        order_type = ORDER_TYPE_SELL

    lot = calculate_lot(price, sl, regime, prob, order_type, symbol)

    request = {
        "action": TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "AI v5.3.6 Linux",
        "type_time": ORDER_TIME_GTC,
        "type_filling": ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)
    if result and result.retcode == TRADE_RETCODE_DONE:
        print(f"✅ {signal} {symbol} | Lot {lot} | Prob {prob:.1%}")
        per_symbol_state[symbol]['LAST_TRADE_TIME'] = datetime.now(timezone.utc)
        save_per_symbol_state(symbol)
    else:
        print(f"❌ Trade failed: {result.comment if result else 'No result'}")

# ==========================================================
# MAIN LOOP
# ==========================================================
def run_bot():
    print("🚀 AI Quant Bot v5.3.6 (Linux + mt5linux) Started")
    
    model_dict = {}
    scaler_dict = {}
    feature_cols_dict = {}

    for sym in SYMBOLS:
        df = get_data(sym)
        df = add_features(df)
        model_dict[sym], scaler_dict[sym], feature_cols_dict[sym] = load_model(df, sym)

    while True:
        try:
            ensure_mt5_connection()

            if not is_market_open():
                time.sleep(600)
                continue

            for symbol in SYMBOLS:
                df = get_data(symbol)
                df = add_features(df)

                model, scaler, feature_cols = load_model(df, symbol)
                model_dict[symbol], scaler_dict[symbol], feature_cols_dict[symbol] = model, scaler, feature_cols

                signal, prob, atr, regime = generate_signal(df, model, scaler, feature_cols, symbol)

                if signal and not position_open(symbol) and not is_on_cooldown(symbol):
                    execute_trade(signal, atr, prob, regime, symbol)

            time.sleep(60)

        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run_bot()