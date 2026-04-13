# ==========================================================
# AI FOREX TRADING BOT (QUANT STYLE) - v5.3.6 "MT5 LOGIN + RECONNECT"
# EURUSD + USDJPY - 12H Retrain + Leakage Fix + MT5 Login
# ==========================================================
import mt5linux as mt5
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
# ==================== MT5 LOGIN ====================
MT5_LOGIN = 134084924         # ← CHANGE TO YOUR MT5 ACCOUNT NUMBER
MT5_PASSWORD = "Dahir@2036"  # ← CHANGE TO YOUR MT5 PASSWORD
MT5_SERVER = "ExnessKE-MT5Real9 " # ← CHANGE TO YOUR BROKER SERVER (e.g. "ICMarketsSC-Demo", "Pepperstone-Demo", etc.)

# ==================== TRADING SETTINGS ====================
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

TIMEFRAME = mt5.TIMEFRAME_M5
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

USE_H1_TREND_FILTER = False   # True = strict trend-following | False = both directions

# ==========================================================
# GLOBAL + PER-SYMBOL STATE
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
# MT5 INITIALIZATION + LOGIN + RECONNECT
# ==========================================================
def mt5_login():
    """Login to MT5 with retry"""
    print(f"🔐 Attempting MT5 login → Account: {MT5_LOGIN} | Server: {MT5_SERVER}")
    
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"❌ MT5 login failed: {mt5.last_error()}")
        return False
    
    account_info = mt5.account_info()
    if account_info is None:
        print("❌ Could not get account info after login")
        mt5.shutdown()
        return False
    
    print(f"✅ MT5 Login Successful!")
    print(f"   Account: {account_info.login} | Balance: ${account_info.balance:.2f} | Equity: ${account_info.equity:.2f}")
    return True

def ensure_mt5_connection():
    """Check and reconnect if MT5 is disconnected"""
    if not mt5.terminal_info():
        print("⚠️ MT5 disconnected. Attempting reconnect...")
        mt5.shutdown()
        time.sleep(2)
        return mt5_login()
    return True

# Initial login
if not mt5_login():
    print("❌ Failed to login to MT5. Please check your credentials and server name.")
    quit()

for sym in SYMBOLS:
    if not mt5.symbol_select(sym, True):
        print(f"⚠️ Failed to select symbol {sym}")

load_global_state()
for sym in SYMBOLS:
    load_per_symbol_state(sym)

# ==========================================================
# MARKET OPEN + NEWS (unchanged)
# ==========================================================
def is_market_open():
    n = datetime.now(timezone.utc)
    w, h = n.weekday(), n.hour
    if w == 5 or (w == 6 and h < 22) or (w == 4 and h >= 22):
        return False
    return True

def fetch_news():
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
# DATA + FEATURES + REGIME + H1 TREND (unchanged from previous version)
# ==========================================================
def get_data(symbol):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, BARS)
    if rates is None or len(rates) < 200:
        raise Exception(f"Insufficient MT5 data for {symbol}")
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
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
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
# MODEL TRAINING (12H + Leakage Fixed) - unchanged
# ==========================================================
def train_model(df, symbol):
    print(f"🔄 Training XGBoost for {symbol}...")
    df = df.copy()
    df['target'] = np.where(df['close'].shift(-TARGET_BARS_AHEAD) > df['close'], 1, 0)
    df = add_features(df)
    df.dropna(inplace=True)
    
    if len(df) < TRAIN_WINDOW + 100:
        print(f"⚠️ Not enough clean data for {symbol}")
        return None, None, None
    
    train_df = df.tail(TRAIN_WINDOW).copy()
    
    feature_cols = ['ema20','ema50','ema200','rsi','macd','momentum',
                    'volatility','volume_ma','adx','volatility_ratio',
                    'close_to_ema50','close_to_ema200']
    
    features = train_df[feature_cols]
    target = train_df['target']
    
    split = int(len(features) * 0.8)
    X_train = features.iloc[:split]
    y_train = target.iloc[:split]
    X_val = features.iloc[split:]
    y_val = target.iloc[split:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    base = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=30
    )
    
    model = CalibratedClassifierCV(estimator=base, method='isotonic', cv=3)
    model.fit(X_train_scaled, y_train)
    
    val_accuracy = model.score(X_val_scaled, y_val)
    print(f"✅ {symbol} Training done | Val Accuracy: {val_accuracy:.2%}")
    
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
            try:
                model, scaler, feature_cols = joblib.load(mf)
                print(f"✅ Model loaded for {symbol} (age: {age_hours:.1f}h)")
                return model, scaler, feature_cols
            except:
                pass
    
    print(f"🔄 Retraining {symbol} model...")
    return train_model(df, symbol)

# ==========================================================
# RISK & POSITION MANAGEMENT (unchanged)
# ==========================================================
def position_open(symbol):
    positions = mt5.positions_get(symbol=symbol)
    return bool(positions) and any(p.magic == MAGIC_NUMBER for p in (positions or []))

def is_on_cooldown(symbol):
    if per_symbol_state[symbol]['LAST_TRADE_TIME'] is None:
        return False
    return (datetime.now(timezone.utc) - per_symbol_state[symbol]['LAST_TRADE_TIME']).total_seconds() < COOLDOWN_MINUTES * 60

def update_peak_equity():
    equity = mt5.account_info().equity
    if account_state['PEAK_EQUITY'] is None or equity > account_state['PEAK_EQUITY']:
        account_state['PEAK_EQUITY'] = equity
        save_global_state()

def get_drawdown():
    if account_state['PEAK_EQUITY'] is None:
        return 0.0
    return max(0.0, (account_state['PEAK_EQUITY'] - mt5.account_info().equity) / account_state['PEAK_EQUITY'] * 100)

def daily_loss_exceeded():
    now = datetime.now(timezone.utc)
    today = now.date()
    if account_state['DAILY_START_EQUITY'] is None or account_state['DAILY_START_EQUITY']['date'] != today:
        account_state['DAILY_START_EQUITY'] = {'date': today, 'equity': mt5.account_info().equity}
        save_global_state()
        return False
    loss = (account_state['DAILY_START_EQUITY']['equity'] - mt5.account_info().equity) / account_state['DAILY_START_EQUITY']['equity']
    return loss >= MAX_DAILY_LOSS

def weekly_loss_exceeded():
    now = datetime.now(timezone.utc)
    week_start = (now - timedelta(days=now.weekday())).date()
    if account_state['WEEKLY_START_EQUITY'] is None or account_state['WEEKLY_START_EQUITY']['date'] != week_start:
        account_state['WEEKLY_START_EQUITY'] = {'date': week_start, 'equity': mt5.account_info().equity}
        save_global_state()
        return False
    loss = (account_state['WEEKLY_START_EQUITY']['equity'] - mt5.account_info().equity) / account_state['WEEKLY_START_EQUITY']['equity']
    return loss >= MAX_WEEKLY_LOSS

def calculate_lot(entry_price, stop_loss, regime, prob, order_type, symbol):
    sym_info = mt5.symbol_info(symbol)
    if not sym_info:
        return 0.01
    min_vol = sym_info.volume_min
    vol_step = sym_info.volume_step
    max_vol = getattr(sym_info, "volume_max", 100.0)
    
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0:
        return min_vol
    
    tick_value = sym_info.trade_tick_value
    tick_size = sym_info.trade_tick_size
    point_value = tick_value * (sym_info.point / tick_size) if tick_size > 0 else 0
    risk_per_lot = stop_distance * point_value if point_value > 0 else 1
    
    dd = get_drawdown()
    conf_factor = 1.6 if prob > 0.85 else 1.2 if prob > 0.75 else 0.8
    regime_factor = 1.0 if regime == "TREND" else 0.5 if regime in ["RANGE", "COMPRESSION"] else 0.7
    dd_factor = 0.3 if dd > 5 else 1.0
    consec_factor = max(0.5, 1.0 - (account_state['CONSECUTIVE_LOSSES'] / (2 * MAX_CONSECUTIVE_LOSSES)))
    
    effective_risk = RISK_PERCENT * conf_factor * regime_factor * dd_factor * consec_factor
    risk_amount = mt5.account_info().balance * effective_risk
    approx_lot = risk_amount / risk_per_lot
    lot = max(min_vol, min(max_vol, round(approx_lot / vol_step) * vol_step))
    
    try:
        free = mt5.account_info().margin_free
        req = mt5.order_calc_margin(order_type, symbol, lot, entry_price)
        while lot >= min_vol and (req is None or req > free):
            lot -= vol_step
            req = mt5.order_calc_margin(order_type, symbol, lot, entry_price)
        lot = max(lot, min_vol)
    except:
        pass
    return round(lot, 2)

# ==========================================================
# SIGNAL + EXECUTION
# ==========================================================
def can_place_trade(tick, regime, df, symbol):
    if is_high_impact_news():
        return False
    max_spread = SYMBOL_CONFIG[symbol]["MAX_SPREAD_POINTS"]
    sym_info = mt5.symbol_info(symbol)
    spread_points = (tick.ask - tick.bid) / sym_info.point
    if spread_points > max_spread:
        print(f"⚠️ Spread too high for {symbol}: {spread_points:.1f} points")
        return False
    if get_drawdown() > MAX_DRAWDOWN_PERCENT:
        return False
    if mt5.account_info().margin_free < 50:
        return False
    if df['atr'].iloc[-1] < 0.0001 * 10:
        return False
    return True

def generate_signal(df, model, scaler, feature_cols, symbol):
    regime = detect_market_regime(df)
    print(f"📊 Regime for {symbol}: {regime}")
    
    latest = scaler.transform(df[feature_cols].iloc[-1:])
    prob = model.predict_proba(latest)[0][1]
    
    atr = df['atr'].iloc[-1]
    ema50 = df['ema50'].iloc[-1]
    ema200 = df['ema200'].iloc[-1]
    adx = df['adx'].iloc[-1]
    
    if adx < 25 or regime in ["RANGE", "COMPRESSION"]:
        return None, None, None, regime
    
    trend_up = get_higher_tf_trend(symbol)
    
    if prob > CONFIDENCE_THRESHOLD and ema50 > ema200:
        if not USE_H1_TREND_FILTER or trend_up:
            return "BUY", prob, atr, regime
    elif prob < (1 - CONFIDENCE_THRESHOLD) and ema50 < ema200:
        if not USE_H1_TREND_FILTER or not trend_up:
            return "SELL", prob, atr, regime
    return None, None, None, regime

def execute_trade(signal, atr, prob, regime, symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return
    
    df = get_data(symbol)
    df = add_features(df)
    if not can_place_trade(tick, regime, df, symbol):
        return
    
    sym_info = mt5.symbol_info(symbol)
    point = sym_info.point
    
    if signal == "BUY":
        price = tick.ask
        sl = round((price - ATR_MULTIPLIER * atr) / point) * point
        tp = round((price + ATR_MULTIPLIER * atr * RISK_REWARD) / point) * point
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        sl = round((price + ATR_MULTIPLIER * atr) / point) * point
        tp = round((price - ATR_MULTIPLIER * atr * RISK_REWARD) / point) * point
        order_type = mt5.ORDER_TYPE_SELL
    
    lot = calculate_lot(price, sl, regime, prob, order_type, symbol)
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "AI v5.3.6",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ {signal} {symbol} | Lot {lot} | Prob {prob:.1%} | Regime {regime}")
        log_trade(datetime.now(timezone.utc), signal, price, sl, tp, prob, regime, lot, symbol)
        per_symbol_state[symbol]['LAST_TRADE_TIME'] = datetime.now(timezone.utc)
        save_per_symbol_state(symbol)
    else:
        print(f"❌ Trade failed for {symbol}: {result.comment}")

def log_trade(time, signal, entry, sl, tp, prob, regime, lot, symbol):
    log_file = SYMBOL_CONFIG[symbol]["LOG_FILE"]
    pd.DataFrame([{
        "time": time, "symbol": symbol, "signal": signal, "entry": entry,
        "sl": sl, "tp": tp, "probability": prob, "regime": regime, "lot": lot
    }]).to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

# ==========================================================
# TRAILING + STATS + DASHBOARD
# ==========================================================
def update_trade_stats():
    now = datetime.now(timezone.utc)
    deals = mt5.history_deals_get(now - timedelta(days=30), now)
    if not deals:
        return
    closed = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT and d.profit != 0]
    if not closed:
        return
    losses = 0
    for d in reversed(closed):
        if d.profit < 0:
            losses += 1
        else:
            break
    account_state['CONSECUTIVE_LOSSES'] = losses
    save_global_state()

def manage_trailing_stop(df, symbol):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    pos = positions[0]
    if pos.magic != MAGIC_NUMBER:
        return
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return
    atr = df['atr'].iloc[-1]
    point = mt5.symbol_info(symbol).point
    
    if pos.type == mt5.POSITION_TYPE_BUY and tick.bid - pos.price_open > TRAILING_ACTIVATION * atr:
        new_sl = round((tick.bid - ATR_MULTIPLIER * atr) / point) * point
        if new_sl > pos.sl + point:
            mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "symbol": symbol, "position": pos.ticket, "sl": new_sl, "tp": pos.tp})
    elif pos.type == mt5.POSITION_TYPE_SELL and pos.price_open - tick.ask > TRAILING_ACTIVATION * atr:
        new_sl = round((tick.ask + ATR_MULTIPLIER * atr) / point) * point
        if new_sl < pos.sl - point:
            mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "symbol": symbol, "position": pos.ticket, "sl": new_sl, "tp": pos.tp})

def print_dashboard():
    print(f"📈 DASHBOARD → Consec Losses: {account_state['CONSECUTIVE_LOSSES']} | Drawdown: {get_drawdown():.1f}%")

# ==========================================================
# MAIN LOOP
# ==========================================================
def run_bot():
    print("🚀 AI Quant Bot v5.3.6 Started - MT5 Login + 12H Retrain")
    update_peak_equity()
    
    model_dict = {}
    scaler_dict = {}
    feature_cols_dict = {}
    
    for sym in SYMBOLS:
        df = get_data(sym)
        df = add_features(df)
        model_dict[sym], scaler_dict[sym], feature_cols_dict[sym] = load_model(df, sym)
    
    last_dashboard = datetime.now(timezone.utc)
    
    while True:
        try:
            # Reconnect if needed
            if not ensure_mt5_connection():
                print("⚠️ Reconnection failed. Retrying in 30 seconds...")
                time.sleep(30)
                continue
            
            if not is_market_open():
                time.sleep(600)
                continue
                
            update_peak_equity()
            update_trade_stats()
            
            if (datetime.now(timezone.utc) - last_dashboard).seconds > 600:
                print_dashboard()
                last_dashboard = datetime.now(timezone.utc)
            
            if daily_loss_exceeded() or weekly_loss_exceeded() or get_drawdown() > MAX_DRAWDOWN_PERCENT:
                print("🚨 Risk limits hit - Bot paused for 1 hour")
                time.sleep(3600)
                continue
            
            if not (TRADING_START_HOUR_UTC <= datetime.now(timezone.utc).hour < TRADING_END_HOUR_UTC):
                time.sleep(300)
                continue
            
            for symbol in SYMBOLS:
                df = get_data(symbol)
                df = add_features(df)
                
                # Check & reload model if 12h passed
                model_dict[symbol], scaler_dict[symbol], feature_cols_dict[symbol] = load_model(df, symbol)
                
                signal, prob, atr, regime = generate_signal(
                    df, model_dict[symbol], scaler_dict[symbol], feature_cols_dict[symbol], symbol
                )
                
                if signal and not position_open(symbol) and not is_on_cooldown(symbol):
                    execute_trade(signal, atr, prob, regime, symbol)
                elif position_open(symbol):
                    manage_trailing_stop(df, symbol)
                    
        except KeyboardInterrupt:
            print("🛑 Bot stopped by user")
            save_global_state()
            for sym in SYMBOLS:
                save_per_symbol_state(sym)
            mt5.shutdown()
            break
        except Exception as e:
            print(f"⚠️ Error in main loop: {e}")
            time.sleep(10)
        
        time.sleep(60)

if __name__ == "__main__":
    run_bot()