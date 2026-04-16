# ==========================================================
# SAFE AI TRADING BOT v6.6.0
# Added symbols | H1 trend filter | Stricter entry rules
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

# ====================== MT5 LINUX/WINDOWS ======================
if platform.system() == "Windows":
    import MetaTrader5 as mt5
else:
    from mt5linux import MetaTrader5
    mt5 = MetaTrader5()

# ====================== CONSTANTS ======================
TIMEFRAME_M5 = 5
TIMEFRAME_H1 = 16385          # MT5 timeframe for H1
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
TRADE_ACTION_DEAL = 1
ORDER_TIME_GTC = 0
ORDER_FILLING_IOC = 1
MAGIC_NUMBER = 20240601

# ====================== CONFIG ======================
MT5_LOGIN = 435112321          # <-- your demo login
MT5_PASSWORD = "Dahir@2036"  # <-- CHANGE THIS
MT5_SERVER = "ExnessKE-MT5Trial9"

# Full symbol list (added all pairs from your request)
SYMBOLS = [
    "EURUSDm", "USDJPYm", "XAUUSDm", "UKOILm", "USOILm", "XNGUSDm",
    "AUDCADm", "AUDCHFm", "AUDCZKm", "AUDDKKm", "AUDHUFm", "AUDJPYm",
    "AUDMXNm", "USDDKKm"
]

# Per-symbol max lot caps (extra safety)
SYMBOL_MAX_LOT = {
    "XAUUSDm": 0.03,
    "UKOILm": 0.05,
    "USOILm": 0.05,
    "XNGUSDm": 0.05,
}

MAX_RISK_PERCENT = 0.005          # 0.5% risk per trade
CONFIDENCE_THRESHOLD = 0.85       # stricter (was 0.78)
COOLDOWN_MINUTES = 30
MAX_DAILY_LOSS_PERCENT = 3.0
BARS = 1200

# Additional strict filters
MIN_ATR_PIPS = 2.0                # minimum ATR in points (avoid dead markets)
MIN_ADX = 20.0                    # require trend strength
MAX_RSI = 85.0                    # skip if RSI > 85
MIN_RSI = 15.0                    # skip if RSI < 15

last_trade_time = {sym: datetime.min.replace(tzinfo=timezone.utc) for sym in SYMBOLS}
daily_start_equity = None
last_date_reset = None

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

# ====================== HELPERS ======================
def has_open_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return False
    return any(p.magic == MAGIC_NUMBER for p in positions)

def is_on_cooldown(symbol):
    return (datetime.now(timezone.utc) - last_trade_time[symbol]).total_seconds() < COOLDOWN_MINUTES * 60

def get_daily_loss():
    global daily_start_equity
    account = mt5.account_info()
    if not account:
        return 0.0
    if daily_start_equity is None:
        daily_start_equity = account.equity
        return 0.0
    loss_pct = (daily_start_equity - account.equity) / daily_start_equity * 100
    return max(0.0, loss_pct)

def reset_daily_equity_if_needed():
    global daily_start_equity, last_date_reset
    now = datetime.now(timezone.utc)
    today = now.date()
    if last_date_reset != today:
        account = mt5.account_info()
        if account:
            daily_start_equity = account.equity
            last_date_reset = today
            logging.info(f"Daily equity reset: {daily_start_equity:.2f}")

def get_data(symbol, timeframe=TIMEFRAME_M5):
    mt5.symbol_select(symbol, True)
    for _ in range(8):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, BARS)
        if rates is not None and len(rates) >= 250:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        time.sleep(2)
    logging.warning(f"Could not load data for {symbol} (tf={timeframe})")
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

def get_h1_trend(symbol):
    """Return 1 (uptrend) or -1 (downtrend) based on H1 EMA50."""
    df = get_data(symbol, TIMEFRAME_H1)
    if df is None or len(df) < 50:
        return 0
    df = add_features(df)
    if df is None:
        return 0
    last_ema50 = df['ema50'].iloc[-1]
    last_close = df['close'].iloc[-1]
    if last_close > last_ema50:
        return 1      # uptrend
    elif last_close < last_ema50:
        return -1     # downtrend
    return 0

def load_or_train_model(df, symbol):
    mf = f"{symbol.lower()}_safe_v66.pkl"
    if os.path.exists(mf):
        try:
            model, scaler, feats = joblib.load(mf)
            logging.info(f"Model loaded for {symbol}")
            return model, scaler, feats
        except:
            pass
    logging.info(f"Training new model for {symbol}...")
    df = df.copy()
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    df.dropna(inplace=True)
    feats = ['ema20', 'ema50', 'rsi', 'atr', 'adx']
    X = df[feats]
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBClassifier(n_estimators=300, max_depth=4, random_state=42)
    model.fit(X_scaled, y)
    joblib.dump((model, scaler, feats), mf)
    return model, scaler, feats

def calculate_lot(symbol, atr, equity):
    info = mt5.symbol_info(symbol)
    if not info:
        return 0.01
    tick_value = info.trade_tick_value
    stop_points = 1.8 * atr
    if stop_points <= 0 or tick_value <= 0:
        return 0.01
    risk_amount = equity * MAX_RISK_PERCENT
    lot = risk_amount / (stop_points * tick_value)
    min_lot = max(info.volume_min, 0.01)
    max_lot = SYMBOL_MAX_LOT.get(symbol, 0.10)
    lot = max(min_lot, min(lot, max_lot))
    lot = round(lot, 2)
    # extra safety: notional value not more than 50% of equity
    notional = lot * info.trade_contract_size * info.point
    if notional > equity * 0.5:
        lot = 0.01
    return lot

def execute_trade(signal, atr, prob, symbol, h1_trend):
    if has_open_position(symbol) or is_on_cooldown(symbol):
        return

    # H1 trend confirmation (strict)
    if signal == "BUY" and h1_trend != 1:
        logging.info(f"{symbol} H1 trend not up → skip BUY")
        return
    if signal == "SELL" and h1_trend != -1:
        logging.info(f"{symbol} H1 trend not down → skip SELL")
        return

    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    account = mt5.account_info()
    if not tick or not info or not account:
        return

    spread = (tick.ask - tick.bid) / info.point
    if spread > 120:
        logging.info(f"High spread on {symbol}: {spread} points - skipping")
        return

    lot = calculate_lot(symbol, atr, account.equity)

    if signal == "BUY":
        price = tick.ask
        sl = price - 1.8 * atr
        tp = price + 3.6 * atr
        order_type = ORDER_TYPE_BUY
    else:
        price = tick.bid
        sl = price + 1.8 * atr
        tp = price - 3.6 * atr
        order_type = ORDER_TYPE_SELL

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
        "comment": "SAFE_v6.6",
        "type_time": ORDER_TIME_GTC,
        "type_filling": ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == 10009:
        logging.info(f"✅ {signal} {symbol} | Lot {lot} | Prob {prob:.1%} | H1 trend: {'UP' if h1_trend==1 else 'DOWN'}")
        last_trade_time[symbol] = datetime.now(timezone.utc)
    else:
        err = result.comment if result else "Unknown"
        logging.error(f"Trade failed {symbol}: {err} (code={result.retcode if result else 'None'})")

# ====================== MAIN ======================
def run_bot():
    global daily_start_equity, last_date_reset
    logging.info("=== SAFE BOT v6.6.0 STARTED (H1 trend filter, stricter rules) ===")

    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        logging.error(f"MT5 init failed: {mt5.last_error()}")
        return

    logging.info("MT5 connected")
    for s in SYMBOLS:
        mt5.symbol_select(s, True)

    models = {}
    for sym in SYMBOLS:
        try:
            df = get_data(sym, TIMEFRAME_M5)
            if df is not None:
                df = add_features(df)
                models[sym] = load_or_train_model(df, sym)
        except Exception as e:
            logging.error(f"Failed to prepare {sym}: {e}")

    logging.info("Bot running. Main loop started.")
    while True:
        try:
            reset_daily_equity_if_needed()
            daily_loss = get_daily_loss()
            if daily_loss > MAX_DAILY_LOSS_PERCENT:
                logging.warning(f"Daily loss {daily_loss:.1f}% > limit. Sleeping 1 hour.")
                time.sleep(3600)
                continue

            for symbol in SYMBOLS:
                if symbol not in models:
                    continue
                if has_open_position(symbol) or is_on_cooldown(symbol):
                    continue

                # Get H1 trend first
                h1_trend = get_h1_trend(symbol)
                if h1_trend == 0:
                    continue   # no clear H1 trend, skip

                df = get_data(symbol, TIMEFRAME_M5)
                if df is None:
                    continue
                df = add_features(df)
                if df is None or len(df) < 2:
                    continue

                # Extract last row features
                last = df.iloc[-1]
                atr = last['atr']
                rsi = last['rsi']
                adx = last['adx']

                # Strict filters
                if atr < MIN_ATR_PIPS:
                    logging.debug(f"{symbol} ATR too low ({atr:.2f}) → skip")
                    continue
                if adx < MIN_ADX:
                    logging.debug(f"{symbol} ADX weak ({adx:.1f}) → skip")
                    continue
                if rsi > MAX_RSI or rsi < MIN_RSI:
                    logging.debug(f"{symbol} RSI extreme ({rsi:.1f}) → skip")
                    continue

                model, scaler, feats = models[symbol]
                latest = scaler.transform(df[feats].iloc[-1:])
                prob = model.predict_proba(latest)[0][1]

                # Determine signal (stricter threshold)
                if prob >= CONFIDENCE_THRESHOLD:
                    signal = "BUY"
                elif prob <= (1 - CONFIDENCE_THRESHOLD):
                    signal = "SELL"
                else:
                    continue

                execute_trade(signal, atr, prob, symbol, h1_trend)

            time.sleep(60)
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            time.sleep(15)

if __name__ == "__main__":
    run_bot()