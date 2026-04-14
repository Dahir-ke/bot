# ==========================================================
# SAFE AI TRADING BOT v6.4.0 
# Fixed Risk Management + One Trade Per Symbol + Daily Protection
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
    logging.info("Running on Windows")
else:
    from mt5linux import MetaTrader5
    mt5 = MetaTrader5()
    logging.info("Running on Linux with mt5linux")

# ====================== MT5 CONSTANTS ======================
TIMEFRAME_M5      = 5
ORDER_TYPE_BUY    = 0
ORDER_TYPE_SELL   = 1
TRADE_ACTION_DEAL = 1
ORDER_TIME_GTC    = 0
ORDER_FILLING_IOC = 1
MAGIC_NUMBER      = 20240601

# ====================== CONFIG ======================
MT5_LOGIN = 435112321
MT5_PASSWORD = "Dahir@2036"
MT5_SERVER = "ExnessKE-MT5Trial9"

SYMBOLS = ["EURUSDm", "USDJPYm", "XAUUSDm", "UKOILm", "USOILm", "XNGUSDm",
           "AUDCADm", "AUDCHFm", "AUDJPYm"]

SYMBOL_CONFIG = {sym: {"MAX_SPREAD": 120} for sym in SYMBOLS}

MAX_RISK_PERCENT = 0.005        # 0.5% risk per trade
CONFIDENCE_THRESHOLD = 0.78
COOLDOWN_MINUTES = 30
MAX_DAILY_LOSS_PERCENT = 3.0
BARS = 1200

# Per-symbol maximum lot size (to further protect against extreme volatility)
SYMBOL_MAX_LOT = {
    "XAUUSDm": 0.03,   # Gold limited to 0.03 lot
    "UKOILm": 0.05,
    "USOILm": 0.05,
    # All others default to 0.10 (see calculate_lot)
}

last_trade_time = {sym: datetime.min.replace(tzinfo=timezone.utc) for sym in SYMBOLS}
daily_start_equity = None
last_date_reset = None

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
    if not account:
        return 0.0
    if daily_start_equity is None:
        daily_start_equity = account.equity
        return 0.0
    loss_pct = (daily_start_equity - account.equity) / daily_start_equity * 100
    return loss_pct

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

def get_data(symbol):
    mt5.symbol_select(symbol, True)
    for _ in range(8):
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_M5, 0, BARS)
        if rates is not None and len(rates) >= 250:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        time.sleep(2)
    logging.warning(f"Could not load data for {symbol}")
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
    mf = f"{symbol.lower()}_safe_v64.pkl"
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
    """Return lot size that risks exactly MAX_RISK_PERCENT of equity."""
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        return 0.01
    # point value in account currency per point for 1 lot
    point_value = symbol_info.trade_tick_value
    # stop distance in points (same unit as tick value)
    stop_distance = 1.8 * atr
    if stop_distance <= 0 or point_value <= 0:
        return 0.01

    risk_amount = equity * MAX_RISK_PERCENT
    lot = risk_amount / (stop_distance * point_value)

    # Apply min/max limits
    min_lot = max(symbol_info.volume_min, 0.01)
    max_lot = SYMBOL_MAX_LOT.get(symbol, 0.10)
    lot = max(min_lot, min(lot, max_lot))
    # Round to 2 decimals (broker lot step)
    lot = round(lot, 2)
    return lot

def execute_trade(signal, atr, prob, symbol):
    if has_open_position(symbol) or is_on_cooldown(symbol):
        return

    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    account = mt5.account_info()
    if not tick or not info or not account:
        return

    spread = (tick.ask - tick.bid) / info.point
    if spread > SYMBOL_CONFIG[symbol]["MAX_SPREAD"]:
        logging.info(f"High spread on {symbol} - skipping")
        return

    lot = calculate_lot(symbol, atr, account.equity)

    if signal == "BUY":
        price = tick.ask
        sl = price - 1.8 * atr
        tp = price + 1.8 * atr * 2.0
        order_type = ORDER_TYPE_BUY
    else:
        price = tick.bid
        sl = price + 1.8 * atr
        tp = price - 1.8 * atr * 2.0
        order_type = ORDER_TYPE_SELL

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
        "comment": "SAFE_v6.4",
        "type_time": ORDER_TIME_GTC,
        "type_filling": ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == 10009:
        risk_usd = lot * (1.8 * atr) * info.trade_tick_value
        logging.info(f"✅ TRADE OPENED → {signal} {symbol} | Lot {lot} | Risk ${risk_usd:.2f} ({MAX_RISK_PERCENT*100:.1f}% of equity)")
        last_trade_time[symbol] = datetime.now(timezone.utc)
    else:
        logging.error(f"Trade failed on {symbol}: {result.comment if result else 'Unknown'} (retcode={result.retcode if result else 'None'})")

# ====================== MAIN ======================
def run_bot():
    global daily_start_equity, last_date_reset
    logging.info("=== SAFE BOT v6.4.0 STARTED ===")

    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        logging.error(f"MT5 login failed: {mt5.last_error()}")
        return

    for s in SYMBOLS:
        mt5.symbol_select(s, True)

    models = {}
    for sym in SYMBOLS:
        try:
            df = get_data(sym)
            if df is not None:
                df = add_features(df)
                models[sym] = load_or_train_model(df, sym)
        except Exception as e:
            logging.error(f"Failed to prepare {sym}: {e}")

    logging.info("Bot running in SAFE mode with dynamic risk sizing...\n")

    while True:
        try:
            # Reset daily equity at start of new UTC day
            reset_daily_equity_if_needed()

            daily_loss = get_daily_loss()
            if daily_loss > MAX_DAILY_LOSS_PERCENT:
                logging.warning(f"DAILY LOSS LIMIT HIT ({daily_loss:.1f}%) → Pausing 1 hour")
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
            logging.error(f"Main loop error: {e}")
            time.sleep(15)

if __name__ == "__main__":
    run_bot()