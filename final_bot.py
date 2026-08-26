# ============================================================
# SAFE AI TRADING BOT v8.0.0
# ============================================================
#
# Major improvements over v7.3.0:
#
# 1. Strict walk-forward validation
# 2. Proper OOF probability handling
# 3. Robust probability calibration
# 4. Model metadata/version validation
# 5. Atomic model persistence
# 6. Safer lot sizing
# 7. Broker volume min/max/step handling
# 8. Broker filling-mode detection
# 9. Margin pre-check
# 10. Stale market-data protection
# 11. Duplicate candle protection
# 12. Robust MT5 order result handling
# 13. Robust position-ticket recovery
# 14. Daily realized + floating drawdown protection
# 15. SQLite decision ledger
# 16. Trade ledger
# 17. Processed-deal protection
# 18. Model quality gate
# 19. Better logging
# 20. Graceful shutdown
#
# Strategy:
#   M5:
#       XGBoost BUY model
#       XGBoost SELL model
#
#   Confirmation:
#       H1 EMA20 / EMA50 / ADX / EMA50 slope
#
#   Risk:
#       Maximum 0.5% equity risk per trade
#       SL = 1.8 ATR
#       TP = 3.6 ATR
#
# ============================================================

import os
import sys
import time
import math
import json
import uuid
import shutil
import signal
import sqlite3
import logging
import platform
from pathlib import Path
from datetime import datetime, timezone, timedelta

import joblib
import numpy as np
import pandas as pd
import ta
import xgboost as xgb

from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ============================================================
# OPTIONAL STATUS MODULE
# ============================================================

try:
    import bot_status as status
except Exception:
    status = None


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# MT5 CONNECTION
# ============================================================

# initialize()'s auto-detect (no path=) looks for a default-named
# "MetaTrader 5" install and fails with -10003 "MetaTrader 5 x64 not
# found" against a broker-branded one - true on native Windows too, not
# just the old Wine path, so this is set unconditionally rather than
# only in the non-Windows branch below. A whole day of "-10005 IPC
# timeout" across Wine, Docker, and even a from-scratch bridge turned
# out to have nothing to do with any of that: the account's server
# (ExnessKE-MT5Real9) belongs to a distinct, separately-licensed
# Exness entity ("Exness (KE) Limited") with its own terminal build,
# installed to this exact path - the generic "MetaTrader 5 EXNESS"
# terminal silently failed account authorization no matter what it ran
# on, native Windows included. See exnesske5setup.exe from exness.ke,
# not the generic exness.com installer, if this ever needs reinstalling.
MT5_TERMINAL_PATH = os.environ.get(
    "MT5_TERMINAL_PATH", r"C:\Program Files\ExnessKE MT5 Terminal\terminal64.exe"
)

if platform.system() == "Windows":

    import MetaTrader5 as mt5

else:

    # Cross-machine bridge path - not what's actually deployed (the bot
    # runs directly on the same Windows VPS as the terminal now, no
    # network hop needed), kept for reference/fallback only.
    from mt5linux import MetaTrader5

    mt5 = MetaTrader5(
        host=os.environ.get("MT5_BRIDGE_HOST", "localhost"),
        port=int(os.environ.get("MT5_BRIDGE_PORT", "18812")),
    )


# ============================================================
# BASIC CONSTANTS
# ============================================================

TIMEFRAME_M5 = 5
TIMEFRAME_H1 = 16385

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1

TRADE_ACTION_DEAL = 1

ORDER_TIME_GTC = 0

MAGIC_NUMBER = 20240601

BOT_VERSION = "8.0.0"

MODEL_VERSION = "SAFE_V8"

# Defaults to next to this file (unchanged local behavior) - the Docker
# deployment sets BOT_DATA_DIR to a bind-mounted volume (see
# docker-compose.yml) so trained models, the SQLite ledger, and the log
# survive `docker compose build`/container recreation instead of living
# in the container's ephemeral layer and vanishing on next deploy.
BOT_DATA_DIR = os.environ.get("BOT_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))

DB_FILE = os.path.join(BOT_DATA_DIR, "trading_bot_v8.db")

LOG_FILE = os.path.join(BOT_DATA_DIR, "trading_bot_v8.log")

# Written by the dashboard (see dashboard_server.py's /api/kill-switch
# route) on the same shared BOT_DATA_DIR volume - read fresh every loop
# iteration rather than cached, so a manual stop takes effect within one
# cycle. Only blocks NEW entries; existing positions keep their broker-
# side SL/TP and are left alone rather than force-closed from here, since
# programmatic mass-closing is its own source of execution risk.
KILL_SWITCH_FILE = os.path.join(BOT_DATA_DIR, "kill_switch.json")

MODEL_DIR = Path(BOT_DATA_DIR) / "models_v8"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MT5 ACCOUNT
# ============================================================

MT5_LOGIN = int(os.environ["MT5_LOGIN"])
MT5_PASSWORD = os.environ["MT5_PASSWORD"]
MT5_SERVER = os.environ["MT5_SERVER"]


# ============================================================
# SYMBOLS
# ============================================================

SYMBOLS = [
    "EURUSDm",
    "USDJPYm",
    "XAUUSDm",
    "UKOILm",
    "USOILm",
    "XNGUSDm",
    "AUDCADm",
    "AUDCHFm",
    "AUDCZKm",
    "AUDDKKm",
    "AUDHUFm",
    "AUDJPYm",
    "AUDMXNm",
    "USDDKKm",
]


# ============================================================
# SYMBOL RISK LIMITS
# ============================================================

SYMBOL_MAX_LOT = {
    "XAUUSDm": 0.03,
    "UKOILm": 0.05,
    "USOILm": 0.05,
    "XNGUSDm": 0.05,
}

DEFAULT_MAX_LOT = 0.10


# ============================================================
# SPREAD LIMITS
# ============================================================

SYMBOL_MAX_SPREAD = {
    "EURUSDm": 30,
    "USDJPYm": 30,
    "XAUUSDm": 80,
    "UKOILm": 50,
    "USOILm": 50,
    "XNGUSDm": 80,
}

DEFAULT_MAX_SPREAD = float(os.environ.get("DEFAULT_MAX_SPREAD", "60"))


# ============================================================
# ATR MINIMUMS
# ============================================================

SYMBOL_MIN_ATR_POINTS = {
    "EURUSDm": 3.0,
    "USDJPYm": 3.0,
    "XAUUSDm": 15.0,
    "UKOILm": 10.0,
    "USOILm": 10.0,
    "XNGUSDm": 20.0,
}


# ============================================================
# STRATEGY CONFIG
# ============================================================

MAX_RISK_PERCENT = float(os.environ.get("MAX_RISK_PERCENT", "0.005"))

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.85"))

MIN_PROB_GAP = float(os.environ.get("MIN_PROB_GAP", "0.10"))

COOLDOWN_MINUTES = float(os.environ.get("COOLDOWN_MINUTES", "30"))

MAX_DAILY_LOSS_PERCENT = 3.0

MAX_CONCURRENT_TRADES = 3

BARS = 10000

MIN_TRAINING_SAMPLES = 1500

MIN_ADX = float(os.environ.get("MIN_ADX", "20.0"))

MAX_RSI = float(os.environ.get("MAX_RSI", "85.0"))

MIN_RSI = float(os.environ.get("MIN_RSI", "15.0"))

SL_ATR_MULT = float(os.environ.get("SL_ATR_MULT", "1.8"))

TP_ATR_MULT = float(os.environ.get("TP_ATR_MULT", "3.6"))

TARGET_LOOKAHEAD = 100

MAX_SPREAD_ATR_RATIO = float(os.environ.get("MAX_SPREAD_ATR_RATIO", "0.30"))

# Strategy-logic filter (not an execution-quality one, unlike the ADX/
# spread checks above) - blocks trading against the H1 trend. Default
# on everywhere; can be turned off per-deployment (e.g. demo testing)
# via env, but doing so changes what the bot considers a valid signal,
# not just when it's willing to act on one.
REQUIRE_H1_ALIGNMENT = os.environ.get(
    "REQUIRE_H1_ALIGNMENT", "true"
).strip().lower() not in ("false", "0", "no")

MAX_DATA_AGE_MINUTES = 10

LOOP_INTERVAL_SECONDS = 60

ORDER_DEVIATION = 20

MODEL_MAX_AGE_HOURS = 168

MIN_MODEL_ROC_AUC = float(os.environ.get("MIN_MODEL_ROC_AUC", "0.55"))

MIN_HIGH_CONF_PRECISION = float(os.environ.get("MIN_HIGH_CONF_PRECISION", "0.50"))

MIN_HIGH_CONF_SIGNALS = int(os.environ.get("MIN_HIGH_CONF_SIGNALS", "10"))

MAX_NOTIONAL_EQUITY_PERCENT = float(
    os.environ.get("MAX_NOTIONAL_EQUITY_PERCENT", "0.30")
)


# ============================================================
# GLOBAL STATE
# ============================================================

last_trade_time = {
    sym: datetime.min.replace(tzinfo=timezone.utc)
    for sym in SYMBOLS
}

last_processed_bar = {}

# Latest per-symbol decision, dashboard-shaped (see bot_status.py /
# dashboard.html) - kept alongside the SQLite decisions table rather than
# replacing it: SQLite is the durable audit ledger, this is just "what to
# show right now" and only ever needs the most recent row per symbol.
signals_snapshot = {}

# Per-symbol/side walk-forward diagnostics from the most recent quality
# gate evaluation (see model_passes_quality_gate) - dashboard-shaped, so
# "why isn't this symbol trading" is answerable from training results,
# not just the current bar's skip_reason.
model_quality_snapshot = {}

daily_start_equity = None

daily_start_balance = None

last_date_reset = None

daily_loss_lock = False

last_deal_check = datetime.now(timezone.utc) - timedelta(days=1)

shutdown_requested = False

model_training_lock = False


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

console = logging.StreamHandler()

console.setLevel(logging.INFO)

logging.getLogger().addHandler(console)


# ============================================================
# HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except Exception:
        return default


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def normalize_probability(value):
    value = safe_float(value, 0.0)
    return clamp(value, 0.0, 1.0)


def atomic_joblib_dump(obj, path):
    """
    Save model atomically to avoid corrupt model files
    if the process stops during serialization.
    """

    path = Path(path)

    tmp_path = path.with_suffix(
        path.suffix + f".tmp_{uuid.uuid4().hex}"
    )

    joblib.dump(obj, tmp_path)

    os.replace(tmp_path, path)


def model_file(symbol, side):
    return MODEL_DIR / f"{symbol.lower()}_{MODEL_VERSION.lower()}_{side.lower()}.pkl"


# ============================================================
# DATABASE
# ============================================================

def init_db():

    with sqlite3.connect(DB_FILE) as conn:

        c = conn.cursor()

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_state (
                date TEXT PRIMARY KEY,
                starting_equity REAL NOT NULL,
                starting_balance REAL NOT NULL,
                daily_loss_lock INTEGER NOT NULL DEFAULT 0
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                position_ticket INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                sl REAL NOT NULL,
                tp REAL NOT NULL,
                initial_volume REAL NOT NULL,
                closed_volume REAL DEFAULT 0,
                initial_risk REAL NOT NULL,
                prob REAL NOT NULL,
                gross_profit REAL DEFAULT 0,
                commission REAL DEFAULT 0,
                swap REAL DEFAULT 0,
                fee REAL DEFAULT 0,
                net_profit REAL DEFAULT 0,
                status TEXT DEFAULT 'OPEN'
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_deals (
                deal_ticket INTEGER PRIMARY KEY,
                position_ticket INTEGER,
                time TEXT
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                bar_time TEXT,
                buy_prob REAL,
                sell_prob REAL,
                prob_gap REAL,
                atr REAL,
                adx REAL,
                rsi REAL,
                h1_trend INTEGER,
                spread_points REAL,
                spread_atr_ratio REAL,
                signal TEXT,
                decision TEXT,
                reason TEXT
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                samples INTEGER,
                positive_rate REAL,
                roc_auc REAL,
                brier REAL,
                logloss REAL,
                precision_high_conf REAL,
                recall_high_conf REAL,
                coverage REAL,
                signals_high_conf INTEGER,
                model_file TEXT
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                volume REAL,
                price REAL,
                sl REAL,
                tp REAL,
                retcode INTEGER,
                order_ticket INTEGER,
                deal_ticket INTEGER,
                position_ticket INTEGER,
                message TEXT
            )
            """
        )

        conn.commit()


# ============================================================
# DECISION LEDGER
# ============================================================

def log_decision(
    symbol,
    buy_prob,
    sell_prob,
    atr,
    adx,
    rsi,
    h1_trend,
    spread_pts,
    spread_atr_ratio,
    signal,
    decision,
    reason,
    bar_time=None,
):

    prob_gap = abs(
        safe_float(buy_prob) -
        safe_float(sell_prob)
    )

    display_prob = (
        safe_float(buy_prob)
        if signal == "BUY"
        else safe_float(sell_prob)
        if signal == "SELL"
        else max(
            safe_float(buy_prob),
            safe_float(sell_prob),
        )
    )

    signals_snapshot[symbol] = {
        "position_open": reason == "NO_TRADE_POSITION_EXISTS",
        "cooldown": reason == "NO_TRADE_COOLDOWN",
        "skip_reason": reason if decision == "SKIP" else None,
        "signal": signal if signal in ("BUY", "SELL") else None,
        "prob": display_prob,
        "atr": safe_float(atr),
        "adx": safe_float(adx),
        "rsi": safe_float(rsi),
        "h1_trend": h1_trend,
    }

    with sqlite3.connect(DB_FILE) as conn:

        conn.execute(
            """
            INSERT INTO decisions (
                timestamp,
                symbol,
                bar_time,
                buy_prob,
                sell_prob,
                prob_gap,
                atr,
                adx,
                rsi,
                h1_trend,
                spread_points,
                spread_atr_ratio,
                signal,
                decision,
                reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now().isoformat(),
                symbol,
                bar_time,
                buy_prob,
                sell_prob,
                prob_gap,
                atr,
                adx,
                rsi,
                h1_trend,
                spread_pts,
                spread_atr_ratio,
                signal,
                decision,
                reason,
            ),
        )

        conn.commit()


# ============================================================
# DAILY STATE
# ============================================================

def load_state():

    global daily_start_equity
    global daily_start_balance
    global last_date_reset
    global daily_loss_lock

    today = utc_now().date().isoformat()

    with sqlite3.connect(DB_FILE) as conn:

        row = conn.execute(
            """
            SELECT
                starting_equity,
                starting_balance,
                daily_loss_lock
            FROM daily_state
            WHERE date = ?
            """,
            (today,),
        ).fetchone()

    if row:

        daily_start_equity = safe_float(row[0])

        daily_start_balance = safe_float(row[1])

        daily_loss_lock = bool(row[2])

        last_date_reset = utc_now().date()

        logging.info(
            "Loaded daily state | "
            f"Equity={daily_start_equity:.2f} | "
            f"Balance={daily_start_balance:.2f} | "
            f"Lock={daily_loss_lock}"
        )

    else:

        logging.info(
            "No state for today. "
            "Daily state will be initialized after MT5 connection."
        )


def save_state():

    if daily_start_equity is None:
        return

    if daily_start_balance is None:
        return

    today = utc_now().date().isoformat()

    with sqlite3.connect(DB_FILE) as conn:

        conn.execute(
            """
            INSERT INTO daily_state (
                date,
                starting_equity,
                starting_balance,
                daily_loss_lock
            )
            VALUES (?, ?, ?, ?)

            ON CONFLICT(date)
            DO UPDATE SET
                starting_equity = excluded.starting_equity,
                starting_balance = excluded.starting_balance,
                daily_loss_lock = excluded.daily_loss_lock
            """,
            (
                today,
                daily_start_equity,
                daily_start_balance,
                int(daily_loss_lock),
            ),
        )

        conn.commit()


def initialize_daily_state():

    global daily_start_equity
    global daily_start_balance
    global last_date_reset
    global daily_loss_lock

    account = mt5.account_info()

    if account is None:
        return False

    today = utc_now().date()

    if last_date_reset == today:
        return True

    daily_start_equity = safe_float(account.equity)

    daily_start_balance = safe_float(account.balance)

    last_date_reset = today

    daily_loss_lock = False

    save_state()

    logging.info(
        "Daily state initialized | "
        f"Equity={daily_start_equity:.2f} | "
        f"Balance={daily_start_balance:.2f}"
    )

    return True


def reset_daily_equity_if_needed():

    global daily_start_equity
    global daily_start_balance
    global last_date_reset
    global daily_loss_lock

    today = utc_now().date()

    if last_date_reset == today:
        return

    account = mt5.account_info()

    if account is None:
        return

    stats = compute_previous_day_stats()

    if stats:

        logging.info(
            "Previous day | "
            f"Trades={stats['total_trades']} | "
            f"WR={stats['win_rate']:.1%} | "
            f"PF={stats['profit_factor']:.2f} | "
            f"Expectancy={stats['expectancy_r']:.2f}R | "
            f"Net={stats['net_profit']:.2f}"
        )

    daily_start_equity = safe_float(account.equity)

    daily_start_balance = safe_float(account.balance)

    last_date_reset = today

    daily_loss_lock = False

    save_state()

    logging.info(
        "Daily reset | "
        f"Equity={daily_start_equity:.2f} | "
        f"Balance={daily_start_balance:.2f}"
    )


# ============================================================
# KILL SWITCH
# ============================================================

def is_kill_switch_active():

    if not os.path.exists(KILL_SWITCH_FILE):
        return False, None

    try:

        with open(KILL_SWITCH_FILE) as f:
            data = json.load(f)

    except (json.JSONDecodeError, OSError):

        return False, None

    if not data.get("active"):
        return False, None

    return True, data.get("reason", "Kill switch active")


# ============================================================
# DAILY RISK
# ============================================================

def check_daily_loss_limits():

    global daily_loss_lock

    account = mt5.account_info()

    if account is None:
        return True

    if daily_start_equity is None:
        return True

    if daily_start_balance is None:
        return True

    equity = safe_float(account.equity)

    balance = safe_float(account.balance)

    floating_dd = max(
        0.0,
        (
            (daily_start_equity - equity)
            / daily_start_equity
            * 100
        ),
    ) if daily_start_equity > 0 else 0

    realized_dd = max(
        0.0,
        (
            (daily_start_balance - balance)
            / daily_start_balance
            * 100
        ),
    ) if daily_start_balance > 0 else 0

    if (
        floating_dd >= MAX_DAILY_LOSS_PERCENT
        or realized_dd >= MAX_DAILY_LOSS_PERCENT
    ):

        if not daily_loss_lock:

            daily_loss_lock = True

            save_state()

            logging.warning(
                "DAILY LOSS LOCK ACTIVATED | "
                f"Floating={floating_dd:.2f}% | "
                f"Realized={realized_dd:.2f}%"
            )

        return True

    return daily_loss_lock


# ============================================================
# DAILY STATISTICS
# ============================================================

def compute_previous_day_stats():

    yesterday = (
        utc_now().date() -
        timedelta(days=1)
    ).isoformat()

    with sqlite3.connect(DB_FILE) as conn:

        rows = conn.execute(
            """
            SELECT
                net_profit,
                initial_risk
            FROM trades
            WHERE status = 'CLOSED'
            AND substr(entry_time, 1, 10) = ?
            """,
            (yesterday,),
        ).fetchall()

    if not rows:
        return None

    valid = [
        (safe_float(p), safe_float(r))
        for p, r in rows
        if safe_float(r) > 0
    ]

    if not valid:
        return None

    wins = [
        x for x in valid
        if x[0] > 0
    ]

    losses = [
        x for x in valid
        if x[0] <= 0
    ]

    win_rate = len(wins) / len(valid)

    avg_win_r = (
        np.mean(
            [p / r for p, r in wins]
        )
        if wins else 0
    )

    avg_loss_r = (
        np.mean(
            [p / r for p, r in losses]
        )
        if losses else 0
    )

    expectancy = (
        win_rate * avg_win_r
        + (1 - win_rate) * avg_loss_r
    )

    gross_profit = sum(
        p for p, _ in wins
    )

    gross_loss = abs(
        sum(p for p, _ in losses)
    )

    return {
        "win_rate": win_rate,
        "profit_factor": (
            gross_profit / gross_loss
            if gross_loss > 0
            else float("inf")
        ),
        "expectancy_r": expectancy,
        "total_trades": len(valid),
        "net_profit": gross_profit - gross_loss,
    }


# ============================================================
# MT5 CONNECTION
# ============================================================

def initialize_mt5():

    logging.info(
        f"Initializing MT5 | Platform={platform.system()}"
    )

    init_kwargs = dict(
        login=MT5_LOGIN,
        password=MT5_PASSWORD,
        server=MT5_SERVER,
    )

    if MT5_TERMINAL_PATH:
        init_kwargs["path"] = MT5_TERMINAL_PATH

    try:

        connected = mt5.initialize(**init_kwargs)

    except TypeError:

        connected = mt5.initialize()

    if not connected:

        logging.error(
            f"MT5 initialization failed: {mt5.last_error()}"
        )

        return False

    account = mt5.account_info()

    if account is None:

        logging.error(
            "MT5 connected but account_info() failed."
        )

        return False

    logging.info(
        "MT5 connected | "
        f"Login={getattr(account, 'login', 'N/A')} | "
        f"Balance={account.balance:.2f} | "
        f"Equity={account.equity:.2f}"
    )

    return True


def ensure_symbol(symbol):

    info = mt5.symbol_info(symbol)

    if info is None:

        logging.error(
            f"{symbol}: symbol_info unavailable"
        )

        return False

    if not info.visible:

        if not mt5.symbol_select(symbol, True):

            logging.error(
                f"{symbol}: failed to select symbol"
            )

            return False

    return True


# ============================================================
# MARKET DATA
# ============================================================

def get_data(
    symbol,
    timeframe=TIMEFRAME_M5,
    closed_only=True,
):

    start_pos = 1 if closed_only else 0

    for attempt in range(5):

        try:

            rates = mt5.copy_rates_from_pos(
                symbol,
                timeframe,
                start_pos,
                BARS,
            )

            if (
                rates is not None
                and len(rates) >= 250
            ):

                df = pd.DataFrame(rates)

                df["time"] = pd.to_datetime(
                    df["time"],
                    unit="s",
                    utc=True,
                )

                df.sort_values(
                    "time",
                    inplace=True,
                )

                df.reset_index(
                    drop=True,
                    inplace=True,
                )

                return df

        except Exception as exc:

            logging.warning(
                f"{symbol}: data attempt {attempt + 1} "
                f"failed: {exc}"
            )

        time.sleep(1)

    return None


def is_data_fresh(df):

    if df is None or df.empty:
        return False

    last_time = df["time"].iloc[-1]

    age = (
        utc_now() -
        last_time.to_pydatetime()
    ).total_seconds() / 60

    return age <= MAX_DATA_AGE_MINUTES


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "ema20",
    "ema50",
    "rsi",
    "atr",
    "adx",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "bb_percent",
    "donchian_breakout",
    "stochrsi_k",
    "stochrsi_d",
    "candle_body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "dist_ema20",
    "dist_ema50",
    "atr_norm_momentum",
    "hour",
    "session",
]


def add_features(df):

    if df is None:
        return None

    df = df.copy()

    df["ema20"] = ta.trend.ema_indicator(
        df["close"],
        window=20,
    )

    df["ema50"] = ta.trend.ema_indicator(
        df["close"],
        window=50,
    )

    df["rsi"] = ta.momentum.rsi(
        df["close"],
        window=14,
    )

    df["atr"] = ta.volatility.average_true_range(
        df["high"],
        df["low"],
        df["close"],
        window=14,
    )

    df["adx"] = ta.trend.adx(
        df["high"],
        df["low"],
        df["close"],
        window=14,
    )

    macd = ta.trend.MACD(
        df["close"],
    )

    df["macd"] = macd.macd()

    df["macd_signal"] = macd.macd_signal()

    df["macd_hist"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(
        df["close"],
        window=20,
        window_dev=2,
    )

    df["bb_width"] = bb.bollinger_wband()

    df["bb_percent"] = bb.bollinger_pband()

    df["donchian_high_prev"] = (
        df["high"]
        .rolling(20)
        .max()
        .shift(1)
    )

    df["donchian_low_prev"] = (
        df["low"]
        .rolling(20)
        .min()
        .shift(1)
    )

    df["donchian_breakout"] = 0

    df.loc[
        df["close"] > df["donchian_high_prev"],
        "donchian_breakout",
    ] = 1

    df.loc[
        df["close"] < df["donchian_low_prev"],
        "donchian_breakout",
    ] = -1

    stochrsi = ta.momentum.StochRSIIndicator(
        df["close"],
        window=14,
        smooth1=3,
        smooth2=3,
    )

    df["stochrsi_k"] = (
        stochrsi.stochrsi_k()
    )

    df["stochrsi_d"] = (
        stochrsi.stochrsi_d()
    )

    df["body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["range"] = (
        df["high"] -
        df["low"]
    )

    safe_range = df["range"].replace(
        0,
        np.nan,
    )

    df["candle_body_ratio"] = (
        df["body"] /
        safe_range
    )

    df["upper_wick"] = (
        df["high"] -
        df[["close", "open"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["close", "open"]].min(axis=1) -
        df["low"]
    )

    df["upper_wick_ratio"] = (
        df["upper_wick"] /
        safe_range
    )

    df["lower_wick_ratio"] = (
        df["lower_wick"] /
        safe_range
    )

    safe_atr = df["atr"].replace(
        0,
        np.nan,
    )

    df["dist_ema20"] = (
        (df["close"] - df["ema20"])
        / safe_atr
    )

    df["dist_ema50"] = (
        (df["close"] - df["ema50"])
        / safe_atr
    )

    df["atr_norm_momentum"] = (
        (df["close"] - df["close"].shift(5))
        / safe_atr
    )

    df["hour"] = df["time"].dt.hour

    df["session"] = np.select(
        [
            df["hour"] < 8,
            (
                (df["hour"] >= 8)
                & (df["hour"] < 16)
            ),
            df["hour"] >= 16,
        ],
        [
            0,
            1,
            2,
        ],
        default=0,
    )

    return df


# ============================================================
# H1 TREND
# ============================================================

def get_h1_trend(symbol):

    df = get_data(
        symbol,
        TIMEFRAME_H1,
        closed_only=True,
    )

    if df is None:
        return 0

    if not is_data_fresh(df):
        return 0

    if len(df) < 60:
        return 0

    df = add_features(df)

    if df is None:
        return 0

    last = df.iloc[-1]

    ema50_slope = (
        last["ema50"] -
        df["ema50"].iloc[-5]
    )

    up = (
        last["close"] > last["ema50"]
        and last["ema20"] > last["ema50"]
        and ema50_slope > 0
        and last["adx"] >= MIN_ADX
    )

    down = (
        last["close"] < last["ema50"]
        and last["ema20"] < last["ema50"]
        and ema50_slope < 0
        and last["adx"] >= MIN_ADX
    )

    if up:
        return 1

    if down:
        return -1

    return 0


# ============================================================
# TARGET GENERATION
# ============================================================

def add_tp_sl_targets(
    df,
    sl_mult=SL_ATR_MULT,
    tp_mult=TP_ATR_MULT,
    lookahead=TARGET_LOOKAHEAD,
    spread_cost=0.0,
):

    # MT5 rate bars are bid-based. A real BUY enters at ask (bid +
    # spread) and exits at bid; a real SELL enters at bid and exits at
    # ask. Previously this labeled a "win" purely off the bid-based
    # bars with no cost at all, which is a materially easier bar to
    # clear than a real trade actually faces - both directions start
    # every position already down the spread. Shifting both TP and SL
    # by spread_cost (up for BUY, down for SELL) makes the win/loss
    # labels reflect that, rather than training on a zero-cost fiction.
    df = df.copy()

    buy_targets = np.full(
        len(df),
        np.nan,
    )

    sell_targets = np.full(
        len(df),
        np.nan,
    )

    highs = df["high"].to_numpy()

    lows = df["low"].to_numpy()

    closes = df["close"].to_numpy()

    atrs = df["atr"].to_numpy()

    for i in range(len(df) - 1):

        atr = atrs[i]

        if (
            not np.isfinite(atr)
            or atr <= 0
        ):
            continue

        buy_tp = (
            closes[i] +
            tp_mult * atr +
            spread_cost
        )

        buy_sl = (
            closes[i] -
            sl_mult * atr +
            spread_cost
        )

        sell_tp = (
            closes[i] -
            tp_mult * atr -
            spread_cost
        )

        sell_sl = (
            closes[i] +
            sl_mult * atr -
            spread_cost
        )

        buy_result = None

        sell_result = None

        end = min(
            i + lookahead + 1,
            len(df),
        )

        for j in range(i + 1, end):

            h = highs[j]

            l = lows[j]

            # -------------------------
            # BUY
            # -------------------------

            if buy_result is None:

                hit_sl = l <= buy_sl

                hit_tp = h >= buy_tp

                if hit_sl and hit_tp:

                    # Same candle touched both.
                    # We deliberately discard it because
                    # the intrabar order is unknowable.
                    buy_result = np.nan

                elif hit_sl:

                    buy_result = 0

                elif hit_tp:

                    buy_result = 1

            # -------------------------
            # SELL
            # -------------------------

            if sell_result is None:

                hit_sl = h >= sell_sl

                hit_tp = l <= sell_tp

                if hit_sl and hit_tp:

                    sell_result = np.nan

                elif hit_sl:

                    sell_result = 0

                elif hit_tp:

                    sell_result = 1

            if (
                buy_result is not None
                and sell_result is not None
            ):
                break

        if buy_result in (0, 1):
            buy_targets[i] = buy_result

        if sell_result in (0, 1):
            sell_targets[i] = sell_result

    df["buy_target"] = buy_targets

    df["sell_target"] = sell_targets

    return df


# ============================================================
# XGBOOST CONFIG
# ============================================================

def create_xgb_model():

    return xgb.XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_alpha=0.05,
        reg_lambda=1.5,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=2,
        tree_method="hist",
    )


# ============================================================
# SAFE CALIBRATION
# ============================================================

def calibrate_model(
    model,
    X_calib,
    y_calib,
):

    unique_classes = np.unique(y_calib)

    if len(unique_classes) < 2:

        logging.warning(
            "Calibration set contains only one class. "
            "Returning uncalibrated model."
        )

        return model

    try:

        # sklearn >=1.6 removed cv="prefit" entirely (silently falling
        # through to the cv=3 branch below on every single fold - it
        # doesn't raise until .fit() runs the actual grid, confirmed by
        # hand: "The 'cv' parameter ... Got 'prefit' instead" fired on
        # every fold in production, meaning calibration was silently
        # never using the already-fit model at all). FrozenEstimator is
        # the current replacement - wraps the pre-fit model so
        # CalibratedClassifierCV calibrates it directly on X_calib/
        # y_calib instead of refitting on a 3-way split of that same
        # already-small calibration slice.
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(
            FrozenEstimator(model),
            method="sigmoid",
        )

        calibrated.fit(
            X_calib,
            y_calib,
        )

        return calibrated

    except Exception as exc:

        logging.warning(
            f"FrozenEstimator calibration unavailable: {exc}"
        )

        try:

            # sklearn <1.6 fallback - the original prefit API.
            calibrated = CalibratedClassifierCV(
                model,
                method="sigmoid",
                cv="prefit",
            )

            calibrated.fit(
                X_calib,
                y_calib,
            )

            return calibrated

        except Exception as exc2:

            logging.warning(
                f"Prefit calibration unavailable: {exc2}"
            )

            try:

                calibrated = CalibratedClassifierCV(
                    model,
                    method="sigmoid",
                    cv=3,
                )

                calibrated.fit(
                    X_calib,
                    y_calib,
                )

                return calibrated

            except Exception as exc3:

                logging.warning(
                    f"Calibration failed completely: {exc3}"
                )

                return model


# ============================================================
# WALK FORWARD TRAINING
# ============================================================

def train_walk_forward_single_model(
    X,
    y,
    symbol,
    side_label,
):

    if len(X) < MIN_TRAINING_SAMPLES:

        raise ValueError(
            f"{symbol} {side_label}: "
            f"only {len(X)} samples"
        )

    y = y.astype(int)

    positive_rate = float(y.mean())

    logging.info(
        f"Training {symbol} {side_label} | "
        f"Samples={len(y)} | "
        f"Positive={positive_rate:.2%}"
    )

    tscv = TimeSeriesSplit(
        n_splits=5
    )

    oof_probs = np.full(
        len(X),
        np.nan,
    )

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        tscv.split(X),
        start=1,
    ):

        if len(train_idx) < 200:
            continue

        split_point = int(
            len(train_idx) * 0.80
        )

        if split_point <= 0:
            continue

        sub_train_idx = train_idx[
            :split_point
        ]

        calib_idx = train_idx[
            split_point:
        ]

        if len(calib_idx) < 50:
            continue

        X_train = X.iloc[sub_train_idx]

        y_train = y.iloc[sub_train_idx]

        X_calib = X.iloc[calib_idx]

        y_calib = y.iloc[calib_idx]

        X_test = X.iloc[test_idx]

        if (
            len(np.unique(y_train)) < 2
            or len(np.unique(y_calib)) < 2
        ):
            logging.warning(
                f"{symbol} {side_label}: "
                f"fold {fold} skipped due to "
                f"single-class split"
            )
            continue

        scaler = StandardScaler()

        X_train_s = scaler.fit_transform(
            X_train
        )

        X_calib_s = scaler.transform(
            X_calib
        )

        X_test_s = scaler.transform(
            X_test
        )

        model = create_xgb_model()

        model.fit(
            X_train_s,
            y_train,
        )

        calibrated = calibrate_model(
            model,
            X_calib_s,
            y_calib,
        )

        probs = calibrated.predict_proba(
            X_test_s
        )[:, 1]

        oof_probs[test_idx] = probs

        logging.info(
            f"{symbol} {side_label}: "
            f"fold {fold} completed"
        )

    valid = np.isfinite(oof_probs)

    if valid.sum() < 100:

        raise ValueError(
            f"{symbol} {side_label}: "
            "insufficient OOF predictions"
        )

    y_eval = y.iloc[
        np.where(valid)[0]
    ].to_numpy()

    p_eval = oof_probs[valid]

    p_eval = np.clip(
        p_eval,
        1e-6,
        1 - 1e-6,
    )

    try:

        roc_auc = roc_auc_score(
            y_eval,
            p_eval,
        )

    except Exception:

        roc_auc = 0.5

    brier = brier_score_loss(
        y_eval,
        p_eval,
    )

    try:

        ll = log_loss(
            y_eval,
            p_eval,
        )

    except Exception:

        ll = float("nan")

    high_conf = (
        p_eval >= CONFIDENCE_THRESHOLD
    )

    high_conf_count = int(
        high_conf.sum()
    )

    if high_conf_count > 0:

        precision_high = precision_score(
            y_eval[high_conf],
            np.ones(high_conf_count),
            zero_division=0,
        )

        recall_high = recall_score(
            y_eval[high_conf],
            np.ones(high_conf_count),
            zero_division=0,
        )

    else:

        precision_high = 0.0

        recall_high = 0.0

    coverage = float(
        high_conf.mean()
    )

    logging.info(
        f"{symbol} {side_label} WF | "
        f"AUC={roc_auc:.3f} | "
        f"Brier={brier:.4f} | "
        f"LogLoss={ll:.4f} | "
        f"Precision@{CONFIDENCE_THRESHOLD:.2f}="
        f"{precision_high:.2%} | "
        f"Coverage={coverage:.2%} | "
        f"Signals={high_conf_count}"
    )

    # --------------------------------------------------------
    # Final model
    # --------------------------------------------------------

    final_split = int(
        len(X) * 0.90
    )

    X_train_final = X.iloc[
        :final_split
    ]

    y_train_final = y.iloc[
        :final_split
    ]

    X_calib_final = X.iloc[
        final_split:
    ]

    y_calib_final = y.iloc[
        final_split:
    ]

    if (
        len(np.unique(y_train_final)) < 2
        or len(np.unique(y_calib_final)) < 2
    ):

        raise ValueError(
            f"{symbol} {side_label}: "
            "final training split contains "
            "only one class"
        )

    final_scaler = StandardScaler()

    X_train_final_s = (
        final_scaler.fit_transform(
            X_train_final
        )
    )

    X_calib_final_s = (
        final_scaler.transform(
            X_calib_final
        )
    )

    final_model = create_xgb_model()

    final_model.fit(
        X_train_final_s,
        y_train_final,
    )

    final_calibrated = calibrate_model(
        final_model,
        X_calib_final_s,
        y_calib_final,
    )

    diagnostics = {
        "samples": int(len(y)),
        "positive_rate": positive_rate,
        "roc_auc": float(roc_auc),
        "brier": float(brier),
        "logloss": float(ll),
        "precision_high_conf": float(
            precision_high
        ),
        "recall_high_conf": float(
            recall_high
        ),
        "coverage": coverage,
        "signals_high_conf": high_conf_count,
    }

    return (
        final_calibrated,
        final_scaler,
        diagnostics,
    )


# ============================================================
# MODEL QUALITY GATE
# ============================================================

def _record_model_quality(
    symbol,
    side,
    diagnostics,
    passed,
    reason,
):

    model_quality_snapshot.setdefault(
        symbol, {}
    )[side] = {
        "auc": diagnostics.get("roc_auc"),
        "precision": diagnostics.get("precision_high_conf"),
        "signals": diagnostics.get("signals_high_conf"),
        "coverage": diagnostics.get("coverage"),
        "passed": passed,
        "reason": reason,
    }


def model_passes_quality_gate(
    diagnostics,
    symbol,
    side,
):

    auc = diagnostics["roc_auc"]

    precision_high = diagnostics[
        "precision_high_conf"
    ]

    signals = diagnostics[
        "signals_high_conf"
    ]

    if auc < MIN_MODEL_ROC_AUC:

        reason = (
            f"AUC {auc:.3f} below minimum "
            f"{MIN_MODEL_ROC_AUC:.3f}"
        )

        logging.warning(
            f"{symbol} {side}: {reason}"
        )

        _record_model_quality(
            symbol,
            side,
            diagnostics,
            False,
            reason,
        )

        return False

    # Was previously conditional on signals >= MIN_HIGH_CONF_SIGNALS,
    # meaning a model with too few high-confidence signals to actually
    # evaluate skipped the precision check entirely and passed on AUC
    # alone - confirmed by hand that this let two models through on
    # essentially no evidence: one with 0 high-confidence signals in
    # its entire walk-forward test, another with exactly 1 (a single
    # coin flip, not a validated track record). Too few signals to
    # measure precision on is now itself a failure, not a free pass -
    # "we don't have enough evidence to trust this" is the honest
    # outcome, not silently defaulting to trust.
    if signals < MIN_HIGH_CONF_SIGNALS:

        reason = (
            f"only {signals} high-confidence signals in "
            f"walk-forward testing, below minimum "
            f"{MIN_HIGH_CONF_SIGNALS} needed to trust precision"
        )

        logging.warning(
            f"{symbol} {side}: {reason}"
        )

        _record_model_quality(
            symbol,
            side,
            diagnostics,
            False,
            reason,
        )

        return False

    if precision_high < MIN_HIGH_CONF_PRECISION:

        reason = (
            f"high-confidence precision "
            f"{precision_high:.2%} below "
            f"{MIN_HIGH_CONF_PRECISION:.2%}"
        )

        logging.warning(
            f"{symbol} {side}: {reason}"
        )

        _record_model_quality(
            symbol,
            side,
            diagnostics,
            False,
            reason,
        )

        return False

    _record_model_quality(
        symbol,
        side,
        diagnostics,
        True,
        None,
    )

    return True


# ============================================================
# MODEL METADATA
# ============================================================

def save_model_bundle(
    symbol,
    side,
    model,
    scaler,
    diagnostics,
):

    path = model_file(
        symbol,
        side,
    )

    bundle = {
        "version": MODEL_VERSION,
        "bot_version": BOT_VERSION,
        "symbol": symbol,
        "side": side,
        "features": FEATURES,
        "created_at": utc_now().isoformat(),
        "model": model,
        "scaler": scaler,
        "diagnostics": diagnostics,
    }

    atomic_joblib_dump(
        bundle,
        path,
    )

    with sqlite3.connect(DB_FILE) as conn:

        conn.execute(
            """
            INSERT INTO model_runs (
                timestamp,
                symbol,
                side,
                samples,
                positive_rate,
                roc_auc,
                brier,
                logloss,
                precision_high_conf,
                recall_high_conf,
                coverage,
                signals_high_conf,
                model_file
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now().isoformat(),
                symbol,
                side,
                diagnostics["samples"],
                diagnostics["positive_rate"],
                diagnostics["roc_auc"],
                diagnostics["brier"],
                diagnostics["logloss"],
                diagnostics["precision_high_conf"],
                diagnostics["recall_high_conf"],
                diagnostics["coverage"],
                diagnostics["signals_high_conf"],
                str(path),
            ),
        )

        conn.commit()

    return path


def load_valid_model(
    symbol,
    side,
):

    path = model_file(
        symbol,
        side,
    )

    if not path.exists():
        return None

    age_hours = (
        time.time() -
        path.stat().st_mtime
    ) / 3600

    if age_hours > MODEL_MAX_AGE_HOURS:

        logging.info(
            f"{symbol} {side}: model expired "
            f"({age_hours:.1f}h)"
        )

        return None

    try:

        bundle = joblib.load(path)

        if not isinstance(
            bundle,
            dict,
        ):
            return None

        if bundle.get("version") != MODEL_VERSION:
            logging.warning(
                f"{symbol} {side}: "
                "model version mismatch"
            )
            return None

        if bundle.get("symbol") != symbol:
            return None

        if bundle.get("side") != side:
            return None

        if bundle.get("features") != FEATURES:
            logging.warning(
                f"{symbol} {side}: "
                "feature configuration mismatch"
            )
            return None

        if "model" not in bundle:
            return None

        if "scaler" not in bundle:
            return None

        diagnostics = bundle.get(
            "diagnostics",
            {},
        )

        if diagnostics:

            if not model_passes_quality_gate(
                diagnostics,
                symbol,
                side,
            ):
                return None

        logging.info(
            f"Loaded valid {side} model | "
            f"{symbol} | age={age_hours:.1f}h"
        )

        return (
            bundle["model"],
            bundle["scaler"],
            FEATURES,
        )

    except Exception as exc:

        logging.warning(
            f"{symbol} {side}: "
            f"model load failed: {exc}"
        )

        return None


# ============================================================
# TRAIN / LOAD MODELS
# ============================================================

def load_or_train_models(
    df,
    symbol,
):

    global model_training_lock

    buy_model = load_valid_model(
        symbol,
        "BUY",
    )

    sell_model = load_valid_model(
        symbol,
        "SELL",
    )

    if (
        buy_model is not None
        and sell_model is not None
    ):

        return {
            "BUY": buy_model,
            "SELL": sell_model,
        }

    if model_training_lock:

        raise RuntimeError(
            "Another model training process "
            "is already active"
        )

    model_training_lock = True

    try:

        logging.info(
            f"Training models for {symbol}"
        )

        # Best available proxy for historical spread: we don't have a
        # per-bar spread series for backtesting, so use the symbol's
        # current live spread as a stand-in for "typical" - an
        # approximation (today's spread won't exactly match every past
        # bar, especially for symbols with volatile spread), but far
        # closer to reality than assuming zero cost, which is what this
        # trained on before.
        spread_cost = 0.0

        symbol_info_for_cost = mt5.symbol_info(symbol)

        if symbol_info_for_cost is not None:

            point_for_cost = safe_float(
                getattr(symbol_info_for_cost, "point", 0)
            )

            spread_points_for_cost = safe_float(
                getattr(symbol_info_for_cost, "spread", 0)
            )

            if point_for_cost > 0:

                spread_cost = (
                    spread_points_for_cost
                    * point_for_cost
                )

        logging.info(
            f"{symbol}: training with spread_cost="
            f"{spread_cost:.6f} "
            f"({spread_points_for_cost if symbol_info_for_cost else 0} pts)"
        )

        train_df = add_tp_sl_targets(
            df.copy(),
            spread_cost=spread_cost,
        )

        train_df.dropna(
            subset=FEATURES + [
                "buy_target",
                "sell_target",
            ],
            inplace=True,
        )

        if len(train_df) < MIN_TRAINING_SAMPLES:

            raise ValueError(
                f"{symbol}: "
                f"only {len(train_df)} valid samples"
            )

        X = train_df[FEATURES].copy()

        y_buy = train_df[
            "buy_target"
        ].astype(int)

        y_sell = train_df[
            "sell_target"
        ].astype(int)

        if len(np.unique(y_buy)) < 2:
            raise ValueError(
                f"{symbol}: BUY target "
                "contains one class"
            )

        if len(np.unique(y_sell)) < 2:
            raise ValueError(
                f"{symbol}: SELL target "
                "contains one class"
            )

        (
            buy_model_obj,
            buy_scaler,
            buy_diag,
        ) = train_walk_forward_single_model(
            X,
            y_buy,
            symbol,
            "BUY",
        )

        (
            sell_model_obj,
            sell_scaler,
            sell_diag,
        ) = train_walk_forward_single_model(
            X,
            y_sell,
            symbol,
            "SELL",
        )

        buy_ok = model_passes_quality_gate(
            buy_diag,
            symbol,
            "BUY",
        )

        sell_ok = model_passes_quality_gate(
            sell_diag,
            symbol,
            "SELL",
        )

        if not buy_ok:
            raise ValueError(
                f"{symbol}: BUY model "
                "failed quality gate"
            )

        if not sell_ok:
            raise ValueError(
                f"{symbol}: SELL model "
                "failed quality gate"
            )

        buy_path = save_model_bundle(
            symbol,
            "BUY",
            buy_model_obj,
            buy_scaler,
            buy_diag,
        )

        sell_path = save_model_bundle(
            symbol,
            "SELL",
            sell_model_obj,
            sell_scaler,
            sell_diag,
        )

        logging.info(
            f"{symbol}: models saved | "
            f"{buy_path.name} | "
            f"{sell_path.name}"
        )

        return {
            "BUY": (
                buy_model_obj,
                buy_scaler,
                FEATURES,
            ),
            "SELL": (
                sell_model_obj,
                sell_scaler,
                FEATURES,
            ),
        }

    finally:

        model_training_lock = False


# ============================================================
# POSITION MANAGEMENT
# ============================================================

def get_magic_positions():

    positions = mt5.positions_get()

    if positions is None:
        return []

    return [
        p for p in positions
        if getattr(
            p,
            "magic",
            None,
        ) == MAGIC_NUMBER
    ]


def has_open_position(symbol):

    positions = mt5.positions_get(
        symbol=symbol
    )

    if positions is None:
        return False

    return any(
        getattr(
            p,
            "magic",
            None,
        ) == MAGIC_NUMBER
        for p in positions
    )


def count_open_positions():

    return len(
        get_magic_positions()
    )


# ============================================================
# COOLDOWN
# ============================================================

def is_on_cooldown(symbol):

    elapsed = (
        utc_now() -
        last_trade_time.get(
            symbol,
            datetime.min.replace(
                tzinfo=timezone.utc
            ),
        )
    ).total_seconds()

    return elapsed < (
        COOLDOWN_MINUTES * 60
    )


# ============================================================
# DEAL PROCESSING
# ============================================================

def update_closed_trades():

    global last_deal_check

    now = utc_now()

    since = (
        last_deal_check -
        timedelta(days=1)
    )

    try:

        deals = mt5.history_deals_get(
            since,
            now,
        )

    except Exception as exc:

        logging.error(
            f"history_deals_get failed: {exc}"
        )

        return

    if deals is None:
        last_deal_check = now
        return

    with sqlite3.connect(DB_FILE) as conn:

        for deal in deals:

            ticket = getattr(
                deal,
                "ticket",
                None,
            )

            if ticket is None:
                continue

            already = conn.execute(
                """
                SELECT 1
                FROM processed_deals
                WHERE deal_ticket = ?
                """,
                (ticket,),
            ).fetchone()

            if already:
                continue

            position_id = getattr(
                deal,
                "position_id",
                0,
            )

            deal_time = getattr(
                deal,
                "time",
                0,
            )

            if deal_time:

                deal_time_str = (
                    datetime.fromtimestamp(
                        deal_time,
                        tz=timezone.utc,
                    ).isoformat()
                )

            else:

                deal_time_str = utc_now().isoformat()

            conn.execute(
                """
                INSERT INTO processed_deals (
                    deal_ticket,
                    position_ticket,
                    time
                )
                VALUES (?, ?, ?)
                """,
                (
                    ticket,
                    position_id,
                    deal_time_str,
                ),
            )

            magic = getattr(
                deal,
                "magic",
                None,
            )

            if magic != MAGIC_NUMBER:
                continue

            entry_type = getattr(
                deal,
                "entry",
                None,
            )

            deal_in = getattr(
                mt5,
                "DEAL_ENTRY_IN",
                0,
            )

            if entry_type == deal_in:
                continue

            deal_type = getattr(
                deal,
                "type",
                None,
            )

            buy_type = getattr(
                mt5,
                "DEAL_TYPE_BUY",
                0,
            )

            sell_type = getattr(
                mt5,
                "DEAL_TYPE_SELL",
                1,
            )

            if deal_type not in (
                buy_type,
                sell_type,
            ):
                continue

            row = conn.execute(
                """
                SELECT
                    initial_volume,
                    closed_volume,
                    initial_risk,
                    symbol,
                    side,
                    entry_price,
                    prob
                FROM trades
                WHERE position_ticket = ?
                """,
                (position_id,),
            ).fetchone()

            if not row:
                continue

            (
                initial_volume,
                closed_volume,
                initial_risk,
                closed_symbol,
                closed_side,
                closed_entry_price,
                closed_prob,
            ) = row

            deal_volume = safe_float(
                getattr(
                    deal,
                    "volume",
                    0,
                )
            )

            deal_profit = safe_float(
                getattr(
                    deal,
                    "profit",
                    0,
                )
            )

            deal_commission = safe_float(
                getattr(
                    deal,
                    "commission",
                    0,
                )
            )

            deal_swap = safe_float(
                getattr(
                    deal,
                    "swap",
                    0,
                )
            )

            deal_fee = safe_float(
                getattr(
                    deal,
                    "fee",
                    0,
                )
            )

            deal_price = safe_float(
                getattr(
                    deal,
                    "price",
                    0,
                )
            )

            deal_net = (
                deal_profit
                + deal_commission
                + deal_swap
                + deal_fee
            )

            new_closed_volume = (
                closed_volume +
                deal_volume
            )

            status_flag = (
                "CLOSED"
                if new_closed_volume
                >= initial_volume - 1e-8
                else "PARTIAL"
            )

            conn.execute(
                """
                UPDATE trades
                SET
                    closed_volume = ?,
                    gross_profit =
                        gross_profit + ?,
                    commission =
                        commission + ?,
                    swap =
                        swap + ?,
                    fee =
                        fee + ?,
                    net_profit =
                        net_profit + ?,
                    status = ?
                WHERE position_ticket = ?
                """,
                (
                    new_closed_volume,
                    deal_profit,
                    deal_commission,
                    deal_swap,
                    deal_fee,
                    deal_net,
                    status_flag,
                    position_id,
                ),
            )

            if status_flag == "CLOSED":

                row2 = conn.execute(
                    """
                    SELECT net_profit
                    FROM trades
                    WHERE position_ticket = ?
                    """,
                    (position_id,),
                ).fetchone()

                if row2:

                    net = safe_float(
                        row2[0]
                    )

                    r_multiple = (
                        net / initial_risk
                        if initial_risk > 0
                        else 0
                    )

                    logging.info(
                        "TRADE CLOSED | "
                        f"Ticket={position_id} | "
                        f"Net={net:.2f} | "
                        f"R={r_multiple:.2f}"
                    )

                    if status is not None:
                        status.log_trade(
                            event="CLOSE",
                            symbol=closed_symbol,
                            side=closed_side,
                            position_ticket=position_id,
                            entry_price=closed_entry_price,
                            close_price=deal_price,
                            net_profit=net,
                            r_multiple=r_multiple,
                            prob=closed_prob,
                        )

        conn.commit()

    last_deal_check = now


# ============================================================
# BROKER VOLUME NORMALIZATION
# ============================================================

def floor_volume(
    volume,
    volume_min,
    volume_max,
    volume_step,
):

    volume = safe_float(
        volume,
        0,
    )

    volume_min = safe_float(
        volume_min,
        0,
    )

    volume_max = safe_float(
        volume_max,
        0,
    )

    volume_step = safe_float(
        volume_step,
        0,
    )

    if (
        volume <= 0
        or volume_min <= 0
        or volume_max <= 0
    ):
        return 0.0

    volume = min(
        volume,
        volume_max,
    )

    if volume_step > 0:

        steps = math.floor(
            volume / volume_step
            + 1e-12
        )

        volume = (
            steps *
            volume_step
        )

    if volume < volume_min:

        return 0.0

    decimals = 8

    if volume_step >= 1:
        decimals = 0
    elif volume_step >= 0.1:
        decimals = 1
    elif volume_step >= 0.01:
        decimals = 2
    elif volume_step >= 0.001:
        decimals = 3

    return round(
        volume,
        decimals,
    )


# ============================================================
# FILLING MODE
# ============================================================

def get_filling_mode(info):

    filling = getattr(
        info,
        "filling_mode",
        0,
    )

    # MT5 filling mode flags commonly:
    # FOK = 1
    # IOC = 2
    # RETURN = 4

    if filling & 1:

        return getattr(
            mt5,
            "ORDER_FILLING_FOK",
            0,
        )

    if filling & 2:

        return getattr(
            mt5,
            "ORDER_FILLING_IOC",
            1,
        )

    return getattr(
        mt5,
        "ORDER_FILLING_RETURN",
        2,
    )


# ============================================================
# RISK CALCULATION
# ============================================================

def calculate_risk_per_lot(
    symbol,
    order_type,
    entry,
    stop_loss,
    info,
):

    try:

        result = mt5.order_calc_profit(
            order_type,
            symbol,
            1.0,
            entry,
            stop_loss,
        )

        if result is not None:

            risk = abs(
                safe_float(result)
            )

            if risk > 0:
                return risk

    except Exception as exc:

        logging.warning(
            f"{symbol}: order_calc_profit "
            f"failed: {exc}"
        )

    tick_size = safe_float(
        getattr(
            info,
            "trade_tick_size",
            0,
        )
    )

    tick_value = safe_float(
        getattr(
            info,
            "trade_tick_value",
            0,
        )
    )

    if (
        tick_size <= 0
        or tick_value <= 0
    ):
        return 0.0

    distance = abs(
        entry - stop_loss
    )

    return (
        distance /
        tick_size
    ) * tick_value


# ============================================================
# TRADE CONDITIONS
# ============================================================

def evaluate_trade_conditions(
    symbol,
    signal,
    atr,
    account,
):

    info = mt5.symbol_info(
        symbol
    )

    tick = mt5.symbol_info_tick(
        symbol
    )

    if info is None:

        return None, "MT5_SYMBOL_ERROR"

    if tick is None:

        return None, "MT5_TICK_ERROR"

    point = safe_float(
        getattr(
            info,
            "point",
            0,
        )
    )

    if point <= 0:

        return None, "INVALID_POINT"

    atr = safe_float(atr)

    if atr <= 0:

        return None, "INVALID_ATR"

    order_type = (
        ORDER_TYPE_BUY
        if signal == "BUY"
        else ORDER_TYPE_SELL
    )

    entry = (
        safe_float(tick.ask)
        if signal == "BUY"
        else safe_float(tick.bid)
    )

    if entry <= 0:

        return None, "INVALID_ENTRY"

    stop_distance = (
        SL_ATR_MULT * atr
    )

    tp_distance = (
        TP_ATR_MULT * atr
    )

    if signal == "BUY":

        sl = (
            entry -
            stop_distance
        )

        tp = (
            entry +
            tp_distance
        )

    else:

        sl = (
            entry +
            stop_distance
        )

        tp = (
            entry -
            tp_distance
        )

    stops_level = safe_float(
        getattr(
            info,
            "trade_stops_level",
            0,
        )
    ) * point

    freeze_level = safe_float(
        getattr(
            info,
            "trade_freeze_level",
            0,
        )
    ) * point

    minimum_distance = max(
        stops_level,
        freeze_level,
        0,
    )

    if (
        abs(entry - sl)
        < minimum_distance
        or
        abs(tp - entry)
        < minimum_distance
    ):

        return (
            None,
            "NO_TRADE_BROKER_STOP_DISTANCE",
        )

    risk_budget = (
        safe_float(account.equity)
        * MAX_RISK_PERCENT
    )

    if risk_budget <= 0:

        return None, "INVALID_RISK_BUDGET"

    risk_per_lot = calculate_risk_per_lot(
        symbol,
        order_type,
        entry,
        sl,
        info,
    )

    if risk_per_lot <= 0:

        return None, "RISK_CALC_FAILED"

    raw_lot = (
        risk_budget /
        risk_per_lot
    )

    broker_min = safe_float(
        getattr(
            info,
            "volume_min",
            0,
        )
    )

    broker_max = safe_float(
        getattr(
            info,
            "volume_max",
            0,
        )
    )

    broker_step = safe_float(
        getattr(
            info,
            "volume_step",
            0,
        )
    )

    configured_max = SYMBOL_MAX_LOT.get(
        symbol,
        DEFAULT_MAX_LOT,
    )

    effective_max = min(
        broker_max
        if broker_max > 0
        else configured_max,
        configured_max,
    )

    if effective_max <= 0:

        return None, "INVALID_MAX_LOT"

    # --------------------------------------------------------
    # Notional cap
    # --------------------------------------------------------

    contract_size = safe_float(
        getattr(
            info,
            "trade_contract_size",
            0,
        )
    )

    if contract_size > 0:

        # `entry` is in the symbol's quote currency, but notional_cap is
        # in account currency (USD) - this only comes out correct when
        # the quote currency IS USD (EURUSDm, XAUUSDm). For any other
        # quote currency it's off by that currency's exchange rate: too
        # tight when the quote currency's per-unit value is large versus
        # the base leg (e.g. AUDMXNm - MXN per AUD is a big number,
        # shrinking max_notional_lot far below what the cap actually
        # intends), too loose the other way (e.g. USDJPYm/USDDKKm - JPY
        # or DKK per USD makes this cap effectively a no-op, though the
        # separately-computed risk-based lot sizing above still applies
        # since it correctly uses MT5's own order_calc_profit). Not fixed
        # here since it's real-money position-sizing math - would need
        # proper base-to-account-currency conversion, not a quick patch.
        notional_cap = (
            safe_float(account.equity)
            * MAX_NOTIONAL_EQUITY_PERCENT
        )

        max_notional_lot = (
            notional_cap /
            (
                contract_size *
                entry
            )
        )

        effective_max = min(
            effective_max,
            max_notional_lot,
        )

    if effective_max < broker_min:

        return (
            None,
            "NO_TRADE_BROKER_MIN_LOT",
        )

    raw_lot = min(
        raw_lot,
        effective_max,
    )

    lot = floor_volume(
        raw_lot,
        broker_min,
        effective_max,
        broker_step,
    )

    if lot <= 0:

        return None, "NO_TRADE_LOT_TOO_SMALL"

    actual_risk = (
        lot *
        risk_per_lot
    )

    # Strict protection against risk rounding errors.
    if actual_risk > (
        risk_budget * 1.001
    ):

        return (
            None,
            "NO_TRADE_RISK_EXCEEDED",
        )

    # --------------------------------------------------------
    # Margin check
    # --------------------------------------------------------

    try:

        margin = mt5.order_calc_margin(
            order_type,
            symbol,
            lot,
            entry,
        )

        margin = safe_float(
            margin,
            0,
        )

        free_margin = safe_float(
            getattr(
                account,
                "margin_free",
                0,
            )
        )

        if (
            margin > 0
            and free_margin > 0
            and margin > free_margin * 0.50
        ):

            return (
                None,
                "NO_TRADE_MARGIN_TOO_HIGH",
            )

    except Exception:

        pass

    digits = int(
        getattr(
            info,
            "digits",
            5,
        )
    )

    return {
        "lot": lot,
        "initial_risk": actual_risk,
        "entry": round(
            entry,
            digits,
        ),
        "sl": round(
            sl,
            digits,
        ),
        "tp": round(
            tp,
            digits,
        ),
        "order_type": order_type,
        "filling": get_filling_mode(
            info
        ),
    }, "TRADE_READY"


# ============================================================
# POSITION TICKET RECOVERY
# ============================================================

def get_position_ticket_after_order(
    symbol,
    result,
):

    deal_ticket = getattr(
        result,
        "deal",
        0,
    )

    if deal_ticket:

        for _ in range(5):

            try:

                deals = mt5.history_deals_get(
                    ticket=deal_ticket
                )

            except Exception:

                deals = None

            if deals:

                for deal in deals:

                    position_id = getattr(
                        deal,
                        "position_id",
                        0,
                    )

                    if position_id:

                        return position_id

            time.sleep(0.5)

    positions = mt5.positions_get(
        symbol=symbol
    )

    if positions:

        candidates = [
            p
            for p in positions
            if getattr(
                p,
                "magic",
                None,
            ) == MAGIC_NUMBER
        ]

        if candidates:

            candidates.sort(
                key=lambda p:
                getattr(
                    p,
                    "time",
                    0,
                ),
                reverse=True,
            )

            return candidates[0].ticket

    return None


# ============================================================
# ORDER LOG
# ============================================================

def log_order(
    symbol,
    side,
    volume,
    price,
    sl,
    tp,
    result,
    message,
    position_ticket=None,
):

    retcode = getattr(
        result,
        "retcode",
        0,
    ) if result else 0

    order_ticket = getattr(
        result,
        "order",
        0,
    ) if result else 0

    deal_ticket = getattr(
        result,
        "deal",
        0,
    ) if result else 0

    with sqlite3.connect(DB_FILE) as conn:

        conn.execute(
            """
            INSERT INTO orders (
                timestamp,
                symbol,
                side,
                volume,
                price,
                sl,
                tp,
                retcode,
                order_ticket,
                deal_ticket,
                position_ticket,
                message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now().isoformat(),
                symbol,
                side,
                volume,
                price,
                sl,
                tp,
                retcode,
                order_ticket,
                deal_ticket,
                position_ticket,
                message,
            ),
        )

        conn.commit()


# ============================================================
# EXECUTION
# ============================================================

def execute_trade(
    params,
    symbol,
    signal,
    prob,
):

    request = {
        "action": TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": params["lot"],
        "type": params["order_type"],
        "price": params["entry"],
        "sl": params["sl"],
        "tp": params["tp"],
        "deviation": ORDER_DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": f"SAFE_{BOT_VERSION}",
        "type_time": ORDER_TIME_GTC,
        "type_filling": params["filling"],
    }

    logging.info(
        f"ORDER SEND | "
        f"{signal} {symbol} | "
        f"Lot={params['lot']} | "
        f"Entry={params['entry']} | "
        f"SL={params['sl']} | "
        f"TP={params['tp']} | "
        f"Risk={params['initial_risk']:.2f}"
    )

    try:

        result = mt5.order_send(
            request
        )

    except Exception as exc:

        logging.error(
            f"{symbol}: order_send exception: "
            f"{exc}"
        )

        log_order(
            symbol,
            signal,
            params["lot"],
            params["entry"],
            params["sl"],
            params["tp"],
            None,
            str(exc),
        )

        if status is not None:
            status.log_trade(
                event="FAILED",
                symbol=symbol,
                side=signal,
                prob=prob,
                error=str(exc),
            )

        return False

    if result is None:

        error = mt5.last_error()

        logging.error(
            f"{symbol}: order_send returned None | "
            f"{error}"
        )

        log_order(
            symbol,
            signal,
            params["lot"],
            params["entry"],
            params["sl"],
            params["tp"],
            None,
            str(error),
        )

        if status is not None:
            status.log_trade(
                event="FAILED",
                symbol=symbol,
                side=signal,
                prob=prob,
                error=str(error),
            )

        return False

    retcode = getattr(
        result,
        "retcode",
        None,
    )

    success_codes = {
        getattr(
            mt5,
            "TRADE_RETCODE_DONE",
            10009,
        ),
        getattr(
            mt5,
            "TRADE_RETCODE_PLACED",
            10008,
        ),
        getattr(
            mt5,
            "TRADE_RETCODE_DONE_PARTIAL",
            10010,
        ),
    }

    if retcode not in success_codes:

        comment = getattr(
            result,
            "comment",
            "",
        )

        logging.error(
            f"ORDER FAILED | "
            f"{symbol} | "
            f"Retcode={retcode} | "
            f"Comment={comment}"
        )

        log_order(
            symbol,
            signal,
            params["lot"],
            params["entry"],
            params["sl"],
            params["tp"],
            result,
            comment,
        )

        if status is not None:
            status.log_trade(
                event="FAILED",
                symbol=symbol,
                side=signal,
                prob=prob,
                error=comment,
                retcode=retcode,
            )

        return False

    position_ticket = (
        get_position_ticket_after_order(
            symbol,
            result,
        )
    )

    if position_ticket is None:

        logging.error(
            f"{symbol}: order accepted but "
            "position ticket could not be recovered"
        )

        log_order(
            symbol,
            signal,
            params["lot"],
            params["entry"],
            params["sl"],
            params["tp"],
            result,
            "POSITION_TICKET_NOT_FOUND",
        )

        if status is not None:
            status.log_trade(
                event="FAILED",
                symbol=symbol,
                side=signal,
                prob=prob,
                error="POSITION_TICKET_NOT_FOUND",
            )

        return False

    with sqlite3.connect(DB_FILE) as conn:

        conn.execute(
            """
            INSERT OR REPLACE INTO trades (
                position_ticket,
                symbol,
                side,
                entry_time,
                entry_price,
                sl,
                tp,
                initial_volume,
                initial_risk,
                prob,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """,
            (
                position_ticket,
                symbol,
                signal,
                utc_now().isoformat(),
                params["entry"],
                params["sl"],
                params["tp"],
                params["lot"],
                params["initial_risk"],
                prob,
            ),
        )

        conn.commit()

    log_order(
        symbol,
        signal,
        params["lot"],
        params["entry"],
        params["sl"],
        params["tp"],
        result,
        "ORDER_ACCEPTED",
        position_ticket,
    )

    last_trade_time[symbol] = utc_now()

    logging.info(
        f"TRADE EXECUTED | "
        f"{signal} {symbol} | "
        f"Lot={params['lot']} | "
        f"Position={position_ticket}"
    )

    if status is not None:
        status.log_trade(
            event="OPEN",
            symbol=symbol,
            side=signal,
            lot=params["lot"],
            price=params["entry"],
            sl=params["sl"],
            tp=params["tp"],
            prob=prob,
        )

    return True


# ============================================================
# CURRENT CANDLE PROTECTION
# ============================================================

def already_processed_bar(
    symbol,
    bar_time,
):

    previous = last_processed_bar.get(
        symbol
    )

    if previous == bar_time:

        return True

    last_processed_bar[
        symbol
    ] = bar_time

    return False


# ============================================================
# MARKET CONDITIONS
# ============================================================

def get_market_snapshot(
    symbol,
    df,
):

    info = mt5.symbol_info(
        symbol
    )

    tick = mt5.symbol_info_tick(
        symbol
    )

    if info is None or tick is None:

        return None

    point = safe_float(
        getattr(
            info,
            "point",
            0,
        )
    )

    atr = safe_float(
        df["atr"].iloc[-1]
    )

    if point <= 0 or atr <= 0:

        return None

    spread = (
        safe_float(tick.ask) -
        safe_float(tick.bid)
    )

    spread_points = (
        spread / point
    )

    spread_atr = (
        spread / atr
    )

    return {
        "info": info,
        "tick": tick,
        "spread_points": spread_points,
        "spread_atr": spread_atr,
    }


# ============================================================
# STATUS
# ============================================================

def update_status(
    state,
    message="",
):

    if status is None:
        return

    try:

        if hasattr(
            status,
            "update",
        ):

            status.update(
                state=state,
                message=message,
            )

    except Exception:

        pass


# ============================================================
# SHUTDOWN
# ============================================================

def request_shutdown(
    signum,
    frame,
):

    global shutdown_requested

    shutdown_requested = True

    logging.warning(
        f"Shutdown signal received: {signum}"
    )


signal.signal(
    signal.SIGINT,
    request_shutdown,
)

signal.signal(
    signal.SIGTERM,
    request_shutdown,
)


# ============================================================
# SIGNAL DECISION
# ============================================================

def generate_signal(
    buy_prob,
    sell_prob,
):

    buy_prob = normalize_probability(
        buy_prob
    )

    sell_prob = normalize_probability(
        sell_prob
    )

    gap = buy_prob - sell_prob

    if (
        buy_prob >= CONFIDENCE_THRESHOLD
        and gap >= MIN_PROB_GAP
    ):

        return (
            "BUY",
            buy_prob,
            "BUY_CONFIDENCE_AND_GAP",
        )

    if (
        sell_prob >= CONFIDENCE_THRESHOLD
        and -gap >= MIN_PROB_GAP
    ):

        return (
            "SELL",
            sell_prob,
            "SELL_CONFIDENCE_AND_GAP",
        )

    if (
        buy_prob >= CONFIDENCE_THRESHOLD
        or sell_prob >= CONFIDENCE_THRESHOLD
    ):

        return (
            None,
            max(
                buy_prob,
                sell_prob,
            ),
            "NO_TRADE_PROB_GAP",
        )

    return (
        None,
        max(
            buy_prob,
            sell_prob,
        ),
        "NO_TRADE_MODEL_CONFIDENCE",
    )


# ============================================================
# PROCESS SYMBOL
# ============================================================

def process_symbol(
    symbol,
    model_bundle,
    account,
):

    if not ensure_symbol(symbol):

        return

    df = get_data(
        symbol,
        TIMEFRAME_M5,
        closed_only=True,
    )

    if df is None:

        return

    if not is_data_fresh(df):

        log_decision(
            symbol,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            "NONE",
            "SKIP",
            "NO_TRADE_STALE_DATA",
        )

        return

    df = add_features(df)

    if df is None:

        return

    if len(df) < 100:

        return

    last = df.iloc[-1]

    bar_time = last["time"].isoformat()

    if already_processed_bar(
        symbol,
        bar_time,
    ):

        return

    atr = safe_float(
        last["atr"]
    )

    adx = safe_float(
        last["adx"]
    )

    rsi = safe_float(
        last["rsi"]
    )

    if atr <= 0:

        log_decision(
            symbol,
            0,
            0,
            atr,
            adx,
            rsi,
            0,
            0,
            0,
            "NONE",
            "SKIP",
            "NO_TRADE_INVALID_ATR",
            bar_time,
        )

        return

    info = mt5.symbol_info(
        symbol
    )

    if info is None:

        return

    point = safe_float(
        getattr(
            info,
            "point",
            0,
        )
    )

    if point <= 0:

        return

    atr_points = (
        atr / point
    )

    minimum_atr = SYMBOL_MIN_ATR_POINTS.get(
        symbol,
        3.0,
    )

    if atr_points < minimum_atr:

        log_decision(
            symbol,
            0,
            0,
            atr,
            adx,
            rsi,
            0,
            0,
            0,
            "NONE",
            "SKIP",
            "NO_TRADE_LOW_ATR",
            bar_time,
        )

        return

    if adx < MIN_ADX:

        log_decision(
            symbol,
            0,
            0,
            atr,
            adx,
            rsi,
            0,
            0,
            0,
            "NONE",
            "SKIP",
            "NO_TRADE_LOW_ADX",
            bar_time,
        )

        return

    market = get_market_snapshot(
        symbol,
        df,
    )

    if market is None:

        return

    spread_points = market[
        "spread_points"
    ]

    spread_atr = market[
        "spread_atr"
    ]

    max_spread = SYMBOL_MAX_SPREAD.get(
        symbol,
        DEFAULT_MAX_SPREAD,
    )

    if (
        spread_points > max_spread
        or spread_atr > MAX_SPREAD_ATR_RATIO
    ):

        log_decision(
            symbol,
            0,
            0,
            atr,
            adx,
            rsi,
            0,
            spread_points,
            spread_atr,
            "NONE",
            "SKIP",
            "NO_TRADE_SPREAD",
            bar_time,
        )

        return

    buy_model, buy_scaler, buy_features = (
        model_bundle["BUY"]
    )

    sell_model, sell_scaler, sell_features = (
        model_bundle["SELL"]
    )

    if (
        buy_features != FEATURES
        or sell_features != FEATURES
    ):

        logging.error(
            f"{symbol}: model feature mismatch"
        )

        return

    latest = df[
        FEATURES
    ].iloc[-1:].copy()

    if latest.isna().any().any():

        log_decision(
            symbol,
            0,
            0,
            atr,
            adx,
            rsi,
            0,
            spread_points,
            spread_atr,
            "NONE",
            "SKIP",
            "NO_TRADE_FEATURE_NAN",
            bar_time,
        )

        return

    try:

        buy_prob = float(
            buy_model.predict_proba(
                buy_scaler.transform(
                    latest
                )
            )[0][1]
        )

        sell_prob = float(
            sell_model.predict_proba(
                sell_scaler.transform(
                    latest
                )
            )[0][1]
        )

    except Exception as exc:

        logging.error(
            f"{symbol}: prediction failed: "
            f"{exc}"
        )

        return

    signal_name, prob, signal_reason = (
        generate_signal(
            buy_prob,
            sell_prob,
        )
    )

    if signal_name is None:

        log_decision(
            symbol,
            buy_prob,
            sell_prob,
            atr,
            adx,
            rsi,
            0,
            spread_points,
            spread_atr,
            "NONE",
            "SKIP",
            signal_reason,
            bar_time,
        )

        return

    h1_trend = get_h1_trend(
        symbol
    )

    if (
        REQUIRE_H1_ALIGNMENT
        and signal_name == "BUY"
        and h1_trend != 1
    ):

        log_decision(
            symbol,
            buy_prob,
            sell_prob,
            atr,
            adx,
            rsi,
            h1_trend,
            spread_points,
            spread_atr,
            signal_name,
            "SKIP",
            "NO_TRADE_H1_MISALIGNMENT",
            bar_time,
        )

        return

    if (
        REQUIRE_H1_ALIGNMENT
        and signal_name == "SELL"
        and h1_trend != -1
    ):

        log_decision(
            symbol,
            buy_prob,
            sell_prob,
            atr,
            adx,
            rsi,
            h1_trend,
            spread_points,
            spread_atr,
            signal_name,
            "SKIP",
            "NO_TRADE_H1_MISALIGNMENT",
            bar_time,
        )

        return

    if (
        signal_name == "BUY"
        and rsi >= MAX_RSI
    ):

        log_decision(
            symbol,
            buy_prob,
            sell_prob,
            atr,
            adx,
            rsi,
            h1_trend,
            spread_points,
            spread_atr,
            signal_name,
            "SKIP",
            "NO_TRADE_RSI_EXTREME",
            bar_time,
        )

        return

    if (
        signal_name == "SELL"
        and rsi <= MIN_RSI
    ):

        log_decision(
            symbol,
            buy_prob,
            sell_prob,
            atr,
            adx,
            rsi,
            h1_trend,
            spread_points,
            spread_atr,
            signal_name,
            "SKIP",
            "NO_TRADE_RSI_EXTREME",
            bar_time,
        )

        return

    if has_open_position(symbol):

        log_decision(
            symbol,
            buy_prob,
            sell_prob,
            atr,
            adx,
            rsi,
            h1_trend,
            spread_points,
            spread_atr,
            signal_name,
            "SKIP",
            "NO_TRADE_POSITION_EXISTS",
            bar_time,
        )

        return

    if is_on_cooldown(symbol):

        log_decision(
            symbol,
            buy_prob,
            sell_prob,
            atr,
            adx,
            rsi,
            h1_trend,
            spread_points,
            spread_atr,
            signal_name,
            "SKIP",
            "NO_TRADE_COOLDOWN",
            bar_time,
        )

        return

    if (
        count_open_positions()
        >= MAX_CONCURRENT_TRADES
    ):

        log_decision(
            symbol,
            buy_prob,
            sell_prob,
            atr,
            adx,
            rsi,
            h1_trend,
            spread_points,
            spread_atr,
            signal_name,
            "SKIP",
            "NO_TRADE_MAX_POSITIONS",
            bar_time,
        )

        return

    params, reason = (
        evaluate_trade_conditions(
            symbol,
            signal_name,
            atr,
            account,
        )
    )

    kill_active, kill_reason = is_kill_switch_active()

    if params and kill_active:

        reason = "KILL_SWITCH_ACTIVE"

        params = None

    log_decision(
        symbol,
        buy_prob,
        sell_prob,
        atr,
        adx,
        rsi,
        h1_trend,
        spread_points,
        spread_atr,
        signal_name,
        "EXECUTE" if params else "SKIP",
        reason,
        bar_time,
    )

    if params:

        execute_trade(
            params,
            symbol,
            signal_name,
            prob,
        )


# ============================================================
# MODEL PREPARATION
# ============================================================

def prepare_models():

    models = {}

    for symbol in SYMBOLS:

        if shutdown_requested:
            break

        try:

            logging.info(
                f"Preparing model: {symbol}"
            )

            if not ensure_symbol(symbol):
                continue

            df = get_data(
                symbol,
                TIMEFRAME_M5,
                closed_only=True,
            )

            if df is None:

                logging.warning(
                    f"{symbol}: no training data"
                )

                continue

            if len(df) < MIN_TRAINING_SAMPLES:

                logging.warning(
                    f"{symbol}: insufficient "
                    f"training bars: {len(df)}"
                )

                continue

            df = add_features(df)

            if df is None:
                continue

            bundle = load_or_train_models(
                df,
                symbol,
            )

            models[symbol] = bundle

            logging.info(
                f"{symbol}: models ready"
            )

        except Exception as exc:

            logging.error(
                f"{symbol}: model preparation "
                f"failed: {exc}"
            )

    return models


# ============================================================
# MAIN LOOP
# ============================================================

def run_bot():

    global shutdown_requested

    bot_started_at = utc_now().isoformat()

    logging.info(
        "================================================"
    )

    logging.info(
        f"SAFE AI TRADING BOT v{BOT_VERSION}"
    )

    logging.info(
        "Starting..."
    )

    logging.info(
        "================================================"
    )

    init_db()

    load_state()

    if not initialize_mt5():

        return

    for symbol in SYMBOLS:

        ensure_symbol(symbol)

    if not initialize_daily_state():

        logging.error(
            "Could not initialize daily state."
        )

        return

    if check_daily_loss_limits():

        logging.warning(
            "Bot starts with daily loss lock active."
        )

    # Retries in-process instead of exiting when nothing qualifies yet.
    # This used to be a one-shot check that returned (ending the script)
    # on zero qualifying symbols - a clean exit, not a crash, so the
    # scheduled task's restart-on-failure policy never caught it and the
    # bot just sat dead until something else relaunched it by hand.
    # Trained-and-found-nothing is a legitimate, expected outcome (model
    # edge can be marginal run to run) and deserves a retry, not a stop.
    models_dict = prepare_models()

    while not models_dict and not shutdown_requested:

        logging.error(
            "No valid models available. "
            f"Retrying in {LOOP_INTERVAL_SECONDS}s."
        )

        # Without this, the dashboard shows the exact same "waiting
        # for first update" blank state whether the bot never started
        # or whether it ran a full training pass and correctly found
        # zero symbols worth trading - those are very different
        # things to know from the outside. write_status() otherwise
        # only ever gets called from inside the main loop below, which
        # this retry path skips entirely.
        if status is not None:

            no_model_account = mt5.account_info()

            status.write_status(
                equity=getattr(no_model_account, "equity", None),
                balance=getattr(no_model_account, "balance", None),
                daily_loss_pct=0.0,
                daily_loss_limit=MAX_DAILY_LOSS_PERCENT,
                paused=True,
                positions=[],
                signals={},
                bot_version=f"v{BOT_VERSION}",
                confidence_threshold=CONFIDENCE_THRESHOLD,
                symbols=SYMBOLS,
                started_at=bot_started_at,
                loop_interval_seconds=LOOP_INTERVAL_SECONDS,
                model_quality=model_quality_snapshot,
                max_concurrent_trades=MAX_CONCURRENT_TRADES,
                max_risk_percent=MAX_RISK_PERCENT,
                pause_reason=(
                    "No symbols currently pass the model quality "
                    "gate (min AUC, min high-confidence precision) - "
                    "trained and evaluated all "
                    f"{len(SYMBOLS)}, none qualified this run. "
                    "Retrying automatically."
                ),
            )

        time.sleep(
            LOOP_INTERVAL_SECONDS
        )

        models_dict = prepare_models()

    if shutdown_requested:

        return

    logging.info(
        "================================================"
    )

    logging.info(
        f"MODELS READY: "
        f"{len(models_dict)}/{len(SYMBOLS)} symbols"
    )

    logging.info(
        "Execution loop started."
    )

    logging.info(
        "================================================"
    )

    update_status(
        "RUNNING",
        "Models ready",
    )

    while not shutdown_requested:

        loop_start = time.time()

        try:

            # ----------------------------------------
            # Synchronize closed deals
            # ----------------------------------------

            update_closed_trades()

            # ----------------------------------------
            # Daily state
            # ----------------------------------------

            reset_daily_equity_if_needed()

            if check_daily_loss_limits():

                update_status(
                    "DAILY_LOCK",
                    "Daily loss limit reached",
                )

                if status is not None:
                    locked_account = mt5.account_info()
                    status.write_status(
                        equity=getattr(locked_account, "equity", None),
                        balance=getattr(locked_account, "balance", None),
                        daily_loss_pct=MAX_DAILY_LOSS_PERCENT,
                        daily_loss_limit=MAX_DAILY_LOSS_PERCENT,
                        paused=True,
                        positions=[],
                        signals=signals_snapshot,
                        bot_version=f"v{BOT_VERSION}",
                        confidence_threshold=CONFIDENCE_THRESHOLD,
                        symbols=SYMBOLS,
                        started_at=bot_started_at,
                loop_interval_seconds=LOOP_INTERVAL_SECONDS,
                        model_quality=model_quality_snapshot,
                        max_concurrent_trades=MAX_CONCURRENT_TRADES,
                max_risk_percent=MAX_RISK_PERCENT,
                    )

                time.sleep(
                    LOOP_INTERVAL_SECONDS
                )

                continue

            account = mt5.account_info()

            if account is None:

                # Previously just logged and hoped the connection healed
                # itself - it doesn't. mt5.initialize() is safe to call
                # again on an already-connected terminal, so actively
                # re-run the full connect sequence (path/login/server)
                # rather than passively waiting on a dead session.
                logging.error(
                    "account_info unavailable - "
                    "attempting MT5 reconnect"
                )

                reconnected = initialize_mt5()

                if reconnected:

                    logging.info(
                        "MT5 reconnected successfully"
                    )

                    account = mt5.account_info()

                if account is None:

                    logging.error(
                        "MT5 reconnect failed, "
                        "will retry next cycle"
                    )

                    time.sleep(10)

                    continue

            # ----------------------------------------
            # Process symbols
            # ----------------------------------------

            for symbol in SYMBOLS:

                if shutdown_requested:
                    break

                if symbol not in models_dict:
                    continue

                try:

                    process_symbol(
                        symbol,
                        models_dict[symbol],
                        account,
                    )

                except Exception as exc:

                    logging.exception(
                        f"{symbol}: "
                        f"processing error: {exc}"
                    )

            # ----------------------------------------
            # Loop timing
            # ----------------------------------------

            elapsed = (
                time.time() -
                loop_start
            )

            sleep_for = max(
                5,
                LOOP_INTERVAL_SECONDS -
                elapsed,
            )

            update_status(
                "RUNNING",
                f"Loop complete | "
                f"Open={count_open_positions()}",
            )

            if status is not None:
    
                positions_snapshot = [
                    {
                        "symbol": p.symbol,
                        "side": "BUY" if p.type == ORDER_TYPE_BUY else "SELL",
                        "volume": p.volume,
                        "price_open": p.price_open,
                        "sl": p.sl,
                        "tp": p.tp,
                        "profit": p.profit,
                    }
                    for p in get_magic_positions()
                ]

                daily_loss_pct = 0.0

                if daily_start_equity and daily_start_equity > 0:
                    daily_loss_pct = max(
                        0.0,
                        (daily_start_equity - safe_float(account.equity))
                        / daily_start_equity
                        * 100,
                    )

                kill_active, kill_reason = is_kill_switch_active()

                status.write_status(
                    equity=account.equity,
                    balance=account.balance,
                    daily_loss_pct=daily_loss_pct,
                    daily_loss_limit=MAX_DAILY_LOSS_PERCENT,
                    paused=daily_loss_lock or kill_active,
                    positions=positions_snapshot,
                    signals=signals_snapshot,
                    bot_version=f"v{BOT_VERSION}",
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    symbols=SYMBOLS,
                    started_at=bot_started_at,
                loop_interval_seconds=LOOP_INTERVAL_SECONDS,
                    model_quality=model_quality_snapshot,
                    max_concurrent_trades=MAX_CONCURRENT_TRADES,
                max_risk_percent=MAX_RISK_PERCENT,
                    kill_switch_active=kill_active,
                    pause_reason=kill_reason if kill_active else None,
                )

                status.log_equity_point(account.equity, account.balance)

            time.sleep(
                sleep_for
            )

        except KeyboardInterrupt:

            shutdown_requested = True

        except Exception as exc:

            logging.exception(
                f"Main loop error: {exc}"
            )

            update_status(
                "ERROR",
                str(exc),
            )

            time.sleep(15)

    # ========================================================
    # SHUTDOWN
    # ========================================================

    logging.info(
        "Shutdown requested. "
        "No new trades will be opened."
    )

    update_status(
        "STOPPED",
        "Bot stopped",
    )

    try:

        update_closed_trades()

    except Exception:

        pass

    try:

        mt5.shutdown()

    except Exception:

        pass

    logging.info(
        "SAFE AI TRADING BOT stopped."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_bot()