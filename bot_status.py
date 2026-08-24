"""Shared state writer used by bot.py / bo.py so a separate dashboard
process can display live status without touching MT5 itself."""

import json
import os
import threading
from datetime import datetime, timezone

_LOCK = threading.Lock()
_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_FILE = os.path.join(_DIR, "status.json")
TRADES_FILE = os.path.join(_DIR, "trades.jsonl")
EQUITY_FILE = os.path.join(_DIR, "equity_curve.jsonl")


def write_status(**fields):
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = STATUS_FILE + ".tmp"
    with _LOCK:
        with open(tmp, "w") as f:
            json.dump(fields, f, default=str)
        os.replace(tmp, STATUS_FILE)


def log_trade(**fields):
    fields["time"] = datetime.now(timezone.utc).isoformat()
    with _LOCK:
        with open(TRADES_FILE, "a") as f:
            f.write(json.dumps(fields, default=str) + "\n")


def log_equity_point(equity, balance):
    point = {
        "time": datetime.now(timezone.utc).isoformat(),
        "equity": equity,
        "balance": balance,
    }
    with _LOCK:
        with open(EQUITY_FILE, "a") as f:
            f.write(json.dumps(point) + "\n")
