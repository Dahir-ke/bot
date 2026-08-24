"""Read-only web dashboard for bot.py / bo.py.

Reads the status.json / trades.jsonl / equity_curve.jsonl files that the
bot writes via bot_status.py and serves them to a browser. Does not talk
to MT5 itself, so it can run as a separate process alongside the bot.
"""

import json
import os

from flask import Flask, jsonify, render_template

BASE_DIR = os.environ.get("BOT_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
STATUS_FILE = os.path.join(BASE_DIR, "status.json")
TRADES_FILE = os.path.join(BASE_DIR, "trades.jsonl")
EQUITY_FILE = os.path.join(BASE_DIR, "equity_curve.jsonl")

app = Flask(__name__)


def _read_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _read_jsonl(path, limit=None):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = f.readlines()
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if limit:
        records = records[-limit:]
    return records


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/status")
def api_status():
    return jsonify(_read_json(STATUS_FILE, {}))


@app.route("/api/trades")
def api_trades():
    trades = _read_jsonl(TRADES_FILE, limit=300)
    trades.reverse()
    return jsonify(trades)


@app.route("/api/equity")
def api_equity():
    return jsonify(_read_jsonl(EQUITY_FILE, limit=3000))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
