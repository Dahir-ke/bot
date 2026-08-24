"""Web dashboard for bo.py - login-gated, since it's reachable at a public
URL (bot.tatuatechnology.com) and shows real account equity/balance/trades.

Reads the status.json / trades.jsonl / equity_curve.jsonl files bo.py
writes via bot_status.py and serves them to a browser. Does not talk to
MT5 itself, so it can run as a separate process/container alongside the
bot - see docker-compose.yml.
"""

import hmac
import json
import os
from functools import wraps

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

BASE_DIR = os.environ.get("BOT_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
STATUS_FILE = os.path.join(BASE_DIR, "status.json")
TRADES_FILE = os.path.join(BASE_DIR, "trades.jsonl")
EQUITY_FILE = os.path.join(BASE_DIR, "equity_curve.jsonl")

DASHBOARD_USERNAME = os.environ.get("DASHBOARD_USERNAME", "")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")

app = Flask(__name__)
app.secret_key = os.environ["DASHBOARD_SECRET_KEY"]
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    # The dashboard is only ever served over HTTPS (see nginx site config) -
    # ok to require it unconditionally rather than branching on an env flag.
    SESSION_COOKIE_SECURE=True,
)


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


def _valid_credentials(username, password):
    # Constant-time comparisons - a username/password check that short-
    # circuits on the first mismatched byte leaks how much of the guess
    # was right via response timing.
    return (
        hmac.compare_digest(username, DASHBOARD_USERNAME)
        and hmac.compare_digest(password, DASHBOARD_PASSWORD)
    )


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if _valid_credentials(request.form.get("username", ""), request.form.get("password", "")):
            session.clear()
            session["logged_in"] = True
            session.permanent = True
            return redirect(request.args.get("next") or url_for("index"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("dashboard.html")


@app.route("/api/status")
@login_required
def api_status():
    return jsonify(_read_json(STATUS_FILE, {}))


@app.route("/api/trades")
@login_required
def api_trades():
    trades = _read_jsonl(TRADES_FILE, limit=300)
    trades.reverse()
    return jsonify(trades)


@app.route("/api/equity")
@login_required
def api_equity():
    return jsonify(_read_jsonl(EQUITY_FILE, limit=3000))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
