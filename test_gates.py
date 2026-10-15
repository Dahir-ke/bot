"""The gates that decide whether money moves.

Written after a month of zero trades that nobody was told about. Each test
here corresponds to something the bot's own ledger proved was happening:

  - 97 of 97 hourly rejections had a NEUTRAL hourly trend, none opposing,
    because get_h1_trend judged risk-override pairs against the global ADX
    floor of 20 while their own gates ran at 12.
  - USDJPYm was switched off entirely on 2026-10-08 because one of its two
    sides produced 3 walk-forward signals instead of 5 - taking the working
    side with it.

The point of these is not that the thresholds are right. It is that a
change to them cannot pass unnoticed again.
"""

import final_bot as bot
import pytest


# ---------- a side with no model cannot trade


def test_a_side_with_no_model_is_never_signalled():
    """The bug that cost a week: a disabled side must be impossible to
    trade, not merely unlikely. Even a certainty of 1.0 is refused."""
    name, prob, reason = bot.generate_signal(
        "USDJPYm", buy_prob=1.0, sell_prob=0.0, allowed_sides={"SELL"},
    )
    assert name != "BUY", f"traded a side with no model: {name} ({reason})"


def test_the_surviving_side_still_trades():
    """And the other half of it: losing one side must not silence the
    pair. This is exactly USDJPYm on 2026-10-08 - BUY passed, SELL did
    not, and the whole symbol went dark."""
    name, prob, reason = bot.generate_signal(
        "USDJPYm", buy_prob=0.95, sell_prob=0.02, allowed_sides={"BUY"},
    )
    assert name == "BUY", f"the passing side did not trade: {name} ({reason})"


def test_both_sides_allowed_behaves_as_before():
    assert bot.generate_signal("USDJPYm", 0.95, 0.02)[0] == "BUY"
    assert bot.generate_signal("USDJPYm", 0.02, 0.95)[0] == "SELL"


def test_no_allowed_sides_means_no_trade():
    name, _, _ = bot.generate_signal("USDJPYm", 0.99, 0.99, allowed_sides=set())
    assert name is None


# ---------- the hourly trend filter


def _h1_frame(adx):
    """An unambiguous hourly uptrend at a given ADX.

    Close above EMA50, EMA20 above EMA50, EMA50 rising. The only thing in
    question is whether the ADX clears the floor this symbol is judged by.
    """
    import pandas as pd

    rows = 60
    frame = pd.DataFrame({
        "close": [1.10] * rows,
        "ema20": [1.09] * rows,
        "ema50": [1.07 + i * 0.0002 for i in range(rows)],
        "adx": [adx] * rows,
    })
    return frame


def _real_h1_trend(monkeypatch, symbol, adx):
    """Calls get_h1_trend itself, with the data layer stubbed out.

    Deliberately not a restatement of the rule - an earlier version of
    this test re-implemented the comparison and would have passed happily
    with the bug still in the source.
    """
    frame = _h1_frame(adx)
    monkeypatch.setattr(bot, "get_data", lambda *a, **k: frame)
    monkeypatch.setattr(bot, "is_data_fresh", lambda *a, **k: True)
    monkeypatch.setattr(bot, "add_features", lambda df: df)
    return bot.get_h1_trend(symbol)


def test_an_override_pair_uses_its_own_adx_floor_on_the_hourly(monkeypatch):
    """ADX 15 is below the global 20 and above the override's 12.

    Before the fix this returned NEUTRAL, and because the alignment check
    rejects anything that is not an exact match, a neutral hourly trend
    rejected every trade. That was 100% of the hourly rejections in the
    ledger - not one of them was a trade going against the trend."""
    symbol = sorted(bot.RISK_OVERRIDE_SYMBOLS)[0]
    assert _real_h1_trend(monkeypatch, symbol, adx=15.0) == 1


def test_a_normal_symbol_still_uses_the_strict_floor(monkeypatch):
    """The override must not leak onto symbols nobody granted it to."""
    normal = "XAUUSDm"
    assert normal not in bot.RISK_OVERRIDE_SYMBOLS
    assert _real_h1_trend(monkeypatch, normal, adx=15.0) == 0
    assert _real_h1_trend(monkeypatch, normal, adx=25.0) == 1


def test_the_override_floor_is_genuinely_lower_than_the_global_one():
    """If these ever converge, the two tests above stop proving anything."""
    assert bot.RISK_OVERRIDE_MIN_ADX < bot.MIN_ADX


# ---------- the confidence gate still applies


def test_an_override_pair_still_has_to_clear_its_threshold():
    name, _, _ = bot.generate_signal(
        sorted(bot.RISK_OVERRIDE_SYMBOLS)[0],
        buy_prob=bot.RISK_OVERRIDE_LIVE_CONFIDENCE - 0.05,
        sell_prob=0.01,
    )
    assert name is None


def test_a_normal_symbol_keeps_the_strict_confidence_bar():
    """0.5 would clear the override bar and must not clear the global one."""
    assert "XAUUSDm" not in bot.RISK_OVERRIDE_SYMBOLS
    name, _, _ = bot.generate_signal("XAUUSDm", buy_prob=0.5, sell_prob=0.01)
    assert name is None


@pytest.mark.parametrize("symbol", sorted(bot.RISK_OVERRIDE_SYMBOLS))
def test_every_override_symbol_can_reach_a_signal(symbol):
    """A live symbol whose thresholds cannot be met by any probability is
    not 'strict', it is off. EURUSDm spent a week in that state and the
    bot reported itself healthy throughout."""
    name, _, _ = bot.generate_signal(symbol, buy_prob=0.99, sell_prob=0.0)
    assert name == "BUY"


# ---------- the watchdog
#
# Each of these is the situation the live bot was actually in while
# reporting itself healthy.


import sqlite3
from datetime import datetime, timedelta, timezone


def _ledger(tmp_path, monkeypatch, decisions=(), orders=()):
    """A throwaway ledger shaped like the real one."""
    path = tmp_path / "ledger.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE decisions (id INTEGER PRIMARY KEY, timestamp TEXT, "
        "symbol TEXT, buy_prob REAL, sell_prob REAL)"
    )
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, timestamp TEXT)")
    conn.executemany(
        "INSERT INTO decisions (timestamp, symbol, buy_prob, sell_prob) "
        "VALUES (?, ?, ?, ?)", decisions,
    )
    conn.executemany("INSERT INTO orders (timestamp) VALUES (?)", orders)
    conn.commit()
    conn.close()
    monkeypatch.setattr(bot, "DB_FILE", str(path))
    return path


def _now():
    return datetime.now(timezone.utc)


def _kinds(alerts):
    return {kind for kind, _ in alerts}


def test_it_shouts_when_no_symbol_has_a_model(tmp_path, monkeypatch):
    _ledger(tmp_path, monkeypatch)
    alerts = bot.run_watchdog({}, started_at=_now())
    assert "NO_MODELS" in _kinds(alerts)


def test_it_shouts_when_the_only_live_symbol_cannot_reach_its_threshold(
    tmp_path, monkeypatch
):
    """EURUSDm, 9-15 October: loaded, evaluated every five minutes, and
    never once producing a probability above 0.44 against a 0.40 bar it
    had to beat by more than the reachable margin. The bot called itself
    RUNNING for a week."""
    recent = (_now() - timedelta(hours=2)).isoformat()
    rows = [(recent, "EURUSDm", 0.10, 0.12) for _ in range(300)]
    _ledger(tmp_path, monkeypatch, decisions=rows)

    alerts = bot.run_watchdog({"EURUSDm": {}}, started_at=_now())
    assert "ALL_SYMBOLS_UNREACHABLE" in _kinds(alerts)


def test_a_symbol_that_does_reach_its_threshold_raises_nothing(
    tmp_path, monkeypatch
):
    recent = (_now() - timedelta(hours=2)).isoformat()
    rows = [(recent, "EURUSDm", 0.80, 0.10) for _ in range(300)]
    _ledger(
        tmp_path, monkeypatch,
        decisions=rows,
        orders=[((_now() - timedelta(hours=5)).isoformat(),)],
    )
    alerts = bot.run_watchdog({"EURUSDm": {}}, started_at=_now())
    assert alerts == []


def test_it_shouts_when_nothing_has_ever_traded(tmp_path, monkeypatch):
    """`orders` had zero rows for the entire life of the bot and nothing
    ever said so."""
    recent = (_now() - timedelta(hours=1)).isoformat()
    _ledger(
        tmp_path, monkeypatch,
        decisions=[(recent, "EURUSDm", 0.80, 0.05) for _ in range(50)],
    )
    alerts = bot.run_watchdog(
        {"EURUSDm": {}}, started_at=_now() - timedelta(days=30),
    )
    assert "NEVER_TRADED" in _kinds(alerts)


def test_a_freshly_started_bot_is_not_accused_of_never_trading(
    tmp_path, monkeypatch
):
    """Up for an hour is not a month of silence."""
    recent = (_now() - timedelta(minutes=10)).isoformat()
    _ledger(
        tmp_path, monkeypatch,
        decisions=[(recent, "EURUSDm", 0.80, 0.05) for _ in range(5)],
    )
    alerts = bot.run_watchdog(
        {"EURUSDm": {}}, started_at=_now() - timedelta(hours=1),
    )
    assert "NEVER_TRADED" not in _kinds(alerts)


def test_it_shouts_when_trading_has_stopped_recently(tmp_path, monkeypatch):
    recent = (_now() - timedelta(hours=1)).isoformat()
    _ledger(
        tmp_path, monkeypatch,
        decisions=[(recent, "EURUSDm", 0.80, 0.05) for _ in range(50)],
        orders=[((_now() - timedelta(days=9)).isoformat(),)],
    )
    alerts = bot.run_watchdog({"EURUSDm": {}}, started_at=_now() - timedelta(days=30))
    assert "NO_RECENT_TRADES" in _kinds(alerts)


def test_every_alert_is_written_down(tmp_path, monkeypatch):
    """An alert that only exists in a log scrolls away. This one had to be
    reconstructed from the decisions table by hand a month later."""
    path = _ledger(tmp_path, monkeypatch)
    bot.run_watchdog({}, started_at=_now())

    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT kind, message FROM alerts").fetchall()
    conn.close()
    assert rows, "the alert was logged but not recorded"
    assert any(kind == "NO_MODELS" for kind, _ in rows)


def test_a_broken_webhook_cannot_stop_the_bot(tmp_path, monkeypatch):
    """Delivery is best effort. An unreachable alert channel must not
    take the trading loop down with it."""
    _ledger(tmp_path, monkeypatch)
    monkeypatch.setattr(bot, "ALERT_WEBHOOK_URL", "http://127.0.0.1:9/nowhere")
    alerts = bot.run_watchdog({}, started_at=_now())
    assert "NO_MODELS" in _kinds(alerts)
