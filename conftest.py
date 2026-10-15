"""Let the bot be imported without a broker attached.

MetaTrader5 is Windows-only and reaches for a terminal as soon as it is
used. The tests here are about decision logic - which side may trade, and
which gate blocks what - so the broker is stubbed out entirely. Nothing in
this file can place an order, by construction.
"""

import os
import sys
import types

_MT5_FUNCTIONS = (
    "initialize", "shutdown", "login", "symbol_info", "symbol_info_tick",
    "positions_get", "account_info", "copy_rates_from_pos", "order_send",
    "last_error", "symbol_select", "history_deals_get", "terminal_info",
    "copy_rates_range", "orders_get",
)

_MT5_CONSTANTS = (
    "TIMEFRAME_M1", "TIMEFRAME_M5", "TIMEFRAME_M15", "TIMEFRAME_H1",
    "TIMEFRAME_H4", "TIMEFRAME_D1", "ORDER_TYPE_BUY", "ORDER_TYPE_SELL",
    "TRADE_ACTION_DEAL", "TRADE_ACTION_SLTP", "ORDER_TIME_GTC",
    "ORDER_FILLING_IOC", "ORDER_FILLING_FOK", "TRADE_RETCODE_DONE",
    "POSITION_TYPE_BUY", "POSITION_TYPE_SELL", "DEAL_TYPE_BUY",
    "DEAL_TYPE_SELL", "DEAL_ENTRY_IN", "DEAL_ENTRY_OUT",
)


def _install_mt5_stub():
    if "MetaTrader5" in sys.modules:
        return
    module = types.ModuleType("MetaTrader5")

    def _refuse(*args, **kwargs):
        raise AssertionError(
            "a test reached MetaTrader5 - these tests must never touch a broker"
        )

    for name in _MT5_FUNCTIONS:
        setattr(module, name, _refuse)
    for index, name in enumerate(_MT5_CONSTANTS):
        setattr(module, name, index)
    sys.modules["MetaTrader5"] = module


_install_mt5_stub()


# The override roster is read from the environment at import. Tests run
# against the same roster production uses, so a test that says "an
# override pair behaves this way" is talking about the real pairs rather
# than an empty set that would make it pass vacuously.
os.environ.setdefault(
    "RISK_OVERRIDE_PAIRS",
    "USDJPYm:BUY,USDJPYm:SELL,EURUSDm:BUY,EURUSDm:SELL",
)
