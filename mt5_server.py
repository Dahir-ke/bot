# Minimal classic-RPyC server exposing the real, freshly pip-installed
# MetaTrader5 package - replaces mt5linux's prebuilt mt5server.exe
# (a third-party PyInstaller binary of unclear provenance/build date,
# not something MetaQuotes builds or tests) with one we control end to
# end, running under a genuine Wine-hosted Windows Python instead.
#
# Talks the same protocol the bot side already expects: mt5linux==0.2.4
# (the client pinned in requirements-linux.txt) does
# rpyc.classic.connect(host, port) and then eval()s/execute()s code
# like "mt5.initialize(*args, **kwargs)" against the connection - that
# only resolves if "mt5" already exists in this process's own
# __main__ namespace before the SlaveService starts serving, which is
# exactly what importing it below at module level achieves.
import sys

import MetaTrader5 as mt5  # noqa: F401 - must be in this module's globals for remote eval()
from rpyc.core.service import SlaveService
from rpyc.utils.server import ThreadedServer

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18812
    print(f"MT5 RPyC bridge (custom) listening on 0.0.0.0:{port}", flush=True)
    ThreadedServer(
        SlaveService,
        hostname="0.0.0.0",
        port=port,
        protocol_config={"allow_all_attrs": True, "allow_public_attrs": True},
    ).start()
