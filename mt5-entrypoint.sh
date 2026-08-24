#!/bin/bash
set -e

# Not set globally in the Dockerfile - the build-time steps use xvfb-run,
# which picks its own free display per invocation and would conflict with
# a hardcoded one. This runtime script manages its own Xvfb directly (a
# long-lived container, not a one-shot build layer, so :99 is fine here).
export DISPLAY=:99

mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"

# `docker restart` (not recreate) reuses the same writable layer, so a
# lock file from a previous run that didn't shut down cleanly can still
# be sitting here - a stale one makes Xvfb silently fail to bind, wine
# gets no real display, and the bridge can't do anything useful even
# though this script itself keeps running. Cheap enough to always clear.
rm -f /tmp/.X99-lock

# Virtual display the Wine-side GUI apps need (the terminal itself, and
# mt5_server.py's Wine-Python process).
Xvfb :99 -screen 0 1024x768x16 &
sleep 3

# Bare Xvfb has no window manager, and the real MetaTrader5 package's
# Python-API handshake is implemented with Windows message-passing
# between hidden windows, which needs something pumping/dispatching
# those messages the way a window manager normally does - bare Xvfb
# just hosts the X server, it doesn't do that. Kept even though it
# alone didn't fix the -10005 "IPC timeout" (confirmed by hand across
# several Wine versions with icewm already running) - not harmful, and
# still plausibly necessary even if not sufficient by itself.
icewm &
sleep 2

# Debug-only VNC, off by default. Even when enabled it's bound to all
# interfaces *inside this container* only - docker-compose.yml maps the
# port to 127.0.0.1 on the host, so it's only ever reachable via an SSH
# tunnel, never the public internet. Password comes from the environment
# at container start, never baked into the image.
if [ -n "$VNC_PASSWORD" ]; then
  x11vnc -display :99 -forever -shared -passwd "$VNC_PASSWORD" -rfbport 5900 &
  echo "VNC debug access enabled on :5900"
fi

echo "Starting MT5 RPyC bridge on :18812..."
# Not exec'd - this script (not wine) stays PID 1, so if the bridge exits
# or crashes the whole container exits cleanly and docker-compose's
# restart policy brings it back up fresh (new Xvfb, new wine session)
# instead of leaving an unmanaged wine process running under a dead script.
#
# mt5_server.py binds 0.0.0.0 itself (see its ThreadedServer hostname=)
# - :18812 is never published to the host in docker-compose.yml, so
# only containers already inside this same compose project's network
# can reach it - never the host, never the internet.
wine "$WINE_PYTHON" Z:/opt/mt5/mt5_server.py 18812
