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
# gets no real display, and mt5server.exe can't do anything useful even
# though this script itself keeps running. Cheap enough to always clear.
rm -f /tmp/.X99-lock

# Virtual display the Wine-side GUI apps need (the terminal itself, and
# mt5server.exe's bundled bridge process).
Xvfb :99 -screen 0 1024x768x16 &
sleep 3

# Bare Xvfb has no window manager - confirmed by hand that without one,
# mt5.initialize() reliably times out with -10005 "IPC timeout" even
# though the terminal fully launches, connects, and streams live quotes
# (checked via VNC). The real MetaTrader5 package's Python-API handshake
# is implemented with Windows message-passing between hidden windows,
# which needs something pumping/dispatching those messages the way a
# window manager normally does - bare Xvfb just hosts the X server, it
# doesn't do that. icewm is a minimal WM added purely to make that
# message-passing work, not for any visible UI purpose here.
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

echo "Starting mt5linux bridge server on :18812..."
# Not exec'd - this script (not wine) stays PID 1, so if the bridge exits
# or crashes the whole container exits cleanly and docker-compose's
# restart policy brings it back up fresh (new Xvfb, new wine session)
# instead of leaving an unmanaged wine process running under a dead script.
#
# --host 0.0.0.0, not the default localhost - confirmed by hand that the
# default made this unreachable from the separate `bot` container even
# over the docker-compose network ("Connection refused" - nothing was
# listening on any interface but this container's own loopback). Safe to
# open up: :18812 is never published to the host in docker-compose.yml,
# so only containers already inside this same compose project's network
# can reach it - never the host, never the internet.
wine /opt/mt5/mt5server.exe --host 0.0.0.0 -p 18812
