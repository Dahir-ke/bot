#!/bin/bash
set -e

# Virtual display the Wine-side GUI apps need (the terminal itself, and
# mt5server.exe's bundled bridge process).
Xvfb :99 -screen 0 1024x768x16 &
sleep 3

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
wine /opt/mt5/mt5server.exe -p 18812
