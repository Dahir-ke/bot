# Deploying the trading bot to Contabo

Runs on the same VPS as grandfinalehotel/DukaSync, but fully isolated in
its own `~/bot` directory and its own Docker image names - nothing here
is routed through the shared nginx proxy, and nothing here is exposed to
the public internet. See `docker-compose.yml` for why.

**This connects to a live account with real money.** The deploy steps
below build and verify the MT5 connection but never start the actual
trading loop - that's a separate, explicit, manual step (step 6).

## 1. Get the code onto the VPS

Push-to-deploy, same pattern as the other stacks on this box:

```bash
# one-time, on the VPS
mkdir -p ~/bot.git && cd ~/bot.git && git init --bare -b main
cat > hooks/post-receive <<'HOOK'
#!/bin/sh
git --work-tree=/home/deploy/bot --git-dir=/home/deploy/bot.git checkout -f main
echo Deployed.
HOOK
chmod +x hooks/post-receive
mkdir -p ~/bot

# from your machine
git remote add contabo deploy@<vps-ip>:bot.git
git push contabo main
```

`.env` (real MT5 credentials) is gitignored on purpose - copy it directly,
never through git:

```bash
scp .env deploy@<vps-ip>:~/bot/.env
ssh deploy@<vps-ip> chmod 600 ~/bot/.env
```

## 2. Build

```bash
cd ~/bot
mkdir -p data
docker compose build
```

The `mt5` image build downloads and silently installs the real Exness MT5
terminal under Wine (`/auto`, MetaQuotes' own documented unattended-install
flag) plus a pinned `mt5server.exe` release (the mt5linux bridge - a
standalone binary, so no separate Windows-Python-under-Wine install is
needed). This step alone can take several minutes.

## 3. Start the MT5 bridge and dashboard only

```bash
docker compose up -d mt5 dashboard
docker compose ps
docker compose logs -f mt5
```

Expect to see `Starting mt5linux bridge server on :18812...` with no
errors. The `bot` service is defined but never started here - see step 6.

## 4. Verify the MT5 connection - read-only, places no trades

From the VPS, run a one-off Python check using the `bot` image (built,
not yet running as a long-lived container):

```bash
docker compose run --rm bot python -c "
from mt5linux import MetaTrader5
import os
mt5 = MetaTrader5(host='mt5', port=18812)
ok = mt5.initialize(
    login=int(os.environ['MT5_LOGIN']),
    password=os.environ['MT5_PASSWORD'],
    server=os.environ['MT5_SERVER'],
)
print('initialize():', ok)
print('account_info():', mt5.account_info())
print('terminal_info():', mt5.terminal_info())
mt5.shutdown()
"
```

`account_info()` returning real balance/equity/leverage confirms the
terminal launched, logged in, and the bridge works end to end - all
without `bo.py`/`bot.py` (and therefore no trading logic) ever running.

If this fails, check `docker compose logs mt5` first. Optional: enable
VNC to see the terminal's own window (`VNC_PASSWORD=<something>` in
`.env`, `docker compose up -d --force-recreate mt5`, then tunnel - see
step 5) and look for a login-error dialog.

## 5. View the dashboard (private only)

Nothing here is on the public internet. From your own machine:

```bash
ssh -L 8800:127.0.0.1:8800 deploy@<vps-ip>
```

Then open `http://localhost:8800` in your own browser. Leave that SSH
session open while you're using it.

For the MT5 terminal's own VNC (only if `VNC_PASSWORD` is set):

```bash
ssh -L 8801:127.0.0.1:8801 deploy@<vps-ip>
# then connect a VNC client to localhost:8801
```

## 6. Start the bot - the actual live-trading step

Only once you've verified step 4 and you're ready for it to place real
trades:

```bash
docker compose up -d bot   # brings the container up idle (sleep infinity)
docker compose exec -d bot python bo.py   # or bot.py - starts the trading loop
docker compose logs -f bot
```

`-d` on `exec` runs it detached so it survives your SSH session ending;
drop it to watch the first few minutes live before backgrounding it.

To stop trading:

```bash
docker compose exec bot pkill -f bo.py   # or bot.py
```

The container itself stays up (still `sleep infinity`'d) either way -
only the trading process inside it stops.

## Updating later

```bash
cd ~/bot && git pull   # or: push from your machine, same as step 1
docker compose build
docker compose up -d mt5 dashboard   # bot is left alone - restart it yourself if it's running
```
