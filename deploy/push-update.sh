#!/usr/bin/env bash
#
# Deploy updates from your MacBook to your server.
#
# Usage:
#   ./deploy/push-update.sh user@your-server.com
#
# This syncs code changes, installs any new dependencies, and restarts the bot.
#

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 user@server-ip"
    echo "Example: $0 root@143.198.100.50"
    exit 1
fi

SERVER="$1"
REMOTE_DIR="/opt/tradingbot"

echo "╭──────────────────────────────────────────────────────╮"
echo "│  Deploying TradingBot Update                         │"
echo "╰──────────────────────────────────────────────────────╯"
echo

# ── Sync code (exclude secrets, data, and venv) ────────────
echo "▸ Syncing code to $SERVER..."
rsync -avz --delete \
    --exclude '.env' \
    --exclude 'token.json' \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'bot/data/' \
    --exclude 'logs/' \
    --exclude 'paper_trades.json' \
    --exclude '.git/' \
    ./ "$SERVER:$REMOTE_DIR/"

# ── Install any new dependencies ────────────────────────────
echo "▸ Installing dependencies..."
ssh "$SERVER" "cd $REMOTE_DIR && sudo -u tradingbot venv/bin/pip install --quiet -r requirements.txt 2>/dev/null || true"

# ── Restart the service ─────────────────────────────────────
echo "▸ Restarting TradingBot..."
ssh "$SERVER" "sudo systemctl restart tradingbot"

# ── Show status ─────────────────────────────────────────────
echo "▸ Checking status..."
ssh "$SERVER" "sudo systemctl status tradingbot --no-pager -l | head -15"

echo
echo "✓ Deployment complete!"
echo "  View logs: ssh $SERVER 'journalctl -u tradingbot -f'"
