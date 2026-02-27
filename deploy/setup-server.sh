#!/usr/bin/env bash
#
# TradingBot Server Setup Script
# ================================
# Run this once on a fresh Ubuntu/Debian VPS to set up TradingBot.
#
# Usage:
#   1. SSH into your server
#   2. Clone the repo:  git clone <your-repo-url> /opt/tradingbot
#   3. Run this script:  sudo bash /opt/tradingbot/deploy/setup-server.sh
#
# After setup:
#   - Edit /opt/tradingbot/.env with your API keys
#   - Run auth:  sudo -u tradingbot /opt/tradingbot/venv/bin/python3 -m bot.auth
#   - Start:     sudo systemctl start tradingbot
#   - Logs:      sudo journalctl -u tradingbot -f
#

set -euo pipefail

INSTALL_DIR="/opt/tradingbot"
SERVICE_NAME="tradingbot"

echo "╭──────────────────────────────────────────────────────╮"
echo "│  TradingBot Server Setup                             │"
echo "╰──────────────────────────────────────────────────────╯"
echo

# ── 1. System dependencies ──────────────────────────────────
echo "▸ Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git curl > /dev/null

# ── 2. Create service user ──────────────────────────────────
if ! id -u "$SERVICE_NAME" &>/dev/null; then
    echo "▸ Creating service user: $SERVICE_NAME"
    useradd --system --shell /usr/sbin/nologin --home-dir "$INSTALL_DIR" "$SERVICE_NAME"
fi

# ── 3. Set permissions ──────────────────────────────────────
echo "▸ Setting directory permissions..."
chown -R "$SERVICE_NAME:$SERVICE_NAME" "$INSTALL_DIR"
chmod 750 "$INSTALL_DIR"

# ── 4. Create virtual environment ───────────────────────────
echo "▸ Creating Python virtual environment..."
sudo -u "$SERVICE_NAME" python3 -m venv "$INSTALL_DIR/venv"

# ── 5. Install dependencies ─────────────────────────────────
echo "▸ Installing Python dependencies..."
sudo -u "$SERVICE_NAME" "$INSTALL_DIR/venv/bin/pip" install --quiet --upgrade pip
if [ -f "$INSTALL_DIR/requirements.txt" ]; then
    sudo -u "$SERVICE_NAME" "$INSTALL_DIR/venv/bin/pip" install --quiet -r "$INSTALL_DIR/requirements.txt"
else
    echo "  ⚠ No requirements.txt found. Install dependencies manually."
fi

# ── 6. Install systemd service ──────────────────────────────
echo "▸ Installing systemd service..."
cp "$INSTALL_DIR/deploy/tradingbot.service" "/etc/systemd/system/$SERVICE_NAME.service"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

# ── 7. Create .env if missing ───────────────────────────────
if [ ! -f "$INSTALL_DIR/.env" ]; then
    echo "▸ Creating .env template..."
    cat > "$INSTALL_DIR/.env" << 'ENVEOF'
# Schwab API credentials
SCHWAB_APP_KEY=
SCHWAB_APP_SECRET=
SCHWAB_CALLBACK_URL=https://127.0.0.1:8182/callback
SCHWAB_ACCOUNT_INDEX=0

# Google AI (for LLM advisor)
GOOGLE_API_KEY=

# Optional: OpenAI
OPENAI_API_KEY=
ENVEOF
    chown "$SERVICE_NAME:$SERVICE_NAME" "$INSTALL_DIR/.env"
    chmod 600 "$INSTALL_DIR/.env"
fi

echo
echo "╭──────────────────────────────────────────────────────╮"
echo "│  ✓ Setup Complete!                                   │"
echo "│                                                      │"
echo "│  Next steps:                                         │"
echo "│  1. Edit .env:    nano /opt/tradingbot/.env           │"
echo "│  2. Authorize:    sudo -u tradingbot \                │"
echo "│       /opt/tradingbot/venv/bin/python3 main.py auth   │"
echo "│  3. Start bot:    sudo systemctl start tradingbot     │"
echo "│  4. View logs:    journalctl -u tradingbot -f         │"
echo "│                                                      │"
echo "│  Manage:                                              │"
echo "│    sudo systemctl stop tradingbot                     │"
echo "│    sudo systemctl restart tradingbot                  │"
echo "│    sudo systemctl status tradingbot                   │"
echo "╰──────────────────────────────────────────────────────╯"
