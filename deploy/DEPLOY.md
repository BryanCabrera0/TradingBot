# Server Deployment Guide

Deploy TradingBot to a cloud VPS for 24/7 operation.

## Recommended Providers

| Provider | Plan | Cost | Notes |
|---|---|---|---|
| **Hetzner** | CX22 | ~$4.50/mo | Best value, EU/US datacenters |
| **DigitalOcean** | Basic Droplet | $6/mo | Easy UI, US datacenters |
| **Vultr** | Cloud Compute | $6/mo | Wide datacenter selection |
| **AWS Lightsail** | Nano | $5/mo | If you want AWS ecosystem |

**Minimum specs:** 1 vCPU, 1GB RAM, 25GB SSD, Ubuntu 22.04+

## Quick Start (5 Minutes)

### 1. Create a VPS
Sign up with any provider above. Choose **Ubuntu 22.04 LTS**.

### 2. Clone the Repo
```bash
ssh root@YOUR_SERVER_IP
git clone https://github.com/YOUR_USERNAME/TradingBot.git /opt/tradingbot
```

### 3. Run Setup
```bash
sudo bash /opt/tradingbot/deploy/setup-server.sh
```
This installs Python, creates a service user, sets up the virtual environment,
installs dependencies, and registers the systemd service.

### 4. Configure
```bash
nano /opt/tradingbot/.env
```
Add your API keys (copy from your MacBook's `.env`):
```
SCHWAB_APP_KEY=your_key
SCHWAB_APP_SECRET=your_secret
SCHWAB_CALLBACK_URL=https://127.0.0.1:8182/callback
SCHWAB_ACCOUNT_INDEX=0
GOOGLE_API_KEY=your_key
```

### 5. Authorize Schwab
```bash
sudo -u tradingbot /opt/tradingbot/venv/bin/python3 main.py auth
```
> **Note:** The Schwab OAuth flow opens a browser. On a headless server, you'll
> need to open the auth URL on your local machine, complete login, then paste
> the redirect URL back in the terminal. The `bot.auth` module handles this.

### 6. Start the Bot
```bash
sudo systemctl start tradingbot
```

### 7. Verify
```bash
# Check status
sudo systemctl status tradingbot

# Stream live logs
journalctl -u tradingbot -f

# Check last 100 log lines
journalctl -u tradingbot -n 100
```

## Managing the Bot

```bash
# Stop
sudo systemctl stop tradingbot

# Restart (after config changes)
sudo systemctl restart tradingbot

# Disable auto-start on boot
sudo systemctl disable tradingbot

# Switch to live mode: edit the service file
sudo nano /etc/systemd/system/tradingbot.service
# Change: ExecStart=... run paper  →  ExecStart=... run live
sudo systemctl daemon-reload
sudo systemctl restart tradingbot
```

## Pushing Updates from Your MacBook

After making code changes locally:
```bash
./deploy/push-update.sh root@YOUR_SERVER_IP
```
This syncs code, installs new dependencies, and restarts the bot automatically.
Your `.env`, `token.json`, trade data, and logs are preserved on the server.

## Copying Training Data to Server

To transfer your paper trading ML training data to the server:
```bash
scp bot/data/closed_trades.json root@YOUR_SERVER_IP:/opt/tradingbot/bot/data/
scp bot/data/ml_model.json root@YOUR_SERVER_IP:/opt/tradingbot/bot/data/
scp bot/data/llm_track_record.json root@YOUR_SERVER_IP:/opt/tradingbot/bot/data/
```

## Token Renewal

Schwab tokens expire every 7 days. The bot will warn you in the logs:
```
⚠  SCHWAB TOKEN EXPIRING SOON — expires in ~18 hours
```

To renew:
```bash
sudo systemctl stop tradingbot
sudo -u tradingbot /opt/tradingbot/venv/bin/python3 main.py auth
sudo systemctl start tradingbot
```

## File Structure on Server

```
/opt/tradingbot/
├── .env                  # API keys (server-specific)
├── token.json            # Schwab OAuth token
├── config.yaml           # Bot configuration
├── main.py               # Entry point
├── bot/                  # Bot source code
│   └── data/             # ML models, trade history
├── logs/                 # Log files
├── deploy/               # Deployment scripts
└── venv/                 # Python virtual environment
```
