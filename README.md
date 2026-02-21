# Options Trading Bot for ThinkorSwim / Charles Schwab

A fully automated options trading bot that connects to your ThinkorSwim (Charles Schwab) account and trades options strategies on your behalf.

## Strategies

| Strategy | Description |
|---|---|
| **Credit Spreads** | Sells bull put spreads and/or bear call spreads to collect premium |
| **Iron Condors** | Sells both a put spread and call spread, profiting when price stays in range |
| **Covered Calls** | Sells calls against stock you already own for income |

## Features

- **Market scanner** — dynamically scans 150+ tickers across all sectors to find the best options-tradeable stocks each cycle, ranked by options volume, IV, bid-ask tightness, and liquidity (no static watchlist needed)
- **Fully automated** — scans, enters, manages, and exits trades with zero manual intervention
- **Risk management** — position sizing, portfolio limits, daily loss caps, per-symbol limits
- **Paper trading** — test strategies with simulated money before going live
- **Configurable** — tune every parameter via `config.yaml`
- **Probability-based** — only enters trades above a minimum probability of profit threshold
- **Multi-factor scoring** — ranks opportunities by probability, premium quality, liquidity, Greeks
- **Automatic exits** — profit targets, stop losses, and DTE-based exits

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy the example env file and add your Schwab credentials:

```bash
cp .env.example .env
# Edit .env with your Schwab API credentials
```

Edit `config.yaml` to adjust strategies, watchlist, risk limits, and schedule.

### 3. Authenticate (first time only)

```bash
python -m bot.auth
```

This opens a browser for Schwab OAuth login. After authorizing, a token is saved locally.

### 4. Run in paper mode (recommended first)

```bash
python main.py
```

### 5. Run in live mode

```bash
python main.py --live
```

## Usage

```
python main.py [options]

Options:
  --config FILE    Path to config file (default: config.yaml)
  --live           Enable live trading (real money)
  --once           Run a single scan and exit
  --report         Show paper trading performance report
  --log-level LVL  Override log level (DEBUG/INFO/WARNING/ERROR)
```

## Getting Schwab API Credentials

1. Go to https://developer.schwab.com/
2. Create a developer account
3. Register a new application
4. Note your **App Key** and **App Secret**
5. Set callback URL to `https://127.0.0.1:8182/callback`
6. Add credentials to your `.env` file

## Configuration

All parameters are in `config.yaml`. Key settings:

- **scanner** — enable/disable dynamic market scanning, set price range, volume filters, result count
- **strategies** — enable/disable each strategy, set delta targets, DTE ranges, spread widths
- **watchlist** — fallback tickers (only used if scanner is disabled)
- **risk** — max portfolio risk, max position size, max positions, daily loss limit
- **schedule** — scan times, position check interval

## Architecture

```
main.py                  CLI entry point
bot/
  orchestrator.py        Main automated trading loop
  market_scanner.py      Dynamic market scanner (finds best stocks)
  schwab_client.py       Schwab API wrapper (auth, data, orders)
  analysis.py            Options analysis (Greeks, probability, scoring)
  risk_manager.py        Risk management and position sizing
  paper_trader.py        Paper trading simulator
  config.py              Configuration management
  auth.py                One-time OAuth flow
  strategies/
    base.py              Strategy base class
    credit_spreads.py    Bull put / bear call spreads
    iron_condors.py      Iron condor strategy
    covered_calls.py     Covered call strategy
```

## Disclaimer

This software is for educational and personal use. Options trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk. Always test thoroughly in paper mode before trading with real money.
