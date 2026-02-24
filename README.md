# Trading Bot

Automated options trading bot for Charles Schwab / ThinkorSwim with:
- rule-based strategy scanning,
- portfolio risk controls,
- LLM trade review using OpenAI GPT (`gpt-5.2-pro`),
- symbol + macro news intelligence for trade context.

## Strategies

- Credit spreads (bull put / bear call)
- Iron condors
- Covered calls

Live execution supports all three strategies with:
- persistent live trade ledger (`live_trades.json`),
- order reconciliation (including stale-order cancellation policy),
- automated exit order placement,
- broker market-hours checks.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure env:

```bash
cp .env.example .env
```

3. Add Schwab credentials and OpenAI key in `.env`:
- `SCHWAB_APP_KEY`
- `SCHWAB_APP_SECRET`
- `SCHWAB_ACCOUNT_HASH` (or `SCHWAB_ACCOUNT_INDEX`)
- `OPENAI_API_KEY`

4. Run guided live setup (credentials check + OAuth + account selection):

```bash
python3 main.py --setup-live
```

5. Run paper mode:

```bash
python3 main.py
```

6. Run live mode (after paper validation):

```bash
python3 main.py --live
```

## LLM + News Intelligence

- LLM provider: OpenAI Responses API
- Default model: `gpt-5.2-pro`
- Review mode: `advisory` or `blocking`
- News scanner:
  - ticker-specific headlines (earnings, analysts, events),
  - market-level headlines (Fed, inflation, volatility, macro risk),
  - lightweight sentiment/topic extraction fed into LLM context.

Tune these in `config.yaml`:
- `llm.*`
- `news.*`
- `execution.*`
- `alerts.*`

## Useful Commands

```bash
python3 main.py --once
python3 main.py --preflight-only
python3 main.py --report
python3 main.py --log-level DEBUG
```

## Operational Alerts

You can enable webhook alerts for runtime failures/incidents:
- `ALERTS_ENABLED=true`
- `ALERTS_WEBHOOK_URL=...`
- `ALERTS_MIN_LEVEL=ERROR`

Live preflight enforces alerting by default (`ALERTS_REQUIRE_IN_LIVE=true`).

## Runtime Hardening

- Rotating logs via `logging.max_bytes` and `logging.backup_count`
- Scanner request throttling + retry/backoff via `scanner.request_pause_seconds`,
  `scanner.max_retry_attempts`, and `scanner.error_backoff_seconds`
- Service templates for daemonized runs:
  - `/Users/bryan/TradingBot/deploy/systemd/tradingbot.service`
  - `/Users/bryan/TradingBot/deploy/launchd/com.bryan.tradingbot.plist`

## Project Layout

```text
main.py
bot/
  orchestrator.py
  market_scanner.py
  news_scanner.py
  llm_advisor.py
  risk_manager.py
  schwab_client.py
  paper_trader.py
  strategies/
tests/
```

## Disclaimer

This software is for educational and personal use. Options trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk. Always test thoroughly in paper mode before trading with real money.
