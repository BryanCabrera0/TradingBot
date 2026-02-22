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
- `OPENAI_API_KEY`

4. Run first-time Schwab OAuth:

```bash
python -m bot.auth
```

5. Run paper mode:

```bash
python main.py
```

6. Run live mode (after paper validation):

```bash
python main.py --live
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

## Useful Commands

```bash
python main.py --once
python main.py --preflight-only
python main.py --report
python main.py --log-level DEBUG
```

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
