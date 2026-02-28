# TradingBot Agent Guide

This document is specifically designed to help AI coding agents, refactoring tools, and language models understand the architecture, data structures, and security patterns of this `TradingBot` repository.

## 1. System Architecture
The bot is a fully automated options trading system that bridges the Charles Schwab API with multi-agent LLM reasoning (Gemini/OpenAI).

### 1.1 Core Pipeline (`bot/orchestrator.py`)
The `TradingBot` class in `orchestrator.py` is the central engine.
*   **Initialization**: Loads `BotConfig`, initializes the Schwab client (or paper trader), and starts background scheduling.
*   **Data Fetching**: Pulls options chains, live quotes, account balances, and historical daily bars.
*   **Strategy Scanning**: Iterates over enabled strategies in `bot/strategies/`.
*   **Risk & Execution**: Sends identified `TradeSignal` objects to `RiskManager`. If approved, they go to the LLM layers (`LLMAdvisor` / `MultiAgentCIO`) for final review before execution via `ExecutionAlgoEngine` or simple routing.

### 1.2 Multi-Agent LLM System
The system relies on LLMs for qualitative market analysis, acting as a filter for algorithmic signals:
*   **`bot/llm_strategist.py`**: The "CIO". Reviews the overall market regime, technicals, and news to determine macro posture.
*   **`bot/llm_advisor.py`**: A per-trade reviewer. Evaluates individual `TradeSignal` analyses against the current market structure. Uses `structured_outputs` (JSON).
*   **`bot/multi_agent_cio.py`**: An advanced debate architecture where multiple agent personas (Macro Economist, Volatility Quant, Risk Manager) debate a trade before the CIO issues a final verdict.
*   **`bot/rl_prompt_optimizer.py`**: A continuous-learning engine. During `TrainingSimulator` runs (or live post-trade analysis), it reviews P&L outcomes and updates `bot/data/learned_rules.json` with new prompt-injection rules.

### 1.3 Strategies (`bot/strategies/base.py`)
All strategies inherit from `BaseStrategy`. They must implement two primary methods:
1.  `scan_for_entries(...) -> list[TradeSignal]`: Generates "open" actions.
2.  `check_exits(...) -> list[TradeSignal]`: Generates "close" or "roll" actions.

**`TradeSignal` Object**: The standard currency of the bot. Agents reviewing or generating trades should look at `action`, `strategy`, `symbol`, and `analysis` (a `SpreadAnalysis` object containing Greeks, PoP, and credit details).

## 2. Security & File Persistence
### 2.1 File Security (`bot/file_security.py`)
**CRITICAL**: *Never* use standard `open(file, 'w')` for sensitive files (API tokens, trade ledgers, rules).
*   Always use `atomic_write_private(path, content, label)` to write files safely with atomic swapping and restricted POSIX permissions (`0o600`).
*   When reading, utilize `validate_sensitive_file` and `tighten_file_permissions`.

### 2.2 Local Data Store (`bot/data_store.py`)
For standard JSON artifacts, use the wrappers in `bot/data_store.py`:
*   `load_json(path, default)`
*   `dump_json(path, payload)`
These automatically utilize the secure `file_security` methods underneath.

## 3. Important Development Rules for Agents
*   **Unit Multipliers**: The `RiskManager` normalizes Greeks. If a contract has a delta of `0.05`, the `RiskManager` will internally scale this by `100.0` (to `5.0` share-equivalents) to calculate portfolio limits. Ensure tests provide normalized per-contract values (e.g., `net_delta=0.05`, `net_vega=0.002`).
*   **Model Strings**: The bot enforces the use of high-reasoning models. The current standard is `gemini-3.1-pro-thinking-preview` for deep reasoning, and `gemini-3.1-flash-thinking-preview` for fallbacks/sentiment. Do not revert these to standard or futuristic non-existent names (like `gpt-5.2`).
*   **Shutdown Handlers**: The bot's `_handle_shutdown` method explicitly cancels **ENTRY** orders (`TO_OPEN`) to prevent unmanaged exposure but preserves **EXIT** orders to maintain protective stops.
*   **Testing**: Run tests using `./.venv/bin/pytest`. The test suite is extensive and highly sensitive to logical changes in risk calculations and config normalization. Run them frequently.
*   **Aggressive AI Tuning**: The AI layers (`LLMAdvisor` and `MultiAgentCIO`) have been deliberately tuned to default to `approve` over `reject`. The system prompts mandate an aggressive stance to prioritize execution volume and raw P&L capture rather than conservative risk aversion.
*   **JSON Fallback Behaviors**: When LLM APIs produce invalid JSON or hit network errors, the `MultiAgentCIO` and `LLMAdvisor` fallbacks are hardcoded to `{"verdict": "approve", "confidence": 100.0}`. This prevents rate limits or malformed responses from throttling trading opportunities.
*   **Simulator P&L Calculations**: When generating synthetic option paths in `simulator.py`, the value of a credit spread must be calculated correctly to prevent artificial massive losses. The true current value of a credit spread is always `(short_leg_value - long_leg_value)`. Profit is then strictly `initial_value - current_value`.
*   **Risk Allowances**: The risk parameters in `bot/config.py` are explicitly set high (e.g., `max_open_positions=25`, `max_daily_loss_pct=15.0`) and the strategy quality thresholds mapped in `bot/strategies/base.py` are deliberately lowered (`min_pop=0.10`, `min_score=5.0`) to force higher signal throughput into the AI review layers.
