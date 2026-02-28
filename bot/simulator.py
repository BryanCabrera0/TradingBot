"""Offline synthetic training simulator for LLM options agents."""

import json
import logging
import random
import time
import uuid
from datetime import datetime
from typing import Any

from bot.analysis import SpreadAnalysis
from bot.config import BotConfig
from bot.data_store import load_json
from bot.llm_advisor import LLMAdvisor
from bot.llm_strategist import LLMStrategist
from bot.multi_agent_cio import MultiAgentCIO
from bot.openai_compat import request_openai_json
from bot.orchestrator import TradingBot
from bot.pricing import black_scholes, bs_greeks, generate_gbm_path
from bot.regime_detector import BEAR_TREND, BULL_TREND, HIGH_VOL_CHOP, LOW_VOL_GRIND
from bot.rl_prompt_optimizer import RLPromptOptimizer
from bot.strategies.base import TradeSignal

logger = logging.getLogger(__name__)

# Regime outcome priors used for regime selection.
# Keys match regime_detector constants exactly.
# Format: {strategy: (win_rate, avg_winner_multiple, avg_loser_multiple)}
REGIME_OUTCOMES: dict[str, dict[str, tuple[float, float, float]]] = {
    BULL_TREND: {
        "bull_put_spread": (0.85, 1.2, 2.5),
        "bear_call_spread": (0.15, 1.2, 2.5),
        "iron_condor": (0.45, 1.5, 3.0),
    },
    BEAR_TREND: {
        "bull_put_spread": (0.20, 1.2, 2.5),
        "bear_call_spread": (0.80, 1.2, 2.5),
        "iron_condor": (0.40, 1.5, 3.0),
    },
    HIGH_VOL_CHOP: {
        "bull_put_spread": (0.60, 1.2, 2.5),
        "bear_call_spread": (0.60, 1.2, 2.5),
        "iron_condor": (0.85, 1.5, 3.0),
    },
    LOW_VOL_GRIND: {
        "bull_put_spread": (0.75, 1.0, 2.5),
        "bear_call_spread": (0.75, 1.0, 2.5),
        "iron_condor": (0.50, 1.5, 3.0),
    },
}

DEFAULT_OUTCOME = (0.50, 1.0, 1.0)


class TrainingSimulator:
    """Generates synthetic options chains to train the RL Prompt Optimizer."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.optimizer = RLPromptOptimizer(
            config.rl_prompt_optimizer,
            explanations_path=config.llm.explanations_file,
            track_record_path=config.llm.track_record_file,
        )
        self.strategist = LLMStrategist(config.llm_strategist)
        self.advisor = LLMAdvisor(config.llm)
        self.cio = MultiAgentCIO(
            query_model=request_openai_json,
            parse_decision=self.advisor._parse_decision,
            learned_rules=self._load_learned_rules(),
        )
        # We instantiate a dummy bot just to fulfill the Advisor signature if needed,
        # but realistically we will mock the pipeline.
        config.terminal_ui.enabled = False
        config.llm.max_output_tokens = 2048
        self.bot = TradingBot(config)
        # Hijack the bot's clients so it absolutely never hits the network
        self.bot.schwab = None  # type: ignore

    def _load_learned_rules(self) -> list[str]:
        """Fetch current rules from the RL optimizer."""
        try:
            data = load_json(self.optimizer.rules_path, {"rules": []})
            return [str(r.get("rule", "")) for r in data.get("rules", [])]
        except Exception:
            return []

    def _generate_synthetic_chain(self, underlying_price: float, dte: int = 45, iv_mean: float = 0.20) -> dict[str, Any]:
        """Generate a mathematically sound SPY option chain using Black-Scholes."""
        calls, puts = [], []
        
        # Strikes from -10% to +10%
        center_strike = int(underlying_price)
        strikes = list(range(center_strike - 50, center_strike + 51, 1))
        
        T = dte / 365.0
        r = 0.05

        for strike in strikes:
            # Introduce a slight volatility smile
            distance = abs(strike - underlying_price) / underlying_price
            iv = max(0.10, iv_mean + (distance * 0.5))
            
            # --- Calls ---
            call_val = black_scholes(underlying_price, strike, T, r, iv, "call")
            call_greeks = bs_greeks(underlying_price, strike, T, r, iv, "call")
            
            # Simulate a realistic spread (wider for further OTM)
            spread_pct = max(0.01, min(0.10, distance * 0.5))
            call_bid = max(0.01, call_val * (1.0 - spread_pct))
            call_ask = max(0.02, call_val * (1.0 + spread_pct))

            calls.append(
                {
                    "strikePrice": strike,
                    "bid": round(float(call_bid), 2),
                    "ask": round(float(call_ask), 2),
                    "mark": round(float(call_val), 2),
                    "volatility": round(float(iv * 100), 2),
                    "delta": round(float(call_greeks["delta"]), 4),
                    "theta": round(float(call_greeks["theta"]), 4),
                    "gamma": round(float(call_greeks["gamma"]), 4),
                    "vega": round(float(call_greeks["vega"]), 4),
                    "openInterest": random.randint(100, 5000),
                    "totalVolume": random.randint(50, 2000),
                }
            )

            # --- Puts ---
            put_val = black_scholes(underlying_price, strike, T, r, iv, "put")
            put_greeks = bs_greeks(underlying_price, strike, T, r, iv, "put")
            
            put_bid = max(0.01, put_val * (1.0 - spread_pct))
            put_ask = max(0.02, put_val * (1.0 + spread_pct))

            puts.append(
                {
                    "strikePrice": strike,
                    "bid": round(float(put_bid), 2),
                    "ask": round(float(put_ask), 2),
                    "mark": round(float(put_val), 2),
                    "volatility": round(float(iv * 100), 2),
                    "delta": round(float(put_greeks["delta"]), 4),
                    "theta": round(float(put_greeks["theta"]), 4),
                    "gamma": round(float(put_greeks["gamma"]), 4),
                    "vega": round(float(put_greeks["vega"]), 4),
                    "openInterest": random.randint(100, 5000),
                    "totalVolume": random.randint(50, 2000),
                }
            )

        exp_date_str = f"2026-04-{dte:02d}" # Dummy string representation

        return {
            "underlying_price": underlying_price,
            "calls": {exp_date_str: calls},  
            "puts": {exp_date_str: puts},
        }

    def train(self, iterations: int = 50) -> None:
        """Run the offline LLM training loop."""
        print(f"\n  Simulator  ·  {iterations} iterations\n")

        regimes = list(REGIME_OUTCOMES.keys())
        wins = 0
        losses = 0
        skipped = 0
        total_pnl = 0.0

        for i in range(1, iterations + 1):
            regime = random.choice(regimes)
            underlying = round(random.uniform(400.0, 550.0), 2)
            chain = self._generate_synthetic_chain(underlying)

            strategy_name = "bull_put_spread" if regime in {BULL_TREND, LOW_VOL_GRIND} else "bear_call_spread"

            analysis = SpreadAnalysis(
                symbol="SPY",
                strategy=strategy_name,
                expiration="2026-04-17",
                dte=45,
                short_strike=underlying - 10 if strategy_name == "bull_put_spread" else underlying + 10,
                long_strike=underlying - 15 if strategy_name == "bull_put_spread" else underlying + 15,
                credit=1.20,
                max_loss=3.80,
                max_profit=1.20,
                risk_reward_ratio=0.31,
                credit_pct_of_width=0.24,
                probability_of_profit=0.72,
                expected_value=0.15,
                net_delta=0.08 if strategy_name == "bull_put_spread" else -0.08,
                net_theta=0.02,
                net_gamma=-0.01,
                net_vega=-0.03,
                score=85.0,
            )

            signal = TradeSignal(
                action="open",
                strategy=strategy_name,
                symbol="SPY",
                analysis=analysis,
                reason=f"Synthetic {regime} mathematically-discovered opportunity",
                order_spec={"type": "NET_CREDIT", "price": 1.20},
                quantity=1,
                size_multiplier=1.0,
            )

            trade = {
                "action": "open",
                "strategy": strategy_name,
                "symbol": "SPY",
                "reason": signal.reason,
                "analysis": analysis.__dict__,
            }

            # Advisor review
            try:
                advisor_result = self.advisor.review_trade(
                    signal=signal,
                    context={"regime": regime, "chain_data": chain},
                )
                approved = advisor_result.verdict == "approve"
                if not approved:
                    skipped += 1
                    print(f"  {i:>4}/{iterations}  {regime:<16}  {strategy_name:<22}  vetoed")
                    continue
            except Exception as e:
                skipped += 1
                logger.debug("Advisor error at iteration %d: %s", i, e)
                print(f"  {i:>4}/{iterations}  {regime:<16}  {strategy_name:<22}  advisor error")
                time.sleep(5)
                continue

            # CIO final decision
            try:
                prompt_json = json.dumps({
                    "proposed_trade": trade,
                    "advisor_feedback": {
                        "verdict": advisor_result.verdict,
                        "confidence": advisor_result.confidence,
                        "reasoning": advisor_result.reasoning,
                    },
                    "market_context": {"regime": regime},
                    "symbol": "SPY",
                    "current_price": underlying,
                })
                cio_result = self.cio.run(prompt=prompt_json)
                go_no_go = str(cio_result.final_payload.get("verdict", "reject")).lower()
                if go_no_go not in {"approve", "reduce_size"}:
                    skipped += 1
                    print(f"  {i:>4}/{iterations}  {regime:<16}  {strategy_name:<22}  rejected")
                    continue
            except Exception as e:
                skipped += 1
                logger.debug("CIO error at iteration %d: %s", i, e)
                print(f"  {i:>4}/{iterations}  {regime:<16}  {strategy_name:<22}  cio error")
                time.sleep(5)
                continue

            # Resolve synthetic outcome via GBM path
            mu, sigma = 0.0, 0.20
            if regime == BULL_TREND:
                mu, sigma = 0.15, 0.15
            elif regime == BEAR_TREND:
                mu, sigma = -0.15, 0.25
            elif regime == HIGH_VOL_CHOP:
                mu, sigma = 0.0, 0.35
            elif regime == LOW_VOL_GRIND:
                mu, sigma = 0.0, 0.10

            is_black_swan = random.random() < 0.15
            path = generate_gbm_path(underlying, mu, sigma, days=45, inject_black_swan=is_black_swan)

            # Calculate initial theoretical value to prevent immediate artificial P&L swings
            if strategy_name == "bull_put_spread":
                initial_value = (
                    black_scholes(underlying, analysis.short_strike, 45/365.0, 0.05, sigma, "put")
                    - black_scholes(underlying, analysis.long_strike, 45/365.0, 0.05, sigma, "put")
                )
            else:
                initial_value = (
                    black_scholes(underlying, analysis.short_strike, 45/365.0, 0.05, sigma, "call")
                    - black_scholes(underlying, analysis.long_strike, 45/365.0, 0.05, sigma, "call")
                )

            final_pnl = 0.0
            pnl = 0.0
            for d, price in enumerate(path):
                T = max(0.001, (45 - d) / 365.0)
                if strategy_name == "bull_put_spread":
                    current_value = (
                        black_scholes(price, analysis.short_strike, T, 0.05, sigma, "put")
                        - black_scholes(price, analysis.long_strike, T, 0.05, sigma, "put")
                    )
                else:
                    current_value = (
                        black_scholes(price, analysis.short_strike, T, 0.05, sigma, "call")
                        - black_scholes(price, analysis.long_strike, T, 0.05, sigma, "call")
                    )
                # P&L is the change in value: if value drops from initial, we profit (credit spread)
                pnl = initial_value - current_value
                if pnl <= -analysis.max_loss * 0.5:
                    final_pnl = pnl * 100
                    break
                elif pnl >= initial_value * 0.75: # Take profit at 75% of max possible gain
                    final_pnl = pnl * 100
                    break
            
            if final_pnl == 0.0:
                final_pnl = pnl * 100

            total_pnl += final_pnl
            if final_pnl >= 0:
                wins += 1
            else:
                losses += 1

            pnl_str = f"+${final_pnl:.0f}" if final_pnl >= 0 else f"-${abs(final_pnl):.0f}"
            swan_note = "  · black swan" if is_black_swan else ""
            print(f"  {i:>4}/{iterations}  {regime:<16}  {strategy_name:<22}  {pnl_str:>8}{swan_note}")

            # Feed RL optimizer (silently)
            position_id = f"sim_{uuid.uuid4().hex[:8]}"
            try:
                self.optimizer.process_closed_trade(
                    position_id=position_id,
                    pnl=final_pnl,
                    trade_context={"symbol": "SPY", "strategy": strategy_name, "regime": regime},
                )
            except Exception as e:
                logger.debug("RL optimizer error at iteration %d: %s", i, e)

            time.sleep(2)

        # Summary
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        pnl_str = f"+${total_pnl:.0f}" if total_pnl >= 0 else f"-${abs(total_pnl):.0f}"
        print(f"\n  {'─' * 38}")
        print(f"  Trades     {total_trades}")
        print(f"  Win rate   {win_rate:.0f}%")
        print(f"  Total P/L  {pnl_str}")
        if skipped:
            print(f"  Skipped    {skipped}")
        print(f"\n  Training complete. LLM memory updated.\n")
