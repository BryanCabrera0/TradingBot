"""Offline synthetic training simulator for LLM options agents."""

import json
import logging
import random
import time
import uuid
from datetime import datetime
from typing import Any, Optional

from bot.analysis import SpreadAnalysis
from bot.config import BotConfig
from bot.data_store import load_json, dump_json
from bot.llm_advisor import LLMAdvisor
from bot.llm_strategist import LLMStrategist
from bot.multi_agent_cio import MultiAgentCIO
from bot.openai_compat import request_openai_json
from bot.orchestrator import TradingBot
from bot.pricing import black_scholes, bs_greeks, generate_gbm_path
from bot.regime_detector import (
    BEAR_TREND,
    BULL_TREND,
    CRASH_CRISIS,
    HIGH_VOL_CHOP,
    LOW_VOL_GRIND,
)
from bot.rl_prompt_optimizer import RLPromptOptimizer
from bot.strategies.base import TradeSignal

logger = logging.getLogger(__name__)

# Regime outcome priors (strategy: (win_rate, avg_winner_multiple, avg_loser_multiple))
# These are used in REGIME_OUTCOMES for reference but actual P&L is GBM-derived.
REGIME_OUTCOMES: dict[str, dict[str, tuple[float, float, float]]] = {
    BULL_TREND: {
        "bull_put_spread": (0.85, 1.2, 2.5),
        "bear_call_spread": (0.15, 1.2, 2.5),
        "iron_condor": (0.55, 1.5, 3.0),
    },
    BEAR_TREND: {
        "bull_put_spread": (0.20, 1.2, 2.5),
        "bear_call_spread": (0.80, 1.2, 2.5),
        "iron_condor": (0.40, 1.5, 3.0),
    },
    HIGH_VOL_CHOP: {
        "bull_put_spread": (0.45, 1.2, 2.5),
        "bear_call_spread": (0.45, 1.2, 2.5),
        "iron_condor": (0.85, 1.5, 3.0),
    },
    LOW_VOL_GRIND: {
        "bull_put_spread": (0.75, 1.0, 2.5),
        "bear_call_spread": (0.30, 1.0, 2.5),
        "iron_condor": (0.65, 1.5, 3.0),
    },
    CRASH_CRISIS: {
        "bull_put_spread": (0.10, 1.2, 4.0),
        "bear_call_spread": (0.65, 1.2, 2.5),
        "iron_condor": (0.20, 1.5, 5.0),
    },
}

DEFAULT_OUTCOME = (0.50, 1.0, 1.0)

# Regime GBM parameters: (mu, sigma, black_swan_prob)
REGIME_GBM: dict[str, tuple[float, float, float]] = {
    BULL_TREND:     (0.15,  0.15, 0.05),
    BEAR_TREND:     (-0.15, 0.25, 0.12),
    HIGH_VOL_CHOP:  (0.00,  0.35, 0.10),
    LOW_VOL_GRIND:  (0.00,  0.10, 0.02),
    CRASH_CRISIS:   (-0.30, 0.50, 0.40),
}

# Primary strategy per regime + a contrarian override for adversarial training
REGIME_STRATEGY_MAP: dict[str, dict[str, str]] = {
    BULL_TREND:    {"primary": "bull_put_spread",  "alt": "iron_condor",    "bad": "bear_call_spread"},
    BEAR_TREND:    {"primary": "bear_call_spread", "alt": "iron_condor",    "bad": "bull_put_spread"},
    HIGH_VOL_CHOP: {"primary": "iron_condor",      "alt": "bear_call_spread","bad": "bull_put_spread"},
    LOW_VOL_GRIND: {"primary": "bull_put_spread",  "alt": "iron_condor",    "bad": "bear_call_spread"},
    CRASH_CRISIS:  {"primary": "bear_call_spread", "alt": "iron_condor",    "bad": "bull_put_spread"},
}

# Fraction of iterations that use an adversarial (wrong-regime) strategy.
# This intentionally generates losses so the RL optimizer can detect failure patterns.
ADVERSARIAL_RATE = 0.25


def _find_strike_near_delta(options: list[dict], target_delta: float) -> Optional[dict]:
    """Return the option closest to target_delta from a sorted option list."""
    if not options:
        return None
    best = min(options, key=lambda o: abs(abs(o.get("delta", 0.0)) - abs(target_delta)))
    return best


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
        config.terminal_ui.enabled = False
        config.llm.max_output_tokens = 2048
        self.bot = TradingBot(config)
        self.bot.schwab = None  # type: ignore
        self._track_record_path = config.llm.track_record_file

    def _load_learned_rules(self) -> list[str]:
        """Fetch current rules from the RL optimizer."""
        try:
            data = load_json(self.optimizer.rules_path, {"rules": []})
            return [str(r.get("rule", "")) for r in data.get("rules", [])]
        except Exception:
            return []

    def _generate_synthetic_chain(
        self, underlying_price: float, dte: int = 45, iv_mean: float = 0.20
    ) -> dict[str, Any]:
        """Generate a mathematically sound SPY option chain using Black-Scholes."""
        calls, puts = [], []

        # 101 strikes from -50 to +50 around the underlying
        center_strike = int(round(underlying_price))
        strikes = list(range(center_strike - 50, center_strike + 51, 1))

        T = dte / 365.0
        r = 0.05

        for strike in strikes:
            distance = abs(strike - underlying_price) / underlying_price
            # Volatility smile: higher IV for OTM options
            iv = max(0.10, iv_mean + (distance * 0.5))

            spread_pct = max(0.01, min(0.10, distance * 0.5))

            # Calls
            call_val = black_scholes(underlying_price, strike, T, r, iv, "call")
            call_greeks = bs_greeks(underlying_price, strike, T, r, iv, "call")
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

            # Puts
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

        exp_date_str = f"2026-04-{dte:02d}"
        return {
            "underlying_price": underlying_price,
            "calls": {exp_date_str: calls},
            "puts": {exp_date_str: puts},
        }

    def _build_analysis_from_chain(
        self,
        strategy_name: str,
        underlying: float,
        chain: dict,
        regime_sigma: float,
    ) -> SpreadAnalysis:
        """Derive realistic trade parameters by finding delta-targeted strikes in chain."""
        exp_date_str = list(chain["calls"].keys())[0]
        calls = chain["calls"][exp_date_str]
        puts = chain["puts"][exp_date_str]

        dte = 45
        T = dte / 365.0
        r = 0.05

        if strategy_name == "bull_put_spread":
            short_opt = _find_strike_near_delta(puts, -0.20) or puts[len(puts)//2]
            long_opt  = _find_strike_near_delta(puts, -0.30) or puts[len(puts)//2 - 5]
            short_strike = float(short_opt.get("strikePrice", underlying - 10))
            long_strike  = float(long_opt.get("strikePrice", underlying - 15))
            # Ensure long is below short for put spreads
            if long_strike >= short_strike:
                long_strike = short_strike - 5
            credit = max(0.10, float(short_opt.get("bid", 1.20)) - float(long_opt.get("ask", 0.05)))
            width = abs(short_strike - long_strike)
            max_loss = max(0.01, width - credit)
            net_delta = float(short_opt.get("delta", -0.20)) - float(long_opt.get("delta", -0.10))
            net_theta = float(short_opt.get("theta", 0.02)) - float(long_opt.get("theta", 0.01))
            net_gamma = float(short_opt.get("gamma", -0.01)) - float(long_opt.get("gamma", -0.005))
            net_vega  = float(short_opt.get("vega", -0.03)) - float(long_opt.get("vega", -0.015))
            pop = max(0.50, min(0.95, 1.0 - abs(net_delta)))

        elif strategy_name == "bear_call_spread":
            short_opt = _find_strike_near_delta(calls, 0.20) or calls[len(calls)//2]
            long_opt  = _find_strike_near_delta(calls, 0.10) or calls[len(calls)//2 + 5]
            short_strike = float(short_opt.get("strikePrice", underlying + 10))
            long_strike  = float(long_opt.get("strikePrice", underlying + 15))
            if long_strike <= short_strike:
                long_strike = short_strike + 5
            credit = max(0.10, float(short_opt.get("bid", 1.20)) - float(long_opt.get("ask", 0.05)))
            width = abs(long_strike - short_strike)
            max_loss = max(0.01, width - credit)
            net_delta = float(short_opt.get("delta", 0.20)) - float(long_opt.get("delta", 0.10))
            net_delta = -abs(net_delta)  # bear call spread has negative net delta
            net_theta = float(short_opt.get("theta", 0.02)) - float(long_opt.get("theta", 0.01))
            net_gamma = -(float(short_opt.get("gamma", 0.01)) - float(long_opt.get("gamma", 0.005)))
            net_vega  = -(float(short_opt.get("vega", 0.03)) - float(long_opt.get("vega", 0.015)))
            pop = max(0.50, min(0.95, 1.0 - abs(net_delta)))

        elif strategy_name == "iron_condor":
            # Short put side
            sp_opt = _find_strike_near_delta(puts,  -0.16) or puts[len(puts)//2]
            lp_opt = _find_strike_near_delta(puts,  -0.25) or puts[len(puts)//2 - 5]
            # Short call side
            sc_opt = _find_strike_near_delta(calls,  0.16) or calls[len(calls)//2]
            lc_opt = _find_strike_near_delta(calls,  0.25) or calls[len(calls)//2 + 5]

            short_strike = float(sp_opt.get("strikePrice", underlying - 10))
            long_strike  = float(lp_opt.get("strikePrice", underlying - 15))
            if long_strike >= short_strike:
                long_strike = short_strike - 5

            sc_strike = float(sc_opt.get("strikePrice", underlying + 10))
            lc_strike = float(lc_opt.get("strikePrice", underlying + 15))
            if lc_strike <= sc_strike:
                lc_strike = sc_strike + 5

            put_credit  = max(0.05, float(sp_opt.get("bid", 0.70)) - float(lp_opt.get("ask", 0.05)))
            call_credit = max(0.05, float(sc_opt.get("bid", 0.70)) - float(lc_opt.get("ask", 0.05)))
            credit = put_credit + call_credit
            put_width  = abs(short_strike - long_strike)
            call_width = abs(lc_strike - sc_strike)
            max_loss = max(0.01, max(put_width, call_width) - credit)

            net_delta = 0.0  # iron condor is approximately delta-neutral
            net_theta = (
                float(sp_opt.get("theta", 0.01)) + float(sc_opt.get("theta", 0.01))
                - float(lp_opt.get("theta", 0.005)) - float(lc_opt.get("theta", 0.005))
            )
            net_gamma = -(
                float(sp_opt.get("gamma", 0.005)) + float(sc_opt.get("gamma", 0.005))
            )
            net_vega = -(
                float(sp_opt.get("vega", 0.02)) + float(sc_opt.get("vega", 0.02))
            )
            pop = 0.70  # iron condors typically have high POP

        else:
            # Fallback defaults
            short_strike = underlying - 10
            long_strike  = underlying - 15
            credit  = 1.20
            max_loss = 3.80
            net_delta = 0.08
            net_theta = 0.02
            net_gamma = -0.01
            net_vega  = -0.03
            pop = 0.72

        width = abs(short_strike - long_strike) if strategy_name != "iron_condor" else max(put_width, call_width)  # type: ignore[possibly-undefined]
        credit_pct = credit / max(1.0, width)
        ev = credit * pop - max_loss * (1.0 - pop)
        rr = credit / max(0.01, max_loss)
        # Score based on probability, EV, and credit efficiency
        score = min(99.0, max(30.0, pop * 60 + ev * 10 + rr * 10))

        return SpreadAnalysis(
            symbol="SPY",
            strategy=strategy_name,
            expiration=f"2026-04-{dte:02d}",
            dte=dte,
            short_strike=round(short_strike, 2),
            long_strike=round(long_strike, 2),
            credit=round(credit, 2),
            max_loss=round(max_loss, 2),
            max_profit=round(credit, 2),
            risk_reward_ratio=round(rr, 4),
            credit_pct_of_width=round(credit_pct, 4),
            probability_of_profit=round(pop, 4),
            expected_value=round(ev, 4),
            net_delta=round(net_delta, 4),
            net_theta=round(net_theta, 4),
            net_gamma=round(net_gamma, 4),
            net_vega=round(net_vega, 4),
            score=round(score, 1),
        )

    def _simulate_pnl(
        self,
        strategy_name: str,
        analysis: SpreadAnalysis,
        underlying: float,
        mu: float,
        sigma: float,
        is_black_swan: bool,
    ) -> float:
        """Run GBM path and compute realistic P&L for the given strategy."""
        dte = 45
        path = generate_gbm_path(underlying, mu, sigma, days=dte, inject_black_swan=is_black_swan)

        if strategy_name in ("bull_put_spread", "bear_call_spread"):
            if strategy_name == "bull_put_spread":
                initial_value = (
                    black_scholes(underlying, analysis.short_strike, dte / 365.0, 0.05, sigma, "put")
                    - black_scholes(underlying, analysis.long_strike, dte / 365.0, 0.05, sigma, "put")
                )
            else:
                initial_value = (
                    black_scholes(underlying, analysis.short_strike, dte / 365.0, 0.05, sigma, "call")
                    - black_scholes(underlying, analysis.long_strike, dte / 365.0, 0.05, sigma, "call")
                )

            stop_loss_threshold = -analysis.max_loss * 0.50
            take_profit_threshold = initial_value * 0.75

            final_pnl = 0.0
            last_pnl = 0.0
            for d, price in enumerate(path):
                T = max(0.001, (dte - d) / 365.0)
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
                pnl = initial_value - current_value
                last_pnl = pnl
                if pnl <= stop_loss_threshold:
                    final_pnl = pnl * 100
                    break
                if pnl >= take_profit_threshold:
                    final_pnl = pnl * 100
                    break
            else:
                final_pnl = last_pnl * 100

        elif strategy_name == "iron_condor":
            # Iron condor: short put spread + short call spread
            short_put  = analysis.short_strike
            long_put   = analysis.long_strike
            # Approximate call strikes as symmetric about underlying
            call_width = abs(short_put - long_put)
            short_call = underlying + (underlying - short_put)
            long_call  = short_call + call_width

            iv_mean = sigma
            put_init  = (
                black_scholes(underlying, short_put,  dte / 365.0, 0.05, iv_mean, "put")
                - black_scholes(underlying, long_put,   dte / 365.0, 0.05, iv_mean, "put")
            )
            call_init = (
                black_scholes(underlying, short_call, dte / 365.0, 0.05, iv_mean, "call")
                - black_scholes(underlying, long_call, dte / 365.0, 0.05, iv_mean, "call")
            )
            initial_value = put_init + call_init

            max_loss_val = max(call_width - initial_value, 0.01)
            stop_loss_threshold = -max_loss_val * 0.50
            take_profit_threshold = initial_value * 0.50  # Iron condors target 50% profit

            final_pnl = 0.0
            last_pnl = 0.0
            for d, price in enumerate(path):
                T = max(0.001, (dte - d) / 365.0)
                # Use higher IV as price moves away from center (vol smile effect)
                distance = abs(price - underlying) / underlying
                iv = iv_mean + distance * 0.3
                put_val  = (
                    black_scholes(price, short_put,  T, 0.05, iv, "put")
                    - black_scholes(price, long_put,   T, 0.05, iv, "put")
                )
                call_val = (
                    black_scholes(price, short_call, T, 0.05, iv, "call")
                    - black_scholes(price, long_call, T, 0.05, iv, "call")
                )
                current_value = put_val + call_val
                pnl = initial_value - current_value
                last_pnl = pnl
                if pnl <= stop_loss_threshold:
                    final_pnl = pnl * 100
                    break
                if pnl >= take_profit_threshold:
                    final_pnl = pnl * 100
                    break
            else:
                final_pnl = last_pnl * 100

        else:
            # Generic fallback
            final_pnl = (analysis.credit if random.random() < 0.65 else -analysis.max_loss) * 100

        return final_pnl

    def _update_track_record_outcome(self, review_id: str, outcome: str, pnl: float) -> None:
        """Write the resolved outcome back into llm_track_record.json for closed-loop feedback."""
        try:
            payload = load_json(self._track_record_path, {"trades": []})
            trades = payload.get("trades", [])
            for trade in reversed(trades):
                if trade.get("review_id") == review_id:
                    trade["outcome"] = outcome
                    trade["realized_pnl"] = round(pnl, 2)
                    break
            dump_json(self._track_record_path, payload)
        except Exception as e:
            logger.debug("Could not update track record outcome for %s: %s", review_id, e)

    def train(self, iterations: int = 50) -> None:
        """Run the offline LLM training loop."""
        print(f"\n  Simulator  ·  {iterations} iterations\n")

        regimes = list(REGIME_OUTCOMES.keys())
        wins = 0
        losses = 0
        skipped = 0
        total_pnl = 0.0
        adversarial_count = 0
        rules_before = len(self._load_learned_rules())

        # Regime-by-strategy performance tracker for summary
        regime_stats: dict[str, dict] = {}

        for i in range(1, iterations + 1):
            regime = random.choice(regimes)
            underlying = round(random.uniform(450.0, 575.0), 2)

            # IV varies by regime: high in crash, low in grind
            iv_map = {
                BULL_TREND: random.uniform(0.15, 0.22),
                BEAR_TREND: random.uniform(0.20, 0.30),
                HIGH_VOL_CHOP: random.uniform(0.25, 0.40),
                LOW_VOL_GRIND: random.uniform(0.10, 0.18),
                CRASH_CRISIS: random.uniform(0.40, 0.70),
            }
            iv_mean = iv_map.get(regime, 0.20)

            chain = self._generate_synthetic_chain(underlying, dte=45, iv_mean=iv_mean)
            strategy_map = REGIME_STRATEGY_MAP[regime]

            # Adversarial training: intentionally use a wrong-regime strategy occasionally
            is_adversarial = random.random() < ADVERSARIAL_RATE
            if is_adversarial:
                strategy_name = strategy_map["bad"]
                adversarial_count += 1
            else:
                # Pick primary or alt with 80/20 split
                strategy_name = (
                    strategy_map["alt"]
                    if random.random() < 0.20
                    else strategy_map["primary"]
                )

            analysis = self._build_analysis_from_chain(strategy_name, underlying, chain, iv_mean)

            signal = TradeSignal(
                action="open",
                strategy=strategy_name,
                symbol="SPY",
                analysis=analysis,
                reason=f"Synthetic {regime} {'[adversarial] ' if is_adversarial else ''}opportunity",
                order_spec={"type": "NET_CREDIT", "price": analysis.credit},
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
            review_id: Optional[str] = None
            try:
                advisor_result = self.advisor.review_trade(
                    signal=signal,
                    context={"regime": regime, "chain_data": chain},
                )
                review_id = getattr(advisor_result, "review_id", None)
                approved = advisor_result.verdict == "approve"
                if not approved:
                    skipped += 1
                    adv_tag = " [adv]" if is_adversarial else ""
                    print(
                        f"  {i:>4}/{iterations}  {regime:<14}  "
                        f"{strategy_name:<22}  vetoed{adv_tag}"
                    )
                    continue
            except Exception as e:
                skipped += 1
                logger.debug("Advisor error at iteration %d: %s", i, e)
                print(
                    f"  {i:>4}/{iterations}  {regime:<14}  "
                    f"{strategy_name:<22}  advisor error"
                )
                time.sleep(3)
                continue

            # CIO final decision
            try:
                prompt_json = json.dumps(
                    {
                        "proposed_trade": trade,
                        "advisor_feedback": {
                            "verdict": advisor_result.verdict,
                            "confidence": advisor_result.confidence,
                            "reasoning": advisor_result.reasoning,
                        },
                        "market_context": {
                            "regime": regime,
                            "iv_environment": f"{iv_mean:.0%}",
                            "adversarial_scenario": is_adversarial,
                        },
                        "symbol": "SPY",
                        "current_price": underlying,
                    }
                )
                cio_result = self.cio.run(prompt=prompt_json)
                go_no_go = str(
                    cio_result.final_payload.get("verdict", "reject")
                ).lower()
                if go_no_go not in {"approve", "reduce_size"}:
                    skipped += 1
                    adv_tag = " [adv]" if is_adversarial else ""
                    print(
                        f"  {i:>4}/{iterations}  {regime:<14}  "
                        f"{strategy_name:<22}  rejected{adv_tag}"
                    )
                    continue
            except Exception as e:
                skipped += 1
                logger.debug("CIO error at iteration %d: %s", i, e)
                print(
                    f"  {i:>4}/{iterations}  {regime:<14}  "
                    f"{strategy_name:<22}  cio error"
                )
                time.sleep(3)
                continue

            # Resolve outcome via GBM path
            mu, sigma, swan_base_prob = REGIME_GBM.get(regime, (0.0, 0.20, 0.10))
            # Adversarial scenarios get extra volatility to surface losses
            if is_adversarial:
                sigma = sigma * 1.5
                swan_base_prob = min(0.50, swan_base_prob * 2.0)

            is_black_swan = random.random() < swan_base_prob
            final_pnl = self._simulate_pnl(
                strategy_name, analysis, underlying, mu, sigma, is_black_swan
            )

            total_pnl += final_pnl
            won = final_pnl >= 0
            if won:
                wins += 1
            else:
                losses += 1

            # Update track record with outcome (closes the feedback loop)
            if review_id:
                self._update_track_record_outcome(
                    review_id,
                    "win" if won else "loss",
                    final_pnl,
                )

            # Feed RL optimizer
            position_id = f"sim_{uuid.uuid4().hex[:8]}"
            try:
                self.optimizer.process_closed_trade(
                    position_id=position_id,
                    pnl=final_pnl,
                    trade_context={
                        "symbol": "SPY",
                        "strategy": strategy_name,
                        "regime": regime,
                        "adversarial": is_adversarial,
                        "confidence": advisor_result.confidence,
                    },
                )
            except Exception as e:
                logger.debug("RL optimizer error at iteration %d: %s", i, e)

            # Track per-regime stats
            key = f"{regime[:8]}/{strategy_name[:10]}"
            rs = regime_stats.setdefault(key, {"w": 0, "l": 0, "pnl": 0.0})
            rs["w" if won else "l"] += 1
            rs["pnl"] += final_pnl

            pnl_str = f"+${final_pnl:.0f}" if final_pnl >= 0 else f"-${abs(final_pnl):.0f}"
            adv_tag = " [adv]" if is_adversarial else ""
            swan_note = " · swan" if is_black_swan else ""
            print(
                f"  {i:>4}/{iterations}  {regime:<14}  "
                f"{strategy_name:<22}  {pnl_str:>8}{adv_tag}{swan_note}"
            )

            time.sleep(1)  # reduced from 2s for throughput

        # ── Summary ──────────────────────────────────────────────────────────
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        pnl_str = f"+${total_pnl:.0f}" if total_pnl >= 0 else f"-${abs(total_pnl):.0f}"
        rules_after = len(self._load_learned_rules())
        new_rules = rules_after - rules_before

        print(f"\n  {'─' * 52}")
        print(f"  Trades       {total_trades:>6}  (adversarial: {adversarial_count})")
        print(f"  Win rate     {win_rate:>5.0f}%")
        print(f"  Total P/L    {pnl_str:>8}")
        if skipped:
            print(f"  Skipped      {skipped:>6}")

        if regime_stats:
            print(f"\n  {'Regime/Strategy':<26}   W   L   Avg P/L")
            print(f"  {'─' * 50}")
            for key, rs in sorted(regime_stats.items()):
                t = rs["w"] + rs["l"]
                avg = rs["pnl"] / max(1, t)
                avg_str = f"+${avg:.0f}" if avg >= 0 else f"-${abs(avg):.0f}"
                print(f"  {key:<26}  {rs['w']:>3} {rs['l']:>3}  {avg_str:>8}")

        if new_rules > 0:
            print(f"\n  RL rules learned: +{new_rules}  (total: {rules_after})")
        else:
            print(f"\n  RL rules: {rules_after} total (need more failure patterns to trigger)")

        print(f"\n  Training complete. LLM memory updated.\n")
