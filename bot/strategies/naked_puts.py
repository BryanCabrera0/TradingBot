"""Naked (cash-secured) put strategy."""

from __future__ import annotations

import logging
from typing import Optional

from bot.analysis import SpreadAnalysis, find_option_by_delta
from bot.iv_history import IVHistory
from bot.number_utils import safe_int
from bot.strategies.base import BaseStrategy, TradeSignal
from bot.technicals import TechnicalContext

logger = logging.getLogger(__name__)


class NakedPutStrategy(BaseStrategy):
    """Sell cash-secured puts on high-conviction symbols in elevated IV."""

    def __init__(self, config: dict):
        super().__init__("naked_puts", config)
        self.iv_history = IVHistory()

    def scan_for_entries(
        self,
        symbol: str,
        chain_data: dict,
        underlying_price: float,
        technical_context: Optional[TechnicalContext] = None,
        market_context: Optional[dict] = None,
    ) -> list[TradeSignal]:
        if underlying_price <= 0:
            return []

        min_dte = int(self.config.get("min_dte", 25))
        max_dte = int(self.config.get("max_dte", 45))
        target_delta = float(self.config.get("short_delta", 0.22))
        puts = chain_data.get("puts", {})
        if not puts:
            return []

        iv_rank = float((market_context or {}).get("iv_rank", 0.0) or 0.0)
        if iv_rank <= 0:
            iv = self._average_put_iv(puts)
            iv_rank = self.iv_history.update_and_rank(symbol, iv) if iv > 0 else 0.0

        # Required gate: only sell naked puts when IV rank is elevated.
        if iv_rank <= 50:
            return []

        allowed_tickers = [
            str(item).upper()
            for item in self.config.get("tickers", [])
            if str(item).strip()
        ]
        if allowed_tickers and symbol.upper() not in allowed_tickers:
            return []

        signals: list[TradeSignal] = []
        for expiration, exp_puts in puts.items():
            if not exp_puts:
                continue
            dte = int(exp_puts[0].get("dte", 0) or 0)
            if dte < min_dte or dte > max_dte:
                continue

            short_put = find_option_by_delta(exp_puts, target_delta)
            if not short_put:
                continue

            strike = float(short_put.get("strike", 0.0) or 0.0)
            credit = float(short_put.get("mid", 0.0) or 0.0)
            if strike <= 0 or credit <= 0:
                continue

            pop = max(
                0.0, min(1.0, 1.0 - abs(float(short_put.get("delta", 0.0) or 0.0)))
            )
            max_loss = max(0.0, strike - credit)
            rr = (credit / max_loss) if max_loss > 0 else 0.0

            analysis = SpreadAnalysis(
                symbol=symbol,
                strategy="naked_put",
                expiration=expiration,
                dte=dte,
                short_strike=strike,
                long_strike=0.0,
                credit=credit,
                max_loss=round(max_loss, 2),
                max_profit=round(credit, 2),
                risk_reward_ratio=round(rr, 4),
                credit_pct_of_width=round(credit / max(strike, 1.0), 4),
                probability_of_profit=round(pop, 4),
                net_delta=round(-float(short_put.get("delta", 0.0) or 0.0), 4),
                net_theta=round(-float(short_put.get("theta", 0.0) or 0.0), 4),
                net_gamma=round(-float(short_put.get("gamma", 0.0) or 0.0), 4),
                net_vega=round(-float(short_put.get("vega", 0.0) or 0.0), 4),
                score=self._score_setup(short_put, pop, rr),
            )

            if self.meets_minimum_quality(analysis):
                signals.append(
                    TradeSignal(
                        action="open",
                        strategy="naked_put",
                        symbol=symbol,
                        analysis=analysis,
                        metadata={"iv_rank": iv_rank},
                    )
                )

        signals.sort(
            key=lambda item: item.analysis.score if item.analysis else 0.0, reverse=True
        )
        return signals

    def check_exits(self, positions: list, market_client) -> list[TradeSignal]:
        signals: list[TradeSignal] = []
        profit_target_pct = float(self.config.get("profit_target_pct", 0.50))
        exit_dte = int(self.config.get("exit_dte", 21))

        for position in positions:
            if str(position.get("status", "open")).lower() != "open":
                continue
            if str(position.get("strategy", "")).lower() != "naked_put":
                continue

            entry_credit = float(position.get("entry_credit", 0.0) or 0.0)
            current_value = float(position.get("current_value", 0.0) or 0.0)
            dte_remaining = safe_int(position.get("dte_remaining"), 999)
            quantity = max(1, int(position.get("quantity", 1)))

            pnl_pct = (
                ((entry_credit - current_value) / entry_credit)
                if entry_credit > 0
                else 0.0
            )
            should_close = pnl_pct >= profit_target_pct or dte_remaining <= exit_dte
            if not should_close:
                continue

            reason = (
                f"Profit target reached ({pnl_pct:.1%})"
                if pnl_pct >= profit_target_pct
                else f"Front DTE reached ({dte_remaining})"
            )
            signals.append(
                TradeSignal(
                    action="close",
                    strategy="naked_put",
                    symbol=str(position.get("symbol", "")),
                    position_id=position.get("position_id"),
                    reason=reason,
                    quantity=quantity,
                )
            )

        return signals

    def meets_minimum_quality(self, analysis: SpreadAnalysis) -> bool:
        """Naked puts require tighter POP and minimum quality thresholds."""
        if analysis.credit <= 0:
            return False
        if analysis.probability_of_profit < 0.60:
            return False
        if analysis.score < 40:
            return False
        return True

    @staticmethod
    def _average_put_iv(puts: dict) -> float:
        values: list[float] = []
        for exp_puts in puts.values():
            for option in exp_puts:
                iv = float(option.get("iv", 0.0) or 0.0)
                if iv > 0:
                    values.append(iv)
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _score_setup(option: dict, pop: float, risk_reward: float) -> float:
        score = 0.0
        score += min(pop, 1.0) * 45.0
        score += min(max(risk_reward, 0.0) / 0.20, 1.0) * 25.0

        oi = float(option.get("open_interest", 0.0) or 0.0)
        volume = float(option.get("volume", 0.0) or 0.0)
        liquidity = min(oi / 500.0, 1.0) * 0.6 + min(volume / 100.0, 1.0) * 0.4
        score += liquidity * 20.0

        spread = float(option.get("ask", 0.0) or 0.0) - float(
            option.get("bid", 0.0) or 0.0
        )
        tightness = max(0.0, 1.0 - spread / 0.25)
        score += tightness * 10.0
        return round(min(score, 100.0), 1)
