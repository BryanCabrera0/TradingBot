"""Credit spread strategy — bull put spreads and bear call spreads."""

import logging
from typing import Optional

from bot.analysis import analyze_credit_spread, find_option_by_delta, find_spread_wing
from bot.dividend_calendar import DividendCalendar
from bot.iv_history import IVHistory
from bot.strategies.base import BaseStrategy, TradeSignal
from bot.technicals import TechnicalContext

logger = logging.getLogger(__name__)


class CreditSpreadStrategy(BaseStrategy):
    """Automated credit spread trading strategy with adaptive IV/technical logic."""

    def __init__(self, config: dict):
        super().__init__("credit_spreads", config)
        self.iv_history = IVHistory()
        self.dividend_calendar = DividendCalendar()

    def scan_for_entries(
        self,
        symbol: str,
        chain_data: dict,
        underlying_price: float,
        technical_context: Optional[TechnicalContext] = None,
        market_context: Optional[dict] = None,
    ) -> list[TradeSignal]:
        """Scan for credit spread opportunities on a symbol."""
        signals: list[TradeSignal] = []
        direction = self.config.get("direction", "both")
        base_min_dte = int(self.config.get("min_dte", 20))
        base_max_dte = int(self.config.get("max_dte", 45))
        target_delta = float(self.config.get("short_delta", 0.30))
        spread_width = float(self.config.get("spread_width", 5))
        min_credit_pct = float(self.config.get("min_credit_pct", 0.30))

        calls = chain_data.get("calls", {})
        puts = chain_data.get("puts", {})
        avg_iv = self._average_chain_iv(calls, puts)
        iv_rank = self.iv_history.update_and_rank(symbol, avg_iv) if avg_iv > 0 else 50.0
        put_call_skew = self._put_call_skew_pct(calls, puts, underlying_price)

        min_dte, max_dte = base_min_dte, base_max_dte
        size_multiplier = 1.0

        if iv_rank > 70:
            spread_width += 1.0
            min_credit_pct *= 1.20
            min_dte, max_dte = 20, 30
        elif iv_rank < 30:
            spread_width = max(1.0, spread_width - 1.0)
            min_dte, max_dte = 35, 50
            size_multiplier = 0.70

        skew_bias = "neutral"
        if put_call_skew > 10:
            skew_bias = "bull_put"
        elif put_call_skew < -10:
            skew_bias = "bear_call"

        mean_reversion_bias = self._mean_reversion_bias(technical_context)

        for exp_date in puts:
            if direction in ("both", "bull_put"):
                put_signals = self._scan_put_spreads(
                    symbol=symbol,
                    underlying_price=underlying_price,
                    puts=puts.get(exp_date, []),
                    exp_date=exp_date,
                    min_dte=min_dte,
                    max_dte=max_dte,
                    target_delta=target_delta,
                    spread_width=spread_width,
                    min_credit_pct=min_credit_pct,
                    iv_rank=iv_rank,
                    skew_bias=skew_bias,
                    skew_pct=put_call_skew,
                    mean_reversion_bias=mean_reversion_bias,
                    size_multiplier=size_multiplier,
                    technical_context=technical_context,
                )
                signals.extend(put_signals)

        for exp_date in calls:
            if direction in ("both", "bear_call"):
                call_signals = self._scan_call_spreads(
                    symbol=symbol,
                    underlying_price=underlying_price,
                    calls=calls.get(exp_date, []),
                    exp_date=exp_date,
                    min_dte=min_dte,
                    max_dte=max_dte,
                    target_delta=target_delta,
                    spread_width=spread_width,
                    min_credit_pct=min_credit_pct,
                    iv_rank=iv_rank,
                    skew_bias=skew_bias,
                    skew_pct=put_call_skew,
                    mean_reversion_bias=mean_reversion_bias,
                    size_multiplier=size_multiplier,
                    technical_context=technical_context,
                )
                signals.extend(call_signals)

        signals.sort(key=lambda s: s.analysis.score if s.analysis else 0, reverse=True)
        return signals

    def _scan_put_spreads(
        self,
        *,
        symbol: str,
        underlying_price: float,
        puts: list,
        exp_date: str,
        min_dte: int,
        max_dte: int,
        target_delta: float,
        spread_width: float,
        min_credit_pct: float,
        iv_rank: float,
        skew_bias: str,
        skew_pct: float,
        mean_reversion_bias: str,
        size_multiplier: float,
        technical_context: Optional[TechnicalContext],
    ) -> list[TradeSignal]:
        """Find bull put spread opportunities."""
        signals: list[TradeSignal] = []
        if not puts:
            return signals

        dte = int(puts[0].get("dte", 0) if puts else 0)
        if dte < min_dte or dte > max_dte:
            return signals

        # Avoid put selling into extreme oversold conditions.
        if technical_context and technical_context.rsi14 < 25:
            return signals

        short_put = find_option_by_delta(puts, target_delta)
        if not short_put:
            return signals
        long_put = find_spread_wing(puts, short_put["strike"], spread_width, "lower")
        if not long_put:
            return signals

        analysis = analyze_credit_spread(underlying_price, short_put, long_put, "PUT")
        analysis.symbol = symbol

        if analysis.credit_pct_of_width < min_credit_pct:
            return signals
        if skew_bias == "bull_put":
            analysis.score = min(100.0, analysis.score + 10.0)
        elif skew_bias == "bear_call":
            analysis.score = max(0.0, analysis.score - 10.0)
        if mean_reversion_bias == "bull_put":
            analysis.score = min(100.0, analysis.score + 15.0)

        div_risk = self.dividend_calendar.assess_trade_risk(
            symbol=symbol,
            strategy="bull_put_spread",
            expiration=exp_date,
            short_strike=float(short_put["strike"]),
            underlying_price=float(underlying_price),
            is_call_side=False,
        )
        analysis.score = round(
            max(0.0, min(100.0, analysis.score + float(div_risk.get("score_adjustment", 0.0)))),
            1,
        )

        if self.meets_minimum_quality(analysis):
            signals.append(
                TradeSignal(
                    action="open",
                    strategy="bull_put_spread",
                    symbol=symbol,
                    analysis=analysis,
                    size_multiplier=size_multiplier,
                    metadata={
                        "iv_rank": iv_rank,
                        "put_call_skew_pct": skew_pct,
                        "dividend_warning": div_risk.get("warning"),
                    },
                )
            )
        return signals

    def _scan_call_spreads(
        self,
        *,
        symbol: str,
        underlying_price: float,
        calls: list,
        exp_date: str,
        min_dte: int,
        max_dte: int,
        target_delta: float,
        spread_width: float,
        min_credit_pct: float,
        iv_rank: float,
        skew_bias: str,
        skew_pct: float,
        mean_reversion_bias: str,
        size_multiplier: float,
        technical_context: Optional[TechnicalContext],
    ) -> list[TradeSignal]:
        """Find bear call spread opportunities."""
        signals: list[TradeSignal] = []
        if not calls:
            return signals

        dte = int(calls[0].get("dte", 0) if calls else 0)
        if dte < min_dte or dte > max_dte:
            return signals

        # Avoid call selling into extreme overbought conditions.
        if technical_context and technical_context.rsi14 > 75:
            return signals

        short_call = find_option_by_delta(calls, target_delta)
        if not short_call:
            return signals
        long_call = find_spread_wing(calls, short_call["strike"], spread_width, "higher")
        if not long_call:
            return signals

        analysis = analyze_credit_spread(underlying_price, short_call, long_call, "CALL")
        analysis.symbol = symbol

        if analysis.credit_pct_of_width < min_credit_pct:
            return signals
        if skew_bias == "bear_call":
            analysis.score = min(100.0, analysis.score + 10.0)
        elif skew_bias == "bull_put":
            analysis.score = max(0.0, analysis.score - 10.0)
        if mean_reversion_bias == "bear_call":
            analysis.score = min(100.0, analysis.score + 15.0)

        div_risk = self.dividend_calendar.assess_trade_risk(
            symbol=symbol,
            strategy="bear_call_spread",
            expiration=exp_date,
            short_strike=float(short_call["strike"]),
            underlying_price=float(underlying_price),
            is_call_side=True,
        )
        analysis.score = round(
            max(0.0, min(100.0, analysis.score + float(div_risk.get("score_adjustment", 0.0)))),
            1,
        )

        if self.meets_minimum_quality(analysis):
            signals.append(
                TradeSignal(
                    action="open",
                    strategy="bear_call_spread",
                    symbol=symbol,
                    analysis=analysis,
                    size_multiplier=size_multiplier,
                    metadata={
                        "iv_rank": iv_rank,
                        "put_call_skew_pct": skew_pct,
                        "dividend_warning": div_risk.get("warning"),
                    },
                )
            )
        return signals

    def check_exits(self, positions: list, market_client) -> list[TradeSignal]:
        """Check open credit spreads for adaptive exit/defense/roll conditions."""
        signals = []
        base_stop_loss_pct = float(self.config.get("stop_loss_pct", 2.0))

        for pos in positions:
            if str(pos.get("status", "open")).lower() != "open":
                continue
            if pos.get("strategy") not in ("bull_put_spread", "bear_call_spread"):
                continue

            entry_credit = float(pos.get("entry_credit", 0) or 0)
            current_value = float(pos.get("current_value", 0) or 0)
            quantity = max(1, int(pos.get("quantity", 1)))
            dte_remaining = int(pos.get("dte_remaining", 999) or 999)
            details = pos.get("details", {}) or {}
            underlying_price = float(pos.get("underlying_price", 0.0) or 0.0)

            pnl = entry_credit - current_value
            pnl_pct = (pnl / entry_credit) if entry_credit > 0 else 0.0
            stop_loss_pct = base_stop_loss_pct

            if self._is_short_strike_tested(pos, underlying_price, details):
                stop_loss_pct = min(stop_loss_pct, 1.5)

            # Offer roll around 21 DTE once some profit is captured.
            if 20 <= dte_remaining <= 22 and pnl_pct >= 0.25:
                signals.append(
                    TradeSignal(
                        action="roll",
                        strategy=pos["strategy"],
                        symbol=pos.get("symbol", ""),
                        position_id=pos.get("position_id"),
                        reason=f"Roll candidate at {dte_remaining} DTE ({pnl_pct:.1%} captured)",
                        quantity=quantity,
                        metadata={"roll_to_next_month": True, "same_delta": True},
                    )
                )
                continue

            # Partial-profit scaling before full target.
            if quantity >= 2 and not pos.get("partial_closed", False) and pnl_pct >= 0.40:
                signals.append(
                    TradeSignal(
                        action="close",
                        strategy=pos["strategy"],
                        symbol=pos.get("symbol", ""),
                        position_id=pos.get("position_id"),
                        reason=f"Partial profit at {pnl_pct:.1%}",
                        quantity=max(1, quantity // 2),
                    )
                )
                continue

            profit_target_pct = self._profit_target_for_dte(dte_remaining)
            if pos.get("partial_closed", False):
                profit_target_pct = max(profit_target_pct, 0.65)

            exit_reason = ""
            if entry_credit > 0:
                if pnl_pct >= profit_target_pct:
                    exit_reason = f"Profit target reached ({pnl_pct:.1%})"
                elif pnl < 0 and abs(pnl) >= entry_credit * stop_loss_pct:
                    exit_reason = f"Stop loss triggered (loss {abs(pnl):.2f})"

            if not exit_reason and dte_remaining <= 5:
                exit_reason = f"Approaching expiration ({dte_remaining} DTE)"

            if exit_reason:
                signals.append(
                    TradeSignal(
                        action="close",
                        strategy=pos["strategy"],
                        symbol=pos.get("symbol", ""),
                        position_id=pos.get("position_id"),
                        reason=exit_reason,
                        quantity=quantity,
                    )
                )

        return signals

    @staticmethod
    def _profit_target_for_dte(dte_remaining: int) -> float:
        if dte_remaining <= 14:
            return 0.25
        if dte_remaining <= 30:
            return 0.50
        return 0.65

    @staticmethod
    def _is_short_strike_tested(position: dict, underlying_price: float, details: dict) -> bool:
        if underlying_price <= 0:
            return False
        strategy = str(position.get("strategy", ""))
        short_strike = float(details.get("short_strike", 0.0) or 0.0)
        if short_strike <= 0:
            return False
        proximity = abs(underlying_price - short_strike) / underlying_price

        if strategy == "bull_put_spread":
            return underlying_price <= short_strike and proximity <= 0.01
        if strategy == "bear_call_spread":
            return underlying_price >= short_strike and proximity <= 0.01
        return False

    @staticmethod
    def _average_chain_iv(calls: dict, puts: dict) -> float:
        ivs: list[float] = []
        for exp_options in calls.values():
            for option in exp_options:
                iv = float(option.get("iv", 0.0) or 0.0)
                if iv > 0:
                    ivs.append(iv)
        for exp_options in puts.values():
            for option in exp_options:
                iv = float(option.get("iv", 0.0) or 0.0)
                if iv > 0:
                    ivs.append(iv)
        if not ivs:
            return 0.0
        return float(sum(ivs) / len(ivs))

    @staticmethod
    def _put_call_skew_pct(calls: dict, puts: dict, underlying_price: float) -> float:
        if underlying_price <= 0:
            return 0.0
        put_ivs: list[float] = []
        call_ivs: list[float] = []

        for exp_options in puts.values():
            for option in exp_options:
                strike = float(option.get("strike", 0.0) or 0.0)
                if strike < underlying_price:
                    iv = float(option.get("iv", 0.0) or 0.0)
                    if iv > 0:
                        put_ivs.append(iv)
        for exp_options in calls.values():
            for option in exp_options:
                strike = float(option.get("strike", 0.0) or 0.0)
                if strike > underlying_price:
                    iv = float(option.get("iv", 0.0) or 0.0)
                    if iv > 0:
                        call_ivs.append(iv)

        if not put_ivs or not call_ivs:
            return 0.0
        put_avg = sum(put_ivs) / len(put_ivs)
        call_avg = sum(call_ivs) / len(call_ivs)
        return round(((put_avg - call_avg) / max(call_avg, 1e-9)) * 100.0, 2)

    @staticmethod
    def _mean_reversion_bias(context: Optional[TechnicalContext]) -> str:
        if not context:
            return "neutral"
        if context.return_5d_zscore <= -2.0:
            return "bull_put"
        if context.return_5d_zscore >= 2.0:
            return "bear_call"
        return "neutral"
