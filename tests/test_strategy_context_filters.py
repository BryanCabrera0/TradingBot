import unittest
from unittest import mock

from bot.analysis import SpreadAnalysis
from bot.strategies.broken_wing_butterfly import BrokenWingButterflyStrategy
from bot.strategies.calendar_spreads import CalendarSpreadStrategy
from bot.strategies.credit_spreads import CreditSpreadStrategy
from bot.strategies.iron_condors import IronCondorStrategy
from bot.strategies.base import TradeSignal
from bot.strategies.strangles import StranglesStrategy


def _spread_chain() -> dict:
    return {
        "calls": {
            "2026-04-17": [
                {"dte": 30, "strike": 105.0, "bid": 1.0, "ask": 1.2, "mid": 1.1, "delta": 0.30, "gamma": 0.01, "theta": -0.03, "vega": 0.04, "iv": 30.0, "volume": 300, "open_interest": 800},
                {"dte": 30, "strike": 110.0, "bid": 0.4, "ask": 0.6, "mid": 0.5, "delta": 0.12, "gamma": 0.01, "theta": -0.02, "vega": 0.03, "iv": 29.0, "volume": 220, "open_interest": 650},
            ]
        },
        "puts": {
            "2026-04-17": [
                {"dte": 30, "strike": 95.0, "bid": 1.1, "ask": 1.3, "mid": 1.2, "delta": -0.30, "gamma": 0.01, "theta": -0.03, "vega": 0.04, "iv": 31.0, "volume": 340, "open_interest": 900},
                {"dte": 30, "strike": 90.0, "bid": 0.3, "ask": 0.5, "mid": 0.4, "delta": -0.12, "gamma": 0.01, "theta": -0.02, "vega": 0.03, "iv": 30.0, "volume": 260, "open_interest": 700},
            ]
        },
    }


def _calendar_chain() -> dict:
    return {
        "calls": {
            "2026-03-20": [
                {"dte": 25, "strike": 100.0, "mid": 2.0, "delta": 0.50, "gamma": 0.02, "theta": -0.05, "vega": 0.12, "iv": 30.0},
            ],
            "2026-05-15": [
                {"dte": 55, "strike": 100.0, "mid": 2.8, "delta": 0.52, "gamma": 0.015, "theta": -0.03, "vega": 0.16, "iv": 25.0},
            ],
        },
        "puts": {},
    }


def _bwb_chain() -> dict:
    return {
        "calls": {
            "2026-04-17": [
                {"dte": 30, "strike": 100.0, "mid": 1.1, "delta": 0.45},
                {"dte": 30, "strike": 105.0, "mid": 1.5, "delta": 0.30},
                {"dte": 30, "strike": 115.0, "mid": 0.4, "delta": 0.10},
            ]
        },
        "puts": {
            "2026-04-17": [
                {"dte": 30, "strike": 95.0, "mid": 1.0, "delta": -0.30},
                {"dte": 30, "strike": 90.0, "mid": 0.4, "delta": -0.15},
                {"dte": 30, "strike": 85.0, "mid": 0.2, "delta": -0.08},
            ]
        },
    }


class StrategyContextFilterTests(unittest.TestCase):
    def test_credit_spread_negative_vol_risk_premium_reduces_size(self) -> None:
        strategy = CreditSpreadStrategy(
            {"direction": "bull_put", "min_dte": 20, "max_dte": 45, "short_delta": 0.30, "spread_width": 5, "min_credit_pct": 0.10}
        )
        strategy.iv_history = mock.Mock(update_and_rank=mock.Mock(return_value=50.0))
        signals = strategy.scan_for_entries(
            "ZZZZ",
            _spread_chain(),
            100.0,
            market_context={"vol_surface": {"realized_implied_spread": -1.5}},
        )

        self.assertTrue(signals)
        self.assertLessEqual(signals[0].size_multiplier, 0.61)

    def test_credit_spread_unusual_institutional_conflict_rejects_signal(self) -> None:
        strategy = CreditSpreadStrategy(
            {"direction": "bull_put", "min_dte": 20, "max_dte": 45, "short_delta": 0.30, "spread_width": 5, "min_credit_pct": 0.10}
        )
        strategy.iv_history = mock.Mock(update_and_rank=mock.Mock(return_value=50.0))
        signals = strategy.scan_for_entries(
            "ZZZZ",
            _spread_chain(),
            100.0,
            market_context={
                "options_flow": {
                    "directional_bias": "bearish",
                    "unusual_activity_flag": True,
                    "institutional_flow_direction": "bearish",
                }
            },
        )

        self.assertEqual(signals, [])

    def test_iron_condor_blocks_high_vol_of_vol(self) -> None:
        strategy = IronCondorStrategy({"min_dte": 20, "max_dte": 45, "short_delta": 0.16, "spread_width": 5})
        signals = strategy.scan_for_entries(
            "SPY",
            _spread_chain(),
            100.0,
            market_context={"vol_surface": {"vol_of_vol": 0.45}},
        )
        self.assertEqual(signals, [])

    def test_strangles_blocks_high_vol_of_vol(self) -> None:
        strategy = StranglesStrategy({"min_iv_rank": 70, "short_delta": 0.16, "min_dte": 20, "max_dte": 45})
        signals = strategy.scan_for_entries(
            "SPY",
            _spread_chain(),
            100.0,
            market_context={
                "iv_rank": 80,
                "regime": "HIGH_VOL_CHOP",
                "account_balance": 100_000,
                "vol_surface": {"vol_of_vol": 0.40},
            },
        )
        self.assertEqual(signals, [])

    def test_calendar_spread_requires_backwardation_in_vol_surface_context(self) -> None:
        strategy = CalendarSpreadStrategy({})
        signals = strategy.scan_for_entries(
            "AAPL",
            _calendar_chain(),
            100.0,
            market_context={"vol_surface": {"term_structure_regime": "contango"}},
        )
        self.assertEqual(signals, [])

    def test_broken_wing_flow_bias_can_override_regime_direction(self) -> None:
        strategy = BrokenWingButterflyStrategy({"min_dte": 20, "max_dte": 45, "short_delta": 0.30})
        signals = strategy.scan_for_entries(
            "SPY",
            _bwb_chain(),
            103.0,
            market_context={
                "regime": "BULL_TREND",
                "options_flow": {
                    "directional_bias": "bearish",
                    "unusual_activity_flag": True,
                    "institutional_flow_direction": "bearish",
                },
            },
        )

        self.assertTrue(signals)
        self.assertEqual(signals[0].metadata["position_details"]["direction"], "bearish")

    def test_credit_spreads_limits_signals_per_symbol(self) -> None:
        strategy = CreditSpreadStrategy({"direction": "bull_put", "min_credit_pct": 0.1})
        strategy.iv_history = mock.Mock(update_and_rank=mock.Mock(return_value=50.0))
        base = SpreadAnalysis(
            symbol="SPY",
            strategy="bull_put_spread",
            expiration="2026-03-20",
            dte=30,
            short_strike=95,
            long_strike=90,
            credit=1.2,
            max_loss=3.8,
            probability_of_profit=0.62,
            score=55.0,
        )
        signal_a = TradeSignal(action="open", strategy="bull_put_spread", symbol="SPY", analysis=base)
        signal_b = TradeSignal(
            action="open",
            strategy="bull_put_spread",
            symbol="SPY",
            analysis=SpreadAnalysis(**{**base.__dict__, "score": 65.0}),
        )
        signal_c = TradeSignal(
            action="open",
            strategy="bull_put_spread",
            symbol="SPY",
            analysis=SpreadAnalysis(**{**base.__dict__, "score": 75.0}),
        )
        strategy._scan_put_spreads = mock.Mock(return_value=[signal_a, signal_b, signal_c])
        strategy._scan_call_spreads = mock.Mock(return_value=[])
        chain = {
            "calls": {"2026-03-20": [{"dte": 30}]},
            "puts": {"2026-03-20": [{"dte": 30}]},
        }

        signals = strategy.scan_for_entries(
            "SPY",
            chain,
            100.0,
            market_context={"max_signals_per_symbol_per_strategy": 2},
        )

        self.assertEqual(len(signals), 2)
        self.assertGreaterEqual(signals[0].analysis.score, signals[1].analysis.score)

    def test_iron_condors_limits_signals_per_symbol(self) -> None:
        strategy = IronCondorStrategy(
            {"min_dte": 20, "max_dte": 45, "short_delta": 0.16, "spread_width": 5, "min_credit_pct": 0.10}
        )
        strategy.iv_history = mock.Mock(update_and_rank=mock.Mock(return_value=50.0))
        chain = {
            "calls": {
                "2026-03-20": [
                    {"dte": 25, "strike": 105.0, "mid": 1.2, "delta": 0.16, "theta": -0.03, "gamma": 0.01, "vega": 0.04, "iv": 30.0},
                    {"dte": 25, "strike": 110.0, "mid": 0.6, "delta": 0.08, "theta": -0.02, "gamma": 0.01, "vega": 0.03, "iv": 29.0},
                ],
                "2026-04-17": [
                    {"dte": 35, "strike": 105.0, "mid": 1.3, "delta": 0.16, "theta": -0.03, "gamma": 0.01, "vega": 0.04, "iv": 30.0},
                    {"dte": 35, "strike": 110.0, "mid": 0.6, "delta": 0.08, "theta": -0.02, "gamma": 0.01, "vega": 0.03, "iv": 29.0},
                ],
                "2026-05-15": [
                    {"dte": 40, "strike": 105.0, "mid": 1.1, "delta": 0.16, "theta": -0.03, "gamma": 0.01, "vega": 0.04, "iv": 30.0},
                    {"dte": 40, "strike": 110.0, "mid": 0.6, "delta": 0.08, "theta": -0.02, "gamma": 0.01, "vega": 0.03, "iv": 29.0},
                ],
            },
            "puts": {
                "2026-03-20": [
                    {"dte": 25, "strike": 95.0, "mid": 1.2, "delta": -0.16, "theta": -0.03, "gamma": 0.01, "vega": 0.04, "iv": 31.0},
                    {"dte": 25, "strike": 90.0, "mid": 0.6, "delta": -0.08, "theta": -0.02, "gamma": 0.01, "vega": 0.03, "iv": 30.0},
                ],
                "2026-04-17": [
                    {"dte": 35, "strike": 95.0, "mid": 1.3, "delta": -0.16, "theta": -0.03, "gamma": 0.01, "vega": 0.04, "iv": 31.0},
                    {"dte": 35, "strike": 90.0, "mid": 0.6, "delta": -0.08, "theta": -0.02, "gamma": 0.01, "vega": 0.03, "iv": 30.0},
                ],
                "2026-05-15": [
                    {"dte": 40, "strike": 95.0, "mid": 1.1, "delta": -0.16, "theta": -0.03, "gamma": 0.01, "vega": 0.04, "iv": 31.0},
                    {"dte": 40, "strike": 90.0, "mid": 0.6, "delta": -0.08, "theta": -0.02, "gamma": 0.01, "vega": 0.03, "iv": 30.0},
                ],
            },
        }

        signals = strategy.scan_for_entries(
            "SPY",
            chain,
            100.0,
            market_context={"max_signals_per_symbol_per_strategy": 2},
        )

        self.assertLessEqual(len(signals), 2)

    def test_strangles_limits_signals_per_symbol(self) -> None:
        strategy = StranglesStrategy({"min_iv_rank": 70, "short_delta": 0.16, "min_dte": 20, "max_dte": 45})
        chain = {
            "calls": {
                "2026-03-20": [
                    {"dte": 25, "strike": 105.0, "mid": 1.2, "delta": 0.16},
                    {"dte": 25, "strike": 100.0, "mid": 2.2, "delta": 0.50},
                ],
                "2026-04-17": [
                    {"dte": 35, "strike": 105.0, "mid": 1.2, "delta": 0.16},
                    {"dte": 35, "strike": 100.0, "mid": 2.2, "delta": 0.50},
                ],
                "2026-05-15": [
                    {"dte": 40, "strike": 105.0, "mid": 1.2, "delta": 0.16},
                    {"dte": 40, "strike": 100.0, "mid": 2.2, "delta": 0.50},
                ],
            },
            "puts": {
                "2026-03-20": [
                    {"dte": 25, "strike": 95.0, "mid": 1.3, "delta": -0.16},
                    {"dte": 25, "strike": 100.0, "mid": 2.3, "delta": -0.50},
                ],
                "2026-04-17": [
                    {"dte": 35, "strike": 95.0, "mid": 1.3, "delta": -0.16},
                    {"dte": 35, "strike": 100.0, "mid": 2.3, "delta": -0.50},
                ],
                "2026-05-15": [
                    {"dte": 40, "strike": 95.0, "mid": 1.3, "delta": -0.16},
                    {"dte": 40, "strike": 100.0, "mid": 2.3, "delta": -0.50},
                ],
            },
        }
        signals = strategy.scan_for_entries(
            "SPY",
            chain,
            100.0,
            market_context={
                "iv_rank": 80.0,
                "regime": "HIGH_VOL_CHOP",
                "account_balance": 100_000,
                "max_signals_per_symbol_per_strategy": 2,
            },
        )

        self.assertLessEqual(len(signals), 2)


if __name__ == "__main__":
    unittest.main()
