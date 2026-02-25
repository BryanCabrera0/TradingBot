import unittest

from bot.analysis import SpreadAnalysis
from bot.config import RiskConfig
from bot.risk_manager import RiskManager
from bot.strategies.base import TradeSignal
from bot.strategies.naked_puts import NakedPutStrategy


def _chain_data() -> dict:
    return {
        "underlying_price": 100.0,
        "calls": {},
        "puts": {
            "2026-03-20": [
                {
                    "strike": 95.0,
                    "mid": 1.2,
                    "bid": 1.1,
                    "ask": 1.3,
                    "delta": -0.22,
                    "gamma": 0.01,
                    "theta": -0.03,
                    "vega": 0.05,
                    "iv": 35.0,
                    "volume": 120,
                    "open_interest": 600,
                    "dte": 30,
                }
            ]
        },
    }


class NakedPutStrategyTests(unittest.TestCase):
    def test_entry_signal_generation(self) -> None:
        strategy = NakedPutStrategy({"min_dte": 25, "max_dte": 45, "short_delta": 0.22})

        signals = strategy.scan_for_entries(
            "AAPL",
            _chain_data(),
            100.0,
            market_context={"iv_rank": 65.0},
        )

        self.assertEqual(len(signals), 1)
        signal = signals[0]
        self.assertEqual(signal.strategy, "naked_put")
        self.assertAlmostEqual(signal.analysis.max_loss, 93.8, places=2)

    def test_iv_rank_gate_rejection(self) -> None:
        strategy = NakedPutStrategy({"min_dte": 25, "max_dte": 45, "short_delta": 0.22})

        signals = strategy.scan_for_entries(
            "AAPL",
            _chain_data(),
            100.0,
            market_context={"iv_rank": 45.0},
        )

        self.assertEqual(signals, [])

    def test_exit_on_profit_target(self) -> None:
        strategy = NakedPutStrategy({"profit_target_pct": 0.50, "exit_dte": 21})
        positions = [
            {
                "position_id": "np1",
                "strategy": "naked_put",
                "symbol": "AAPL",
                "entry_credit": 1.0,
                "current_value": 0.4,
                "dte_remaining": 35,
                "status": "open",
                "quantity": 1,
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, "close")
        self.assertIn("Profit target", signals[0].reason)

    def test_exit_on_dte(self) -> None:
        strategy = NakedPutStrategy({"profit_target_pct": 0.50, "exit_dte": 21})
        positions = [
            {
                "position_id": "np2",
                "strategy": "naked_put",
                "symbol": "AAPL",
                "entry_credit": 1.0,
                "current_value": 0.95,
                "dte_remaining": 20,
                "status": "open",
                "quantity": 1,
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, "close")
        self.assertIn("DTE", signals[0].reason)

    def test_risk_calculation_accuracy(self) -> None:
        manager = RiskManager(RiskConfig())
        signal = TradeSignal(
            action="open",
            strategy="naked_put",
            symbol="AAPL",
            quantity=1,
            analysis=SpreadAnalysis(
                symbol="AAPL",
                strategy="naked_put",
                expiration="2026-03-20",
                dte=30,
                short_strike=100.0,
                long_strike=0.0,
                credit=2.0,
                max_loss=0.0,
                probability_of_profit=0.7,
                score=60.0,
            ),
        )

        loss = manager.effective_max_loss_per_contract(signal)

        self.assertEqual(loss, 98.0)


if __name__ == "__main__":
    unittest.main()
