import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bot.strategies.earnings_vol_crush import EarningsVolCrushStrategy


def _chain() -> dict:
    return {
        "calls": {
            "2026-03-05": [
                {
                    "dte": 2,
                    "strike": 105.0,
                    "bid": 1.0,
                    "ask": 1.2,
                    "mid": 1.1,
                    "delta": 0.12,
                    "gamma": 0.01,
                    "theta": -0.04,
                    "vega": 0.06,
                    "iv": 58.0,
                    "volume": 500,
                    "open_interest": 800,
                },
                {
                    "dte": 2,
                    "strike": 115.0,
                    "bid": 0.2,
                    "ask": 0.3,
                    "mid": 0.25,
                    "delta": 0.03,
                    "gamma": 0.005,
                    "theta": -0.01,
                    "vega": 0.03,
                    "iv": 55.0,
                    "volume": 300,
                    "open_interest": 500,
                },
            ]
        },
        "puts": {
            "2026-03-05": [
                {
                    "dte": 2,
                    "strike": 95.0,
                    "bid": 1.0,
                    "ask": 1.2,
                    "mid": 1.1,
                    "delta": -0.12,
                    "gamma": 0.01,
                    "theta": -0.04,
                    "vega": 0.06,
                    "iv": 59.0,
                    "volume": 500,
                    "open_interest": 800,
                },
                {
                    "dte": 2,
                    "strike": 85.0,
                    "bid": 0.2,
                    "ask": 0.3,
                    "mid": 0.25,
                    "delta": -0.03,
                    "gamma": 0.005,
                    "theta": -0.01,
                    "vega": 0.03,
                    "iv": 56.0,
                    "volume": 300,
                    "open_interest": 500,
                },
            ]
        },
    }


class EarningsVolCrushStrategyTests(unittest.TestCase):
    def test_requires_iv_rank(self) -> None:
        strategy = EarningsVolCrushStrategy({"min_iv_rank": 75.0})
        strategy.earnings_calendar = mock.Mock(
            earnings_within_window=mock.Mock(return_value=(True, "2026-03-04"))
        )
        signals = strategy.scan_for_entries(
            "TSLA",
            _chain(),
            100.0,
            market_context={"iv_rank": 50.0, "max_expiration": "2026-03-05"},
        )
        self.assertEqual(signals, [])

    def test_requires_earnings_window(self) -> None:
        strategy = EarningsVolCrushStrategy({"min_iv_rank": 70.0})
        strategy.earnings_calendar = mock.Mock(
            earnings_within_window=mock.Mock(return_value=(False, None))
        )
        signals = strategy.scan_for_entries(
            "TSLA",
            _chain(),
            100.0,
            market_context={"iv_rank": 80.0, "max_expiration": "2026-03-05"},
        )
        self.assertEqual(signals, [])

    def test_generates_signal_when_filters_pass(self) -> None:
        strategy = EarningsVolCrushStrategy({"min_iv_rank": 70.0, "wing_width": 10.0})
        strategy.earnings_calendar = mock.Mock(
            earnings_within_window=mock.Mock(return_value=(True, "2026-03-04"))
        )
        signals = strategy.scan_for_entries(
            "TSLA",
            _chain(),
            100.0,
            market_context={"iv_rank": 80.0, "max_expiration": "2026-03-05"},
        )
        self.assertTrue(signals)
        self.assertEqual(signals[0].strategy, "earnings_vol_crush")

    def test_exit_on_profit(self) -> None:
        strategy = EarningsVolCrushStrategy({})
        positions = [
            {
                "position_id": "e1",
                "strategy": "earnings_vol_crush",
                "symbol": "TSLA",
                "status": "open",
                "entry_credit": 1.0,
                "current_value": 0.6,
                "quantity": 1,
                "dte_remaining": 1,
            }
        ]
        signals = strategy.check_exits(positions, market_client=None)
        self.assertEqual(len(signals), 1)

    def test_exit_on_event_complete(self) -> None:
        strategy = EarningsVolCrushStrategy({})
        positions = [
            {
                "position_id": "e2",
                "strategy": "earnings_vol_crush",
                "symbol": "TSLA",
                "status": "open",
                "entry_credit": 1.0,
                "current_value": 0.95,
                "quantity": 1,
                "dte_remaining": 0,
            }
        ]
        signals = strategy.check_exits(positions, market_client=None)
        self.assertEqual(len(signals), 1)

    def test_skips_symbol_when_historical_moves_exceed_implied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            moves = Path(tmp_dir) / "moves.json"
            moves.write_text(
                '{"TSLA":{"avg_earnings_move_pct":9.5,"implied_move_pct":8.5,"times_exceeded_implied":7,"total_events":10}}',
                encoding="utf-8",
            )
            strategy = EarningsVolCrushStrategy(
                {
                    "min_iv_rank": 70.0,
                    "wing_width": 10.0,
                    "earnings_moves_file": str(moves),
                }
            )
            strategy.earnings_calendar = mock.Mock(
                earnings_within_window=mock.Mock(return_value=(True, "2026-03-04"))
            )

            signals = strategy.scan_for_entries(
                "TSLA",
                _chain(),
                100.0,
                market_context={"iv_rank": 80.0, "max_expiration": "2026-03-05"},
            )

            self.assertEqual(signals, [])

    def test_boosts_score_when_historical_move_is_below_implied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            moves = Path(tmp_dir) / "moves.json"
            moves.write_text(
                '{"TSLA":{"avg_earnings_move_pct":4.0,"implied_move_pct":10.0,"times_exceeded_implied":2,"total_events":10}}',
                encoding="utf-8",
            )

            boosted = EarningsVolCrushStrategy(
                {
                    "min_iv_rank": 70.0,
                    "wing_width": 10.0,
                    "earnings_moves_file": str(moves),
                }
            )
            boosted.earnings_calendar = mock.Mock(
                earnings_within_window=mock.Mock(return_value=(True, "2026-03-04"))
            )
            baseline = EarningsVolCrushStrategy(
                {
                    "min_iv_rank": 70.0,
                    "wing_width": 10.0,
                    "earnings_moves_file": str(Path(tmp_dir) / "empty.json"),
                }
            )
            baseline.earnings_calendar = mock.Mock(
                earnings_within_window=mock.Mock(return_value=(True, "2026-03-04"))
            )

            boosted_signal = boosted.scan_for_entries(
                "TSLA",
                _chain(),
                100.0,
                market_context={"iv_rank": 80.0, "max_expiration": "2026-03-05"},
            )[0]
            baseline_signal = baseline.scan_for_entries(
                "TSLA",
                _chain(),
                100.0,
                market_context={"iv_rank": 80.0, "max_expiration": "2026-03-05"},
            )[0]

            self.assertGreater(
                boosted_signal.analysis.score, baseline_signal.analysis.score
            )


if __name__ == "__main__":
    unittest.main()
