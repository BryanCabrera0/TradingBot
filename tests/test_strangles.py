import unittest

from bot.strategies.strangles import StranglesStrategy


def _chain() -> dict:
    return {
        "calls": {
            "2026-03-20": [
                {
                    "dte": 28,
                    "strike": 110.0,
                    "bid": 1.0,
                    "ask": 1.2,
                    "mid": 1.1,
                    "delta": 0.16,
                    "gamma": 0.01,
                    "theta": -0.03,
                    "vega": 0.04,
                    "iv": 35.0,
                    "volume": 800,
                    "open_interest": 1000,
                },
                {
                    "dte": 28,
                    "strike": 100.0,
                    "bid": 3.4,
                    "ask": 3.8,
                    "mid": 3.6,
                    "delta": 0.50,
                    "gamma": 0.02,
                    "theta": -0.06,
                    "vega": 0.08,
                    "iv": 34.0,
                    "volume": 900,
                    "open_interest": 1200,
                },
            ]
        },
        "puts": {
            "2026-03-20": [
                {
                    "dte": 28,
                    "strike": 90.0,
                    "bid": 0.9,
                    "ask": 1.1,
                    "mid": 1.0,
                    "delta": -0.16,
                    "gamma": 0.01,
                    "theta": -0.03,
                    "vega": 0.04,
                    "iv": 36.0,
                    "volume": 850,
                    "open_interest": 1100,
                },
                {
                    "dte": 28,
                    "strike": 100.0,
                    "bid": 3.6,
                    "ask": 4.0,
                    "mid": 3.8,
                    "delta": -0.50,
                    "gamma": 0.02,
                    "theta": -0.06,
                    "vega": 0.08,
                    "iv": 35.0,
                    "volume": 900,
                    "open_interest": 1200,
                },
            ]
        },
    }


class StranglesStrategyTests(unittest.TestCase):
    def test_iv_rank_gate_blocks_entries(self) -> None:
        strategy = StranglesStrategy(
            {"min_iv_rank": 70, "short_delta": 0.16, "min_dte": 20, "max_dte": 45}
        )
        signals = strategy.scan_for_entries(
            "SPY",
            _chain(),
            100.0,
            market_context={
                "iv_rank": 45,
                "regime": "HIGH_VOL_CHOP",
                "account_balance": 100000,
            },
        )
        self.assertEqual(signals, [])

    def test_regime_gate_blocks_crash(self) -> None:
        strategy = StranglesStrategy(
            {"min_iv_rank": 70, "short_delta": 0.16, "min_dte": 20, "max_dte": 45}
        )
        signals = strategy.scan_for_entries(
            "SPY",
            _chain(),
            100.0,
            market_context={
                "iv_rank": 80,
                "regime": "CRASH/CRISIS",
                "account_balance": 100000,
            },
        )
        self.assertEqual(signals, [])

    def test_generates_strangle_signal(self) -> None:
        strategy = StranglesStrategy(
            {
                "min_iv_rank": 70,
                "short_delta": 0.16,
                "min_dte": 20,
                "max_dte": 45,
                "allow_straddles_on": ["SPY"],
            }
        )
        signals = strategy.scan_for_entries(
            "SPY",
            _chain(),
            100.0,
            market_context={
                "iv_rank": 80,
                "regime": "HIGH_VOL_CHOP",
                "account_balance": 100000,
            },
        )
        self.assertTrue(any(signal.strategy == "short_strangle" for signal in signals))

    def test_account_balance_gate(self) -> None:
        strategy = StranglesStrategy(
            {
                "min_iv_rank": 70,
                "short_delta": 0.16,
                "min_dte": 20,
                "max_dte": 45,
                "min_account_balance": 50000,
            }
        )
        signals = strategy.scan_for_entries(
            "SPY",
            _chain(),
            100.0,
            market_context={
                "iv_rank": 80,
                "regime": "HIGH_VOL_CHOP",
                "account_balance": 10000,
            },
        )
        self.assertEqual(signals, [])

    def test_exit_profit_target(self) -> None:
        strategy = StranglesStrategy(
            {"profit_target_pct": 0.5, "stop_loss_multiple": 2.0}
        )
        positions = [
            {
                "position_id": "s1",
                "strategy": "short_strangle",
                "symbol": "SPY",
                "status": "open",
                "entry_credit": 2.0,
                "current_value": 1.0,
                "quantity": 1,
            }
        ]
        signals = strategy.check_exits(positions, market_client=None)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, "close")


if __name__ == "__main__":
    unittest.main()
