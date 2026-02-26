import unittest

from bot.monte_carlo import MonteCarloRiskEngine


class MonteCarloTests(unittest.TestCase):
    def test_simulation_returns_expected_shape(self) -> None:
        engine = MonteCarloRiskEngine(simulations=500, var_limit_pct=3.0, random_seed=42)
        positions = [
            {
                "symbol": "SPY",
                "quantity": 2,
                "underlying_price": 500.0,
                "details": {
                    "net_delta": 0.12,
                    "net_gamma": 0.01,
                    "net_theta": 0.02,
                    "net_vega": -0.05,
                    "entry_iv": 0.22,
                    "vol_of_vol": 0.15,
                },
            }
        ]

        result = engine.simulate(positions, account_balance=100000.0)
        payload = result.to_dict()

        self.assertEqual(payload["simulations"], 500)
        self.assertIn("1d", payload["horizons"])
        self.assertIn("5d", payload["horizons"])
        self.assertIn("21d", payload["horizons"])
        self.assertIn("var99", payload["horizons"]["1d"])

    def test_var_limit_flags_high_risk(self) -> None:
        engine = MonteCarloRiskEngine(simulations=1000, var_limit_pct=0.1, random_seed=7)
        positions = [
            {
                "symbol": "QQQ",
                "quantity": 5,
                "underlying_price": 450.0,
                "details": {
                    "net_delta": 0.50,
                    "net_gamma": 0.03,
                    "net_theta": -0.05,
                    "net_vega": 0.25,
                    "entry_iv": 0.40,
                    "vol_of_vol": 0.40,
                },
            }
        ]

        result = engine.simulate(positions, account_balance=50000.0)

        self.assertTrue(result.high_risk)
        self.assertGreater(result.var99_pct_account, 0.1)

    def test_empty_positions_return_zero_risk(self) -> None:
        engine = MonteCarloRiskEngine(simulations=250, random_seed=1)

        result = engine.simulate([], account_balance=100000.0)

        self.assertFalse(result.high_risk)
        self.assertEqual(result.var99_pct_account, 0.0)
        self.assertEqual(result.horizons["1d"]["var99"], 0.0)


if __name__ == "__main__":
    unittest.main()
