import unittest

from bot.analysis import (
    analyze_credit_spread,
    analyze_iron_condor,
    compute_probability_of_profit_condor,
)


class AnalysisTests(unittest.TestCase):
    def test_condor_pop_is_positive_for_reasonable_range(self) -> None:
        pop = compute_probability_of_profit_condor(
            underlying_price=100,
            put_short_strike=90,
            call_short_strike=110,
            credit=2,
            dte=30,
            iv=20,
        )
        self.assertGreater(pop, 0.1)
        self.assertLess(pop, 1.0)

    def test_credit_spread_net_greeks_use_position_signs(self) -> None:
        short_put = {
            "symbol": "SPY",
            "strike": 95,
            "mid": 1.60,
            "bid": 1.50,
            "ask": 1.70,
            "delta": -0.30,
            "theta": -0.12,
            "vega": 0.09,
            "iv": 25.0,
            "dte": 30,
            "open_interest": 1200,
            "volume": 450,
            "expiration": "2026-03-20",
        }
        long_put = {
            "symbol": "SPY",
            "strike": 90,
            "mid": 0.90,
            "bid": 0.85,
            "ask": 0.95,
            "delta": -0.18,
            "theta": -0.07,
            "vega": 0.06,
            "iv": 24.0,
            "dte": 30,
            "open_interest": 900,
            "volume": 300,
            "expiration": "2026-03-20",
        }

        analysis = analyze_credit_spread(
            underlying_price=100,
            short_option=short_put,
            long_option=long_put,
            contract_type="PUT",
        )

        self.assertAlmostEqual(analysis.net_delta, 0.12, places=4)
        self.assertAlmostEqual(analysis.net_theta, 0.05, places=4)
        self.assertAlmostEqual(analysis.net_vega, -0.03, places=4)

    def test_iron_condor_net_greeks_use_position_signs(self) -> None:
        put_short = {
            "strike": 95,
            "mid": 1.4,
            "delta": -0.22,
            "theta": -0.10,
            "vega": 0.08,
            "iv": 25.0,
            "dte": 30,
            "expiration": "2026-03-20",
        }
        put_long = {
            "strike": 90,
            "mid": 0.8,
            "delta": -0.14,
            "theta": -0.06,
            "vega": 0.06,
            "iv": 24.0,
            "dte": 30,
            "expiration": "2026-03-20",
        }
        call_short = {
            "strike": 105,
            "mid": 1.3,
            "delta": 0.24,
            "theta": -0.09,
            "vega": 0.08,
            "iv": 26.0,
            "dte": 30,
            "expiration": "2026-03-20",
        }
        call_long = {
            "strike": 110,
            "mid": 0.7,
            "delta": 0.16,
            "theta": -0.05,
            "vega": 0.05,
            "iv": 24.0,
            "dte": 30,
            "expiration": "2026-03-20",
        }

        analysis = analyze_iron_condor(
            underlying_price=100,
            put_short=put_short,
            put_long=put_long,
            call_short=call_short,
            call_long=call_long,
        )

        self.assertAlmostEqual(analysis.net_delta, -0.0, places=4)
        self.assertAlmostEqual(analysis.net_theta, 0.08, places=4)
        self.assertAlmostEqual(analysis.net_vega, -0.05, places=4)


if __name__ == "__main__":
    unittest.main()
