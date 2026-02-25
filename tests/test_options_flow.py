import unittest

from bot.options_flow import OptionsFlowAnalyzer


def _chain(call_vol: int = 200, put_vol: int = 400, call_oi: int = 300, put_oi: int = 500) -> dict:
    return {
        "calls": {
            "2026-03-20": [
                {"volume": call_vol, "open_interest": call_oi, "iv": 24.0, "delta": 0.25, "strike": 105.0},
                {"volume": call_vol // 2, "open_interest": call_oi, "iv": 23.0, "delta": 0.15, "strike": 110.0},
            ]
        },
        "puts": {
            "2026-03-20": [
                {"volume": put_vol, "open_interest": put_oi, "iv": 27.0, "delta": -0.25, "strike": 95.0},
                {"volume": put_vol // 2, "open_interest": put_oi, "iv": 26.0, "delta": -0.15, "strike": 90.0},
            ]
        },
    }


class OptionsFlowTests(unittest.TestCase):
    def test_directional_bias_from_put_call_ratio(self) -> None:
        analyzer = OptionsFlowAnalyzer(unusual_volume_multiple=4.0)
        context = analyzer.analyze(symbol="SPY", chain_data=_chain(150, 800))
        self.assertEqual(context.directional_bias, "bearish")
        self.assertGreater(context.put_call_volume_ratio, 1.0)

    def test_unusual_activity_flag_when_volume_spikes(self) -> None:
        analyzer = OptionsFlowAnalyzer(unusual_volume_multiple=2.0)
        context = analyzer.analyze(symbol="QQQ", chain_data=_chain(100, 1200))
        self.assertTrue(context.unusual_activity_flag)

    def test_open_interest_change_uses_previous_snapshot(self) -> None:
        analyzer = OptionsFlowAnalyzer(unusual_volume_multiple=5.0)
        previous = _chain(call_vol=100, put_vol=100, call_oi=200, put_oi=200)
        current = _chain(call_vol=100, put_vol=100, call_oi=500, put_oi=500)
        context = analyzer.analyze(symbol="IWM", chain_data=current, previous_chain_data=previous)
        self.assertGreater(context.open_interest_change, 0.0)

    def test_to_dict_contains_required_fields(self) -> None:
        analyzer = OptionsFlowAnalyzer()
        context = analyzer.analyze(symbol="AAPL", chain_data=_chain())
        payload = context.to_dict()
        self.assertIn("directional_bias", payload)
        self.assertIn("institutional_flow_direction", payload)
        self.assertIn("metrics", payload)


if __name__ == "__main__":
    unittest.main()

