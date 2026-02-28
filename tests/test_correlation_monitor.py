import unittest

import numpy as np

from bot.correlation_monitor import (
    CRISIS,
    NORMAL,
    STRESSED,
    CrossAssetCorrelationMonitor,
)


def _rows(values: np.ndarray) -> list[dict]:
    return [{"close": float(v)} for v in values]


def _prices_from_returns(start: float, returns: np.ndarray) -> np.ndarray:
    prices = [float(start)]
    for value in returns:
        prices.append(prices[-1] * (1.0 + float(value)))
    return np.array(prices, dtype=float)


class CorrelationMonitorTests(unittest.TestCase):
    def test_normal_regime_with_healthy_inverse_spy_vix(self) -> None:
        t = np.linspace(0.0, 6.0, 39)
        spy_returns = 0.001 + (0.003 * np.sin(t)) + (0.001 * np.cos(2 * t))
        base = _prices_from_returns(100.0, spy_returns)
        qqq = _prices_from_returns(120.0, (spy_returns * 0.6) + (0.002 * np.sin(3 * t)))
        iwm = _prices_from_returns(90.0, (spy_returns * 0.4) + (0.004 * np.sin(4 * t)))
        vix = _prices_from_returns(
            20.0, (-spy_returns * 0.9) + (0.0002 * np.cos(5 * t))
        )
        hyg = _prices_from_returns(80.0, (spy_returns * 0.7) + 0.0005)
        tlt = _prices_from_returns(130.0, (-spy_returns * 0.4) + 0.0002)
        gld = _prices_from_returns(170.0, (0.0003 * np.sin(2 * t)))

        by_symbol = {
            "SPY": _rows(base),
            "QQQ": _rows(qqq),
            "IWM": _rows(iwm),
            "^VIX": _rows(vix),
            "HYG": _rows(hyg),
            "TLT": _rows(tlt),
            "GLD": _rows(gld),
        }
        monitor = CrossAssetCorrelationMonitor(
            get_price_history=lambda symbol, days: by_symbol.get(symbol, []),
            lookback_days=20,
            crisis_threshold=0.95,
            stress_threshold=0.85,
        )

        state = monitor.get_correlation_state()

        self.assertEqual(state["correlation_regime"], NORMAL)
        self.assertLess(state["correlations"]["SPY_VIX"], 0.0)
        self.assertFalse(state["flags"]["spy_vix_positive"])

    def test_crisis_when_spy_vix_flips_positive(self) -> None:
        t = np.linspace(0.0, 6.0, 39)
        spy_returns = 0.001 + (0.003 * np.sin(t)) + (0.001 * np.cos(2 * t))
        base = _prices_from_returns(100.0, spy_returns)
        qqq = _prices_from_returns(
            120.0, (spy_returns * 1.0) + (0.0001 * np.sin(3 * t))
        )
        iwm = _prices_from_returns(
            95.0, (spy_returns * 0.95) + (0.0001 * np.cos(4 * t))
        )
        vix = _prices_from_returns(
            18.0, (spy_returns * 0.85) + (0.0002 * np.sin(5 * t))
        )
        hyg = _prices_from_returns(85.0, (spy_returns * 0.2))
        tlt = _prices_from_returns(125.0, (-spy_returns * 0.2))
        gld = _prices_from_returns(175.0, 0.0002 * np.cos(2 * t))

        by_symbol = {
            "SPY": _rows(base),
            "QQQ": _rows(qqq),
            "IWM": _rows(iwm),
            "^VIX": _rows(vix),
            "HYG": _rows(hyg),
            "TLT": _rows(tlt),
            "GLD": _rows(gld),
        }
        monitor = CrossAssetCorrelationMonitor(
            get_price_history=lambda symbol, days: by_symbol.get(symbol, []),
            lookback_days=20,
            crisis_threshold=0.95,
            stress_threshold=0.85,
        )

        state = monitor.get_correlation_state()

        self.assertEqual(state["correlation_regime"], CRISIS)
        self.assertTrue(state["flags"]["spy_vix_positive"])

    def test_stressed_when_equity_correlations_spike(self) -> None:
        t = np.linspace(0.0, 6.0, 39)
        spy_returns = 0.001 + (0.003 * np.sin(t)) + (0.001 * np.cos(2 * t))
        base = _prices_from_returns(100.0, spy_returns)
        qqq = _prices_from_returns(
            120.0, (spy_returns * 0.8) + (0.0004 * np.sin(3 * t))
        )
        iwm = _prices_from_returns(
            95.0, (spy_returns * 0.75) + (0.0004 * np.cos(2 * t))
        )
        vix = _prices_from_returns(
            20.0, (-spy_returns * 0.4) + (0.0003 * np.sin(5 * t))
        )
        hyg = _prices_from_returns(84.0, spy_returns * 0.6)
        tlt = _prices_from_returns(126.0, -spy_returns * 0.3)
        gld = _prices_from_returns(176.0, 0.0002 * np.sin(2 * t))

        by_symbol = {
            "SPY": _rows(base),
            "QQQ": _rows(qqq),
            "IWM": _rows(iwm),
            "^VIX": _rows(vix),
            "HYG": _rows(hyg),
            "TLT": _rows(tlt),
            "GLD": _rows(gld),
        }
        monitor = CrossAssetCorrelationMonitor(
            get_price_history=lambda symbol, days: by_symbol.get(symbol, []),
            lookback_days=20,
            crisis_threshold=0.995,
            stress_threshold=0.85,
        )

        state = monitor.get_correlation_state()

        self.assertEqual(state["correlation_regime"], STRESSED)
        self.assertTrue(state["flags"]["equity_corr_spike"])


if __name__ == "__main__":
    unittest.main()
