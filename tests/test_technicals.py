import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bot.technicals import TechnicalAnalyzer, build_technical_context


def _bars(count: int = 80) -> list[dict]:
    out = []
    price = 100.0
    for idx in range(count):
        price += 0.5
        out.append(
            {
                "datetime": idx,
                "open": price - 0.2,
                "high": price + 0.6,
                "low": price - 0.7,
                "close": price,
                "volume": 1_000_000 + idx * 1_000,
            }
        )
    return out


class TechnicalsTests(unittest.TestCase):
    def test_build_technical_context_calculates_key_fields(self) -> None:
        context = build_technical_context("SPY", _bars())

        self.assertIsNotNone(context)
        self.assertGreater(context.sma20, 0)
        self.assertGreater(context.sma50, 0)
        self.assertGreaterEqual(context.bollinger_position, 0.0)
        self.assertLessEqual(context.bollinger_position, 1.0)

    def test_technical_analyzer_uses_daily_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = TechnicalAnalyzer(Path(tmp_dir) / "cache.json")
            schwab = mock.Mock()
            schwab.get_price_history.return_value = _bars()

            first = analyzer.get_context("SPY", schwab)
            second = analyzer.get_context("SPY", schwab)

            self.assertIsNotNone(first)
            self.assertEqual(first.as_of, second.as_of)
            schwab.get_price_history.assert_called_once()


if __name__ == "__main__":
    unittest.main()
