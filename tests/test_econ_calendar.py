import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bot.econ_calendar import EconomicCalendar
from bot.strategies.base import TradeSignal


class EconCalendarTests(unittest.TestCase):
    def test_policy_for_trade_returns_skip_for_high_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            calendar = EconomicCalendar(
                cache_path=Path(tmp_dir) / "econ.json",
                static_events=[{"name": "FOMC", "event_date": "2026-03-18", "severity": "high", "impact": "macro"}],
                policy={"high": "skip", "medium": "widen", "low": "none"},
            )
            decision = calendar.policy_for_trade(expiration="2026-03-20", as_of=None)
            self.assertEqual(decision["action"], "skip")
            self.assertTrue(decision["events"])

    def test_adjust_signal_reduce_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            calendar = EconomicCalendar(
                cache_path=Path(tmp_dir) / "econ.json",
                static_events=[{"name": "CPI", "event_date": "2026-03-12", "severity": "high", "impact": "macro"}],
                policy={"high": "reduce_size", "medium": "widen", "low": "none"},
            )
            analysis = SimpleNamespace(expiration="2026-03-20", short_strike=100.0, long_strike=95.0)
            signal = TradeSignal(action="open", strategy="bull_put_spread", symbol="SPY", analysis=analysis, size_multiplier=1.0)

            allowed, reason = calendar.adjust_signal(signal)

            self.assertTrue(allowed)
            self.assertIn("Reduced size", reason)
            self.assertLess(signal.size_multiplier, 1.0)

    def test_adjust_signal_skip_blocks_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            calendar = EconomicCalendar(
                cache_path=Path(tmp_dir) / "econ.json",
                static_events=[{"name": "NFP", "event_date": "2026-03-06", "severity": "high", "impact": "macro"}],
                policy={"high": "skip", "medium": "widen", "low": "none"},
            )
            analysis = SimpleNamespace(expiration="2026-03-20")
            signal = TradeSignal(action="open", strategy="iron_condor", symbol="QQQ", analysis=analysis)

            allowed, reason = calendar.adjust_signal(signal)

            self.assertFalse(allowed)
            self.assertIn("Macro event", reason)

    def test_context_includes_upcoming_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            calendar = EconomicCalendar(cache_path=Path(tmp_dir) / "econ.json")
            context = calendar.context(days=30)
            self.assertIn("upcoming_macro_events", context)
            self.assertIn("horizon_days", context)


if __name__ == "__main__":
    unittest.main()

