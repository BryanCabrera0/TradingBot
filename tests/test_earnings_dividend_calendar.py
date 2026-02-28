import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bot.dividend_calendar import DividendCalendar
from bot.earnings_calendar import EarningsCalendar


class CalendarTests(unittest.TestCase):
    def test_earnings_window_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            calendar = EarningsCalendar(Path(tmp_dir) / "earnings.json")
            response = mock.Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "quoteSummary": {
                    "result": [
                        {
                            "calendarEvents": {
                                "earnings": {"earningsDate": [{"fmt": "2026-03-10"}]}
                            }
                        }
                    ]
                }
            }

            with mock.patch(
                "bot.earnings_calendar.requests.get", return_value=response
            ):
                blocked, earnings_date = calendar.earnings_within_window(
                    "AAPL", "2026-03-20"
                )

            self.assertTrue(blocked)
            self.assertEqual(earnings_date, "2026-03-10")

    def test_dividend_itm_covered_call_penalty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            calendar = DividendCalendar(Path(tmp_dir) / "dividend.json")
            response = mock.Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "quoteSummary": {
                    "result": [
                        {"summaryDetail": {"exDividendDate": {"fmt": "2026-03-05"}}}
                    ]
                }
            }

            with mock.patch(
                "bot.dividend_calendar.requests.get", return_value=response
            ):
                risk = calendar.assess_trade_risk(
                    symbol="MSFT",
                    strategy="covered_call",
                    expiration="2026-03-20",
                    short_strike=100.0,
                    underlying_price=110.0,
                    is_call_side=True,
                )

            self.assertEqual(risk["score_adjustment"], -20.0)
            self.assertIn("early-assignment", risk["warning"])


if __name__ == "__main__":
    unittest.main()
