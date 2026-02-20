import unittest
from datetime import date, timedelta

from bot.config import RiskConfig
from bot.risk_manager import RiskManager


class RiskManagerTests(unittest.TestCase):
    def test_update_portfolio_keeps_supplied_daily_pnl_on_new_day(self) -> None:
        manager = RiskManager(RiskConfig())
        manager.portfolio.daily_pnl_date = date.today() - timedelta(days=1)

        manager.update_portfolio(
            account_balance=10_000,
            open_positions=[],
            daily_pnl=-150.0,
        )

        self.assertEqual(manager.portfolio.daily_pnl, -150.0)


if __name__ == "__main__":
    unittest.main()
