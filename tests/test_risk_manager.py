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

    def test_register_open_position_updates_capacity_and_risk(self) -> None:
        manager = RiskManager(RiskConfig(max_open_positions=2))
        manager.update_portfolio(
            account_balance=25_000,
            open_positions=[],
            daily_pnl=0.0,
        )

        self.assertTrue(manager.can_open_more_positions())

        manager.register_open_position(
            symbol="SPY",
            max_loss_per_contract=3.5,
            quantity=2,
        )

        self.assertTrue(manager.can_open_more_positions())
        self.assertEqual(len(manager.portfolio.open_positions), 1)
        self.assertEqual(manager.portfolio.total_risk_deployed, 700.0)

        manager.register_open_position(
            symbol="QQQ",
            max_loss_per_contract=2.0,
            quantity=1,
        )

        self.assertFalse(manager.can_open_more_positions())


if __name__ == "__main__":
    unittest.main()
