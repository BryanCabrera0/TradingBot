import unittest

from bot.hedger import PortfolioHedger


class HedgerTests(unittest.TestCase):
    def test_disabled_hedger_returns_no_actions(self) -> None:
        hedger = PortfolioHedger({"enabled": False})
        actions = hedger.propose(account_value=100000, net_delta=80, sector_exposure={}, regime="normal")
        self.assertEqual(actions, [])

    def test_delta_hedge_triggers(self) -> None:
        hedger = PortfolioHedger({"enabled": True, "delta_hedge_trigger": 50, "max_hedge_cost_pct": 2.0})
        actions = hedger.propose(account_value=100000, net_delta=90, sector_exposure={}, regime="normal")
        self.assertTrue(any(action.symbol == "SPY" for action in actions))

    def test_tail_risk_hedge_when_low_vol(self) -> None:
        hedger = PortfolioHedger({"enabled": True, "tail_risk_enabled": True, "max_hedge_cost_pct": 2.0})
        actions = hedger.propose(account_value=100000, net_delta=0, sector_exposure={}, regime="LOW_VOL_GRIND")
        self.assertTrue(any(action.symbol == "VIX" for action in actions))

    def test_sector_concentration_hedge(self) -> None:
        hedger = PortfolioHedger({"enabled": True, "max_hedge_cost_pct": 2.0})
        actions = hedger.propose(
            account_value=100000,
            net_delta=0,
            sector_exposure={"Information Technology": 0.45},
            regime="normal",
        )
        self.assertTrue(any(action.symbol == "XLK" for action in actions))

    def test_max_hedge_cost_cap_applied(self) -> None:
        hedger = PortfolioHedger(
            {"enabled": True, "tail_risk_enabled": True, "delta_hedge_trigger": 10, "max_hedge_cost_pct": 0.05}
        )
        actions = hedger.propose(
            account_value=100000,
            net_delta=120,
            sector_exposure={"Information Technology": 0.5},
            regime="LOW_VOL_GRIND",
        )
        total_cost = sum(action.estimated_cost for action in actions)
        self.assertLessEqual(total_cost, 50.0)


if __name__ == "__main__":
    unittest.main()

