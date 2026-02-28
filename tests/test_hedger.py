import unittest
from unittest import mock

from bot.config import BotConfig
from bot.hedger import HedgeAction, PortfolioHedger
from bot.orchestrator import TradingBot


class HedgerTests(unittest.TestCase):
    def test_disabled_hedger_returns_no_actions(self) -> None:
        hedger = PortfolioHedger({"enabled": False})
        actions = hedger.propose(
            account_value=100000, net_delta=80, sector_exposure={}, regime="normal"
        )
        self.assertEqual(actions, [])

    def test_delta_hedge_triggers(self) -> None:
        hedger = PortfolioHedger(
            {"enabled": True, "delta_hedge_trigger": 50, "max_hedge_cost_pct": 2.0}
        )
        actions = hedger.propose(
            account_value=100000, net_delta=90, sector_exposure={}, regime="normal"
        )
        self.assertTrue(any(action.symbol == "SPY" for action in actions))

    def test_tail_risk_hedge_when_low_vol(self) -> None:
        hedger = PortfolioHedger(
            {"enabled": True, "tail_risk_enabled": True, "max_hedge_cost_pct": 2.0}
        )
        actions = hedger.propose(
            account_value=100000,
            net_delta=0,
            sector_exposure={},
            regime="LOW_VOL_GRIND",
        )
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
            {
                "enabled": True,
                "tail_risk_enabled": True,
                "delta_hedge_trigger": 10,
                "max_hedge_cost_pct": 0.05,
            }
        )
        actions = hedger.propose(
            account_value=100000,
            net_delta=120,
            sector_exposure={"Information Technology": 0.5},
            regime="LOW_VOL_GRIND",
        )
        total_cost = sum(action.estimated_cost for action in actions)
        self.assertLessEqual(total_cost, 50.0)

    def test_hedger_auto_execution(self) -> None:
        cfg = BotConfig()
        cfg.trading_mode = "paper"
        cfg.scanner.enabled = False
        cfg.news.enabled = False
        cfg.llm.enabled = False
        cfg.hedging.enabled = True
        cfg.hedging.auto_execute = True
        cfg.hedging.max_hedge_cost_pct = 2.0
        bot = TradingBot(cfg)
        bot.risk_manager.update_portfolio(
            account_balance=100_000, open_positions=[], daily_pnl=0.0
        )
        bot.hedger = mock.Mock()
        bot.hedger.propose.return_value = [
            HedgeAction(
                symbol="SPY",
                instrument="options",
                direction="buy_put",
                quantity=1,
                estimated_cost=80.0,
                reason="Delta hedge",
            )
        ]
        bot._monthly_hedge_cost = mock.Mock(return_value=0.0)
        bot._execute_hedge_action = mock.Mock(return_value=True)
        bot._record_hedge_action = mock.Mock()
        bot._alert = mock.Mock()

        bot._apply_hedging_layer()

        bot._execute_hedge_action.assert_called_once()
        bot._record_hedge_action.assert_called_once()
        bot._alert.assert_called_once()

    def test_execute_hedge_action_opens_hedge_position(self) -> None:
        cfg = BotConfig()
        cfg.trading_mode = "paper"
        cfg.scanner.enabled = False
        cfg.news.enabled = False
        cfg.llm.enabled = False
        bot = TradingBot(cfg)
        bot.risk_manager.update_portfolio(
            account_balance=100_000, open_positions=[], daily_pnl=0.0
        )
        bot._select_hedge_option = mock.Mock(
            return_value={
                "expiration": "2026-04-17",
                "dte": 35,
                "contract_type": "P",
                "strike": 460.0,
                "mid": 0.8,
                "delta": -0.20,
                "theta": -0.02,
                "gamma": 0.01,
                "vega": 0.03,
            }
        )
        bot.risk_manager.approve_trade = mock.Mock(return_value=(True, "ok"))
        bot._review_entry_with_llm = mock.Mock(return_value=True)

        ok = bot._execute_hedge_action(
            HedgeAction(
                symbol="SPY",
                instrument="options",
                direction="buy_put",
                quantity=1,
                estimated_cost=90.0,
                reason="delta",
            )
        )

        self.assertTrue(ok)
        self.assertTrue(
            any(pos.get("strategy") == "hedge" for pos in bot.paper_trader.positions)
        )


if __name__ == "__main__":
    unittest.main()
