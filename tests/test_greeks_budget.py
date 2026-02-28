import unittest
from unittest import mock

from bot.analysis import SpreadAnalysis
from bot.config import RiskConfig
from bot.risk_manager import RiskManager
from bot.strategies.base import TradeSignal


def _signal(*, net_delta: float, net_vega: float, quantity: int = 1) -> TradeSignal:
    return TradeSignal(
        action="open",
        strategy="bull_put_spread",
        symbol="SPY",
        quantity=quantity,
        analysis=SpreadAnalysis(
            symbol="SPY",
            strategy="bull_put_spread",
            expiration="2026-04-17",
            dte=32,
            short_strike=540.0,
            long_strike=535.0,
            credit=1.2,
            max_loss=3.8,
            probability_of_profit=0.62,
            score=58.0,
            net_delta=net_delta,
            net_vega=net_vega,
        ),
        metadata={"regime": "BULL_TREND"},
    )


class GreeksBudgetTests(unittest.TestCase):
    def _manager(self) -> RiskManager:
        manager = RiskManager(
            RiskConfig(
                max_sector_risk_pct=100.0,
                max_portfolio_delta_abs=1000.0,
                max_portfolio_vega_pct_of_account=100.0,
            )
        )
        manager.earnings_calendar = mock.Mock(
            earnings_within_window=mock.Mock(return_value=(False, None))
        )
        manager.update_portfolio(
            account_balance=100_000, open_positions=[], daily_pnl=0.0
        )
        return manager

    def test_regime_budget_rejects_when_breached(self) -> None:
        manager = self._manager()
        manager.portfolio.net_delta = 75.0
        manager.set_greeks_budget_config({"enabled": True, "reduce_size_to_fit": False})

        ok, qty, reason = manager.evaluate_greeks_budget(
            _signal(net_delta=0.08, net_vega=-0.002, quantity=1),
            regime="BULL_TREND",
            quantity=1,
        )

        self.assertFalse(ok)
        self.assertEqual(qty, 0)
        self.assertIn("Greeks budget breach", reason)

    def test_regime_budget_reduces_size_to_fit(self) -> None:
        manager = self._manager()
        manager.portfolio.net_delta = 20.0
        manager.set_greeks_budget_config({"enabled": True, "reduce_size_to_fit": True})

        ok, qty, reason = manager.evaluate_greeks_budget(
            _signal(net_delta=0.08, net_vega=-0.6, quantity=3),
            regime="HIGH_VOL_CHOP",
            quantity=3,
        )

        self.assertTrue(ok)
        self.assertEqual(qty, 1)
        self.assertIn("resized", reason.lower())

    def test_approve_trade_rejects_when_budget_still_breached(self) -> None:
        manager = self._manager()
        manager.portfolio.net_delta = 75.0
        manager.set_greeks_budget_config({"enabled": True, "reduce_size_to_fit": True})
        signal = _signal(net_delta=0.08, net_vega=-0.002, quantity=1)

        approved, reason = manager.approve_trade(signal)

        self.assertFalse(approved)
        self.assertIn("Greeks budget", reason)

    def test_crisis_regime_alias_uses_crash_budget(self) -> None:
        manager = self._manager()
        manager.portfolio.net_delta = 9.0
        signal = _signal(net_delta=0.02, net_vega=0.002, quantity=1)
        ok, qty, reason = manager.evaluate_greeks_budget(
            signal,
            regime="crisis",
            quantity=1,
            allow_resize=False,
        )

        self.assertFalse(ok)
        self.assertEqual(qty, 0)
        self.assertIn("CRASH/CRISIS", reason)


if __name__ == "__main__":
    unittest.main()
