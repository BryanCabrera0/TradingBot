import unittest
from datetime import date, timedelta
from unittest import mock

from bot.analysis import SpreadAnalysis
from bot.config import RiskConfig
from bot.risk_manager import RiskManager
from bot.strategies.base import TradeSignal


class RiskManagerTests(unittest.TestCase):
    @staticmethod
    def _make_signal(
        *,
        symbol: str = "AAPL",
        net_delta: float = 5.0,
        net_vega: float = 0.2,
        max_loss: float = 4.0,
    ) -> TradeSignal:
        return TradeSignal(
            action="open",
            strategy="bull_put_spread",
            symbol=symbol,
            quantity=1,
            analysis=SpreadAnalysis(
                symbol=symbol,
                strategy="bull_put_spread",
                expiration="2026-03-20",
                dte=30,
                short_strike=95,
                long_strike=90,
                credit=1.0,
                max_loss=max_loss,
                probability_of_profit=0.65,
                score=60,
                net_delta=net_delta,
                net_vega=net_vega,
            ),
        )

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

    def test_covered_call_uses_notional_risk_proxy_when_max_loss_missing(self) -> None:
        manager = RiskManager(RiskConfig(covered_call_notional_risk_pct=20.0))
        signal = TradeSignal(
            action="open",
            strategy="covered_call",
            symbol="AAPL",
            quantity=1,
            analysis=SpreadAnalysis(
                symbol="AAPL",
                strategy="covered_call",
                expiration="2026-03-20",
                dte=30,
                short_strike=200,
                long_strike=0,
                credit=1.0,
                max_loss=0.0,
                probability_of_profit=0.7,
                score=55,
            ),
        )

        loss = manager.effective_max_loss_per_contract(signal)

        self.assertEqual(loss, 39.0)

    def test_approve_trade_blocks_earnings_in_window(self) -> None:
        manager = RiskManager(RiskConfig())
        manager.earnings_calendar = mock.Mock(
            earnings_within_window=mock.Mock(return_value=(True, "2026-03-10"))
        )
        manager.update_portfolio(account_balance=100_000, open_positions=[], daily_pnl=0.0)
        signal = self._make_signal(symbol="AAPL", net_delta=5.0, net_vega=0.5, max_loss=3.8)

        approved, reason = manager.approve_trade(signal)

        self.assertFalse(approved)
        self.assertIn("Earnings", reason)

    def test_approve_trade_blocks_excess_delta_direction(self) -> None:
        manager = RiskManager(RiskConfig(max_portfolio_delta_abs=50.0))
        manager.earnings_calendar = mock.Mock(
            earnings_within_window=mock.Mock(return_value=(False, None))
        )
        manager.update_portfolio(
            account_balance=100_000,
            open_positions=[{"symbol": "SPY", "quantity": 1, "max_loss": 1.0, "details": {"net_delta": 52.0}}],
            daily_pnl=0.0,
        )
        signal = self._make_signal(symbol="QQQ", net_delta=6.0, net_vega=0.1, max_loss=4.0)

        approved, reason = manager.approve_trade(signal)

        self.assertFalse(approved)
        self.assertIn("delta limit", reason.lower())

    def test_check9_earnings_blocks_trade(self) -> None:
        manager = RiskManager(RiskConfig())
        manager.earnings_calendar = mock.Mock(
            earnings_within_window=mock.Mock(return_value=(True, "2026-03-10"))
        )
        manager.update_portfolio(account_balance=100_000, open_positions=[], daily_pnl=0.0)

        approved, reason = manager.approve_trade(self._make_signal())

        self.assertFalse(approved)
        self.assertIn("Earnings", reason)

    def test_check10_delta_guard(self) -> None:
        manager = RiskManager(RiskConfig(max_portfolio_delta_abs=50.0))
        manager.earnings_calendar = mock.Mock(
            earnings_within_window=mock.Mock(return_value=(False, None))
        )
        manager.update_portfolio(
            account_balance=100_000,
            open_positions=[{"symbol": "SPY", "quantity": 1, "max_loss": 1.0, "details": {"net_delta": 48.0}}],
            daily_pnl=0.0,
        )

        approved, reason = manager.approve_trade(self._make_signal(symbol="QQQ", net_delta=5.0))

        self.assertFalse(approved)
        self.assertIn("delta limit", reason.lower())

    def test_check10_delta_opposite_direction_allowed(self) -> None:
        manager = RiskManager(
            RiskConfig(
                max_portfolio_delta_abs=50.0,
                max_sector_risk_pct=100.0,
            )
        )
        manager.earnings_calendar = mock.Mock(
            earnings_within_window=mock.Mock(return_value=(False, None))
        )
        manager.update_portfolio(
            account_balance=100_000,
            open_positions=[{"symbol": "SPY", "quantity": 1, "max_loss": 1.0, "details": {"net_delta": 48.0}}],
            daily_pnl=0.0,
        )

        approved, _ = manager.approve_trade(self._make_signal(symbol="QQQ", net_delta=-5.0))

        self.assertTrue(approved)

    def test_check11_vega_guard(self) -> None:
        manager = RiskManager(RiskConfig(max_portfolio_vega_pct_of_account=0.5))
        manager.earnings_calendar = mock.Mock(
            earnings_within_window=mock.Mock(return_value=(False, None))
        )
        manager.update_portfolio(account_balance=100_000, open_positions=[], daily_pnl=0.0)

        approved, reason = manager.approve_trade(self._make_signal(symbol="QQQ", net_vega=6.0))

        self.assertFalse(approved)
        self.assertIn("vega limit", reason.lower())

    def test_check12_sector_concentration(self) -> None:
        manager = RiskManager(
            RiskConfig(
                max_sector_risk_pct=40.0,
                max_portfolio_risk_pct=20.0,
            )
        )
        manager.earnings_calendar = mock.Mock(
            earnings_within_window=mock.Mock(return_value=(False, None))
        )
        manager.update_portfolio(
            account_balance=100_000,
            open_positions=[
                {"symbol": "AAPL", "quantity": 1, "max_loss": 39.0, "details": {}},
                {"symbol": "JPM", "quantity": 1, "max_loss": 61.0, "details": {}},
            ],
            daily_pnl=0.0,
        )

        approved, reason = manager.approve_trade(self._make_signal(symbol="MSFT", max_loss=3.0))

        self.assertFalse(approved)
        self.assertIn("Sector concentration", reason)

    def test_correlation_guard(self) -> None:
        manager = RiskManager(
            RiskConfig(
                max_positions_per_symbol=1,
                max_sector_risk_pct=100.0,
                correlation_threshold=0.8,
            )
        )
        manager.earnings_calendar = mock.Mock(
            earnings_within_window=mock.Mock(return_value=(False, None))
        )
        manager.update_portfolio(
            account_balance=100_000,
            open_positions=[{"symbol": "AAPL", "quantity": 1, "max_loss": 2.0, "details": {}}],
            daily_pnl=0.0,
        )

        base = [100 + i for i in range(70)]
        highly_corr = [200 + (i * 2) for i in range(70)]

        def provider(symbol: str, days: int):
            if symbol == "AAPL":
                return [{"close": value} for value in base]
            if symbol == "MSFT":
                return [{"close": value} for value in highly_corr]
            return []

        manager.set_price_history_provider(provider)
        approved, reason = manager.approve_trade(self._make_signal(symbol="MSFT", max_loss=2.0))

        self.assertFalse(approved)
        self.assertIn("Correlation guard", reason)

    def test_equity_curve_scaling_positive_slope_scales_up(self) -> None:
        manager = RiskManager(RiskConfig())
        manager.set_sizing_config(
            {
                "equity_curve_scaling": True,
                "equity_curve_lookback": 20,
                "max_scale_up": 1.25,
                "max_scale_down": 0.50,
            }
        )
        manager.update_portfolio(account_balance=100_000, open_positions=[], daily_pnl=0.0)
        manager.update_trade_history([{"pnl": 120.0} for _ in range(20)])

        scalar = manager._equity_curve_risk_scalar()

        self.assertGreater(scalar, 1.0)
        self.assertLessEqual(scalar, 1.25)

    def test_equity_curve_scaling_negative_slope_scales_down(self) -> None:
        manager = RiskManager(RiskConfig())
        manager.set_sizing_config(
            {
                "equity_curve_scaling": True,
                "equity_curve_lookback": 20,
                "max_scale_up": 1.25,
                "max_scale_down": 0.50,
            }
        )
        manager.update_portfolio(account_balance=100_000, open_positions=[], daily_pnl=0.0)
        manager.update_trade_history([{"pnl": -180.0} for _ in range(20)])

        scalar = manager._equity_curve_risk_scalar()

        self.assertLess(scalar, 1.0)
        self.assertGreaterEqual(scalar, 0.50)


if __name__ == "__main__":
    unittest.main()
