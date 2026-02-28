import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from bot.analysis import SpreadAnalysis
from bot.backtester import Backtester, _daily_returns, _sharpe_ratio
from bot.config import BotConfig
from bot.strategies.base import TradeSignal
from bot.strategies.credit_spreads import CreditSpreadStrategy


def _write_snapshot(
    path: Path,
    snapshot_date: str,
    short_mid: float,
    long_mid: float,
    symbol: str = "SPY",
) -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "snapshot_date": snapshot_date,
                "expiration": "2026-03-20",
                "side": "PUT",
                "strike": 95.0,
                "bid": short_mid - 0.05,
                "ask": short_mid + 0.05,
                "mid": short_mid,
                "delta": -0.30,
                "gamma": 0.01,
                "theta": -0.04,
                "vega": 0.05,
                "iv": 25.0,
                "volume": 200,
                "open_interest": 1000,
                "dte": 30,
                "underlying_price": 100.0,
            },
            {
                "symbol": symbol,
                "snapshot_date": snapshot_date,
                "expiration": "2026-03-20",
                "side": "PUT",
                "strike": 90.0,
                "bid": long_mid - 0.05,
                "ask": long_mid + 0.05,
                "mid": long_mid,
                "delta": -0.10,
                "gamma": 0.01,
                "theta": -0.02,
                "vega": 0.03,
                "iv": 24.0,
                "volume": 150,
                "open_interest": 900,
                "dte": 30,
                "underlying_price": 100.0,
            },
            {
                "symbol": symbol,
                "snapshot_date": snapshot_date,
                "expiration": "2026-03-20",
                "side": "CALL",
                "strike": 105.0,
                "bid": 0.9,
                "ask": 1.1,
                "mid": 1.0,
                "delta": 0.30,
                "gamma": 0.01,
                "theta": -0.03,
                "vega": 0.05,
                "iv": 26.0,
                "volume": 150,
                "open_interest": 1000,
                "dte": 30,
                "underlying_price": 100.0,
            },
            {
                "symbol": symbol,
                "snapshot_date": snapshot_date,
                "expiration": "2026-03-20",
                "side": "CALL",
                "strike": 110.0,
                "bid": 0.2,
                "ask": 0.3,
                "mid": 0.25,
                "delta": 0.10,
                "gamma": 0.01,
                "theta": -0.02,
                "vega": 0.03,
                "iv": 24.0,
                "volume": 110,
                "open_interest": 800,
                "dte": 30,
                "underlying_price": 100.0,
            },
        ]
    )
    frame.to_csv(path, index=False, compression="gzip")


class BacktesterTests(unittest.TestCase):
    def test_backtester_generates_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            _write_snapshot(
                data_dir / "SPY_2026-02-20.parquet.csv.gz",
                "2026-02-20",
                short_mid=1.2,
                long_mid=0.2,
            )
            _write_snapshot(
                data_dir / "SPY_2026-02-23.parquet.csv.gz",
                "2026-02-23",
                short_mid=0.4,
                long_mid=0.1,
            )

            cfg = BotConfig()
            cfg.scanner.enabled = False
            cfg.iron_condors.enabled = False
            cfg.covered_calls.enabled = False
            cfg.credit_spreads.enabled = True
            cfg.credit_spreads.min_dte = 1
            cfg.credit_spreads.max_dte = 45

            backtester = Backtester(cfg, data_dir=data_dir, initial_balance=100_000.0)
            backtester.risk_manager.earnings_calendar = mock.Mock(
                earnings_within_window=mock.Mock(return_value=(False, None))
            )

            result = backtester.run(start="2026-02-20", end="2026-02-23")

            self.assertTrue(Path(result.report_path).exists())
            self.assertIn("total_return", result.report)
            self.assertIn("monthly_returns", result.report)

    def test_entries_respect_max_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            for symbol in ("SPY", "QQQ", "IWM", "AAPL", "MSFT"):
                _write_snapshot(
                    data_dir / f"{symbol}_2026-02-20.parquet.csv.gz",
                    "2026-02-20",
                    short_mid=1.2,
                    long_mid=0.2,
                    symbol=symbol,
                )

            cfg = BotConfig()
            cfg.iron_condors.enabled = False
            cfg.covered_calls.enabled = False
            cfg.credit_spreads.enabled = True
            cfg.credit_spreads.min_dte = 1
            cfg.credit_spreads.max_dte = 45
            cfg.risk.max_open_positions = 2

            backtester = Backtester(cfg, data_dir=data_dir, initial_balance=100_000.0)
            backtester.risk_manager.earnings_calendar = mock.Mock(
                earnings_within_window=mock.Mock(return_value=(False, None))
            )

            result = backtester.run(start="2026-02-20", end="2026-02-20")

            self.assertLessEqual(len(backtester.positions), 2)
            self.assertLessEqual(int(result.report["open_positions"]), 2)

    def test_exit_on_profit_target(self) -> None:
        strategy = CreditSpreadStrategy(
            {"profit_target_pct": 0.5, "stop_loss_pct": 2.0}
        )
        positions = [
            {
                "position_id": "p-profit",
                "strategy": "bull_put_spread",
                "symbol": "SPY",
                "entry_credit": 1.0,
                "current_value": 0.5,
                "dte_remaining": 30,
                "status": "open",
                "quantity": 1,
                "details": {"short_strike": 95.0},
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, "close")
        self.assertIn("Profit target", signals[0].reason)

    def test_exit_on_stop_loss(self) -> None:
        strategy = CreditSpreadStrategy(
            {"profit_target_pct": 0.5, "stop_loss_pct": 2.0}
        )
        positions = [
            {
                "position_id": "p-loss",
                "strategy": "bull_put_spread",
                "symbol": "SPY",
                "entry_credit": 1.0,
                "current_value": 3.2,
                "dte_remaining": 30,
                "status": "open",
                "quantity": 1,
                "details": {"short_strike": 95.0},
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, "close")
        self.assertIn("Stop loss", signals[0].reason)

    def test_sharpe_calculation(self) -> None:
        equity_curve = [100000, 100500, 101000, 100200, 100800]
        returns = _daily_returns(equity_curve)
        sharpe = _sharpe_ratio(returns)

        mean_ret = sum(returns) / len(returns)
        variance = sum((item - mean_ret) ** 2 for item in returns) / (len(returns) - 1)
        expected = (mean_ret / (variance**0.5)) * (252**0.5)

        self.assertAlmostEqual(sharpe, expected, places=2)

    def test_empty_data_produces_empty_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = BotConfig()
            cfg.iron_condors.enabled = False
            cfg.covered_calls.enabled = False
            cfg.credit_spreads.enabled = True

            backtester = Backtester(cfg, data_dir=tmp_dir, initial_balance=100_000.0)
            backtester.risk_manager.earnings_calendar = mock.Mock(
                earnings_within_window=mock.Mock(return_value=(False, None))
            )

            result = backtester.run(start="2026-02-20", end="2026-02-20")

            self.assertEqual(result.report["closed_trades"], 0)
            self.assertEqual(result.report["open_positions"], 0)
            self.assertEqual(result.report["ending_equity"], 100_000.0)

    def test_open_position_persists_greeks_for_later_portfolio_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = BotConfig()
            cfg.credit_spreads.enabled = True
            cfg.iron_condors.enabled = False
            cfg.covered_calls.enabled = False
            cfg.naked_puts.enabled = False
            cfg.calendar_spreads.enabled = False
            bt = Backtester(cfg, data_dir=tmp_dir, initial_balance=100_000.0)

            analysis = SpreadAnalysis(
                symbol="SPY",
                strategy="bull_put_spread",
                expiration="2026-03-20",
                dte=30,
                short_strike=95.0,
                long_strike=90.0,
                credit=1.2,
                max_loss=3.8,
                probability_of_profit=0.65,
                score=60.0,
                net_delta=0.08,
                net_theta=0.02,
                net_gamma=-0.01,
                net_vega=-0.03,
            )
            signal = TradeSignal(
                action="open",
                strategy="bull_put_spread",
                symbol="SPY",
                analysis=analysis,
                quantity=1,
            )
            bt._open_position(signal)

            bt.risk_manager.update_portfolio(
                account_balance=100_000.0,
                open_positions=bt.positions,
                daily_pnl=0.0,
            )

            self.assertEqual(bt.risk_manager.portfolio.net_delta, 8.0)
            self.assertEqual(bt.risk_manager.portfolio.net_theta, 2.0)
            self.assertEqual(bt.risk_manager.portfolio.net_gamma, -1.0)
            self.assertEqual(bt.risk_manager.portfolio.net_vega, -3.0)

    def test_backtester_report_includes_regime_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            _write_snapshot(
                data_dir / "SPY_2026-02-20.parquet.csv.gz",
                "2026-02-20",
                short_mid=1.2,
                long_mid=0.2,
            )
            _write_snapshot(
                data_dir / "SPY_2026-02-23.parquet.csv.gz",
                "2026-02-23",
                short_mid=0.4,
                long_mid=0.1,
            )

            cfg = BotConfig()
            cfg.scanner.enabled = False
            cfg.iron_condors.enabled = False
            cfg.covered_calls.enabled = False
            cfg.credit_spreads.enabled = True
            cfg.credit_spreads.min_dte = 1
            cfg.credit_spreads.max_dte = 45

            backtester = Backtester(cfg, data_dir=data_dir, initial_balance=100_000.0)
            backtester.risk_manager.earnings_calendar = mock.Mock(
                earnings_within_window=mock.Mock(return_value=(False, None))
            )

            result = backtester.run(start="2026-02-20", end="2026-02-23")

            self.assertIn("regime_performance", result.report)
            self.assertIsInstance(result.report["regime_performance"], dict)
            if backtester.closed_trades:
                self.assertIn("regime", backtester.closed_trades[0])

    def test_backtester_report_includes_analytics_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            _write_snapshot(
                data_dir / "SPY_2026-02-20.parquet.csv.gz",
                "2026-02-20",
                short_mid=1.2,
                long_mid=0.2,
            )
            _write_snapshot(
                data_dir / "SPY_2026-02-23.parquet.csv.gz",
                "2026-02-23",
                short_mid=0.4,
                long_mid=0.1,
            )

            cfg = BotConfig()
            cfg.scanner.enabled = False
            cfg.iron_condors.enabled = False
            cfg.covered_calls.enabled = False
            cfg.credit_spreads.enabled = True
            cfg.credit_spreads.min_dte = 1
            cfg.credit_spreads.max_dte = 45

            backtester = Backtester(cfg, data_dir=data_dir, initial_balance=100_000.0)
            backtester.risk_manager.earnings_calendar = mock.Mock(
                earnings_within_window=mock.Mock(return_value=(False, None))
            )

            result = backtester.run(start="2026-02-20", end="2026-02-23")

            self.assertIn("analytics", result.report)
            self.assertIn("analytics_core", result.report)
            self.assertIn("risk_adjusted_return", result.report)
            self.assertIn("expectancy_per_trade", result.report)


if __name__ == "__main__":
    unittest.main()
