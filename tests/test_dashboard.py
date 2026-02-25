import tempfile
import unittest
from pathlib import Path

from bot.dashboard import generate_dashboard


class DashboardTests(unittest.TestCase):
    def test_generate_dashboard_writes_html(self) -> None:
        payload = {
            "equity_curve": [{"date": "2026-02-20", "equity": 100000}, {"date": "2026-02-21", "equity": 100500}],
            "monthly_pnl": {"2026-02": 500.0},
            "strategy_breakdown": {
                "bull_put_spread": {"win_rate": 60.0, "avg_pnl": 50.0, "total_pnl": 500.0}
            },
            "top_winners": [{"symbol": "SPY", "pnl": 200.0}],
            "top_losers": [{"symbol": "QQQ", "pnl": -100.0}],
            "risk_metrics": {"sharpe": 1.1, "sortino": 1.4, "max_drawdown": 0.05, "current_drawdown": 0.01},
            "portfolio_greeks": {"delta": 10.0, "theta": 2.0, "gamma": 0.5, "vega": 15.0},
            "sector_exposure": {"Technology": 40.0, "Financials": 20.0},
            "circuit_breakers": {"regime": "normal", "halt_entries": False},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "dashboard.html"
            output = generate_dashboard(payload, output_path=path)

            html = Path(output).read_text(encoding="utf-8")
            self.assertIn("TradingBot Dashboard", html)
            self.assertIn("Equity Curve", html)
            self.assertIn("Technology", html)


if __name__ == "__main__":
    unittest.main()
