import tempfile
import unittest
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from bot.config import StrategySandboxConfig
from bot.strategy_sandbox import StrategySandboxManager


@dataclass
class FakeBacktestResult:
    report: dict


class FakeBacktester:
    def __init__(self, report: dict):
        self._report = report

    def run(self, *, start: str, end: str) -> FakeBacktestResult:
        _ = (start, end)
        return FakeBacktestResult(report=self._report)


class StrategySandboxTests(unittest.TestCase):
    def test_synthesis_trigger_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = StrategySandboxConfig(
                enabled=True,
                min_failing_score=40.0,
                consecutive_fail_cycles=3,
                state_file=str(Path(tmp_dir) / "state.json"),
            )
            manager = StrategySandboxManager(
                cfg,
                config_path=Path(tmp_dir) / "config.yaml",
                backtester_factory=lambda: FakeBacktester({"sharpe_ratio": 1.0, "max_drawdown": 0.05}),
            )
            scores = {"credit_spreads": 25.0, "iron_condors": 35.0}
            enabled = {"credit_spreads", "iron_condors"}

            self.assertFalse(manager.update_trigger(regime="HIGH_VOL_CHOP", strategy_scores=scores, enabled_strategies=enabled))
            self.assertFalse(manager.update_trigger(regime="HIGH_VOL_CHOP", strategy_scores=scores, enabled_strategies=enabled))
            self.assertTrue(manager.update_trigger(regime="HIGH_VOL_CHOP", strategy_scores=scores, enabled_strategies=enabled))

    def test_llm_proposal_parsing(self) -> None:
        parsed = StrategySandboxManager.parse_strategy_proposal(
            {
                "name": "Crisis Put Debit Spread",
                "type": "debit_spread",
                "direction": "bearish",
                "dte_range": [21, 7],
                "delta_range": [0.45, 0.30],
                "max_risk_per_trade": "500",
            }
        )

        self.assertEqual(parsed["name"], "crisis_put_debit_spread")
        self.assertEqual(parsed["dte_range"], [7, 21])
        self.assertEqual(parsed["delta_range"], [0.3, 0.45])
        self.assertEqual(parsed["max_risk_per_trade"], 500.0)

    def test_sandbox_backtest_pass_fail_logic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = StrategySandboxConfig(
                enabled=True,
                min_sharpe=0.5,
                max_drawdown_pct=15.0,
                state_file=str(Path(tmp_dir) / "state.json"),
            )

            pass_manager = StrategySandboxManager(
                cfg,
                config_path=Path(tmp_dir) / "config_pass.yaml",
                backtester_factory=lambda: FakeBacktester({"sharpe_ratio": 0.8, "max_drawdown": 0.12}),
            )
            fail_manager = StrategySandboxManager(
                cfg,
                config_path=Path(tmp_dir) / "config_fail.yaml",
                backtester_factory=lambda: FakeBacktester({"sharpe_ratio": 0.1, "max_drawdown": 0.22}),
            )

            proposal = {
                "name": "sandbox_alpha",
                "type": "debit_spread",
                "direction": "bearish",
                "dte_range": [7, 21],
                "delta_range": [0.3, 0.45],
                "max_risk_per_trade": 500,
            }
            passed = pass_manager.run_sandbox_backtest(proposal, as_of=date(2026, 2, 26))
            failed = fail_manager.run_sandbox_backtest(proposal, as_of=date(2026, 2, 26))

            self.assertTrue(passed["passed"])
            self.assertFalse(failed["passed"])

    def test_auto_disable_after_deployment_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text("strategies: {}\n", encoding="utf-8")
            cfg = StrategySandboxConfig(
                enabled=True,
                deployment_days=1,
                sizing_scalar=0.5,
                state_file=str(Path(tmp_dir) / "state.json"),
            )
            manager = StrategySandboxManager(
                cfg,
                config_path=config_path,
                backtester_factory=lambda: FakeBacktester({"sharpe_ratio": 1.0, "max_drawdown": 0.1}),
            )

            proposal = {
                "name": "sandbox_beta",
                "type": "credit_spread",
                "direction": "bullish",
                "dte_range": [14, 28],
                "delta_range": [0.25, 0.35],
                "max_risk_per_trade": 300,
            }
            deployed = manager.deploy_strategy(proposal, backtest={"passed": True}, deployed_on=date(2026, 2, 20))
            self.assertTrue(deployed)

            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertTrue(data["strategies"]["sandbox"]["enabled"])

            manager.expire_if_needed(today=date(2026, 2, 22))
            data_after = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertFalse(bool(data_after["strategies"]["sandbox"]["enabled"]))


if __name__ == "__main__":
    unittest.main()
