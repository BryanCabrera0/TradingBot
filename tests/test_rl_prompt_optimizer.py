import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bot.config import LLMConfig, RLPromptOptimizerConfig
from bot.llm_advisor import LLMAdvisor
from bot.rl_prompt_optimizer import RLPromptOptimizer


class RLPromptOptimizerTests(unittest.TestCase):
    def test_pattern_detection_and_rule_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            cfg = RLPromptOptimizerConfig(
                enabled=True,
                min_trades_for_pattern=3,
                loss_rate_threshold=0.60,
                max_rules=25,
                rolling_window_size=100,
                learned_rules_file=str(base / "learned_rules.json"),
                audit_log_file=str(base / "audit.jsonl"),
            )
            optimizer = RLPromptOptimizer(
                cfg,
                explanations_path=base / "trade_explanations.json",
                track_record_path=base / "llm_track_record.json",
                attribution_path=base / "pnl_attribution.json",
            )

            outcomes = [-120.0, -80.0, 50.0, -40.0, -25.0]
            for idx, pnl in enumerate(outcomes):
                optimizer.process_closed_trade(
                    position_id=f"pos_{idx}",
                    pnl=pnl,
                    trade_context={
                        "strategy": "iron_condor",
                        "regime": "HIGH_VOL_CHOP",
                    },
                )

            payload = json.loads(
                (base / "learned_rules.json").read_text(encoding="utf-8")
            )
            rules = payload.get("rules", [])
            self.assertTrue(rules)
            self.assertTrue(
                any(
                    str(row.get("pattern_key", "")).startswith("strategy_regime:")
                    for row in rules
                )
            )
            self.assertTrue(
                all(str(row.get("rule", "")).startswith("RULE:") for row in rules)
            )

    def test_max_rules_pruning_keeps_most_recent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            cfg = RLPromptOptimizerConfig(
                enabled=True,
                min_trades_for_pattern=3,
                loss_rate_threshold=0.50,
                max_rules=2,
                rolling_window_size=200,
                learned_rules_file=str(base / "learned_rules.json"),
                audit_log_file=str(base / "audit.jsonl"),
            )
            optimizer = RLPromptOptimizer(
                cfg,
                explanations_path=base / "trade_explanations.json",
                track_record_path=base / "llm_track_record.json",
                attribution_path=base / "pnl_attribution.json",
            )

            for strategy, regime in [
                ("iron_condor", "HIGH_VOL_CHOP"),
                ("calendar_spread", "CRASH"),
                ("bull_put_spread", "BEAR_TREND"),
            ]:
                for idx, pnl in enumerate([-100.0, -120.0, -90.0], start=1):
                    optimizer.process_closed_trade(
                        position_id=f"{strategy}_{idx}",
                        pnl=pnl,
                        trade_context={"strategy": strategy, "regime": regime},
                    )

            payload = json.loads(
                (base / "learned_rules.json").read_text(encoding="utf-8")
            )
            rules = payload.get("rules", [])
            self.assertEqual(len(rules), 2)
            keys = [str(row.get("pattern_key", "")) for row in rules]
            self.assertNotIn("strategy_regime:iron_condor|HIGH_VOL_CHOP", keys)

    def test_learned_rules_are_injected_into_cio_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rules_path = Path(tmp_dir) / "learned_rules.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "rules": [
                            {
                                "rule_id": "r1",
                                "pattern_key": "k1",
                                "rule": "RULE: Avoid low-confidence approvals below 65 unless sized down.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            advisor = LLMAdvisor(
                LLMConfig(
                    enabled=True,
                    provider="openai",
                    ensemble_enabled=True,
                    multi_turn_enabled=False,
                )
            )
            advisor.learned_rules_path = rules_path

            system_prompts: list[str] = []

            def _fake_query(
                prompt: str,
                *,
                provider=None,
                model=None,
                system_prompt=None,
            ) -> str:
                system_prompts.append(str(system_prompt or ""))
                return (
                    '{"verdict":"approve","confidence":82,'
                    '"reasoning":"ok","suggested_adjustment":null,'
                    '"risk_adjustment":1.0,"capital_allocation_scalar":1.0}'
                )

            advisor._query_model = mock.Mock(side_effect=_fake_query)
            advisor._review_trade_ensemble("{}")

            rendered = "\n".join(system_prompts)
            self.assertIn(
                "Hard constraints from post-trade reinforcement learning", rendered
            )
            self.assertIn("RULE: Avoid low-confidence approvals", rendered)


if __name__ == "__main__":
    unittest.main()
