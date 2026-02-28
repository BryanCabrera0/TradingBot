import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

from bot.analysis import SpreadAnalysis
from bot.config import LLMConfig
from bot.data_store import dump_json
from bot.llm_advisor import LLMAdvisor
from bot.strategies.base import TradeSignal


def make_signal() -> TradeSignal:
    return TradeSignal(
        action="open",
        strategy="bull_put_spread",
        symbol="SPY",
        quantity=2,
        analysis=SpreadAnalysis(
            symbol="SPY",
            strategy="bull_put_spread",
            expiration="2026-03-20",
            dte=30,
            short_strike=95,
            long_strike=90,
            credit=1.25,
            max_loss=3.75,
            probability_of_profit=0.68,
            score=64,
        ),
    )


class LLMAdvisorTests(unittest.TestCase):
    def test_review_trade_parses_structured_json(self) -> None:
        advisor = LLMAdvisor(LLMConfig(enabled=True, provider="ollama"))
        advisor._query_model = mock.Mock(
            return_value=(
                '{"verdict":"reject","confidence":82,'
                '"risk_adjustment":0.6,"reasoning":"volatility spike"}'
            )
        )

        decision = advisor.review_trade(make_signal(), {"account_balance": 100000})

        self.assertFalse(decision.approve)
        self.assertAlmostEqual(decision.confidence, 0.82, places=2)
        self.assertAlmostEqual(decision.risk_adjustment, 0.6, places=2)
        self.assertEqual(decision.reason, "volatility spike")

    def test_review_trade_handles_invalid_response(self) -> None:
        advisor = LLMAdvisor(LLMConfig(enabled=True, provider="ollama"))
        advisor._query_model = mock.Mock(return_value="not-json")

        decision = advisor.review_trade(make_signal(), None)

        self.assertTrue(decision.approve)
        self.assertEqual(decision.confidence, 0.0)
        self.assertEqual(decision.risk_adjustment, 1.0)
        self.assertIn("missing reason", decision.reason.lower())

    def test_prompt_includes_conservative_policy_thresholds(self) -> None:
        advisor = LLMAdvisor(
            LLMConfig(enabled=True, provider="ollama", risk_style="conservative")
        )
        prompt = advisor._build_prompt(make_signal(), {"account_balance": 100000})
        payload = json.loads(prompt)

        self.assertEqual(payload["risk_style"], "conservative")
        self.assertEqual(payload["risk_policy"]["min_probability_of_profit"], 0.65)
        self.assertEqual(payload["risk_policy"]["min_score"], 60.0)

    def test_invalid_risk_style_falls_back_to_moderate(self) -> None:
        advisor = LLMAdvisor(
            LLMConfig(enabled=True, provider="ollama", risk_style="unknown-style")
        )
        prompt = advisor._build_prompt(make_signal(), None)
        payload = json.loads(prompt)

        self.assertEqual(payload["risk_style"], "moderate")

    def test_openai_uses_responses_api(self) -> None:
        advisor = LLMAdvisor(
            LLMConfig(enabled=True, provider="openai", model="gpt-4.1")
        )
        response = mock.Mock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "output_text": (
                '{"approve":true,"confidence":0.9,"risk_adjustment":1.0,"reason":"ok"}'
            )
        }

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with mock.patch(
                "bot.openai_compat.requests.post", return_value=response
            ) as post:
                raw = advisor._query_openai("{}")

        self.assertIn('"approve":true', raw.replace(" ", ""))
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.openai.com/v1/responses")
        self.assertEqual(kwargs["json"]["model"], "gpt-4.1")
        self.assertEqual(kwargs["json"]["text"]["format"]["type"], "json_schema")

    def test_openai_fallback_extracts_text_from_output_blocks(self) -> None:
        advisor = LLMAdvisor(
            LLMConfig(enabled=True, provider="openai", model="gpt-4.1")
        )
        response = mock.Mock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "output_text": "",
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                '{"approve":false,"confidence":0.3,'
                                '"risk_adjustment":0.6,"reason":"headline risk"}'
                            ),
                        }
                    ]
                }
            ],
        }

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with mock.patch("bot.openai_compat.requests.post", return_value=response):
                raw = advisor._query_openai("{}")

        self.assertIn("headline risk", raw)

    def test_openai_fallbacks_to_chat_completions_on_404(self) -> None:
        advisor = LLMAdvisor(
            LLMConfig(enabled=True, provider="openai", model="gpt-4.1")
        )
        responses_404 = mock.Mock()
        responses_404.status_code = 404
        responses_404.raise_for_status.side_effect = Exception("not used")

        chat_response = mock.Mock()
        chat_response.raise_for_status.return_value = None
        chat_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"verdict":"approve","confidence":85,'
                            '"reasoning":"ok","suggested_adjustment":null}'
                        )
                    }
                }
            ]
        }

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with mock.patch(
                "bot.openai_compat.requests.post",
                side_effect=[responses_404, chat_response],
            ) as post:
                raw = advisor._query_openai("{}")

        self.assertIn('"verdict":"approve"', raw.replace(" ", ""))
        self.assertEqual(
            post.call_args_list[0].args[0], "https://api.openai.com/v1/responses"
        )
        self.assertEqual(
            post.call_args_list[1].args[0], "https://api.openai.com/v1/chat/completions"
        )

    def test_openai_o1_payload_uses_reasoning_without_temp_or_schema(
        self,
    ) -> None:
        advisor = LLMAdvisor(
            LLMConfig(
                enabled=True,
                provider="openai",
                model="o1",
                reasoning_effort="high",
                text_verbosity="low",
                max_output_tokens=777,
            )
        )
        response = mock.Mock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "output_text": (
                '{"verdict":"approve","confidence":92,'
                '"reasoning":"ok","suggested_adjustment":null}'
            )
        }

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with mock.patch(
                "bot.openai_compat.requests.post", return_value=response
            ) as post:
                _ = advisor._query_openai("{}")

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "o1")
        self.assertEqual(payload["reasoning"]["effort"], "high")
        self.assertEqual(payload["text"]["verbosity"], "low")
        self.assertEqual(payload["max_output_tokens"], 777)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("format", payload["text"])

    def test_openai_gpt52pro_uses_chat_fallback_model_on_404(self) -> None:
        advisor = LLMAdvisor(
            LLMConfig(
                enabled=True,
                provider="openai",
                model="gpt-5.2-pro",
                chat_fallback_model="gpt-4.1",
            )
        )
        responses_404 = mock.Mock()
        responses_404.status_code = 404
        responses_404.raise_for_status.side_effect = Exception("not used")

        chat_response = mock.Mock()
        chat_response.raise_for_status.return_value = None
        chat_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"verdict":"approve","confidence":85,'
                            '"reasoning":"ok","suggested_adjustment":null}'
                        )
                    }
                }
            ]
        }

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with mock.patch(
                "bot.openai_compat.requests.post",
                side_effect=[responses_404, chat_response],
            ) as post:
                raw = advisor._query_openai("{}")

        self.assertIn('"verdict":"approve"', raw.replace(" ", ""))
        fallback_payload = post.call_args_list[1].kwargs["json"]
        self.assertEqual(fallback_payload["model"], "gpt-4.1")

    def test_openai_health_check_rejects_placeholder_key(self) -> None:
        advisor = LLMAdvisor(LLMConfig(enabled=True, provider="openai"))

        with mock.patch.dict(
            os.environ, {"OPENAI_API_KEY": "your_openai_key_here"}, clear=True
        ):
            ok, message = advisor.health_check()

        self.assertFalse(ok)
        self.assertIn("missing", message.lower())

    def test_google_uses_generate_content_api(self) -> None:
        advisor = LLMAdvisor(
            LLMConfig(enabled=True, provider="google", model="gemini-3.1-pro-thinking-preview")
        )
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    '{"verdict":"approve","confidence":85,'
                                    '"reasoning":"ok","suggested_adjustment":null}'
                                )
                            }
                        ]
                    }
                }
            ]
        }

        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
            with mock.patch(
                "bot.llm_advisor.requests.post", return_value=response
            ) as post:
                raw = advisor._query_google("{}")

        self.assertIn('"verdict":"approve"', raw.replace(" ", ""))
        args, kwargs = post.call_args
        self.assertIn(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-thinking-preview:generateContent",
            args[0],
        )
        self.assertEqual(kwargs["params"]["key"], "test-key")
        self.assertEqual(
            kwargs["json"]["generationConfig"]["responseMimeType"], "application/json"
        )

    def test_google_health_check_rejects_placeholder_key(self) -> None:
        advisor = LLMAdvisor(LLMConfig(enabled=True, provider="google"))
        with mock.patch.dict(
            os.environ, {"GOOGLE_API_KEY": "your_google_key_here"}, clear=True
        ):
            ok, message = advisor.health_check()

        self.assertFalse(ok)
        self.assertIn("missing", message.lower())

    def test_google_health_check_probe_verifies_primary_and_fallback_models(self) -> None:
        advisor = LLMAdvisor(
            LLMConfig(
                enabled=True,
                provider="google",
                model="gemini-3.1-pro-thinking-preview",
                chat_fallback_model="gemini-3.1-flash-thinking-preview",
            )
        )
        advisor._query_google = mock.Mock(return_value='{"status":"ok"}')
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
            ok, message = advisor.health_check(
                probe_google=True,
                probe_models=[
                    "gemini-3.1-pro-thinking-preview",
                    "gemini-3.1-flash-thinking-preview",
                ],
            )

        self.assertTrue(ok)
        self.assertIn("gemini-3.1-pro-thinking-preview", message)
        self.assertIn("gemini-3.1-flash-thinking-preview", message)
        self.assertEqual(advisor._query_google.call_count, 2)

    def test_google_health_check_probe_rejects_non_json(self) -> None:
        advisor = LLMAdvisor(LLMConfig(enabled=True, provider="google"))
        advisor._query_google = mock.Mock(return_value="not-json")
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
            ok, message = advisor.health_check(probe_google=True)

        self.assertFalse(ok)
        self.assertIn("non-json", message.lower())

    def test_retry_when_initial_response_is_not_json(self) -> None:
        advisor = LLMAdvisor(LLMConfig(enabled=True, provider="ollama"))
        advisor._query_model = mock.Mock(
            side_effect=[
                "bad response",
                '{"verdict":"approve","confidence":85,"reasoning":"ok","suggested_adjustment":null}',
            ]
        )

        decision = advisor.review_trade(make_signal(), {})

        self.assertTrue(decision.approve)
        self.assertEqual(advisor._query_model.call_count, 2)

    def test_parse_new_verdict_schema(self) -> None:
        advisor = LLMAdvisor(LLMConfig(enabled=True, provider="ollama"))
        advisor._query_model = mock.Mock(
            return_value='{"verdict":"reduce_size","confidence":75,"reasoning":"event risk","suggested_adjustment":"reduce 40%"}'
        )

        decision = advisor.review_trade(make_signal(), {})

        self.assertEqual(decision.verdict, "reduce_size")
        self.assertLess(decision.risk_adjustment, 1.0)

    def test_deep_debate_conflict_forces_rebuttal_and_reject(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            provider="openai",
            ensemble_enabled=True,
            multi_turn_enabled=False,
        )
        advisor = LLMAdvisor(cfg)
        advisor._query_model = mock.Mock(
            side_effect=[
                '{"verdict":"approve","confidence":80,"reasoning":"macro supports entry","suggested_adjustment":null,"risk_adjustment":1.0,"capital_allocation_scalar":1.0}',
                '{"verdict":"reject","confidence":83,"reasoning":"vol regime hostile","suggested_adjustment":null,"risk_adjustment":1.0,"capital_allocation_scalar":1.0}',
                '{"verdict":"reduce_size","confidence":74,"reasoning":"tail risk elevated","suggested_adjustment":"reduce 30%","risk_adjustment":0.7,"capital_allocation_scalar":0.7}',
                '{"verdict":"reduce_size","confidence":71,"reasoning":"conflict detected; request rebuttal","suggested_adjustment":"reduce 20%","risk_adjustment":0.8,"capital_allocation_scalar":0.8,"force_debate":true}',
                '{"verdict":"reject","confidence":79,"reasoning":"macro downside risk now dominates","suggested_adjustment":null,"risk_adjustment":1.0,"capital_allocation_scalar":1.0}',
                '{"verdict":"reject","confidence":81,"reasoning":"vol skew now adverse","suggested_adjustment":null,"risk_adjustment":1.0,"capital_allocation_scalar":1.0}',
                '{"verdict":"reject","confidence":84,"reasoning":"portfolio tail risk unacceptable","suggested_adjustment":null,"risk_adjustment":1.0,"capital_allocation_scalar":1.0}',
                '{"verdict":"reject","confidence":88,"reasoning":"CIO final no-go after rebuttal","suggested_adjustment":null,"risk_adjustment":1.0,"capital_allocation_scalar":1.0}',
            ]
        )

        decision, votes = advisor._review_trade_ensemble("{}")

        self.assertGreaterEqual(len(votes), 7)
        self.assertEqual(decision.verdict, "reject")
        self.assertEqual(max(int(v.get("round", 1)) for v in votes), 2)
        self.assertTrue(all(str(v.get("provider")) == "openai" for v in votes))

    def test_deep_debate_falls_back_to_gpt52_when_primary_fails(self) -> None:
        cfg = LLMConfig(
            enabled=True,
            provider="openai",
            model="gpt-5.2-pro",
            ensemble_enabled=True,
            multi_turn_enabled=False,
        )
        advisor = LLMAdvisor(cfg)
        models_seen: list[str] = []

        def _fake_query(
            prompt: str,
            *,
            provider: Optional[str] = None,
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,
            **kwargs,
        ) -> str:
            models_seen.append(str(model or ""))
            if model == "gpt-5.2-pro":
                raise RuntimeError("rate limited")
            return (
                '{"verdict":"approve","confidence":90,"reasoning":"fallback succeeded",'
                '"suggested_adjustment":"reduce 10%","risk_adjustment":0.9,'
                '"capital_allocation_scalar":0.9}'
            )

        advisor._query_model = mock.Mock(side_effect=_fake_query)

        decision, votes = advisor._review_trade_ensemble("{}")

        self.assertEqual(decision.verdict, "approve")
        self.assertTrue(any(model == "gpt-5.2-pro" for model in models_seen))
        self.assertGreaterEqual(len(votes), 4)

    def test_prompt_includes_journal_context_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            journal_path = Path(tmp_dir) / "journal.json"
            dump_json(
                journal_path,
                {
                    "entries": [
                        {
                            "symbol": "SPY",
                            "strategy": "bull_put_spread",
                            "verdict": "approve",
                            "outcome": 120.0,
                            "analysis": {"adjustment": "keep wider wings"},
                        }
                    ]
                },
            )
            advisor = LLMAdvisor(
                LLMConfig(
                    enabled=True,
                    provider="ollama",
                    journal_enabled=True,
                    journal_file=str(journal_path),
                    journal_context_entries=5,
                )
            )
            prompt = advisor._build_prompt(make_signal(), {"account_balance": 100000})
            payload = json.loads(prompt)
            self.assertTrue(payload["sections"]["recent_trade_journal"])

    def test_prompt_builds_relevant_bounded_history_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            track_path = Path(tmp_dir) / "track.json"
            trades = []
            for idx in range(20):
                symbol = "SPY" if idx % 2 == 0 else "QQQ"
                strategy = "bull_put_spread" if idx % 3 else "iron_condor"
                verdict = "approve" if idx % 2 == 0 else "reject"
                outcome = (
                    -50.0
                    if idx in {2, 4, 6}
                    else (80.0 if verdict == "approve" else -40.0)
                )
                trades.append(
                    {
                        "timestamp": f"2026-02-{idx + 1:02d}",
                        "symbol": symbol,
                        "strategy": strategy,
                        "verdict": verdict,
                        "confidence": 70,
                        "outcome": outcome,
                        "trade_snapshot": {
                            "dte": 30,
                            "score": 60 + (idx % 5),
                            "probability_of_profit": 0.60,
                        },
                    }
                )
            dump_json(track_path, {"trades": trades, "meta": {}})
            advisor = LLMAdvisor(
                LLMConfig(
                    enabled=True,
                    provider="ollama",
                    track_record_file=str(track_path),
                )
            )

            prompt = advisor._build_prompt(make_signal(), {"account_balance": 100000})
            payload = json.loads(prompt)
            hist = payload["sections"]["historical_context"]

            self.assertLessEqual(len(hist["same_symbol_recent"]), 5)
            self.assertLessEqual(len(hist["same_strategy_recent"]), 5)
            self.assertLessEqual(len(hist["recent_mistakes"]), 3)
            self.assertLessEqual(len(payload["sections"]["similar_trades"]), 3)
            self.assertIn("approval accuracy", hist["success_rate_summary"].lower())

    def test_model_weight_uses_accuracy_after_minimum_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            track = Path(tmp_dir) / "llm_track.json"
            dump_json(
                track,
                {
                    "trades": [],
                    "meta": {
                        "model_stats": {
                            "openai:gpt-5.2-pro": {
                                "trades": 20,
                                "hits": 15,
                                "accuracy": 0.75,
                            }
                        }
                    },
                },
            )
            advisor = LLMAdvisor(
                LLMConfig(
                    enabled=True,
                    provider="openai",
                    model="gpt-5.2-pro",
                    track_record_file=str(track),
                )
            )

            weight = advisor._model_weight("openai:gpt-5.2-pro")

            self.assertAlmostEqual(weight, 1.5, places=4)

    def test_calibrated_confidence_uses_calibration_curve_after_50_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            track = Path(tmp_dir) / "llm_track.json"
            trades = []
            for idx in range(55):
                trades.append(
                    {
                        "verdict": "approve",
                        "outcome": 10.0 if idx % 2 == 0 else -5.0,
                        "confidence": 85.0,
                    }
                )
            dump_json(
                track,
                {
                    "trades": trades,
                    "meta": {
                        "calibration": {
                            "80-90": {
                                "actual_win_rate": 0.55,
                                "expected_confidence": 0.85,
                                "trades": 55,
                            }
                        }
                    },
                },
            )
            advisor = LLMAdvisor(
                LLMConfig(enabled=True, provider="ollama", track_record_file=str(track))
            )

            adjusted = advisor._calibrated_confidence_pct(84.0)

            self.assertAlmostEqual(adjusted, 55.0, places=4)

    def test_review_trade_applies_confidence_calibration(self) -> None:
        advisor = LLMAdvisor(LLMConfig(enabled=True, provider="ollama"))
        advisor._query_model = mock.Mock(
            return_value='{"verdict":"approve","confidence":84,"reasoning":"ok","suggested_adjustment":null}'
        )
        advisor._calibrated_confidence_pct = mock.Mock(return_value=55.0)

        decision = advisor.review_trade(make_signal(), {})

        self.assertAlmostEqual(decision.confidence, 0.55, places=4)

    def test_update_model_accuracy_tracks_model_id_votes(self) -> None:
        advisor = LLMAdvisor(LLMConfig(enabled=True, provider="ollama"))
        payload = {"trades": [], "meta": {}}
        trade = {
            "outcome": 120.0,
            "model_votes": [{"model_id": "openai:gpt-5.2-pro", "verdict": "approve"}],
        }

        advisor._update_model_accuracy(payload, trade)

        stats = payload["meta"]["model_stats"]["openai:gpt-5.2-pro"]
        self.assertEqual(stats["trades"], 1)
        self.assertEqual(stats["hits"], 1)

    def test_multi_turn_rechecks_uncertain_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            track = Path(tmp_dir) / "llm_track.json"
            advisor = LLMAdvisor(
                LLMConfig(
                    enabled=True,
                    provider="ollama",
                    track_record_file=str(track),
                    multi_turn_enabled=True,
                    multi_turn_confidence_threshold=70,
                )
            )
            advisor._query_model = mock.Mock(
                side_effect=[
                    '{"verdict":"reduce_size","confidence":65,"reasoning":"uncertain","suggested_adjustment":"reduce 20%"}',
                    '{"verdict":"approve","confidence":82,"reasoning":"resolved","suggested_adjustment":null}',
                ]
            )
            signal = make_signal()

            decision = advisor.review_trade(
                signal, {"portfolio_exposure": {"total_delta": 12}}
            )

            self.assertEqual(advisor._query_model.call_count, 2)
            self.assertEqual(decision.verdict, "approve")
            self.assertTrue(bool(signal.metadata.get("llm_multi_turn_used")))
            self.assertTrue(bool(signal.metadata.get("llm_multi_turn_changed_verdict")))

    def test_multi_turn_skips_high_confidence_non_reduce(self) -> None:
        advisor = LLMAdvisor(
            LLMConfig(
                enabled=True,
                provider="ollama",
                multi_turn_enabled=True,
                multi_turn_confidence_threshold=70,
            )
        )
        advisor._query_model = mock.Mock(
            return_value='{"verdict":"approve","confidence":91,"reasoning":"clear","suggested_adjustment":null}'
        )
        signal = make_signal()

        decision = advisor.review_trade(signal, {})

        self.assertEqual(decision.verdict, "approve")
        self.assertEqual(advisor._query_model.call_count, 1)
        self.assertFalse(bool(signal.metadata.get("llm_multi_turn_used")))

    def test_trade_explanations_saved_and_outcome_appended(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            track = Path(tmp_dir) / "llm_track.json"
            explanations = Path(tmp_dir) / "explanations.json"
            advisor = LLMAdvisor(
                LLMConfig(
                    enabled=True,
                    provider="ollama",
                    track_record_file=str(track),
                    explanations_file=str(explanations),
                )
            )
            advisor._query_model = mock.Mock(
                return_value=(
                    '{"verdict":"approve","confidence":88,"reasoning":"good setup",'
                    '"suggested_adjustment":null,"bull_case":"premium rich",'
                    '"bear_case":"gap risk","key_risk":"earnings surprise",'
                    '"expected_duration":"10 days","confidence_drivers":["score","pop","iv"]}'
                )
            )
            signal = make_signal()

            decision = advisor.review_trade(signal, {})
            advisor.bind_position(decision.review_id, "pos_1")
            advisor.record_outcome("pos_1", 125.0)

            payload = json.loads(explanations.read_text(encoding="utf-8"))
            row = payload["positions"]["pos_1"]
            self.assertEqual(row["bull_case"], "premium rich")
            self.assertEqual(float(row["outcome"]), 125.0)

    def test_adversarial_review_can_trigger_exit(self) -> None:
        advisor = LLMAdvisor(
            LLMConfig(
                enabled=True,
                provider="ollama",
                adversarial_review_enabled=True,
            )
        )
        advisor._query_model = mock.Mock(
            side_effect=[
                '{"conviction":85,"reasoning":"risk is asymmetric"}',
                '{"conviction":50,"reasoning":"can recover"}',
            ]
        )

        result = advisor.adversarial_review_position(
            {"symbol": "SPY", "strategy": "bull_put_spread", "dte_remaining": 7}
        )

        self.assertTrue(result["should_exit"])
        self.assertGreater(result["close_conviction"], result["hold_conviction"])


if __name__ == "__main__":
    unittest.main()
