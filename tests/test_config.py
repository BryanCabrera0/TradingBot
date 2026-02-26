import os
import tempfile
import textwrap
import unittest
from unittest import mock

from bot.config import BotConfig, format_validation_report, load_config, validate_config


class ConfigTests(unittest.TestCase):
    def _write_config(self, yaml_text: str) -> str:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        tmp.write(textwrap.dedent(yaml_text))
        tmp.flush()
        tmp.close()
        self.addCleanup(lambda: os.remove(tmp.name) if os.path.exists(tmp.name) else None)
        return tmp.name

    def test_load_config_normalizes_enum_like_fields(self) -> None:
        config_path = self._write_config(
            """
            trading_mode: " LIVE "
            llm:
              provider: " OPENAI "
              mode: " BLOCKING "
              risk_style: " AGGRESSIVE "
            logging:
              level: warning
            """
        )

        with mock.patch("bot.config.load_dotenv", return_value=False), mock.patch.dict(
            os.environ, {}, clear=True
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.trading_mode, "live")
        self.assertEqual(cfg.llm.provider, "openai")
        self.assertEqual(cfg.llm.mode, "blocking")
        self.assertEqual(cfg.llm.risk_style, "aggressive")
        self.assertEqual(cfg.log_level, "WARNING")

    def test_load_config_invalid_enum_values_fall_back_to_safe_defaults(self) -> None:
        config_path = self._write_config(
            """
            trading_mode: risky
            llm:
              provider: local
              mode: veto
              risk_style: yolo
            logging:
              level: noisy
            """
        )

        with mock.patch("bot.config.load_dotenv", return_value=False), mock.patch.dict(
            os.environ, {}, clear=True
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.trading_mode, "paper")
        self.assertEqual(cfg.llm.provider, "ollama")
        self.assertEqual(cfg.llm.mode, "advisory")
        self.assertEqual(cfg.llm.risk_style, "moderate")
        self.assertEqual(cfg.log_level, "INFO")

    def test_env_overrides_clamp_llm_numeric_settings(self) -> None:
        config_path = self._write_config("llm:\n  enabled: true\n")

        with mock.patch("bot.config.load_dotenv", return_value=False), mock.patch.dict(
            os.environ,
            {
                "LLM_TIMEOUT_SECONDS": "0",
                "LLM_TEMPERATURE": "3.5",
                "LLM_MIN_CONFIDENCE": "-0.2",
                "LLM_MAX_OUTPUT_TOKENS": "0",
            },
            clear=True,
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.llm.timeout_seconds, 1)
        self.assertEqual(cfg.llm.temperature, 2.0)
        self.assertEqual(cfg.llm.min_confidence, 0.0)
        self.assertEqual(cfg.llm.max_output_tokens, 64)

    def test_env_overrides_llm_ensemble_models(self) -> None:
        config_path = self._write_config("llm:\n  enabled: true\n")
        with mock.patch("bot.config.load_dotenv", return_value=False), mock.patch.dict(
            os.environ,
            {
                "LLM_ENSEMBLE_MODELS": "openai:gpt-5.2-pro, anthropic:claude-sonnet-4-20250514 ,ollama:llama3.1",
            },
            clear=True,
        ):
            cfg = load_config(config_path)

        self.assertEqual(
            cfg.llm.ensemble_models,
            [
                "openai:gpt-5.2-pro",
                "anthropic:claude-sonnet-4-20250514",
                "ollama:llama3.1",
            ],
        )

    def test_llm_openai_capability_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            llm:
              reasoning_effort: turbo
              text_verbosity: very-high
              chat_fallback_model: "  "
            """
        )

        with mock.patch("bot.config.load_dotenv", return_value=False), mock.patch.dict(
            os.environ, {}, clear=True
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.llm.reasoning_effort, "high")
        self.assertEqual(cfg.llm.text_verbosity, "low")
        self.assertEqual(cfg.llm.chat_fallback_model, "gpt-4.1")

    def test_news_config_normalizes_and_clamps(self) -> None:
        config_path = self._write_config(
            """
            news:
              provider: " GOOGLE_RSS "
              cache_seconds: -5
              request_timeout_seconds: 0
              max_symbol_headlines: 0
              max_market_headlines: -3
              market_queries: ["stock market", " STOCK MARKET ", "", "inflation report"]
            """
        )

        with mock.patch("bot.config.load_dotenv", return_value=False), mock.patch.dict(
            os.environ, {}, clear=True
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.news.provider, "google_rss")
        self.assertEqual(cfg.news.cache_seconds, 0)
        self.assertEqual(cfg.news.request_timeout_seconds, 1)
        self.assertEqual(cfg.news.max_symbol_headlines, 1)
        self.assertEqual(cfg.news.max_market_headlines, 1)
        self.assertEqual(cfg.news.market_queries, ["stock market", "inflation report"])

    def test_news_llm_capability_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            news:
              llm_reasoning_effort: turbo
              llm_text_verbosity: verbose
              llm_max_output_tokens: 0
              llm_chat_fallback_model: ""
            """
        )

        with mock.patch("bot.config.load_dotenv", return_value=False), mock.patch.dict(
            os.environ, {}, clear=True
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.news.llm_reasoning_effort, "medium")
        self.assertEqual(cfg.news.llm_text_verbosity, "low")
        self.assertEqual(cfg.news.llm_max_output_tokens, 64)
        self.assertEqual(cfg.news.llm_chat_fallback_model, "gpt-4.1")

    def test_execution_and_alert_env_overrides(self) -> None:
        config_path = self._write_config(
            """
            execution:
              stale_order_minutes: 5
            alerts:
              min_level: warning
            """
        )

        with mock.patch("bot.config.load_dotenv", return_value=False), mock.patch.dict(
            os.environ,
            {
                "EXECUTION_STALE_ORDER_MINUTES": "0",
                "EXECUTION_CANCEL_STALE_ORDERS": "false",
                "ALERTS_ENABLED": "true",
                "ALERTS_MIN_LEVEL": "critical",
                "ALERTS_TIMEOUT_SECONDS": "0",
                "ALERTS_REQUIRE_IN_LIVE": "false",
                "LOG_MAX_BYTES": "512",
                "LOG_BACKUP_COUNT": "0",
            },
            clear=True,
        ):
            cfg = load_config(config_path)

        self.assertFalse(cfg.execution.cancel_stale_orders)
        self.assertEqual(cfg.execution.stale_order_minutes, 1)
        self.assertTrue(cfg.alerts.enabled)
        self.assertEqual(cfg.alerts.min_level, "CRITICAL")
        self.assertEqual(cfg.alerts.timeout_seconds, 1)
        self.assertFalse(cfg.alerts.require_in_live)
        self.assertEqual(cfg.log_max_bytes, 1024)
        self.assertEqual(cfg.log_backup_count, 1)

    def test_execution_smart_ladder_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            execution:
              smart_ladder_enabled: true
              ladder_width_fractions: [0.0, 0.1, 5.0, -1.0]
              ladder_step_timeouts_seconds: [45, "x", 3, 30]
              max_ladder_attempts: 10
            """
        )
        with mock.patch("bot.config.load_dotenv", return_value=False), mock.patch.dict(
            os.environ, {}, clear=True
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.execution.smart_ladder_enabled)
        self.assertEqual(cfg.execution.ladder_width_fractions, [0.0, 0.1, 1.0, 0.0])
        self.assertEqual(cfg.execution.ladder_step_timeouts_seconds, [45, 5, 30])
        self.assertEqual(cfg.execution.max_ladder_attempts, 10)

    def test_signal_ranking_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            signal_ranking:
              enabled: true
              weight_score: 3
              weight_pop: -1
              weight_credit: 0.2
              weight_vol_premium: 0.1
              top_ranked_to_log: 0
            max_signals_per_symbol_per_strategy: 0
            """
        )
        with mock.patch("bot.config.load_dotenv", return_value=False), mock.patch.dict(
            os.environ, {}, clear=True
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.signal_ranking.enabled)
        self.assertEqual(cfg.signal_ranking.weight_score, 1.0)
        self.assertEqual(cfg.signal_ranking.weight_pop, 0.0)
        self.assertEqual(cfg.signal_ranking.weight_credit, 0.2)
        self.assertEqual(cfg.signal_ranking.weight_vol_premium, 0.1)
        self.assertEqual(cfg.signal_ranking.top_ranked_to_log, 1)
        self.assertEqual(cfg.max_signals_per_symbol_per_strategy, 1)

    def test_sizing_equity_curve_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            sizing:
              equity_curve_scaling: true
              equity_curve_lookback: 0
              max_scale_up: 0.2
              max_scale_down: 2.0
            """
        )
        with mock.patch("bot.config.load_dotenv", return_value=False), mock.patch.dict(
            os.environ, {}, clear=True
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.sizing.equity_curve_scaling)
        self.assertEqual(cfg.sizing.equity_curve_lookback, 5)
        self.assertEqual(cfg.sizing.max_scale_up, 1.0)
        self.assertEqual(cfg.sizing.max_scale_down, 1.0)

    def test_risk_correlated_and_gamma_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            risk:
              correlated_loss_threshold: 0
              correlated_loss_pct: 2.0
              correlated_loss_cooldown_hours: 0
              gamma_week_tight_stop: 0.5
              expiration_day_close_pct: 0.8
            """
        )
        with mock.patch("bot.config.load_dotenv", return_value=False), mock.patch.dict(
            os.environ, {}, clear=True
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.risk.correlated_loss_threshold, 1)
        self.assertEqual(cfg.risk.correlated_loss_pct, 1.0)
        self.assertEqual(cfg.risk.correlated_loss_cooldown_hours, 1)
        self.assertEqual(cfg.risk.gamma_week_tight_stop, 1.0)
        self.assertEqual(cfg.risk.expiration_day_close_pct, 0.25)

    def test_placeholder_live_secrets_are_treated_as_missing(self) -> None:
        config_path = self._write_config(
            """
            trading_mode: live
            """
        )

        with mock.patch("bot.config.load_dotenv", return_value=False), mock.patch.dict(
            os.environ,
            {
                "SCHWAB_APP_KEY": "your_app_key_here",
                "SCHWAB_APP_SECRET": "your_app_secret_here",
                "SCHWAB_ACCOUNT_HASH": "your_account_hash_here",
            },
            clear=True,
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.schwab.app_key, "")
        self.assertEqual(cfg.schwab.app_secret, "")
        self.assertEqual(cfg.schwab.account_hash, "")

    def test_validation_rejects_portfolio_risk_below_position_risk(self) -> None:
        cfg = BotConfig()
        cfg.risk.max_portfolio_risk_pct = 1.0
        cfg.risk.max_position_risk_pct = 2.0
        report = validate_config(cfg)

        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("max_portfolio_risk_pct" in msg for msg in report.failed)
        )

    def test_validation_rejects_kelly_fraction_out_of_bounds(self) -> None:
        cfg = BotConfig()
        cfg.sizing.method = "kelly"
        cfg.sizing.kelly_fraction = 0.05
        report = validate_config(cfg)

        self.assertFalse(report.is_valid)
        self.assertTrue(any("kelly_fraction" in msg for msg in report.failed))

    def test_validation_rejects_invalid_ensemble_model_pair(self) -> None:
        cfg = BotConfig()
        cfg.llm.ensemble_models = ["openai-gpt-5.2-pro", "anthropic:"]
        report = validate_config(cfg)

        self.assertFalse(report.is_valid)
        self.assertTrue(any("ensemble_models" in msg for msg in report.failed))

    def test_validation_rejects_hedging_budget_above_portfolio_risk(self) -> None:
        cfg = BotConfig()
        cfg.hedging.enabled = True
        cfg.hedging.max_hedge_cost_pct = 8.0
        cfg.risk.max_portfolio_risk_pct = 5.0
        report = validate_config(cfg)

        self.assertFalse(report.is_valid)
        self.assertTrue(any("hedging.max_hedge_cost_pct" in msg for msg in report.failed))

    def test_validation_rejects_overlapping_dte_windows_when_multiple_symbol_positions_allowed(self) -> None:
        cfg = BotConfig()
        cfg.credit_spreads.enabled = True
        cfg.iron_condors.enabled = True
        cfg.credit_spreads.min_dte = 20
        cfg.credit_spreads.max_dte = 45
        cfg.iron_condors.min_dte = 25
        cfg.iron_condors.max_dte = 50
        cfg.risk.max_positions_per_symbol = 2
        report = validate_config(cfg)

        self.assertFalse(report.is_valid)
        self.assertTrue(any("Overlapping strategy DTE windows" in msg for msg in report.failed))

    def test_validation_formats_human_readable_report(self) -> None:
        cfg = BotConfig()
        cfg.credit_spreads.enabled = True
        cfg.iron_condors.enabled = True
        cfg.risk.max_positions_per_symbol = 2
        report = validate_config(cfg)
        output = format_validation_report(report)

        self.assertIn("Configuration validation report", output)
        self.assertIn("Failures:", output)


if __name__ == "__main__":
    unittest.main()
