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
        self.addCleanup(
            lambda: os.remove(tmp.name) if os.path.exists(tmp.name) else None
        )
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

        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
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

        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.trading_mode, "paper")
        self.assertEqual(cfg.llm.provider, "google")
        self.assertEqual(cfg.llm.mode, "advisory")
        self.assertEqual(cfg.llm.risk_style, "moderate")
        self.assertEqual(cfg.log_level, "INFO")

    def test_load_config_normalizes_google_provider_alias(self) -> None:
        config_path = self._write_config(
            """
            llm:
              provider: " GEMINI "
              model: gpt-5.2-pro
            llm_strategist:
              provider: " GOOGLE "
              model: gpt-5.2-pro
            """
        )

        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.llm.provider, "google")
        self.assertEqual(cfg.llm.model, "gemini-3.1-pro-thinking-preview")
        self.assertEqual(cfg.llm_strategist.provider, "google")
        self.assertEqual(cfg.llm_strategist.model, "gemini-3.1-pro-thinking-preview")

    def test_load_config_enforces_google_gemini_model_pair(self) -> None:
        config_path = self._write_config(
            """
            llm:
              provider: google
              model: gemini-3.1-flash-thinking-preview
              chat_fallback_model: gpt-4.1
            llm_strategist:
              provider: google
              model: gemini-3.1-flash-thinking-preview
            """
        )

        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.llm.model, "gemini-3.1-pro-thinking-preview")
        self.assertEqual(
            cfg.llm.chat_fallback_model, "gemini-3.1-flash-thinking-preview"
        )
        self.assertEqual(cfg.llm_strategist.model, "gemini-3.1-pro-thinking-preview")

    def test_env_overrides_clamp_llm_numeric_settings(self) -> None:
        config_path = self._write_config("llm:\n  enabled: true\n")

        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(
                os.environ,
                {
                    "LLM_TIMEOUT_SECONDS": "0",
                    "LLM_TEMPERATURE": "3.5",
                    "LLM_MIN_CONFIDENCE": "-0.2",
                    "LLM_MAX_OUTPUT_TOKENS": "0",
                },
                clear=True,
            ),
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.llm.timeout_seconds, 1)
        self.assertEqual(cfg.llm.temperature, 2.0)
        self.assertEqual(cfg.llm.min_confidence, 0.0)
        self.assertEqual(cfg.llm.max_output_tokens, 64)

    def test_env_overrides_llm_ensemble_models(self) -> None:
        config_path = self._write_config("llm:\n  enabled: true\n")
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(
                os.environ,
                {
                    "LLM_ENSEMBLE_MODELS": "openai:gpt-5.2-pro, anthropic:claude-sonnet-4-20250514 ,ollama:llama3.1",
                },
                clear=True,
            ),
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

        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.llm.reasoning_effort, "xhigh")
        self.assertEqual(cfg.llm.text_verbosity, "low")
        self.assertEqual(cfg.llm.chat_fallback_model, "gemini-3.1-flash-thinking-preview")

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

        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
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

        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.news.llm_reasoning_effort, "xhigh")
        self.assertEqual(cfg.news.llm_text_verbosity, "low")
        self.assertEqual(cfg.news.llm_max_output_tokens, 64)
        self.assertEqual(cfg.news.llm_chat_fallback_model, "gemini-3.1-flash-thinking-preview")

    def test_execution_and_alert_env_overrides(self) -> None:
        config_path = self._write_config(
            """
            execution:
              stale_order_minutes: 5
            alerts:
              min_level: warning
            """
        )

        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(
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
            ),
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

    def test_terminal_ui_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            terminal_ui:
              enabled: true
              refresh_rate: 0
              max_activity_events: 1
              show_rejected_trades: false
              compact_mode: true
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.terminal_ui.enabled)
        self.assertEqual(cfg.terminal_ui.refresh_rate, 0.1)
        self.assertEqual(cfg.terminal_ui.max_activity_events, 10)
        self.assertFalse(cfg.terminal_ui.show_rejected_trades)
        self.assertTrue(cfg.terminal_ui.compact_mode)

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
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.execution.smart_ladder_enabled)
        self.assertEqual(cfg.execution.ladder_width_fractions, [0.0, 0.1, 1.0, 0.0])
        self.assertEqual(cfg.execution.ladder_step_timeouts_seconds, [45, 5, 30])
        self.assertEqual(cfg.execution.max_ladder_attempts, 10)

    def test_execution_partial_fill_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            execution:
              partial_fill_handling: true
              market_move_cancel_pct: -1.0
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.execution.partial_fill_handling)
        self.assertEqual(cfg.execution.market_move_cancel_pct, 0.0)

    def test_signal_ranking_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            signal_ranking:
              enabled: true
              weight_score: 3
              weight_pop: -1
              weight_credit: 0.2
              weight_vol_premium: 0.1
              ml_score_weight: 4
              top_ranked_to_log: 0
            max_signals_per_symbol_per_strategy: 0
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.signal_ranking.enabled)
        self.assertEqual(cfg.signal_ranking.weight_score, 1.0)
        self.assertEqual(cfg.signal_ranking.weight_pop, 0.0)
        self.assertEqual(cfg.signal_ranking.weight_credit, 0.2)
        self.assertEqual(cfg.signal_ranking.weight_vol_premium, 0.1)
        self.assertEqual(cfg.signal_ranking.ml_score_weight, 1.0)
        self.assertEqual(cfg.signal_ranking.top_ranked_to_log, 1)
        self.assertEqual(cfg.max_signals_per_symbol_per_strategy, 1)

    def test_ml_scorer_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            ml_scorer:
              enabled: true
              min_training_trades: 0
              retrain_day: " FUNDAY "
              retrain_time: ""
              feature_importance_log: false
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.ml_scorer.enabled)
        self.assertEqual(cfg.ml_scorer.min_training_trades, 1)
        self.assertEqual(cfg.ml_scorer.retrain_day, "sunday")
        self.assertEqual(cfg.ml_scorer.retrain_time, "18:00")
        self.assertFalse(cfg.ml_scorer.feature_importance_log)

    def test_theta_harvest_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            theta_harvest:
              enabled: true
              satisfied_pct: 2.0
              underperform_pct: -2.0
              cutoff_hour: 30
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.theta_harvest.enabled)
        self.assertEqual(cfg.theta_harvest.satisfied_pct, 1.0)
        self.assertEqual(cfg.theta_harvest.underperform_pct, 0.0)
        self.assertEqual(cfg.theta_harvest.cutoff_hour, 23)

    def test_correlation_monitor_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            correlation_monitor:
              enabled: true
              lookback_days: 1
              crisis_threshold: 2.0
              stress_threshold: -2.0
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.correlation_monitor.enabled)
        self.assertEqual(cfg.correlation_monitor.lookback_days, 10)
        self.assertEqual(cfg.correlation_monitor.crisis_threshold, 1.0)
        self.assertEqual(cfg.correlation_monitor.stress_threshold, 0.0)

    def test_execution_timing_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            execution_timing:
              adaptive: true
              min_fills_per_bucket: 0
              analysis_file: ""
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.execution_timing.adaptive)
        self.assertEqual(cfg.execution_timing.min_fills_per_bucket, 1)
        self.assertEqual(
            cfg.execution_timing.analysis_file,
            "bot/data/execution_timing_analysis.json",
        )

    def test_pillar_two_to_five_config_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            strategies:
              sandbox:
                name: sandbox_alpha
                enabled: true
            rl_prompt_optimizer:
              enabled: true
              min_trades_for_pattern: 1
              loss_rate_threshold: 0.1
              max_rules: 0
              rolling_window_size: 5
            alt_data:
              gex_enabled: true
              dark_pool_proxy_enabled: true
              social_sentiment_enabled: true
              social_sentiment_cache_minutes: 0
              social_sentiment_model: ""
            execution_algos:
              enabled: true
              algo_type: turbo
              twap_slices: 0
              twap_window_seconds: 0
              iceberg_visible_qty: 0
              adaptive_spread_pause_threshold: 0.1
              adaptive_spread_accelerate_threshold: 9
            strategy_sandbox:
              enabled: true
              min_failing_score: -2
              consecutive_fail_cycles: 0
              backtest_days: 0
              max_drawdown_pct: 0
              deployment_days: 0
              sizing_scalar: 9
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.rl_prompt_optimizer.enabled)
        self.assertEqual(cfg.rl_prompt_optimizer.min_trades_for_pattern, 3)
        self.assertEqual(cfg.rl_prompt_optimizer.loss_rate_threshold, 0.5)
        self.assertEqual(cfg.rl_prompt_optimizer.max_rules, 1)
        self.assertEqual(cfg.rl_prompt_optimizer.rolling_window_size, 20)

        self.assertEqual(cfg.alt_data.social_sentiment_cache_minutes, 1)
        self.assertEqual(cfg.alt_data.social_sentiment_model, "gemini-3.1-pro-thinking-preview")

        self.assertTrue(cfg.execution_algos.enabled)
        self.assertEqual(cfg.execution_algos.algo_type, "smart_ladder")
        self.assertEqual(cfg.execution_algos.twap_slices, 1)
        self.assertEqual(cfg.execution_algos.twap_window_seconds, 10)
        self.assertEqual(cfg.execution_algos.iceberg_visible_qty, 1)
        self.assertEqual(cfg.execution_algos.adaptive_spread_pause_threshold, 1.0)
        self.assertEqual(cfg.execution_algos.adaptive_spread_accelerate_threshold, 1.0)

        self.assertTrue(cfg.strategy_sandbox.enabled)
        self.assertEqual(cfg.strategy_sandbox.min_failing_score, 0.0)
        self.assertEqual(cfg.strategy_sandbox.consecutive_fail_cycles, 1)
        self.assertEqual(cfg.strategy_sandbox.backtest_days, 5)
        self.assertEqual(cfg.strategy_sandbox.max_drawdown_pct, 1.0)
        self.assertEqual(cfg.strategy_sandbox.deployment_days, 1)
        self.assertEqual(cfg.strategy_sandbox.sizing_scalar, 1.0)
        self.assertEqual(
            cfg.strategy_sandbox.active_strategy.get("name"), "sandbox_alpha"
        )

    def test_scaling_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            scaling:
              enabled: true
              scale_in_delay_minutes: 0
              scale_in_max_adds: -2
              partial_exit_pct: 2.0
              partial_exit_size: -1.0
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.scaling.enabled)
        self.assertEqual(cfg.scaling.scale_in_delay_minutes, 1)
        self.assertEqual(cfg.scaling.scale_in_max_adds, 0)
        self.assertEqual(cfg.scaling.partial_exit_pct, 1.0)
        self.assertEqual(cfg.scaling.partial_exit_size, 0.0)

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
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.sizing.equity_curve_scaling)
        self.assertEqual(cfg.sizing.equity_curve_lookback, 5)
        self.assertEqual(cfg.sizing.max_scale_up, 1.0)
        self.assertEqual(cfg.sizing.max_scale_down, 1.0)

    def test_sizing_high_conviction_boost_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            sizing:
              high_conviction_size_boost_enabled: true
              high_conviction_size_boost_multiplier: 0.2
              high_conviction_size_boost_min_score: -10
              high_conviction_size_boost_min_pop: 2.0
              high_conviction_size_boost_min_ml_score: -3.0
              high_conviction_size_boost_min_composite: -1
              high_conviction_size_boost_max_extra_contracts: -2
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.sizing.high_conviction_size_boost_enabled)
        self.assertEqual(cfg.sizing.high_conviction_size_boost_multiplier, 1.0)
        self.assertEqual(cfg.sizing.high_conviction_size_boost_min_score, 0.0)
        self.assertEqual(cfg.sizing.high_conviction_size_boost_min_pop, 1.0)
        self.assertEqual(cfg.sizing.high_conviction_size_boost_min_ml_score, 0.0)
        self.assertEqual(cfg.sizing.high_conviction_size_boost_min_composite, 0.0)
        self.assertEqual(cfg.sizing.high_conviction_size_boost_max_extra_contracts, 0)

    def test_walkforward_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            walkforward:
              enabled: true
              train_days: 0
              test_days: 0
              step_days: 0
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.walkforward.enabled)
        self.assertEqual(cfg.walkforward.train_days, 10)
        self.assertEqual(cfg.walkforward.test_days, 5)
        self.assertEqual(cfg.walkforward.step_days, 1)

    def test_monte_carlo_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            monte_carlo:
              enabled: true
              simulations: 1
              var_limit_pct: -2
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.monte_carlo.enabled)
        self.assertEqual(cfg.monte_carlo.simulations, 100)
        self.assertEqual(cfg.monte_carlo.var_limit_pct, 0.0)

    def test_reconciliation_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            reconciliation:
              interval_minutes: 0
              auto_import: false
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.reconciliation.interval_minutes, 5)
        self.assertFalse(cfg.reconciliation.auto_import)

    def test_multi_timeframe_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            multi_timeframe:
              enabled: true
              min_agreement: 9
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.multi_timeframe.enabled)
        self.assertEqual(cfg.multi_timeframe.min_agreement, 3)

    def test_cooldown_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            cooldown:
              graduated: true
              level_1_losses: 0
              level_1_reduction: -2
              level_2_losses: 1
              level_2_reduction: 2
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.cooldown.level_1_losses, 1)
        self.assertEqual(cfg.cooldown.level_1_reduction, 0.0)
        self.assertEqual(cfg.cooldown.level_2_losses, 1)
        self.assertEqual(cfg.cooldown.level_2_reduction, 0.95)

    def test_strategy_allocation_fields_normalize(self) -> None:
        config_path = self._write_config(
            """
            strategy_allocation:
              enabled: true
              lookback_trades: 0
              min_sharpe_for_boost: 2.1
              cold_start_penalty: 9.0
              cold_start_window_days: 0
              cold_start_min_trades: 0
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertTrue(cfg.strategy_allocation.enabled)
        self.assertEqual(cfg.strategy_allocation.lookback_trades, 5)
        self.assertEqual(cfg.strategy_allocation.min_sharpe_for_boost, 2.1)
        self.assertEqual(cfg.strategy_allocation.cold_start_penalty, 1.0)
        self.assertEqual(cfg.strategy_allocation.cold_start_window_days, 7)
        self.assertEqual(cfg.strategy_allocation.cold_start_min_trades, 1)

    def test_greeks_budget_limits_normalize(self) -> None:
        config_path = self._write_config(
            """
            greeks_budget:
              enabled: true
              reduce_size_to_fit: true
              limits:
                crash:
                  delta_min: 5
                  delta_max: -5
                  vega_min: "bad"
                  vega_max: 100
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        crash = cfg.greeks_budget.limits["CRASH/CRISIS"]
        self.assertEqual(crash["delta_min"], -5.0)
        self.assertEqual(crash["delta_max"], 5.0)
        self.assertEqual(crash["vega_min"], 0.0)
        self.assertEqual(crash["vega_max"], 100.0)

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
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.risk.correlated_loss_threshold, 1)
        self.assertEqual(cfg.risk.correlated_loss_pct, 1.0)
        self.assertEqual(cfg.risk.correlated_loss_cooldown_hours, 1)
        self.assertEqual(cfg.risk.gamma_week_tight_stop, 1.0)
        self.assertEqual(cfg.risk.expiration_day_close_pct, 0.25)

    def test_risk_quality_and_contract_caps_normalize(self) -> None:
        config_path = self._write_config(
            """
            risk:
              min_trade_score: -15
              min_trade_pop: 2.0
              max_contracts_per_trade: 0
            """
        )
        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.risk.min_trade_score, 0.0)
        self.assertEqual(cfg.risk.min_trade_pop, 1.0)
        self.assertEqual(cfg.risk.max_contracts_per_trade, 1)

    def test_placeholder_live_secrets_are_treated_as_missing(self) -> None:
        config_path = self._write_config(
            """
            trading_mode: live
            """
        )

        with (
            mock.patch("bot.config.load_dotenv", return_value=False),
            mock.patch.dict(
                os.environ,
                {
                    "SCHWAB_APP_KEY": "your_app_key_here",
                    "SCHWAB_APP_SECRET": "your_app_secret_here",
                    "SCHWAB_ACCOUNT_HASH": "your_account_hash_here",
                },
                clear=True,
            ),
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
        self.assertTrue(any("max_portfolio_risk_pct" in msg for msg in report.failed))

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

    def test_validation_accepts_google_ensemble_model_pair(self) -> None:
        cfg = BotConfig()
        cfg.llm.ensemble_models = ["google:gemini-3.1-pro-thinking-preview"]
        report = validate_config(cfg)

        self.assertTrue(report.is_valid)

    def test_validation_rejects_hedging_budget_above_portfolio_risk(self) -> None:
        cfg = BotConfig()
        cfg.hedging.enabled = True
        cfg.hedging.max_hedge_cost_pct = 8.0
        cfg.risk.max_portfolio_risk_pct = 5.0
        report = validate_config(cfg)

        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("hedging.max_hedge_cost_pct" in msg for msg in report.failed)
        )

    def test_validation_allows_overlapping_dte_windows_when_multiple_symbol_positions_allowed(
        self,
    ) -> None:
        cfg = BotConfig()
        cfg.credit_spreads.enabled = True
        cfg.iron_condors.enabled = True
        cfg.credit_spreads.min_dte = 20
        cfg.credit_spreads.max_dte = 45
        cfg.iron_condors.min_dte = 25
        cfg.iron_condors.max_dte = 50
        cfg.risk.max_positions_per_symbol = 2
        report = validate_config(cfg)

        self.assertTrue(report.is_valid)
        self.assertTrue(
            any("strategy DTE overlap check" in msg for msg in report.passed)
        )

    def test_validation_formats_human_readable_report(self) -> None:
        cfg = BotConfig()
        cfg.credit_spreads.enabled = True
        cfg.iron_condors.enabled = True
        cfg.risk.max_positions_per_symbol = 1
        report = validate_config(cfg)
        output = format_validation_report(report)

        self.assertIn("Configuration validation report", output)
        self.assertIn("Warnings:", output)


if __name__ == "__main__":
    unittest.main()
