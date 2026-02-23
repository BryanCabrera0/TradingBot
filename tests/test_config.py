import os
import tempfile
import textwrap
import unittest
from unittest import mock

from bot.config import load_config


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

        with mock.patch.dict(os.environ, {}, clear=True):
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

        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = load_config(config_path)

        self.assertEqual(cfg.trading_mode, "paper")
        self.assertEqual(cfg.llm.provider, "ollama")
        self.assertEqual(cfg.llm.mode, "advisory")
        self.assertEqual(cfg.llm.risk_style, "moderate")
        self.assertEqual(cfg.log_level, "INFO")

    def test_env_overrides_clamp_llm_numeric_settings(self) -> None:
        config_path = self._write_config("llm:\n  enabled: true\n")

        with mock.patch.dict(
            os.environ,
            {
                "LLM_TIMEOUT_SECONDS": "0",
                "LLM_TEMPERATURE": "3.5",
                "LLM_MIN_CONFIDENCE": "-0.2",
            },
            clear=True,
        ):
            cfg = load_config(config_path)

        self.assertEqual(cfg.llm.timeout_seconds, 1)
        self.assertEqual(cfg.llm.temperature, 2.0)
        self.assertEqual(cfg.llm.min_confidence, 0.0)

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

        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = load_config(config_path)

        self.assertEqual(cfg.news.provider, "google_rss")
        self.assertEqual(cfg.news.cache_seconds, 0)
        self.assertEqual(cfg.news.request_timeout_seconds, 1)
        self.assertEqual(cfg.news.max_symbol_headlines, 1)
        self.assertEqual(cfg.news.max_market_headlines, 1)
        self.assertEqual(cfg.news.market_queries, ["stock market", "inflation report"])

    def test_execution_and_alert_env_overrides(self) -> None:
        config_path = self._write_config(
            """
            execution:
              stale_order_minutes: 5
            alerts:
              min_level: warning
            """
        )

        with mock.patch.dict(
            os.environ,
            {
                "EXECUTION_STALE_ORDER_MINUTES": "0",
                "EXECUTION_CANCEL_STALE_ORDERS": "false",
                "ALERTS_ENABLED": "true",
                "ALERTS_MIN_LEVEL": "critical",
                "ALERTS_TIMEOUT_SECONDS": "0",
            },
            clear=True,
        ):
            cfg = load_config(config_path)

        self.assertFalse(cfg.execution.cancel_stale_orders)
        self.assertEqual(cfg.execution.stale_order_minutes, 1)
        self.assertTrue(cfg.alerts.enabled)
        self.assertEqual(cfg.alerts.min_level, "CRITICAL")
        self.assertEqual(cfg.alerts.timeout_seconds, 1)


if __name__ == "__main__":
    unittest.main()
