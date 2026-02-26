import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bot.config import BotConfig
from bot.orchestrator import TradingBot


class AuditLogTests(unittest.TestCase):
    def test_append_audit_event_writes_jsonl_record(self) -> None:
        cfg = BotConfig()
        cfg.scanner.enabled = False
        cfg.news.enabled = False
        cfg.llm.enabled = False
        bot = TradingBot(cfg)

        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_path = Path(tmp_dir) / "audit_log.jsonl"
            with mock.patch("bot.orchestrator.AUDIT_LOG_PATH", audit_path):
                bot._append_audit_event(
                    event_type="signal_generated",
                    details={"symbol": "SPY", "strategy": "bull_put_spread"},
                    correlation_id="corr-123",
                )

            lines = audit_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            row = json.loads(lines[0])
            self.assertEqual(row["event_type"], "signal_generated")
            self.assertEqual(row["correlation_id"], "corr-123")
            self.assertEqual(row["details"]["symbol"], "SPY")


if __name__ == "__main__":
    unittest.main()
