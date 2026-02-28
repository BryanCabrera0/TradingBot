import tempfile
import unittest
from pathlib import Path

from dotenv import dotenv_values

from bot.live_setup import _ensure_optional_runtime_defaults, _render_service_templates


class LiveSetupTests(unittest.TestCase):
    def test_ensure_optional_runtime_defaults_disables_optional_integrations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "LLM_ENABLED=true",
                        "LLM_PROVIDER=google",
                        "GOOGLE_API_KEY=",
                        "ALERTS_REQUIRE_IN_LIVE=true",
                        "ALERTS_WEBHOOK_URL=",
                    ]
                ),
                encoding="utf-8",
            )

            _ensure_optional_runtime_defaults(env_path)
            values = dotenv_values(env_path)

            self.assertEqual(str(values.get("LLM_ENABLED")).lower(), "false")
            self.assertEqual(
                str(values.get("ALERTS_REQUIRE_IN_LIVE")).lower(),
                "false",
            )

    def test_render_service_templates_replaces_project_root_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "deploy/systemd").mkdir(parents=True, exist_ok=True)
            (root / "deploy/launchd").mkdir(parents=True, exist_ok=True)

            (root / "deploy/systemd/tradingbot.service").write_text(
                "WorkingDirectory=__PROJECT_ROOT__\n",
                encoding="utf-8",
            )
            (root / "deploy/launchd/com.bryan.tradingbot.plist").write_text(
                "<string>__PROJECT_ROOT__/main.py</string>\n",
                encoding="utf-8",
            )

            generated = _render_service_templates(root)
            self.assertEqual(len(generated), 2)

            systemd_out = (
                root / "deploy/generated/systemd/tradingbot.service"
            ).read_text(encoding="utf-8")
            launchd_out = (
                root / "deploy/generated/launchd/com.bryan.tradingbot.plist"
            ).read_text(encoding="utf-8")

            self.assertIn(str(root), systemd_out)
            self.assertIn(str(root), launchd_out)


if __name__ == "__main__":
    unittest.main()
