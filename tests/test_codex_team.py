import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_codex_team_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "codex_team.py"
    spec = importlib.util.spec_from_file_location("codex_team_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/codex_team.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


codex_team = _load_codex_team_module()


class CodexTeamTests(unittest.TestCase):
    def test_parse_roles_rejects_duplicate_role(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            codex_team._parse_roles("architect,reviewer,architect")

        self.assertIn("Duplicate role", str(ctx.exception))

    def test_specialist_prompt_avoids_recursive_trigger_phrase(self) -> None:
        prompt = codex_team._build_specialist_prompt(
            "agent team: debug this project",
            codex_team.SPECIALIST_ROLES["architect"],
            "evidence",
        )

        self.assertNotIn("agent team", prompt.lower())
        self.assertIn("agent-team", prompt.lower())

    def test_collect_specialist_reports_flags_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ok_report = root / "architect.md"
            empty_report = root / "reviewer.md"
            ok_report.write_text("## Findings\n- good\n", encoding="utf-8")
            empty_report.write_text("", encoding="utf-8")

            results = [
                codex_team.SpecialistResult(
                    role_name="architect",
                    report_path=ok_report,
                    exit_code=0,
                    detail="",
                    is_dry_run=False,
                ),
                codex_team.SpecialistResult(
                    role_name="reviewer",
                    report_path=empty_report,
                    exit_code=0,
                    detail="",
                    is_dry_run=False,
                ),
                codex_team.SpecialistResult(
                    role_name="tester",
                    report_path=root / "tester.md",
                    exit_code=3,
                    detail="boom",
                    is_dry_run=False,
                ),
            ]

            reports, failures = codex_team._collect_specialist_reports(results)

        self.assertIn("architect", reports)
        self.assertIn("reviewer", reports)
        self.assertIn("tester", reports)
        self.assertIn("reviewer: empty report", failures)
        self.assertIn("tester: exit code 3", failures)
        self.assertIn("tester: empty report", failures)

    @mock.patch.object(codex_team, "_terminate_pid", return_value=True)
    @mock.patch.object(codex_team, "_launch_terminal_window")
    @mock.patch.object(codex_team.time, "sleep", return_value=None)
    @mock.patch.object(codex_team.time, "monotonic")
    def test_terminal_timeout_attempts_pid_cleanup(
        self,
        monotonic_mock: mock.Mock,
        _sleep_mock: mock.Mock,
        launch_mock: mock.Mock,
        terminate_mock: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "architect.md"
            pid_file = output_path.with_suffix(output_path.suffix + ".pid")

            def _fake_launch(_command: str) -> tuple[bool, str]:
                pid_file.write_text("4242", encoding="utf-8")
                return (True, "")

            launch_mock.side_effect = _fake_launch
            monotonic_mock.side_effect = [0.0, 0.2, 1.2]

            exit_code, detail, is_dry_run = codex_team._run_codex(
                prompt="debug prompt",
                cwd=Path(tmp),
                output_file=output_path,
                sandbox="read-only",
                model=None,
                dry_run=False,
                timeout_seconds=1,
                run_in_terminal_window=True,
            )

        self.assertEqual(exit_code, 124)
        self.assertFalse(is_dry_run)
        self.assertIn("terminated pid=4242", detail)
        terminate_mock.assert_called_once_with(4242)


if __name__ == "__main__":
    unittest.main()
