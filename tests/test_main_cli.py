import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import main


class MainCliParserTests(unittest.TestCase):
    def test_run_paper_mode_is_parseable(self) -> None:
        parser = main._build_parser()
        args = parser.parse_args(["run", "paper"])

        self.assertEqual(args.command, "run")
        self.assertEqual(args.mode, "paper")

    def test_run_live_mode_is_parseable(self) -> None:
        parser = main._build_parser()
        args = parser.parse_args(["run", "live"])

        self.assertEqual(args.command, "run")
        self.assertEqual(args.mode, "live")

    def test_run_command_defaults_to_paper_mode(self) -> None:
        with mock.patch("sys.argv", ["main.py", "run"]):
            args, interactive_menu_requested = main._parse_args()
        self.assertFalse(interactive_menu_requested)
        self.assertEqual(args.command, "run")
        self.assertEqual(args.mode, "paper")

    def test_run_simulator_mode_is_rejected(self) -> None:
        parser = main._build_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["run", "simulator"])

        self.assertEqual(ctx.exception.code, 2)

    def test_parse_args_non_interactive_defaults_to_paper(self) -> None:
        with (
            mock.patch("sys.argv", ["main.py"]),
            mock.patch("sys.stdin.isatty", return_value=False),
        ):
            args, interactive_menu_requested = main._parse_args()

        self.assertTrue(interactive_menu_requested)
        self.assertEqual(args.command, "run")
        self.assertEqual(args.mode, "paper")
        self.assertFalse(args.live)
        self.assertFalse(args.once)

    def test_prompt_run_menu_shows_only_paper_and_live(self) -> None:
        out = io.StringIO()
        with (
            mock.patch("builtins.input", return_value="1"),
            redirect_stdout(out),
        ):
            mode, once = main.prompt_run_menu()

        rendered = out.getvalue()
        self.assertEqual((mode, once), ("paper", False))
        self.assertIn("1) Paper Trading (Live Market Data)", rendered)
        self.assertIn("2) Live Trading (Real Money)", rendered)
        self.assertNotIn("Simulator", rendered)
        self.assertNotIn("Bootstrap", rendered)


if __name__ == "__main__":
    unittest.main()
