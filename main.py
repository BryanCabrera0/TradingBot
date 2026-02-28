#!/usr/bin/env python3
"""
TradingBot — Fully Automated Options Trading Bot
=================================================

Connects to Charles Schwab's API (via ThinkorSwim account) and
automatically trades options strategies including:
  - Bull put spreads / Bear call spreads (credit spreads)
  - Iron condors
  - Covered calls
  - Naked puts (cash-secured)
  - Calendar spreads
  - Short strangles / index straddles
  - Broken-wing butterflies
  - Earnings volatility-crush plays

Usage:
    # With no command, an interactive menu appears to pick a run mode
    python3 main.py
    python3 main.py run simulator

    # Run the offline training simulator
    python3 main.py run simulator --iterations 100

    # Run continuous live mode (non-interactive confirmation bypass shown)
    python3 main.py run live --yes

    # Run one live scan cycle and exit
    python3 main.py run live once --yes

    # Integrated diagnostics (auth + SPY chain + entry-gate snapshot)
    python3 main.py run simulator once --diagnose
"""

import argparse
import copy
import json
import logging
import select
import sys
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from bot.config import format_validation_report, load_config, validate_config
from bot.file_security import validate_sensitive_file
from bot.number_utils import safe_float

if TYPE_CHECKING:
    from bot.orchestrator import TradingBot


def setup_logging(
    level: str = "INFO",
    console_level: Optional[str] = None,
    log_file: str = "logs/tradingbot.log",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> None:
    """Configure logging to both console and file."""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, level.upper(), logging.INFO)
    resolved_console_level = getattr(
        logging,
        (console_level or level).upper(),
        log_level,
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(resolved_console_level)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console.setFormatter(console_fmt)
    root_logger.addHandler(console)

    # Rotating file handler for long-running process safety.
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max(1024, int(max_bytes)),
            backupCount=max(1, int(backup_count)),
        )
        file_handler.setLevel(log_level)
        file_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        root_logger.addHandler(file_handler)
    except PermissionError:
        root_logger.warning(
            f"Permission denied writing to {log_file}, skipping file logging."
        )


def print_banner() -> None:
    print("\n  TradingBot")
    print("  Fully Automated · Risk Managed\n")


def show_report() -> None:
    """Display paper trading performance report."""
    from bot.paper_trader import PaperTrader

    trader = PaperTrader()
    summary = trader.get_performance_summary()

    print("\n  Performance Report")
    print("  " + "─" * 30)
    print(f"  Balance:        ${summary['balance']:>12,.2f}")
    print(f"  Total Trades:   {summary['total_trades']:>12}")
    if summary["total_trades"] > 0:
        print(f"  Wins:           {summary['wins']:>12}")
        print(f"  Losses:         {summary['losses']:>12}")
        print(f"  Win Rate:       {summary['win_rate']:>11.1f}%")
        print(f"  Total P/L:      ${summary['total_pnl']:>12,.2f}")
        print(f"  Avg P/L:        ${summary['avg_pnl']:>12,.2f}")
        print(f"  Return:         {summary.get('return_pct', 0):>11.2f}%")
    print(f"  Open Positions: {summary['open_positions']:>12}\n")


def _run_inline_auth() -> None:
    """Run the Schwab OAuth re-authorization flow inline."""
    from bot.auth import run_auth_flow

    print("\n  Schwab Token Re-Authorization\n")
    try:
        run_auth_flow()
        print("  ✓ Token refreshed successfully.\n")
    except Exception as exc:
        print(f"  ✗ Re-authorization failed: {exc}\n")


def _token_age_days(token_path: Path) -> Optional[float]:
    """Return token age in days when creation timestamp is available."""
    try:
        raw = token_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        created_ts = payload.get("creation_timestamp")
        if created_ts is None:
            return None
        created = datetime.fromtimestamp(float(created_ts), tz=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (now - created).total_seconds() / 86400.0)
    except Exception:
        return None


def _ensure_token_ready(config, *, interactive: bool) -> bool:
    """Validate token presence and optionally run inline auth when needed."""
    token_path = Path(config.schwab.token_path).expanduser()
    if not token_path.is_absolute():
        # Try CWD first, then fall back to the package root directory
        cwd_candidate = (Path.cwd() / token_path).resolve()
        pkg_candidate = (Path(__file__).resolve().parent / token_path).resolve()
        if cwd_candidate.exists():
            token_path = cwd_candidate
        elif pkg_candidate.exists():
            token_path = pkg_candidate
        else:
            token_path = cwd_candidate  # default for the "missing" error message

    def _auth_prompt() -> bool:
        if not interactive:
            return False
        try:
            answer = input("Run Schwab auth flow now? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in {"", "y", "yes"}

    try:
        validate_sensitive_file(
            token_path, label="Schwab token file", allow_missing=False
        )
    except PermissionError:
        pass  # File exists but is locked; assume it is valid for now.
    except FileNotFoundError:
        print(f"  Token missing at {token_path}")
        if _auth_prompt():
            _run_inline_auth()
            try:
                validate_sensitive_file(
                    token_path, label="Schwab token file", allow_missing=False
                )
            except PermissionError:
                pass
            except Exception:
                print("  Token check failed after auth. Please re-run.")
                return False
            return True
        print("  Run `python3 main.py auth` to create a token.")
        return False
    except Exception as exc:
        print(f"  Token check failed: {exc}")
        return False

    age_days = _token_age_days(token_path)
    if age_days is not None and age_days >= 7.0:
        print(f"  Token expired (~{age_days:.1f} days old).")
        if _auth_prompt():
            _run_inline_auth()
            return True
        print("  Run `python3 main.py auth` to renew.")
        return False

    if age_days is not None:
        print(f"  Token OK  (~{age_days:.1f} days old)")
    else:
        print("  Token found.")
    return True


def _flatten_chain_side(chain_side: dict) -> list[dict]:
    rows: list[dict] = []
    if not isinstance(chain_side, dict):
        return rows
    for contracts in chain_side.values():
        if isinstance(contracts, list):
            rows.extend([row for row in contracts if isinstance(row, dict)])
    return rows


def _is_valid_contract(contract: dict) -> bool:
    return bool(
        str(contract.get("symbol", "")).strip()
        and str(contract.get("expiration", "")).strip()
        and safe_float(contract.get("strike"), 0.0) > 0
        and safe_float(contract.get("dte"), -1) >= 0
    )


def _has_price(contract: dict) -> bool:
    return (
        safe_float(contract.get("bid"), 0.0) > 0.0
        or safe_float(contract.get("ask"), 0.0) > 0.0
        or safe_float(contract.get("mid"), 0.0) > 0.0
    )


def run_integrated_diagnostics(
    config, mode_hint: str = "paper", symbol: str = "SPY"
) -> int:
    """Run one-shot diagnostics for auth, chain parsing, and entry blockers."""
    print(f"\n  Diagnostics  ·  {mode_hint.upper()}  ·  {symbol}\n")
    from bot.schwab_client import SchwabClient
    client: Optional[SchwabClient] = None
    bot: Optional["TradingBot"] = None

    def _cleanup_streaming() -> None:
        for candidate in (
            getattr(bot, "schwab", None),
            client,
        ):
            try:
                if candidate is not None:
                    candidate.stop_streaming()
            except Exception:
                pass

    token_path = Path(config.schwab.token_path).expanduser()
    if not token_path.is_absolute():
        token_path = (Path.cwd() / token_path).resolve()
    try:
        validate_sensitive_file(
            token_path, label="Schwab token file", allow_missing=False
        )
        mode = token_path.stat().st_mode & 0o777
        print(f"  token       {token_path}  (perm {oct(mode)})")
    except Exception as exc:
        print(f"  token       failed  {exc}")
        return 1

    try:
        client = SchwabClient(config.schwab)
        client.connect()
        print("  auth        ok")
    except Exception as exc:
        print(f"  auth        failed  {exc}")
        _cleanup_streaming()
        return 1

    try:
        raw = client.get_option_chain(symbol)
        parsed = SchwabClient.parse_option_chain(raw)
        calls = _flatten_chain_side(parsed.get("calls", {}))
        puts = _flatten_chain_side(parsed.get("puts", {}))
        valid_calls = [row for row in calls if _is_valid_contract(row)]
        valid_puts = [row for row in puts if _is_valid_contract(row)]
        priced_calls = [row for row in valid_calls if _has_price(row)]
        priced_puts = [row for row in valid_puts if _has_price(row)]
        underlying_px = safe_float(parsed.get('underlying_price'), 0.0)
        print(
            f"  chain       {symbol} ${underlying_px:.2f}  "
            f"calls {len(priced_calls)}/{len(calls)}  puts {len(priced_puts)}/{len(puts)}"
        )
    except Exception as exc:
        print(f"  chain       failed  {exc}")
        _cleanup_streaming()
        return 1

    try:
        diag_cfg = copy.deepcopy(config)
        diag_cfg.terminal_ui.enabled = False
        bot = TradingBot(diag_cfg, warn_unclean_shutdown=False)
        bot.connect()
        bot._update_portfolio_state()
        entries_allowed = bot._entries_allowed()
        timing = bot._entry_timing_state()
        open_positions = len(bot.risk_manager.portfolio.open_positions)
        max_positions = int(bot.config.risk.max_open_positions)
        print(
            f"  entries     allowed={entries_allowed}  "
            f"timing={timing.get('allowed')}  optimal={timing.get('optimal')}  "
            f"reason={timing.get('reason')}  "
            f"positions={open_positions}/{max_positions}"
        )

        pause_fields = [
            "halt_entries",
            "consecutive_loss_pause_until",
            "weekly_loss_pause_until",
            "correlated_loss_pause_until",
            "correlation_pause_until",
        ]
        pauses = {
            key: bot.circuit_state.get(key)
            for key in pause_fields
            if bot.circuit_state.get(key)
        }
        if pauses:
            print(f"  circuit     {pauses}")

        chain_data, underlying = bot._get_chain_data(symbol)
        if not chain_data or safe_float(underlying, 0.0) <= 0:
            print(
                "Orchestrator chain pull: FAILED (empty chain or non-positive underlying)"
            )
            return 1

        technical_context = None
        try:
            technical_context = bot.technicals.get_context(symbol, bot.schwab)
        except Exception:
            technical_context = None
        market_context = bot._build_market_context(symbol, chain_data)

        strategy_signal_counts: dict[str, int] = {}
        candidates = []
        for strategy in bot.strategies:
            signals = strategy.scan_for_entries(
                symbol,
                chain_data,
                underlying,
                technical_context=technical_context,
                market_context=market_context,
            )
            filtered = bot._filter_signals_by_context(signals, market_context)
            strategy_signal_counts[str(strategy.name)] = len(filtered)
            candidates.extend(filtered)

        print(f"  signals     {strategy_signal_counts}")
        if not candidates:
            print("  candidates  0")
            _cleanup_streaming()
            return 0

        candidates.sort(
            key=lambda s: safe_float(s.analysis.score if s.analysis else 0.0, 0.0),
            reverse=True,
        )
        signal = candidates[0]
        analysis = signal.analysis
        if analysis is None:
            print("  top candidate  no analysis payload")
            _cleanup_streaming()
            return 0

        mtf_ok, mtf_agreement, _ = bot._passes_multi_timeframe_confirmation(signal)
        min_score = bot._strategy_regime_min_score(signal)
        risk_ok, risk_reason = bot.risk_manager.approve_trade(signal)
        budget_ok, budget_qty, budget_reason = bot.risk_manager.evaluate_greeks_budget(
            signal,
            regime=str(
                signal.metadata.get("regime", bot.circuit_state.get("regime", "NORMAL"))
            ),
            quantity=max(1, int(signal.quantity or 1)),
            allow_resize=True,
        )
        slippage_penalty = bot._symbol_slippage_penalty(signal.symbol)
        width = bot._signal_width(signal)
        min_credit_pct = bot._strategy_min_credit_pct(signal.strategy)
        required_credit = (
            (min_credit_pct * width) + slippage_penalty
            if width > 0 and min_credit_pct > 0
            else 0.0
        )
        credit_ok = float(analysis.credit or 0.0) >= required_credit

        mtf_min = int(getattr(bot.config.multi_timeframe, 'min_agreement', 2) or 2)
        print(
            f"  top signal  {signal.strategy} {signal.symbol}  "
            f"score={safe_float(analysis.score, 0.0):.1f}  "
            f"pop={safe_float(analysis.probability_of_profit, 0.0) * 100.0:.1f}%  "
            f"credit={safe_float(analysis.credit, 0.0):.2f}"
        )
        print(
            f"  gates       mtf={mtf_ok} ({mtf_agreement}/{mtf_min})  "
            f"score={safe_float(analysis.score, 0.0):.1f}>={min_score:.1f}  "
            f"risk={risk_ok}  greeks={budget_ok} qty={budget_qty}  "
            f"credit_ok={credit_ok} ({safe_float(analysis.credit, 0.0):.2f}/{required_credit:.2f})"
        )
    except Exception as exc:
        print(f"  diagnostics failed  {exc}")
        _cleanup_streaming()
        return 1

    print("\n  Done.\n")
    _cleanup_streaming()
    return 0


RUN_MENU_OPTIONS: dict[str, tuple[str, bool]] = {
    "1": ("simulator", False),
    "2": ("live", False),
}

RUN_MENU_ALIASES: dict[str, tuple[str, bool]] = {
    "run simulator": ("simulator", False),
    "run simulator once": ("simulator", True),
    "run live": ("live", False),
    "run live once": ("live", True),
    "auth": ("auth", False),
    "reauth": ("auth", False),
    "re-auth": ("auth", False),
    "token": ("auth", False),
    "auth paper": ("auth_paper", False),
    "auth live": ("auth_live", False),
    "simulator": ("simulator", False),
    "live": ("live", False),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fully automated options trading bot for ThinkorSwim/Schwab"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Acknowledge live-trading confirmation prompt "
            "(useful for non-interactive live runs)"
        ),
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level from config",
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Suppress routine console logs (warnings/errors still shown). "
            "Full logs remain in log file. Enabled by default; use --no-quiet "
            "to show routine logs."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run trading bot")
    run_parser.set_defaults(mode="simulator", once_token=None)
    run_subparsers = run_parser.add_subparsers(dest="mode")

    run_simulator = run_subparsers.add_parser("simulator", help="Run offline training simulator")
    run_simulator.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of simulated training iterations (default 50)",
    )
    run_simulator.add_argument(
        "once_token",
        nargs="?",
        choices=["once"],
        help="Run one scan cycle and exit",
    )
    run_simulator.add_argument(
        "--diagnose",
        action="store_true",
        help="Run integrated diagnostics (auth/chain/gate checks) and exit",
    )
    run_simulator.set_defaults(mode="simulator")

    run_live = run_subparsers.add_parser("live", help="Run in live mode")
    run_live.add_argument(
        "once_token",
        nargs="?",
        choices=["once"],
        help="Run one scan cycle and exit",
    )
    run_live.add_argument(
        "--diagnose",
        action="store_true",
        help="Run integrated diagnostics (auth/chain/gate checks) and exit",
    )
    run_live.set_defaults(mode="live")

    subparsers.add_parser("auth", help="Re-authorize Schwab OAuth token")
    return parser


def _resolve_menu_selection() -> tuple[str, bool]:
    mode, once = prompt_run_menu()
    while mode == "auth":
        _run_inline_auth()
        mode, once = prompt_run_menu()
    if mode == "auth_paper":
        _run_inline_auth()
        return "simulator", once
    if mode == "auth_live":
        _run_inline_auth()
        return "live", once
    return mode, once

def _parse_args() -> tuple[argparse.Namespace, bool]:
    args = _build_parser().parse_args()
    interactive_menu_requested = not args.command

    if interactive_menu_requested:
        args.command = "run"
        if sys.stdin.isatty():
            mode, once = _resolve_menu_selection()
            if mode == "bootstrap":
                args.mode = "simulator"
                args.iterations = 100
                args.once_token = None
            else:
                args.mode = mode
                args.once_token = "once" if once else None
        else:
            # Avoid hanging in non-interactive contexts (scripts/services).
            args.mode = "simulator"
            args.once_token = None

    args.live = str(getattr(args, "mode", "simulator") or "simulator") == "live"
    args.once = str(getattr(args, "once_token", "") or "") == "once"
    args.diagnose = bool(getattr(args, "diagnose", False))
    args.iterations = getattr(args, "iterations", 50)
    args.iterations = int(args.iterations) if args.iterations else 50
    return args, interactive_menu_requested

def _count_enabled_strategies(config) -> int:
    return sum(
        1
        for strategy in (
            config.credit_spreads,
            config.iron_condors,
            config.covered_calls,
            config.naked_puts,
            config.calendar_spreads,
        )
        if getattr(strategy, "enabled", False)
    )


def _print_runtime_summary(config) -> None:
    mode_str = "LIVE" if config.trading_mode == "live" else ("SIMULATOR" if config.trading_mode == "simulator" else "PAPER")
    print(f"  Mode:       {mode_str}")
    print(f"  Strategies: {_count_enabled_strategies(config)} enabled")
    if not config.watchlist:
        print(
            "  Scanner:    ON — dynamically finds best options stocks from 150+ tickers"
        )
    else:
        prefix = ", ".join(config.watchlist[:5])
        suffix = "..." if len(config.watchlist) > 5 else ""
        print(f"  Watchlist:  {prefix}{suffix}")
    print()


def _is_bootstrap_complete() -> bool:
    try:
        path = Path("bot/data/learned_rules.json")
        if not path.exists():
            return False
        import json
        with path.open("r") as f:
            data = json.load(f)
        return "meta" in data or "rules" in data
    except Exception:
        return False

def prompt_run_menu() -> tuple[str, bool]:
    """Prompt for run mode."""
    print("\n  Mode Selection")
    print("  " + "─" * 14)
    
    bootstrap_done = _is_bootstrap_complete()
    options = {}
    
    if not bootstrap_done:
        print("  1) One-time Bootstrap (~10-15m)")
        print("  2) Simulator (Offline / Synthetic Data)")
        print("  3) Paper Trading (Live Market Data)")
        print("  4) Live Trading (Real Money)\n")
        options = {"1": ("bootstrap", False), "2": ("simulator", False), "3": ("paper", False), "4": ("live", False)}
        valid_range = "[1-4]"
    else:
        print("  1) Simulator (Offline / Synthetic Data)")
        print("  2) Paper Trading (Live Market Data)")
        print("  3) Live Trading (Real Money)\n")
        options = {"1": ("simulator", False), "2": ("paper", False), "3": ("live", False)}
        valid_range = "[1-3]"

    while True:
        try:
            raw = input(f"  Select mode {valid_range}: ").strip().lower()
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(130)
        except EOFError:
            print("\nNo selection received. Aborted.")
            sys.exit(2)

        if raw in options:
            return options[raw]
        if raw in RUN_MENU_ALIASES:
            return RUN_MENU_ALIASES[raw]
        print(f"Invalid selection. Enter a number {valid_range}.")


def prompt_after_run_menu() -> bool:
    """Prompt whether to return to the mode menu after a run exits."""
    print("\n  Session Complete\n")
    print("  1) Menu")
    print("  2) Exit\n")
    while True:
        try:
            raw = input("  Select action [1-2]: ").strip().lower()
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)
        except EOFError:
            print("\nNo selection received. Exiting.")
            sys.exit(0)

        if raw in {"1", "menu", "m"}:
            return True
        if raw in {"2", "exit", "quit", "q"}:
            return False
        print("Invalid selection. Enter 1 or 2.")


def start_dashboard_command_listener(
    bot: "TradingBot",
) -> tuple[dict, threading.Event, threading.Thread]:
    """Listen for menu commands while the bot is running.

    When the TUI is active the keybinding footer already shows the
    options and the TUI's own stdin listener handles input.  When
    running without the TUI we fall back to a plain stdin reader.
    """
    action: dict[str, Optional[str]] = {"selection": None}
    stop_event = threading.Event()

    def _on_command(cmd: str) -> None:
        action["selection"] = cmd
        stop_event.set()
        bot.stop()

    # Prefer the TUI's built-in listener (it shows a visible [m] Menu · [q] Quit bar)
    if bot.ui is not None:
        bot.ui.start_command_listener(_on_command)
        # Return a no-op thread so the caller API stays the same
        noop = threading.Thread(target=lambda: None, daemon=True)
        noop.start()
        return action, stop_event, noop

    # Fallback: plain stdin listener when TUI is disabled
    def _listen() -> None:
        print("Commands: [m] Return to mode menu  [q] Exit")
        while not stop_event.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.25)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            raw = sys.stdin.readline()
            if raw == "":
                return
            choice = raw.strip().lower()
            if choice in {"1", "menu", "return", "m"}:
                _on_command("menu")
                return
            if choice in {"2", "exit", "quit", "q"}:
                _on_command("exit")
                return

    listener = threading.Thread(
        target=_listen,
        name="dashboard-command-listener",
        daemon=True,
    )
    listener.start()
    return action, stop_event, listener


def main() -> None:
    args, interactive_menu_requested = _parse_args()
    args.report = False
    # Keep advanced execution paths disabled in the simplified 4-command CLI.
    args.dashboard = False
    args.backtest = False
    args.fetch_data = False
    args.update_econ_calendar = False
    args.audit_trail = ""
    args.start = ""
    args.end = ""
    args.preflight_only = False
    args.setup_live = False
    args.prepare_live = False
    args.live_readiness_only = False

    # Handle `python3 main.py auth` subcommand
    if args.command == "auth":
        _run_inline_auth()
        return

    if args.report:
        show_report()
        return

    if args.prepare_live:
        from bot.live_setup import run_live_setup
        try:
            run_live_setup(
                config_path=args.config,
                auto_auth=False,
                auto_select_account=False,
                prepare_only=True,
                render_service_files=True,
            )
        except Exception as exc:
            print(f"Live preparation failed: {exc}")
            sys.exit(1)
        return

    if args.live_readiness_only:
        args.live = True

    if args.setup_live:
        from bot.live_setup import run_live_setup
        try:
            run_live_setup(
                config_path=args.config,
                auto_auth=True,
                auto_select_account=True,
                prepare_only=False,
                render_service_files=True,
            )
        except Exception as exc:
            print(f"Live setup failed: {exc}")
            sys.exit(1)
        return

    # Load config
    config = load_config(args.config)
    
    # Apply CLI mode
    if hasattr(args, "mode"):
        config.trading_mode = args.mode

    # Always enable the live terminal dashboard for the simplified run commands.
    config.terminal_ui.enabled = True
    validation = validate_config(config)
    print(format_validation_report(validation))
    if validation.failed:
        print(
            "Configuration validation failed. Resolve the failures above before continuing."
        )
        sys.exit(2)

    # Apply token checks across all run-mode entry points (menu + CLI).
    needs_token = (
        args.command == "run"
        and config.trading_mode != "simulator"
        and not args.dashboard
        and not args.backtest
        and not args.fetch_data
        and not args.update_econ_calendar
        and not args.audit_trail
        and not args.report
    )
    if needs_token and not _ensure_token_ready(config, interactive=sys.stdin.isatty()):
        sys.exit(1)

    if args.diagnose:
        code = run_integrated_diagnostics(
            config=config,
            mode_hint=str(
                getattr(args, "mode", config.trading_mode) or config.trading_mode
            ),
            symbol="SPY",
        )
        if code != 0:
            sys.exit(code)
        return

    if args.fetch_data:
        if not args.start or not args.end:
            print("--fetch-data requires --start and --end in YYYY-MM-DD format.")
            sys.exit(2)
        from bot.data_fetcher import HistoricalDataFetcher
        from bot.schwab_client import SchwabClient
        try:
            schwab = SchwabClient(config.schwab)
            schwab.connect()
            symbols = list(
                dict.fromkeys(
                    config.watchlist + list(config.covered_calls.tickers or [])
                )
            )
            fetcher = HistoricalDataFetcher(schwab)
            results = fetcher.fetch_range(
                start=args.start, end=args.end, symbols=symbols
            )
            created = sum(1 for row in results if not row.skipped)
            skipped = sum(1 for row in results if row.skipped)
            print(f"Fetch complete: {created} snapshots created, {skipped} skipped.")
        except Exception as exc:
            print(f"Data fetch failed: {exc}")
            sys.exit(1)
        return

    if args.update_econ_calendar:
        from bot.econ_calendar import refresh_static_calendar_file
        try:
            output = refresh_static_calendar_file()
            print(f"Economic calendar updated: {output}")
        except Exception as exc:
            print(f"Economic calendar update failed: {exc}")
            sys.exit(1)
        return

    if args.audit_trail:
        symbol = str(args.audit_trail).upper().strip()
        path = Path("bot/data/audit_log.jsonl")
        if not path.exists():
            print("No audit trail file found.")
            return
        matched = 0
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            details = row.get("details", {}) if isinstance(row, dict) else {}
            row_symbol = str(details.get("symbol", "")).upper()
            if row_symbol != symbol:
                continue
            matched += 1
            print(json.dumps(row, indent=2, default=str))
        if matched == 0:
            print(f"No audit events found for {symbol}.")
        return

    if args.backtest:
        if not args.start or not args.end:
            print("--backtest requires --start and --end in YYYY-MM-DD format.")
            sys.exit(2)
        from bot.backtester import Backtester
        try:
            backtester = Backtester(config)
            result = backtester.run(start=args.start, end=args.end)
            print(f"Backtest complete. Report: {result.report_path}")
        except Exception as exc:
            print(f"Backtest failed: {exc}")
            sys.exit(1)
        return

    if args.dashboard:
        from bot.dashboard import generate_dashboard
        from bot.live_trade_ledger import LiveTradeLedger
        from bot.paper_trader import PaperTrader
        from bot.schwab_client import SchwabClient
        try:
            if config.trading_mode in {"paper", "simulator"}:
                paper = PaperTrader()
                closed = list(paper.closed_trades)
                open_positions = paper.get_positions()
                balance = paper.get_account_balance()
            else:
                ledger = LiveTradeLedger()
                closed = ledger.list_positions(statuses={"closed", "closed_external"})
                open_positions = ledger.list_positions(
                    statuses={"open", "opening", "closing"}
                )
                try:
                    schwab = SchwabClient(config.schwab)
                    schwab.connect()
                    balance = schwab.get_account_balance()
                except Exception:
                    balance = 0.0

            monthly: dict[str, float] = {}
            by_strategy: dict[str, dict] = {}
            winners: list[dict] = []
            losers: list[dict] = []
            for trade in closed:
                pnl = float(trade.get("pnl", trade.get("realized_pnl", 0.0)) or 0.0)
                close_date = str(trade.get("close_date", ""))
                month = close_date[:7] if len(close_date) >= 7 else "unknown"
                monthly[month] = monthly.get(month, 0.0) + pnl
                strategy = str(trade.get("strategy", "unknown"))
                stats = by_strategy.setdefault(
                    strategy, {"wins": 0, "count": 0, "total_pnl": 0.0}
                )
                stats["count"] += 1
                if pnl > 0:
                    stats["wins"] += 1
                stats["total_pnl"] += pnl
                row = {"symbol": trade.get("symbol", ""), "pnl": pnl}
                (winners if pnl >= 0 else losers).append(row)

            strategy_breakdown = {}
            for name, stats in by_strategy.items():
                count = max(1, stats["count"])
                strategy_breakdown[name] = {
                    "win_rate": (stats["wins"] / count) * 100.0,
                    "avg_pnl": stats["total_pnl"] / count,
                    "total_pnl": stats["total_pnl"],
                }

            payload = {
                "equity_curve": [],
                "monthly_pnl": monthly,
                "strategy_breakdown": strategy_breakdown,
                "top_winners": sorted(winners, key=lambda x: x["pnl"], reverse=True)[
                    :5
                ],
                "top_losers": sorted(losers, key=lambda x: x["pnl"])[:5],
                "risk_metrics": {
                    "sharpe": 0.0,
                    "sortino": 0.0,
                    "max_drawdown": 0.0,
                    "current_drawdown": 0.0,
                },
                "portfolio_greeks": {
                    "delta": 0.0,
                    "theta": 0.0,
                    "gamma": 0.0,
                    "vega": 0.0,
                },
                "sector_exposure": {},
                "circuit_breakers": {
                    "regime": "n/a",
                    "halt_entries": False,
                    "consecutive_loss_pause_until": "-",
                    "weekly_loss_pause_until": "-",
                },
                "balance": balance,
                "open_positions": len(open_positions),
            }
            output = generate_dashboard(payload)
            print(f"Dashboard generated: {output}")
        except Exception as exc:
            print(f"Dashboard generation failed: {exc}")
            sys.exit(1)
        return

    if args.live:
        config.trading_mode = "live"
        # Safety check
        if not config.schwab.app_key or not config.schwab.app_secret:
            print("  SCHWAB_APP_KEY and SCHWAB_APP_SECRET are required for live trading.")
            print("  Set them in .env (see .env.example), then re-run.")
            sys.exit(1)

    log_level = args.log_level or config.log_level
    console_level = "WARNING" if args.quiet else log_level
    setup_logging(
        log_level,
        console_level=console_level,
        log_file=config.log_file,
        max_bytes=config.log_max_bytes,
        backup_count=config.log_backup_count,
    )

    logger = logging.getLogger(__name__)

    print_banner()
    _print_runtime_summary(config)

    if config.trading_mode == "live":
        print("  Live trading  ·  real money at risk\n")
        needs_confirmation = (
            not args.preflight_only and not args.yes and not args.live_readiness_only
        )
        if needs_confirmation:
            if not sys.stdin.isatty():
                print("  Confirmation required. Re-run with --yes to proceed.")
                sys.exit(2)
            confirm = input("  Type YES to confirm: ")
            if confirm.strip() != "YES":
                print("  Aborted.")
                return

    # Create and run the simulator or bot
    if config.trading_mode == "simulator":
        from bot.simulator import TrainingSimulator
        sim = TrainingSimulator(config)
        sim.train(iterations=args.iterations)
        if interactive_menu_requested and sys.stdin.isatty():
            if prompt_after_run_menu():
                main()
        return

    from bot.orchestrator import TradingBot

    bot = TradingBot(config)
    preflight_message = "Preflight checks passed."
    dashboard_action: dict[str, Optional[str]] = {"selection": None}
    dashboard_listener_stop: Optional[threading.Event] = None
    dashboard_listener: Optional[threading.Thread] = None

    if args.live_readiness_only:
        logger.info(
            "Running live configuration readiness checks (no broker connectivity)..."
        )
        try:
            bot.validate_live_configuration(require_token_file=False)
        except Exception as e:
            logger.error("Live configuration readiness failed: %s", e)
            bot._alert(
                level="ERROR",
                title="Live configuration readiness failed",
                message=str(e),
            )
            sys.exit(1)
        print("Live configuration readiness checks passed.")
        return

    if config.trading_mode == "live":
        logger.info("Running live preflight checks...")
        try:
            bot.validate_live_readiness()
        except Exception as e:
            logger.error("Live preflight failed: %s", e)
            bot._alert(
                level="ERROR",
                title="Live preflight failed",
                message=str(e),
            )
            sys.exit(1)
    elif args.preflight_only:
        logger.info("Running paper-mode preflight...")
        try:
            bot.connect()
        except Exception as e:
            # Allow local/offline paper preflight when Schwab auth has not been set up yet.
            if "Token file not found" in str(e):
                logger.warning(
                    "Paper preflight market-data check skipped: %s",
                    e,
                )
                preflight_message = (
                    "Preflight checks passed (offline paper mode; "
                    "Schwab market data not connected)."
                )
            else:
                logger.error("Paper preflight failed: %s", e)
                bot._alert(
                    level="ERROR",
                    title="Paper preflight failed",
                    message=str(e),
                )
                sys.exit(1)
        try:
            bot.validate_llm_readiness()
        except Exception as e:
            logger.error("Paper preflight failed: %s", e)
            bot._alert(
                level="ERROR",
                title="Paper preflight failed",
                message=str(e),
            )
            sys.exit(1)

    if args.preflight_only:
        print(preflight_message)
        return

    once_clean_shutdown = False
    try:
        if args.once:
            logger.info("Running single scan cycle...")
            bot._write_runtime_state(clean_shutdown=False, note="once_starting")
            bot.connect()
            bot.validate_llm_readiness()
            bot.scan_and_trade()
            if config.trading_mode in {"paper", "simulator"}:
                show_report()
            once_clean_shutdown = True
        else:
            if interactive_menu_requested and sys.stdin.isatty():
                dashboard_action, dashboard_listener_stop, dashboard_listener = (
                    start_dashboard_command_listener(bot)
                )
            bot.run()
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        if args.once:
            try:
                bot._write_runtime_state(
                    clean_shutdown=once_clean_shutdown,
                    note="once_complete" if once_clean_shutdown else "once_failed",
                )
            except Exception:
                pass
            try:
                bot.schwab.stop_streaming()
            except Exception:
                pass
        if dashboard_listener_stop:
            dashboard_listener_stop.set()
        if dashboard_listener and dashboard_listener.is_alive():
            dashboard_listener.join(timeout=0.2)

    if interactive_menu_requested and sys.stdin.isatty():
        if dashboard_action.get("selection") == "menu":
            main()
            return
        if dashboard_action.get("selection") == "exit":
            return
        if prompt_after_run_menu():
            main()


if __name__ == "__main__":
    main()
