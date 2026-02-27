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
    python3 main.py run paper

    # Run one paper scan cycle and exit
    python3 main.py run paper once

    # Run continuous live mode (non-interactive confirmation bypass shown)
    python3 main.py run live --yes

    # Run one live scan cycle and exit
    python3 main.py run live once --yes
"""

import argparse
import json
import logging
import select
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from bot.config import format_validation_report, load_config, validate_config
from bot.data_fetcher import HistoricalDataFetcher
from bot.backtester import Backtester
from bot.dashboard import generate_dashboard
from bot.econ_calendar import refresh_static_calendar_file
from bot.live_setup import run_live_setup
from bot.orchestrator import TradingBot
from bot.paper_trader import PaperTrader
from bot.live_trade_ledger import LiveTradeLedger
from bot.schwab_client import SchwabClient


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


def print_banner() -> None:
    print("""
╭──────────────────────────────────────────────────────────╮
│              TradingBot                                  │
│              Fully Automated · Risk Managed              │
╰──────────────────────────────────────────────────────────╯
    """)


def show_report() -> None:
    """Display paper trading performance report."""
    trader = PaperTrader()
    summary = trader.get_performance_summary()

    print("\n" + "=" * 50)
    print("  PERFORMANCE REPORT")
    print("=" * 50)
    print(f"  Balance:        ${summary['balance']:>12,.2f}")
    print(f"  Total Trades:   {summary['total_trades']:>12}")
    if summary["total_trades"] > 0:
        print(f"  Wins:           {summary['wins']:>12}")
        print(f"  Losses:         {summary['losses']:>12}")
        print(f"  Win Rate:       {summary['win_rate']:>11.1f}%")
        print(f"  Total P/L:      ${summary['total_pnl']:>12,.2f}")
        print(f"  Avg P/L:        ${summary['avg_pnl']:>12,.2f}")
        print(f"  Return:         {summary.get('return_pct', 0):>11.2f}%")
    print(f"  Open Positions: {summary['open_positions']:>12}")
    print("=" * 50 + "\n")


def _run_inline_auth() -> None:
    """Run the Schwab OAuth re-authorization flow inline."""
    from bot.auth import run_auth_flow

    print("\n╭──────────────────────────────────────────────────────╮")
    print("│  Schwab Token Re-Authorization                      │")
    print("╰──────────────────────────────────────────────────────╯\n")
    try:
        run_auth_flow()
        print("\n✓ Token refreshed successfully.\n")
    except Exception as exc:
        print(f"\n✗ Re-authorization failed: {exc}\n")


def prompt_run_menu() -> tuple[str, bool]:
    """Prompt for one of the four supported run modes (or re-auth)."""
    options = {
        "1": ("paper", False),
        "2": ("paper", True),
        "3": ("live", False),
        "4": ("live", True),
        "5": ("auth", False),
    }
    aliases = {
        "run paper": ("paper", False),
        "run paper once": ("paper", True),
        "run live": ("live", False),
        "run live once": ("live", True),
        "auth": ("auth", False),
        "reauth": ("auth", False),
        "re-auth": ("auth", False),
        "token": ("auth", False),
    }

    print("\nSelect TradingBot mode:")
    print("  1) run paper")
    print("  2) run paper once")
    print("  3) run live")
    print("  4) run live once")
    print("  5) re-authorize Schwab token")

    while True:
        try:
            raw = input("Choose 1-5: ").strip().lower()
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(130)
        except EOFError:
            print("\nNo selection received. Aborted.")
            sys.exit(2)

        if raw in options:
            return options[raw]
        if raw in aliases:
            return aliases[raw]
        print("Invalid selection. Enter 1, 2, 3, 4, or 5.")


def prompt_after_run_menu() -> bool:
    """Prompt whether to return to the mode menu after a run exits."""
    print("\nRun ended.")
    print("  1) Return to mode menu")
    print("  2) Exit")
    while True:
        try:
            raw = input("Choose 1-2: ").strip().lower()
        except KeyboardInterrupt:
            print("\nAborted.")
            return False
        except EOFError:
            print("\nNo selection received. Exiting.")
            return False

        if raw in {"1", "return", "menu", "m"}:
            return True
        if raw in {"2", "exit", "quit", "q"}:
            return False
        print("Invalid selection. Enter 1 or 2.")


def start_dashboard_command_listener(bot: TradingBot) -> tuple[dict, threading.Event, threading.Thread]:
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
    parser = argparse.ArgumentParser(
        description="Fully automated options trading bot for ThinkorSwim/Schwab"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Acknowledge live-trading confirmation prompt (useful for non-interactive live runs)",
    )
    parser.add_argument(
        "--log-level", default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level from config",
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Suppress routine console logs (warnings/errors still shown). "
            "Full logs remain in log file. Enabled by default; use --no-quiet to show routine logs."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run trading bot")
    run_parser.set_defaults(mode="paper", once_token=None)
    run_subparsers = run_parser.add_subparsers(dest="mode")

    run_paper = run_subparsers.add_parser("paper", help="Run in paper mode")
    run_paper.add_argument(
        "once_token",
        nargs="?",
        choices=["once"],
        help="Run one scan cycle and exit",
    )
    run_paper.set_defaults(mode="paper")

    run_live = run_subparsers.add_parser("live", help="Run in live mode")
    run_live.add_argument(
        "once_token",
        nargs="?",
        choices=["once"],
        help="Run one scan cycle and exit",
    )
    run_live.set_defaults(mode="live")

    subparsers.add_parser("auth", help="Re-authorize Schwab OAuth token")

    args = parser.parse_args()
    interactive_menu_requested = not args.command
    # Default no-command invocation to interactive mode selection.
    if interactive_menu_requested:
        args.command = "run"
        if sys.stdin.isatty():
            mode, once = prompt_run_menu()
            # Handle re-auth selection: run auth flow, then loop back
            while mode == "auth":
                _run_inline_auth()
                mode, once = prompt_run_menu()
            args.mode = mode
            args.once_token = "once" if once else None
        else:
            # Avoid hanging in non-interactive contexts (scripts/services).
            args.mode = "paper"
            args.once_token = None

    args.live = str(getattr(args, "mode", "paper") or "paper") == "live"
    args.once = str(getattr(args, "once_token", "") or "") == "once"
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
    # Always enable the live terminal dashboard for the simplified run commands.
    config.terminal_ui.enabled = True
    validation = validate_config(config)
    print(format_validation_report(validation))
    if validation.failed:
        print("Configuration validation failed. Resolve the failures above before continuing.")
        sys.exit(2)

    if args.fetch_data:
        if not args.start or not args.end:
            print("--fetch-data requires --start and --end in YYYY-MM-DD format.")
            sys.exit(2)
        try:
            schwab = SchwabClient(config.schwab)
            schwab.connect()
            symbols = list(dict.fromkeys(config.watchlist + list(config.covered_calls.tickers or [])))
            fetcher = HistoricalDataFetcher(schwab)
            results = fetcher.fetch_range(start=args.start, end=args.end, symbols=symbols)
            created = sum(1 for row in results if not row.skipped)
            skipped = sum(1 for row in results if row.skipped)
            print(f"Fetch complete: {created} snapshots created, {skipped} skipped.")
        except Exception as exc:
            print(f"Data fetch failed: {exc}")
            sys.exit(1)
        return

    if args.update_econ_calendar:
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
        try:
            backtester = Backtester(config)
            result = backtester.run(start=args.start, end=args.end)
            print(f"Backtest complete. Report: {result.report_path}")
        except Exception as exc:
            print(f"Backtest failed: {exc}")
            sys.exit(1)
        return

    if args.dashboard:
        try:
            if config.trading_mode == "paper":
                paper = PaperTrader()
                closed = list(paper.closed_trades)
                open_positions = paper.get_positions()
                balance = paper.get_account_balance()
            else:
                ledger = LiveTradeLedger()
                closed = ledger.list_positions(statuses={"closed", "closed_external"})
                open_positions = ledger.list_positions(statuses={"open", "opening", "closing"})
                try:
                    schwab = SchwabClient(config.schwab)
                    schwab.connect()
                    balance = schwab.get_account_balance()
                except Exception:
                    balance = 0.0

            monthly: dict[str, float] = {}
            by_strategy: dict[str, dict] = {}
            winners, losers = [], []
            for trade in closed:
                pnl = float(trade.get("pnl", trade.get("realized_pnl", 0.0)) or 0.0)
                close_date = str(trade.get("close_date", ""))
                month = close_date[:7] if len(close_date) >= 7 else "unknown"
                monthly[month] = monthly.get(month, 0.0) + pnl
                strategy = str(trade.get("strategy", "unknown"))
                stats = by_strategy.setdefault(strategy, {"wins": 0, "count": 0, "total_pnl": 0.0})
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
                "top_winners": sorted(winners, key=lambda x: x["pnl"], reverse=True)[:5],
                "top_losers": sorted(losers, key=lambda x: x["pnl"])[:5],
                "risk_metrics": {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "current_drawdown": 0.0},
                "portfolio_greeks": {"delta": 0.0, "theta": 0.0, "gamma": 0.0, "vega": 0.0},
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
            print("ERROR: Live trading requires SCHWAB_APP_KEY and SCHWAB_APP_SECRET")
            print("       Set them in your .env file. See .env.example.")
            print("       Then run: python3 main.py run live")
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

    mode_str = "LIVE" if config.trading_mode == "live" else "PAPER"
    print(f"  Mode:       {mode_str}")
    print(
        "  Strategies: "
        f"{sum(1 for s in [config.credit_spreads, config.iron_condors, config.covered_calls, config.naked_puts, config.calendar_spreads] if getattr(s, 'enabled', False))} enabled"
    )
    if config.scanner.enabled:
        print(f"  Scanner:    ON — dynamically finds best options stocks from 150+ tickers")
    else:
        print(f"  Watchlist:  {', '.join(config.watchlist[:5])}{'...' if len(config.watchlist) > 5 else ''}")
    print()

    if config.trading_mode == "live":
        print("  ⚠  LIVE TRADING MODE — REAL MONEY AT RISK")
        print("  ⚠  Make sure you understand the risks before proceeding.")
        print()
        needs_confirmation = (
            not args.preflight_only
            and not args.yes
            and not args.live_readiness_only
        )
        if needs_confirmation:
            if not sys.stdin.isatty():
                print("ERROR: Live confirmation required in non-interactive mode.")
                print("       Re-run with --yes once you have validated your setup.")
                sys.exit(2)
            confirm = input("  Type 'YES' to confirm live trading: ")
            if confirm.strip() != "YES":
                print("  Aborted.")
                return

    # Create and run the bot
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

    try:
        if args.once:
            logger.info("Running single scan cycle...")
            bot.connect()
            bot.validate_llm_readiness()
            bot.scan_and_trade()
            if config.trading_mode == "paper":
                show_report()
        else:
            if interactive_menu_requested and sys.stdin.isatty():
                dashboard_action, dashboard_listener_stop, dashboard_listener = (
                    start_dashboard_command_listener(bot)
                )
            bot.run()
    except KeyboardInterrupt:
        print("\nRun interrupted by user.")
    finally:
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
