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
    # Paper trading (default — no real money)
    python3 main.py

    # Paper trading with custom config
    python3 main.py --config my_config.yaml

    # Live trading (requires Schwab API credentials in .env)
    python3 main.py --live

    # Live trading (non-interactive confirmation bypass)
    python3 main.py --live --yes

    # Validate live configuration without broker connectivity
    python3 main.py --live-readiness-only

    # Prepare all local live runtime files/settings before broker access is active
    python3 main.py --prepare-live

    # Run a single scan (no continuous loop)
    python3 main.py --once

    # Show performance report
    python3 main.py --report
"""

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from bot.config import load_config
from bot.data_fetcher import HistoricalDataFetcher
from bot.backtester import Backtester
from bot.dashboard import generate_dashboard
from bot.live_setup import run_live_setup
from bot.orchestrator import TradingBot
from bot.paper_trader import PaperTrader
from bot.live_trade_ledger import LiveTradeLedger
from bot.schwab_client import SchwabClient


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/tradingbot.log",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> None:
    """Configure logging to both console and file."""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
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
╔══════════════════════════════════════════════════════════╗
║           OPTIONS TRADING BOT                            ║
║           Fully Automated • Risk Managed                 ║
╚══════════════════════════════════════════════════════════╝
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fully automated options trading bot for ThinkorSwim/Schwab"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Enable live trading (default is paper trading)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single scan cycle and exit (no continuous loop)",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Show paper trading performance report and exit",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Generate HTML dashboard and exit",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run historical backtest and exit",
    )
    parser.add_argument(
        "--fetch-data", action="store_true",
        help="Fetch historical option-chain snapshots and exit",
    )
    parser.add_argument(
        "--start", default="",
        help="Start date for backtest/fetch in YYYY-MM-DD",
    )
    parser.add_argument(
        "--end", default="",
        help="End date for backtest/fetch in YYYY-MM-DD",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Run startup checks and exit without starting the bot",
    )
    parser.add_argument(
        "--setup-live", action="store_true",
        help="Run guided live setup (credentials, OAuth token, account selection) and exit",
    )
    parser.add_argument(
        "--prepare-live", action="store_true",
        help="Prepare local live runtime (env defaults + rendered service files) without broker connectivity",
    )
    parser.add_argument(
        "--live-readiness-only", action="store_true",
        help="Validate live configuration prerequisites without broker API calls",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Acknowledge live-trading confirmation prompt (useful for non-interactive runs)",
    )
    parser.add_argument(
        "--log-level", default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level from config",
    )

    args = parser.parse_args()

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
            print("       Or run: python3 main.py --setup-live")
            sys.exit(1)

    log_level = args.log_level or config.log_level
    setup_logging(
        log_level,
        config.log_file,
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

    if args.once:
        logger.info("Running single scan cycle...")
        bot.connect()
        bot.validate_llm_readiness()
        bot.scan_and_trade()
        if config.trading_mode == "paper":
            show_report()
    else:
        bot.run()


if __name__ == "__main__":
    main()
