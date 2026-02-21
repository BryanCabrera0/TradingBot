#!/usr/bin/env python3
"""
TradingBot — Fully Automated Options Trading Bot
=================================================

Connects to Charles Schwab's API (via ThinkorSwim account) and
automatically trades options strategies including:
  - Bull put spreads / Bear call spreads (credit spreads)
  - Iron condors
  - Covered calls

Usage:
    # Paper trading (default — no real money)
    python main.py

    # Paper trading with custom config
    python main.py --config my_config.yaml

    # Live trading (requires Schwab API credentials in .env)
    python main.py --live

    # Run a single scan (no continuous loop)
    python main.py --once

    # Show performance report
    python main.py --report
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from bot.config import load_config
from bot.orchestrator import TradingBot
from bot.paper_trader import PaperTrader


def setup_logging(level: str = "INFO", log_file: str = "logs/tradingbot.log") -> None:
    """Configure logging to both console and file."""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console.setFormatter(console_fmt)
    root_logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(log_file)
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
        "--log-level", default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level from config",
    )

    args = parser.parse_args()

    if args.report:
        show_report()
        return

    # Load config
    config = load_config(args.config)

    if args.live:
        config.trading_mode = "live"
        # Safety check
        if not config.schwab.app_key or not config.schwab.app_secret:
            print("ERROR: Live trading requires SCHWAB_APP_KEY and SCHWAB_APP_SECRET")
            print("       Set them in your .env file. See .env.example.")
            sys.exit(1)

    log_level = args.log_level or config.log_level
    setup_logging(log_level, config.log_file)

    logger = logging.getLogger(__name__)

    print_banner()

    mode_str = "LIVE" if config.trading_mode == "live" else "PAPER"
    print(f"  Mode:       {mode_str}")
    print(f"  Strategies: {sum(1 for s in [config.credit_spreads, config.iron_condors, config.covered_calls] if getattr(s, 'enabled', False))} enabled")
    if config.scanner.enabled:
        print(f"  Scanner:    ON — dynamically finds best options stocks from 150+ tickers")
    else:
        print(f"  Watchlist:  {', '.join(config.watchlist[:5])}{'...' if len(config.watchlist) > 5 else ''}")
    print()

    if config.trading_mode == "live":
        print("  ⚠  LIVE TRADING MODE — REAL MONEY AT RISK")
        print("  ⚠  Make sure you understand the risks before proceeding.")
        print()
        confirm = input("  Type 'YES' to confirm live trading: ")
        if confirm.strip() != "YES":
            print("  Aborted.")
            return

    # Create and run the bot
    bot = TradingBot(config)

    if args.once:
        logger.info("Running single scan cycle...")
        bot.connect()
        bot.scan_and_trade()
        if config.trading_mode == "paper":
            show_report()
    else:
        bot.run()


if __name__ == "__main__":
    main()
