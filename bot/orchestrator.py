"""Main bot orchestrator — fully automated trading loop.

This is the brain of the bot. It:
1. Scans the entire market to find the best options-tradeable stocks
2. Runs every enabled strategy against each top-ranked symbol
3. Filters through risk management
4. Executes trades automatically (paper or live)
5. Monitors open positions and exits at targets/stops
6. Runs continuously with no manual intervention
"""

import logging
import time
import traceback
from datetime import datetime, date
from typing import Optional

import schedule

from bot.config import BotConfig, load_config
from bot.schwab_client import SchwabClient
from bot.paper_trader import PaperTrader
from bot.risk_manager import RiskManager
from bot.market_scanner import MarketScanner
from bot.strategies.credit_spreads import CreditSpreadStrategy
from bot.strategies.iron_condors import IronCondorStrategy
from bot.strategies.covered_calls import CoveredCallStrategy
from bot.strategies.base import TradeSignal

logger = logging.getLogger(__name__)


class TradingBot:
    """Fully automated options trading bot."""

    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or load_config()
        self.is_paper = self.config.trading_mode == "paper"
        self._running = False

        # Initialize components
        self.risk_manager = RiskManager(self.config.risk)
        self.paper_trader = PaperTrader() if self.is_paper else None

        # Initialize Schwab client (needed for both modes — scanner needs market data)
        self.schwab: Optional[SchwabClient] = None
        if not self.is_paper or self.config.scanner.enabled:
            self.schwab = SchwabClient(self.config.schwab)

        # Initialize market scanner
        self.scanner: Optional[MarketScanner] = None
        if self.config.scanner.enabled:
            self.scanner = MarketScanner(self.schwab, self.config.scanner)

        # Initialize strategies
        self.strategies = []
        if self.config.credit_spreads.enabled:
            self.strategies.append(
                CreditSpreadStrategy(vars(self.config.credit_spreads))
            )
        if self.config.iron_condors.enabled:
            self.strategies.append(
                IronCondorStrategy(vars(self.config.iron_condors))
            )
        if self.config.covered_calls.enabled:
            self.strategies.append(
                CoveredCallStrategy(vars(self.config.covered_calls))
            )

        scanner_status = "ON (dynamic market scan)" if self.scanner else "OFF (static watchlist)"
        logger.info(
            "TradingBot initialized | Mode: %s | Strategies: %d | Scanner: %s",
            self.config.trading_mode,
            len(self.strategies),
            scanner_status,
        )

    # ── Connection ───────────────────────────────────────────────────

    def connect(self) -> None:
        """Connect to the Schwab API (live mode) or initialize paper trader."""
        if self.is_paper:
            logger.info(
                "Paper trading mode — no API connection needed. "
                "Balance: $%s", f"{self.paper_trader.get_account_balance():,.2f}"
            )
            return

        if self.schwab is None:
            raise RuntimeError("Schwab client not initialized")
        self.schwab.connect()
        balance = self.schwab.get_account_balance()
        logger.info("Connected to Schwab API. Account balance: $%s", f"{balance:,.2f}")

    # ── Market Hours Check ───────────────────────────────────────────

    @staticmethod
    def is_market_open() -> bool:
        """Check if US stock market is currently open (rough check)."""
        now = datetime.now()
        # Weekday check (Mon=0 ... Fri=4)
        if now.weekday() > 4:
            return False
        # Market hours: 9:30 AM - 4:00 PM ET (approximate, doesn't handle holidays)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close

    # ── Core Trading Loop ────────────────────────────────────────────

    def scan_and_trade(self) -> None:
        """Main scan cycle: look for opportunities and execute trades.

        This runs automatically on schedule.
        """
        logger.info("=" * 60)
        logger.info("SCAN CYCLE STARTED at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("=" * 60)

        try:
            # Update portfolio state
            self._update_portfolio_state()

            # Check exits on existing positions first
            self._check_exits()

            # Scan for new entries
            self._scan_for_entries()

            logger.info("Scan cycle complete.")

        except Exception as e:
            logger.error("Error during scan cycle: %s", e)
            logger.debug(traceback.format_exc())

    def _update_portfolio_state(self) -> None:
        """Refresh account balance and positions for risk management."""
        if self.is_paper:
            balance = self.paper_trader.get_account_balance()
            positions = self.paper_trader.get_positions()
            daily_pnl = self.paper_trader.get_daily_pnl()
        else:
            balance = self.schwab.get_account_balance()
            positions = self._get_tracked_positions()
            daily_pnl = 0.0  # TODO: compute from order history

        self.risk_manager.update_portfolio(balance, positions, daily_pnl)
        logger.info(
            "Portfolio: Balance=$%s | Open positions: %d | Daily P/L: $%.2f",
            f"{balance:,.2f}", len(positions), daily_pnl,
        )

    def _get_scan_targets(self) -> list[str]:
        """Get the list of symbols to scan for opportunities.

        If the market scanner is enabled, dynamically discover the best
        options-tradeable stocks. Otherwise fall back to the static watchlist.
        """
        if self.scanner:
            try:
                targets = self.scanner.get_cached_results()
                if targets:
                    logger.info(
                        "Scanner found %d top tickers: %s",
                        len(targets), ", ".join(targets[:10]),
                    )
                    return targets
                logger.warning("Scanner returned no results. Falling back to watchlist.")
            except Exception as e:
                logger.error("Scanner failed: %s. Falling back to watchlist.", e)

        return self.config.watchlist

    def _scan_for_entries(self) -> None:
        """Scan the market for new trade opportunities.

        Uses the market scanner to dynamically find the best stocks,
        then runs each strategy against the top-ranked tickers.
        """
        targets = self._get_scan_targets()

        for symbol in targets:
            logger.info("Scanning %s...", symbol)

            try:
                chain_data, underlying_price = self._get_chain_data(symbol)
                if not chain_data or underlying_price <= 0:
                    logger.warning("No chain data for %s, skipping.", symbol)
                    continue

                # Run each strategy
                all_signals = []
                for strategy in self.strategies:
                    signals = strategy.scan_for_entries(
                        symbol, chain_data, underlying_price
                    )
                    all_signals.extend(signals)

                if not all_signals:
                    logger.info("No opportunities found on %s.", symbol)
                    continue

                # Sort by score and take the best
                all_signals.sort(
                    key=lambda s: s.analysis.score if s.analysis else 0,
                    reverse=True,
                )

                # Try to execute the best signal (risk manager may reject)
                for signal in all_signals[:3]:  # Try top 3 at most
                    self._try_execute_entry(signal)

            except Exception as e:
                logger.error("Error scanning %s: %s", symbol, e)
                logger.debug(traceback.format_exc())

    def _check_exits(self) -> None:
        """Check all open positions for exit conditions."""
        if self.is_paper:
            positions = self.paper_trader.get_positions()
        else:
            positions = self._get_tracked_positions()

        if not positions:
            return

        logger.info("Checking %d open positions for exits...", len(positions))

        all_exit_signals = []
        for strategy in self.strategies:
            exit_signals = strategy.check_exits(positions, self.schwab)
            all_exit_signals.extend(exit_signals)

        for signal in all_exit_signals:
            self._execute_exit(signal)

    # ── Trade Execution ──────────────────────────────────────────────

    def _try_execute_entry(self, signal: TradeSignal) -> bool:
        """Attempt to execute an entry trade after risk approval."""
        # Calculate position size
        quantity = self.risk_manager.calculate_position_size(signal)
        signal.quantity = quantity

        # Risk check
        approved, reason = self.risk_manager.approve_trade(signal)
        if not approved:
            logger.info(
                "Trade REJECTED by risk manager: %s %s on %s — %s",
                signal.strategy, signal.action, signal.symbol, reason,
            )
            return False

        analysis = signal.analysis
        logger.info(
            "EXECUTING TRADE: %s on %s | %d contracts | "
            "Credit: $%.2f | Max loss: $%.2f | POP: %.1f%% | Score: %.1f",
            signal.strategy, signal.symbol, quantity,
            analysis.credit, analysis.max_loss,
            analysis.probability_of_profit * 100, analysis.score,
        )

        if self.is_paper:
            details = {
                "expiration": analysis.expiration,
                "dte": analysis.dte,
                "short_strike": analysis.short_strike,
                "long_strike": analysis.long_strike,
                "put_short_strike": analysis.put_short_strike,
                "put_long_strike": analysis.put_long_strike,
                "call_short_strike": analysis.call_short_strike,
                "call_long_strike": analysis.call_long_strike,
                "probability_of_profit": analysis.probability_of_profit,
                "score": analysis.score,
            }
            result = self.paper_trader.execute_open(
                strategy=signal.strategy,
                symbol=signal.symbol,
                credit=analysis.credit,
                max_loss=analysis.max_loss,
                quantity=quantity,
                details=details,
            )
            logger.info("Paper trade opened: %s", result)
            return True
        else:
            return self._execute_live_entry(signal)

    def _execute_live_entry(self, signal: TradeSignal) -> bool:
        """Execute a real trade via Schwab API."""
        analysis = signal.analysis
        quantity = signal.quantity

        try:
            if signal.strategy == "bull_put_spread":
                order = self.schwab.build_bull_put_spread(
                    symbol=signal.symbol,
                    expiration=analysis.expiration,
                    short_strike=analysis.short_strike,
                    long_strike=analysis.long_strike,
                    quantity=quantity,
                    price=analysis.credit,
                )
            elif signal.strategy == "bear_call_spread":
                order = self.schwab.build_bear_call_spread(
                    symbol=signal.symbol,
                    expiration=analysis.expiration,
                    short_strike=analysis.short_strike,
                    long_strike=analysis.long_strike,
                    quantity=quantity,
                    price=analysis.credit,
                )
            elif signal.strategy == "iron_condor":
                order = self.schwab.build_iron_condor(
                    symbol=signal.symbol,
                    expiration=analysis.expiration,
                    put_long_strike=analysis.put_long_strike,
                    put_short_strike=analysis.put_short_strike,
                    call_short_strike=analysis.call_short_strike,
                    call_long_strike=analysis.call_long_strike,
                    quantity=quantity,
                    price=analysis.credit,
                )
            else:
                logger.warning("Live execution not implemented for: %s", signal.strategy)
                return False

            result = self.schwab.place_order(order)
            logger.info("LIVE order placed: %s", result)
            return True

        except Exception as e:
            logger.error("Failed to place live order: %s", e)
            logger.debug(traceback.format_exc())
            return False

    def _execute_exit(self, signal: TradeSignal) -> None:
        """Execute an exit trade."""
        logger.info(
            "CLOSING POSITION: %s | Reason: %s",
            signal.position_id, signal.reason,
        )

        if self.is_paper:
            result = self.paper_trader.execute_close(
                position_id=signal.position_id,
                reason=signal.reason,
            )
            logger.info("Paper close result: %s", result)
        else:
            # For live trading, we'd need to build closing orders
            # This depends on the specific position legs
            logger.info(
                "Live closing not yet fully implemented for %s. "
                "Manual review recommended.", signal.position_id,
            )

    # ── Data Fetching ────────────────────────────────────────────────

    def _get_chain_data(self, symbol: str) -> tuple[dict, float]:
        """Fetch and parse options chain data for a symbol."""
        if self.is_paper:
            # In paper mode, we still need real market data for realistic simulation
            if self.schwab is None:
                logger.warning(
                    "Paper mode with no API connection — "
                    "cannot fetch live chain data for %s.", symbol,
                )
                return {}, 0.0

        raw_chain = self.schwab.get_option_chain(symbol)
        parsed = SchwabClient.parse_option_chain(raw_chain)
        return parsed, parsed.get("underlying_price", 0.0)

    def _get_tracked_positions(self) -> list:
        """Get positions from Schwab and match with our tracked trades."""
        # In a production bot, you'd maintain a local DB of positions
        # For now, return positions from the API
        try:
            return self.schwab.get_positions()
        except Exception as e:
            logger.error("Failed to fetch positions: %s", e)
            return []

    # ── Scheduling & Main Loop ───────────────────────────────────────

    def setup_schedule(self) -> None:
        """Configure the automated trading schedule."""
        sched_config = self.config.schedule

        # Schedule market scans
        for scan_time in sched_config.scan_times:
            schedule.every().day.at(scan_time).do(self._scheduled_scan)
            logger.info("Scheduled scan at %s", scan_time)

        # Schedule position monitoring
        interval = sched_config.position_check_interval
        schedule.every(interval).minutes.do(self._scheduled_position_check)
        logger.info("Scheduled position checks every %d minutes", interval)

        # Daily performance report
        schedule.every().day.at("16:05").do(self._daily_report)
        logger.info("Scheduled daily report at 16:05")

    def _scheduled_scan(self) -> None:
        """Scan wrapper that checks market hours."""
        today = datetime.now().strftime("%A").lower()
        if today not in self.config.schedule.trading_days:
            logger.info("Not a trading day (%s). Skipping scan.", today)
            return

        if not self.is_market_open():
            logger.info("Market is closed. Skipping scan.")
            return

        self.scan_and_trade()

    def _scheduled_position_check(self) -> None:
        """Position monitoring wrapper."""
        if not self.is_market_open() and not self.is_paper:
            return

        try:
            self._update_portfolio_state()
            self._check_exits()
        except Exception as e:
            logger.error("Error during position check: %s", e)

    def _daily_report(self) -> None:
        """Log end-of-day performance summary."""
        if self.is_paper:
            summary = self.paper_trader.get_performance_summary()
            logger.info("=" * 60)
            logger.info("DAILY PERFORMANCE REPORT")
            logger.info("=" * 60)
            logger.info("Balance: $%s", f"{summary['balance']:,.2f}")
            logger.info("Total trades: %d", summary["total_trades"])
            logger.info("Win rate: %.1f%%", summary["win_rate"])
            logger.info("Total P/L: $%s", f"{summary['total_pnl']:,.2f}")
            logger.info("Return: %.2f%%", summary.get("return_pct", 0))
            logger.info("Open positions: %d", summary["open_positions"])
            logger.info("=" * 60)

    def run(self) -> None:
        """Start the fully automated trading loop.

        This method blocks and runs indefinitely, executing trades
        on the configured schedule with no manual intervention needed.
        """
        logger.info("=" * 60)
        logger.info("TRADING BOT STARTING")
        logger.info("Mode: %s", self.config.trading_mode.upper())
        logger.info("Strategies: %s", ", ".join(s.name for s in self.strategies))
        if self.scanner:
            logger.info("Scanner: ENABLED — dynamically finding best options stocks")
        else:
            logger.info("Watchlist: %s", ", ".join(self.config.watchlist))
        logger.info("=" * 60)

        self.connect()
        self.setup_schedule()

        # Run an initial scan immediately
        logger.info("Running initial scan...")
        self.scan_and_trade()

        self._running = True
        logger.info("Bot is now running. Press Ctrl+C to stop.")

        try:
            while self._running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            self._running = False

        # Final report
        if self.is_paper:
            self._daily_report()

    def stop(self) -> None:
        """Stop the bot gracefully."""
        self._running = False
        logger.info("Bot stop requested.")
