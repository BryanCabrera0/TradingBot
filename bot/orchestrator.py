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
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import schedule

from bot.config import BotConfig, load_config
from bot.schwab_client import SchwabClient
from bot.llm_advisor import LLMAdvisor
from bot.paper_trader import PaperTrader
from bot.risk_manager import RiskManager
from bot.market_scanner import MarketScanner
from bot.strategies.credit_spreads import CreditSpreadStrategy
from bot.strategies.iron_condors import IronCondorStrategy
from bot.strategies.covered_calls import CoveredCallStrategy
from bot.strategies.base import TradeSignal

logger = logging.getLogger(__name__)

LIVE_SUPPORTED_ENTRY_STRATEGIES = {"credit_spreads", "covered_calls"}
EASTERN_TZ = ZoneInfo("America/New_York")


class TradingBot:
    """Fully automated options trading bot."""

    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or load_config()
        self.is_paper = self.config.trading_mode == "paper"
        self._running = False

        # Initialize components
        self.risk_manager = RiskManager(self.config.risk)
        self.paper_trader = PaperTrader() if self.is_paper else None
        self.llm_advisor: Optional[LLMAdvisor] = None
        if self.config.llm.enabled:
            self.llm_advisor = LLMAdvisor(self.config.llm)

        # Market data is required in both paper and live modes.
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
        if self.llm_advisor:
            logger.info(
                "LLM advisor enabled | Provider: %s | Model: %s | Mode: %s | Risk style: %s",
                self.config.llm.provider,
                self.config.llm.model,
                self.config.llm.mode,
                self.config.llm.risk_style,
            )

    # ── Connection ───────────────────────────────────────────────────

    def connect(self) -> None:
        """Connect to Schwab API for market data and/or live trading."""
        self.schwab.connect()

        if self.is_paper:
            logger.info(
                "Paper trading mode — Schwab market data connected. "
                "Paper balance: $%s",
                f"{self.paper_trader.get_account_balance():,.2f}",
            )
            return

        account_hash = self.schwab.resolve_account_hash(require_unique=True)
        balance = self.schwab.get_account_balance()
        logger.info(
            "Connected to Schwab API | Account: ...%s | Balance: $%s",
            account_hash[-4:],
            f"{balance:,.2f}",
        )

    def validate_live_readiness(self) -> None:
        """Run live-mode startup checks and fail fast on unsafe setup."""
        if self.is_paper:
            return

        enabled = self._enabled_strategy_names()
        unsupported = sorted(enabled - LIVE_SUPPORTED_ENTRY_STRATEGIES)
        if unsupported:
            raise RuntimeError(
                "Live execution currently supports only "
                "credit_spreads and covered_calls. "
                f"Disable unsupported strategies: {', '.join(unsupported)}"
            )
        if not enabled:
            raise RuntimeError("No enabled strategies in config.")

        self.connect()
        self._validate_llm_readiness()

        probe_symbol = self.config.watchlist[0] if self.config.watchlist else "SPY"
        quote = self.schwab.get_quote(probe_symbol)
        quote_ref = quote.get("quote", quote)
        probe_price = float(quote_ref.get("lastPrice", quote_ref.get("mark", 0.0)))
        if probe_price <= 0:
            raise RuntimeError(
                f"Could not retrieve a valid live quote for {probe_symbol}."
            )
        logger.info("Live preflight quote check passed: %s @ $%.2f", probe_symbol, probe_price)

        chain_data, underlying_price = self._get_chain_data(probe_symbol)
        if not chain_data or underlying_price <= 0:
            raise RuntimeError(
                f"Could not retrieve valid option-chain data for {probe_symbol}."
            )
        logger.info(
            "Live preflight option-chain check passed: %s @ $%.2f",
            probe_symbol,
            underlying_price,
        )

        logger.warning(
            "Live exit automation currently requires tracked strategy metadata. "
            "Monitor open live positions closely."
        )

    def _validate_llm_readiness(self) -> None:
        """Validate optional LLM advisor connectivity/configuration."""
        if not self.llm_advisor:
            return

        ok, message = self.llm_advisor.health_check()
        if ok:
            logger.info("LLM advisor check passed: %s", message)
            return

        if self.config.llm.mode == "blocking":
            raise RuntimeError(f"LLM advisor check failed in blocking mode: {message}")

        logger.warning("LLM advisor check failed (advisory mode): %s", message)

    def validate_llm_readiness(self) -> None:
        """Public wrapper for LLM readiness checks."""
        self._validate_llm_readiness()

    def _enabled_strategy_names(self) -> set[str]:
        """Return enabled strategy identifiers from config."""
        enabled: set[str] = set()
        if self.config.credit_spreads.enabled:
            enabled.add("credit_spreads")
        if self.config.iron_condors.enabled:
            enabled.add("iron_condors")
        if self.config.covered_calls.enabled:
            enabled.add("covered_calls")
        return enabled

    # ── Market Hours Check ───────────────────────────────────────────

    @classmethod
    def _now_eastern(cls) -> datetime:
        """Return current time in US/Eastern."""
        return datetime.now(EASTERN_TZ)

    @classmethod
    def is_market_open(cls, now: Optional[datetime] = None) -> bool:
        """Check if US stock market is currently open (rough check)."""
        if now is None:
            now_et = cls._now_eastern()
        elif now.tzinfo is None:
            # Treat naive datetimes as already in Eastern to avoid accidental local-time drift.
            now_et = now.replace(tzinfo=EASTERN_TZ)
        else:
            now_et = now.astimezone(EASTERN_TZ)

        # Weekday check (Mon=0 ... Fri=4)
        if now_et.weekday() > 4:
            return False
        # Market hours: 9:30 AM - 4:00 PM ET (approximate, doesn't handle holidays)
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now_et <= market_close

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
            self._refresh_paper_position_values()
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

    def _refresh_paper_position_values(self) -> None:
        """Refresh paper position marks using the latest option-chain mids."""
        if not self.paper_trader:
            return

        positions = self.paper_trader.get_positions()
        if not positions:
            return

        chain_cache: dict[str, dict] = {}
        position_marks: dict[str, float] = {}

        for position in positions:
            position_id = position.get("position_id")
            symbol = position.get("symbol")
            if not position_id or not symbol:
                continue

            if symbol not in chain_cache:
                chain_data, _ = self._get_chain_data(symbol)
                chain_cache[symbol] = chain_data

            chain_data = chain_cache.get(symbol, {})
            if not chain_data:
                continue

            mark = self._estimate_paper_position_value(position, chain_data)
            if mark is not None:
                position_marks[position_id] = mark

        if position_marks:
            self.paper_trader.update_position_values(position_marks)

    def _estimate_paper_position_value(
        self, position: dict, chain_data: dict
    ) -> Optional[float]:
        """Estimate current debit-to-close for a paper position."""
        details = position.get("details", {})
        strategy = position.get("strategy", "")

        expiration = self._extract_expiration_key(
            details.get("expiration", position.get("expiration", ""))
        )
        if not expiration:
            return None

        calls = chain_data.get("calls", {}).get(expiration, [])
        puts = chain_data.get("puts", {}).get(expiration, [])

        if strategy == "bull_put_spread":
            short_mid = self._find_option_mid(puts, details.get("short_strike"))
            long_mid = self._find_option_mid(puts, details.get("long_strike"))
            if short_mid is None or long_mid is None:
                return None
            return round(max(short_mid - long_mid, 0.0), 2)

        if strategy == "bear_call_spread":
            short_mid = self._find_option_mid(calls, details.get("short_strike"))
            long_mid = self._find_option_mid(calls, details.get("long_strike"))
            if short_mid is None or long_mid is None:
                return None
            return round(max(short_mid - long_mid, 0.0), 2)

        if strategy == "iron_condor":
            put_short = self._find_option_mid(puts, details.get("put_short_strike"))
            put_long = self._find_option_mid(puts, details.get("put_long_strike"))
            call_short = self._find_option_mid(calls, details.get("call_short_strike"))
            call_long = self._find_option_mid(calls, details.get("call_long_strike"))
            if None in (put_short, put_long, call_short, call_long):
                return None
            return round(max((put_short - put_long) + (call_short - call_long), 0.0), 2)

        if strategy == "covered_call":
            short_mid = self._find_option_mid(calls, details.get("short_strike"))
            if short_mid is None:
                return None
            return round(max(short_mid, 0.0), 2)

        return None

    @staticmethod
    def _extract_expiration_key(raw_expiration: object) -> str:
        """Normalize expiration formats to YYYY-MM-DD."""
        if raw_expiration is None:
            return ""

        exp = str(raw_expiration).strip()
        if not exp:
            return ""

        if "T" in exp:
            exp = exp.split("T", 1)[0]
        if ":" in exp:
            exp = exp.split(":", 1)[0]
        return exp

    @staticmethod
    def _find_option_mid(options: list[dict], strike: Optional[float]) -> Optional[float]:
        """Locate an option by strike and return its mid price."""
        if strike is None:
            return None

        for option in options:
            try:
                if abs(float(option.get("strike", 0.0)) - float(strike)) < 0.01:
                    return float(option.get("mid", 0.0))
            except (TypeError, ValueError):
                continue

        return None

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

        if not self._review_entry_with_llm(signal):
            return False

        analysis = signal.analysis
        logger.info(
            "EXECUTING TRADE: %s on %s | %d contracts | "
            "Credit: $%.2f | Max loss: $%.2f | POP: %.1f%% | Score: %.1f",
            signal.strategy, signal.symbol, signal.quantity,
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
                quantity=signal.quantity,
                details=details,
            )
            logger.info("Paper trade opened: %s", result)
            return True
        else:
            return self._execute_live_entry(signal)

    def _review_entry_with_llm(self, signal: TradeSignal) -> bool:
        """Optionally review an entry with the configured LLM advisor."""
        if not self.llm_advisor or signal.action != "open":
            return True

        context = {
            "trading_mode": self.config.trading_mode,
            "account_balance": self.risk_manager.portfolio.account_balance,
            "open_positions": len(self.risk_manager.portfolio.open_positions),
            "daily_pnl": self.risk_manager.portfolio.daily_pnl,
            "deployed_risk": self.risk_manager.portfolio.total_risk_deployed,
        }

        try:
            decision = self.llm_advisor.review_trade(signal, context)
        except Exception as e:
            if self.config.llm.mode == "blocking":
                logger.error("LLM review failed in blocking mode: %s", e)
                return False

            logger.warning("LLM review failed in advisory mode: %s", e)
            return True

        if signal.quantity > 1 and decision.risk_adjustment < 1.0:
            adjusted_quantity = max(1, int(signal.quantity * decision.risk_adjustment))
            if adjusted_quantity < signal.quantity:
                logger.info(
                    "LLM reduced quantity for %s on %s: %d -> %d (confidence %.2f)",
                    signal.strategy,
                    signal.symbol,
                    signal.quantity,
                    adjusted_quantity,
                    decision.confidence,
                )
                signal.quantity = adjusted_quantity

        if (
            self.config.llm.mode == "blocking"
            and decision.confidence < self.config.llm.min_confidence
        ):
            logger.info(
                "Trade REJECTED by LLM confidence gate: %.2f < %.2f | %s",
                decision.confidence,
                self.config.llm.min_confidence,
                decision.reason,
            )
            return False

        if not decision.approve:
            if self.config.llm.mode == "blocking":
                logger.info(
                    "Trade REJECTED by LLM: %s %s on %s | %s",
                    signal.strategy,
                    signal.action,
                    signal.symbol,
                    decision.reason,
                )
                return False

            logger.warning(
                "LLM flagged trade but advisory mode allows execution: %s",
                decision.reason,
            )
        else:
            logger.info(
                "LLM approved trade: %s (confidence %.2f)",
                decision.reason,
                decision.confidence,
            )

        return True

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
                logger.warning(
                    "Live iron condor execution is not implemented in this "
                    "schwab-py integration."
                )
                return False
            elif signal.strategy == "covered_call":
                if signal.analysis is None:
                    logger.warning("Missing analysis for live covered call on %s.", signal.symbol)
                    return False

                available_shares = self._get_live_equity_shares(signal.symbol)
                required_shares = quantity * 100
                if available_shares < required_shares:
                    logger.warning(
                        "Insufficient shares for covered call on %s: need %d, have %d",
                        signal.symbol,
                        required_shares,
                        available_shares,
                    )
                    return False

                order = self.schwab.build_covered_call_open(
                    symbol=signal.symbol,
                    expiration=analysis.expiration,
                    short_strike=analysis.short_strike,
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
        try:
            raw_chain = self.schwab.get_option_chain(symbol)
            parsed = SchwabClient.parse_option_chain(raw_chain)
            return parsed, parsed.get("underlying_price", 0.0)
        except Exception as e:
            logger.error("Failed to fetch option chain for %s: %s", symbol, e)
            logger.debug(traceback.format_exc())
            return {}, 0.0

    def _get_tracked_positions(self) -> list:
        """Get positions from Schwab and match with our tracked trades."""
        # In a production bot, you'd maintain a local DB of positions
        # For now, return positions from the API
        try:
            return self.schwab.get_positions()
        except Exception as e:
            logger.error("Failed to fetch positions: %s", e)
            return []

    def _get_live_equity_shares(self, symbol: str) -> int:
        """Return net long shares for a symbol from live account positions."""
        total = 0.0
        for pos in self._get_tracked_positions():
            instrument = pos.get("instrument", {})
            if instrument.get("assetType") != "EQUITY":
                continue
            if instrument.get("symbol") != symbol:
                continue

            long_qty = float(pos.get("longQuantity", 0.0))
            short_qty = float(pos.get("shortQuantity", 0.0))
            total += long_qty - short_qty

        return max(0, int(total))

    # ── Scheduling & Main Loop ───────────────────────────────────────

    def setup_schedule(self) -> None:
        """Configure the automated trading schedule."""
        sched_config = self.config.schedule

        # Schedule market scans
        for scan_time in sched_config.scan_times:
            schedule.every().day.at(scan_time, tz="America/New_York").do(
                self._scheduled_scan
            )
            logger.info("Scheduled scan at %s ET", scan_time)

        # Schedule position monitoring
        interval = sched_config.position_check_interval
        schedule.every(interval).minutes.do(self._scheduled_position_check)
        logger.info("Scheduled position checks every %d minutes", interval)

        # Daily performance report
        schedule.every().day.at("16:05", tz="America/New_York").do(self._daily_report)
        logger.info("Scheduled daily report at 16:05 ET")

    def _scheduled_scan(self) -> None:
        """Scan wrapper that checks market hours."""
        today = self._now_eastern().strftime("%A").lower()
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
        self._validate_llm_readiness()
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
