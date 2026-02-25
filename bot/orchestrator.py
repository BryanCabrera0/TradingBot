"""Main bot orchestrator — fully automated trading loop.

This is the brain of the bot. It:
1. Scans the entire market to find the best options-tradeable stocks
2. Runs every enabled strategy against each top-ranked symbol
3. Filters through risk management
4. Executes trades automatically (paper or live)
5. Monitors open positions and exits at targets/stops
6. Runs continuously with no manual intervention
"""

from collections import defaultdict, deque
import logging
from pathlib import Path
import re
import time
import traceback
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import schedule

from bot.alerts import AlertManager
from bot.config import BotConfig, load_config
from bot.live_trade_ledger import LiveTradeLedger
from bot.schwab_client import SchwabClient
from bot.llm_advisor import LLMAdvisor
from bot.news_scanner import NewsScanner
from bot.number_utils import safe_float, safe_int
from bot.paper_trader import PaperTrader
from bot.risk_manager import RiskManager
from bot.market_scanner import MarketScanner
from bot.strategies.credit_spreads import CreditSpreadStrategy
from bot.strategies.iron_condors import IronCondorStrategy
from bot.strategies.covered_calls import CoveredCallStrategy
from bot.strategies.base import TradeSignal

logger = logging.getLogger(__name__)

LIVE_SUPPORTED_ENTRY_STRATEGIES = {"credit_spreads", "covered_calls", "iron_condors"}
EASTERN_TZ = ZoneInfo("America/New_York")
OPEN_LONG_INSTRUCTIONS = {"BUY_TO_OPEN"}
OPEN_SHORT_INSTRUCTIONS = {"SELL_TO_OPEN", "SELL_SHORT"}
CLOSE_LONG_INSTRUCTIONS = {"SELL_TO_CLOSE"}
CLOSE_SHORT_INSTRUCTIONS = {"BUY_TO_CLOSE", "BUY_TO_COVER"}
ENTRY_ORDER_TERMINAL = {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}
EXIT_ORDER_TERMINAL = ENTRY_ORDER_TERMINAL
PENDING_ORDER_STATUSES = {
    "AWAITING_PARENT_ORDER",
    "AWAITING_CONDITION",
    "AWAITING_MANUAL_REVIEW",
    "ACCEPTED",
    "AWAITING_UR_OUT",
    "PENDING_ACTIVATION",
    "QUEUED",
    "WORKING",
    "PENDING_CANCEL",
    "PENDING_REPLACE",
}
PARTIAL_ORDER_STATUSES = {"PARTIALLY_FILLED", "WORKING", "QUEUED", "ACCEPTED"}
OCC_COMPACT_PATTERN = re.compile(r"^([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")
UNDERSCORE_OPTION_PATTERN = re.compile(
    r"^([A-Z]{1,6})_(\d{2})(\d{2})(\d{2})([CP])(\d+(?:\.\d+)?)$"
)


class TradingBot:
    """Fully automated options trading bot."""

    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or load_config()
        self.is_paper = self.config.trading_mode == "paper"
        self._running = False
        self._market_open_cache: Optional[tuple[datetime, bool]] = None
        self._live_bootstrap_done = False

        # Initialize components
        self.risk_manager = RiskManager(self.config.risk)
        self.paper_trader = PaperTrader() if self.is_paper else None
        self.live_ledger = None if self.is_paper else LiveTradeLedger()
        self.alerts = AlertManager(self.config.alerts)
        self.llm_advisor: Optional[LLMAdvisor] = None
        if self.config.llm.enabled:
            self.llm_advisor = LLMAdvisor(self.config.llm)
        self.news_scanner: Optional[NewsScanner] = None
        if self.config.news.enabled:
            self.news_scanner = NewsScanner(self.config.news)

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
        if self.news_scanner:
            logger.info(
                "News scanner enabled | Provider: %s | Symbol headlines: %d | Market headlines: %d",
                self.config.news.provider,
                self.config.news.max_symbol_headlines,
                self.config.news.max_market_headlines,
            )

    def _alert(
        self,
        *,
        level: str,
        title: str,
        message: str,
        context: Optional[dict] = None,
    ) -> None:
        """Send an operational alert if alerting is configured."""
        self.alerts.send(
            level=level,
            title=title,
            message=message,
            context=context or {},
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
        if not self._live_bootstrap_done:
            try:
                imported = self._bootstrap_live_ledger_from_broker()
                self._live_bootstrap_done = True
                if imported:
                    logger.info("Imported %d existing live positions into ledger.", imported)
            except Exception as exc:
                logger.error("Failed to bootstrap live ledger: %s", exc)
                logger.debug(traceback.format_exc())
                self._alert(
                    level="ERROR",
                    title="Live ledger bootstrap failed",
                    message=str(exc),
                )

    def validate_live_readiness(self) -> None:
        """Run live-mode startup checks and fail fast on unsafe setup."""
        if self.is_paper:
            return

        self.validate_live_configuration(require_token_file=True)
        self.connect()
        self._validate_live_covered_call_readiness()

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

        logger.info(
            "Live ledger + exit automation enabled. Existing broker positions "
            "will be bootstrapped into ledger on connect."
        )

    def validate_live_configuration(self, *, require_token_file: bool = False) -> None:
        """Validate live config/safety prerequisites without broker API calls."""
        if self.is_paper:
            return

        if not self.config.schwab.app_key or not self.config.schwab.app_secret:
            raise RuntimeError(
                "Live mode requires SCHWAB_APP_KEY and SCHWAB_APP_SECRET."
            )

        enabled = self._enabled_strategy_names()
        unsupported = sorted(enabled - LIVE_SUPPORTED_ENTRY_STRATEGIES)
        if unsupported:
            raise RuntimeError(
                "Live execution currently supports only "
                "credit_spreads, iron_condors, and covered_calls. "
                f"Disable unsupported strategies: {', '.join(unsupported)}"
            )
        if not enabled:
            raise RuntimeError("No enabled strategies in config.")

        token_path = Path(self.config.schwab.token_path).expanduser()
        if require_token_file:
            if not token_path.exists():
                raise RuntimeError(
                    f"Schwab token file not found at {token_path}. "
                    "Run `python3 main.py --setup-live`."
                )
        elif not token_path.exists():
            logger.warning(
                "Schwab token file not found at %s. "
                "This is expected before OAuth is completed.",
                token_path,
            )

        if (
            not self.config.schwab.account_hash
            and int(self.config.schwab.account_index) < 0
        ):
            logger.warning(
                "SCHWAB_ACCOUNT_HASH / SCHWAB_ACCOUNT_INDEX not set. "
                "Auto-resolution requires exactly one linked account."
            )

        self._validate_live_alerting_readiness()
        self._validate_live_covered_call_configuration()
        self._validate_llm_readiness()

    def _validate_live_alerting_readiness(self) -> None:
        """Enforce runtime alert sink for live mode unless explicitly disabled."""
        if self.is_paper:
            return
        if not self.config.alerts.require_in_live:
            return
        if not self.config.alerts.enabled:
            raise RuntimeError(
                "Live mode requires alerts.enabled=true (set ALERTS_ENABLED=true)."
            )
        if not str(self.config.alerts.webhook_url).strip():
            raise RuntimeError(
                "Live mode requires ALERTS_WEBHOOK_URL when alerts.require_in_live=true."
            )

    def _validate_live_covered_call_configuration(self) -> None:
        """Validate covered-call config that can be checked without broker calls."""
        if self.is_paper or not self.config.covered_calls.enabled:
            return

        tickers = list(self.config.covered_calls.tickers or [])
        if not tickers:
            raise RuntimeError(
                "Live covered_calls is enabled but covered_calls.tickers is empty."
            )

    def _validate_live_covered_call_readiness(self) -> None:
        """Validate covered-call live prerequisites before order flow starts."""
        if self.is_paper or not self.config.covered_calls.enabled:
            return

        tickers = list(self.config.covered_calls.tickers or [])

        broker_positions = self._get_broker_positions() or []
        shares_by_symbol: dict[str, int] = defaultdict(int)
        for pos in broker_positions:
            instrument = pos.get("instrument", {})
            if instrument.get("assetType") != "EQUITY":
                continue
            symbol = str(instrument.get("symbol", "")).upper()
            long_qty = safe_int(pos.get("longQuantity"), 0)
            short_qty = safe_int(pos.get("shortQuantity"), 0)
            shares_by_symbol[symbol] += max(0, long_qty - short_qty)

        eligible = [sym for sym in tickers if shares_by_symbol.get(sym, 0) >= 100]
        if not eligible:
            raise RuntimeError(
                "Live covered_calls enabled but none of covered_calls.tickers has "
                "at least 100 shares in the broker account."
            )

        logger.info(
            "Live covered-call readiness passed: %d/%d configured tickers have >=100 shares.",
            len(eligible),
            len(tickers),
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

    def _is_market_open_now(self) -> bool:
        """Check market-open state, preferring broker market-hours in live mode."""
        now_et = self._now_eastern()
        if self.is_paper:
            return self.is_market_open(now_et)

        if self._market_open_cache:
            checked_at, cached_value = self._market_open_cache
            if now_et - checked_at < timedelta(minutes=5):
                return cached_value

        try:
            is_open = self.schwab.is_equity_market_open(now=now_et)
            self._market_open_cache = (now_et, is_open)
            return is_open
        except Exception as exc:
            logger.warning(
                "Falling back to approximate market-hours check: %s",
                exc,
            )
            fallback = self.is_market_open(now_et)
            self._market_open_cache = (now_et, fallback)
            return fallback

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
            self._alert(
                level="ERROR",
                title="Scan cycle failed",
                message=str(e),
            )

    def _update_portfolio_state(self) -> None:
        """Refresh account balance and positions for risk management."""
        if self.is_paper:
            balance = self.paper_trader.get_account_balance()
            positions = self.paper_trader.get_positions()
            daily_pnl = self.paper_trader.get_daily_pnl()
        else:
            self._reconcile_live_orders()
            balance = self.schwab.get_account_balance()
            positions = self._get_tracked_positions()
            daily_pnl = self._compute_live_daily_pnl()

        self.risk_manager.update_portfolio(balance, positions, daily_pnl)
        logger.info(
            "Portfolio: Balance=$%s | Open positions: %d | Daily P/L: $%.2f",
            f"{balance:,.2f}", len(positions), daily_pnl,
        )

    def _reconcile_live_orders(self) -> None:
        """Poll pending live orders and advance ledger state on terminal statuses."""
        if self.is_paper or not self.live_ledger:
            return

        cancel_stale = bool(self.config.execution.cancel_stale_orders)
        stale_minutes = int(self.config.execution.stale_order_minutes)
        include_partials = bool(self.config.execution.include_partials_in_ledger)

        for order_id in self.live_ledger.pending_entry_order_ids():
            try:
                order = self.schwab.get_order(order_id)
            except Exception as exc:
                logger.warning("Failed to fetch live entry order %s: %s", order_id, exc)
                continue

            status = str(order.get("status", "")).upper()
            fill_summary = self._summarize_order_fill(order)
            filled_qty = self._extract_filled_contracts(order, fill_summary)

            if include_partials and status in PARTIAL_ORDER_STATUSES and filled_qty > 0:
                self.live_ledger.apply_partial_entry_fill(
                    order_id,
                    filled_quantity=filled_qty,
                    entry_credit=(
                        fill_summary.get("per_contract")
                        if fill_summary and fill_summary.get("net_cash", 0.0) > 0
                        else None
                    ),
                )

            if status not in ENTRY_ORDER_TERMINAL:
                if (
                    cancel_stale
                    and status in PENDING_ORDER_STATUSES
                    and self._is_order_stale(order, order_id=order_id, side="entry", stale_minutes=stale_minutes)
                ):
                    self._cancel_stale_live_order(order_id, side="entry")
                continue

            entry_credit = None
            if fill_summary and fill_summary.get("net_cash", 0.0) > 0:
                entry_credit = fill_summary.get("per_contract")
            filled_at = (
                self._extract_order_terminal_time(order).isoformat()
                if status == "FILLED"
                else None
            )
            self.live_ledger.reconcile_entry_order(
                order_id,
                status=status,
                filled_at=filled_at,
                entry_credit=entry_credit,
                filled_quantity=int(filled_qty) if filled_qty > 0 else None,
            )

        for order_id in self.live_ledger.pending_exit_order_ids():
            try:
                order = self.schwab.get_order(order_id)
            except Exception as exc:
                logger.warning("Failed to fetch live exit order %s: %s", order_id, exc)
                continue

            status = str(order.get("status", "")).upper()
            fill_summary = self._summarize_order_fill(order)
            filled_qty = self._extract_filled_contracts(order, fill_summary)

            if include_partials and status in PARTIAL_ORDER_STATUSES and filled_qty > 0:
                self.live_ledger.apply_partial_exit_fill(
                    order_id,
                    filled_quantity=filled_qty,
                    close_value=fill_summary.get("per_contract") if fill_summary else None,
                )

            if status not in EXIT_ORDER_TERMINAL:
                if (
                    cancel_stale
                    and status in PENDING_ORDER_STATUSES
                    and self._is_order_stale(order, order_id=order_id, side="exit", stale_minutes=stale_minutes)
                ):
                    self._cancel_stale_live_order(order_id, side="exit")
                continue

            close_value = fill_summary.get("per_contract") if fill_summary else None
            filled_at = (
                self._extract_order_terminal_time(order).isoformat()
                if status == "FILLED"
                else None
            )
            self.live_ledger.reconcile_exit_order(
                order_id,
                status=status,
                filled_at=filled_at,
                close_value=close_value,
            )

    def _compute_live_daily_pnl(self) -> float:
        """Best-effort realized P/L for today's live fills using order history."""
        try:
            orders = self.schwab.get_orders(days_back=60)
        except Exception as exc:
            logger.warning("Failed to fetch live order history for daily P/L: %s", exc)
            return 0.0

        if not isinstance(orders, list):
            return 0.0

        pnl = self._compute_daily_pnl_from_orders(
            orders=orders,
            target_date=self._now_eastern().date(),
        )
        return round(pnl, 2)

    @classmethod
    def _compute_daily_pnl_from_orders(
        cls,
        orders: list[dict],
        target_date: date,
    ) -> float:
        """Compute realized P/L for ``target_date`` from a list of filled orders."""
        fills = cls._extract_order_fills(orders)
        fills.sort(key=lambda fill: fill["time"])

        long_lots: dict[str, deque[dict[str, float]]] = defaultdict(deque)
        short_lots: dict[str, deque[dict[str, float]]] = defaultdict(deque)
        daily_pnl = 0.0

        for fill in fills:
            key = fill["key"]
            instruction = fill["instruction"]
            quantity = fill["quantity"]
            price = fill["price"]
            multiplier = fill["multiplier"]
            fill_date = fill["time"].astimezone(EASTERN_TZ).date()

            if quantity <= 0 or price < 0:
                continue

            if instruction in OPEN_LONG_INSTRUCTIONS:
                long_lots[key].append({"quantity": quantity, "price": price})
                continue

            if instruction in OPEN_SHORT_INSTRUCTIONS:
                short_lots[key].append({"quantity": quantity, "price": price})
                continue

            if instruction in CLOSE_LONG_INSTRUCTIONS:
                pnl, remaining = cls._close_lots_fifo(
                    lots=long_lots[key],
                    close_quantity=quantity,
                    close_price=price,
                    multiplier=multiplier,
                    closing_long=True,
                )
                if remaining > 0:
                    short_lots[key].append({"quantity": remaining, "price": price})
                if fill_date == target_date:
                    daily_pnl += pnl
                continue

            if instruction in CLOSE_SHORT_INSTRUCTIONS:
                pnl, remaining = cls._close_lots_fifo(
                    lots=short_lots[key],
                    close_quantity=quantity,
                    close_price=price,
                    multiplier=multiplier,
                    closing_long=False,
                )
                if remaining > 0:
                    long_lots[key].append({"quantity": remaining, "price": price})
                if fill_date == target_date:
                    daily_pnl += pnl
                continue

            # Equity fills may use BUY/SELL without explicit position effect.
            if instruction == "BUY":
                pnl, remaining = cls._close_lots_fifo(
                    lots=short_lots[key],
                    close_quantity=quantity,
                    close_price=price,
                    multiplier=multiplier,
                    closing_long=False,
                )
                if fill_date == target_date:
                    daily_pnl += pnl
                if remaining > 0:
                    long_lots[key].append({"quantity": remaining, "price": price})
                continue

            if instruction == "SELL":
                pnl, remaining = cls._close_lots_fifo(
                    lots=long_lots[key],
                    close_quantity=quantity,
                    close_price=price,
                    multiplier=multiplier,
                    closing_long=True,
                )
                if fill_date == target_date:
                    daily_pnl += pnl
                if remaining > 0:
                    short_lots[key].append({"quantity": remaining, "price": price})

        return daily_pnl

    @staticmethod
    def _close_lots_fifo(
        lots: deque[dict[str, float]],
        close_quantity: float,
        close_price: float,
        multiplier: float,
        *,
        closing_long: bool,
    ) -> tuple[float, float]:
        """Close lots FIFO and return (realized_pnl, unmatched_quantity)."""
        remaining = max(0.0, float(close_quantity))
        realized = 0.0
        epsilon = 1e-9

        while remaining > epsilon and lots:
            lot = lots[0]
            lot_qty = max(0.0, safe_float(lot.get("quantity"), 0.0))
            if lot_qty <= epsilon:
                lots.popleft()
                continue

            matched = min(remaining, lot_qty)
            open_price = safe_float(lot.get("price"), 0.0)
            if closing_long:
                realized += (close_price - open_price) * matched * multiplier
            else:
                realized += (open_price - close_price) * matched * multiplier

            lot["quantity"] = lot_qty - matched
            remaining -= matched
            if lot["quantity"] <= epsilon:
                lots.popleft()

        return realized, max(0.0, remaining)

    @classmethod
    def _extract_order_fills(cls, orders: list[dict]) -> list[dict]:
        """Extract normalized fill events from raw Schwab order payloads."""
        fills: list[dict] = []

        for order in orders:
            if not isinstance(order, dict):
                continue

            leg_meta: dict[int, dict[str, object]] = {}
            default_meta: Optional[dict[str, object]] = None
            for leg in order.get("orderLegCollection", []):
                if not isinstance(leg, dict):
                    continue
                instruction = str(leg.get("instruction", "")).upper()
                instrument = leg.get("instrument", {})
                if not isinstance(instrument, dict):
                    continue
                symbol = str(instrument.get("symbol", "")).strip()
                if not symbol:
                    continue
                asset_type = str(instrument.get("assetType", "")).upper()
                multiplier = 100.0 if asset_type == "OPTION" else 1.0

                meta = {
                    "instruction": instruction,
                    "symbol": symbol,
                    "multiplier": multiplier,
                }
                if default_meta is None:
                    default_meta = meta

                raw_leg_id = leg.get("legId")
                leg_id = cls._parse_leg_id(raw_leg_id)
                if leg_id is not None:
                    leg_meta[leg_id] = meta

            if default_meta is None:
                continue

            found_execution_legs = False
            activities = order.get("orderActivityCollection", [])
            if isinstance(activities, list):
                for activity in activities:
                    if not isinstance(activity, dict):
                        continue
                    execution_legs = activity.get("executionLegs", [])
                    if not isinstance(execution_legs, list):
                        continue

                    for execution in execution_legs:
                        if not isinstance(execution, dict):
                            continue
                        found_execution_legs = True

                        leg_id = cls._parse_leg_id(execution.get("legId"))
                        meta = leg_meta.get(leg_id, default_meta)

                        quantity = safe_float(execution.get("quantity"), 0.0)
                        price = safe_float(execution.get("price"), 0.0)
                        if quantity <= 0 or price <= 0:
                            continue

                        fill_time = cls._parse_order_timestamp(
                            execution.get("time")
                            or activity.get("executionTime")
                            or activity.get("time")
                            or order.get("closeTime")
                            or order.get("enteredTime")
                        )
                        if fill_time is None:
                            continue

                        fills.append(
                            {
                                "time": fill_time,
                                "key": cls._normalize_fill_key(
                                    str(meta["symbol"]),
                                    safe_float(meta["multiplier"], 1.0),
                                ),
                                "instruction": str(meta["instruction"]).upper(),
                                "quantity": quantity,
                                "price": price,
                                "multiplier": safe_float(meta["multiplier"], 1.0),
                            }
                        )

            if found_execution_legs:
                continue

            # Fallback for payloads that only expose top-level fill data.
            if str(order.get("status", "")).upper() != "FILLED":
                continue

            quantity = safe_float(order.get("filledQuantity"), 0.0)
            if quantity <= 0:
                quantity = safe_float(order.get("quantity"), 0.0)
            price = safe_float(order.get("price"), 0.0)
            if quantity <= 0 or price <= 0:
                continue

            fill_time = cls._parse_order_timestamp(
                order.get("closeTime") or order.get("enteredTime")
            )
            if fill_time is None:
                continue

            fills.append(
                {
                    "time": fill_time,
                    "key": cls._normalize_fill_key(
                        str(default_meta["symbol"]),
                        safe_float(default_meta["multiplier"], 1.0),
                    ),
                    "instruction": str(default_meta["instruction"]).upper(),
                    "quantity": quantity,
                    "price": price,
                    "multiplier": safe_float(default_meta["multiplier"], 1.0),
                }
            )

        return fills

    @staticmethod
    def _parse_leg_id(raw_leg_id: object) -> Optional[int]:
        """Parse a Schwab leg ID that may be int-like or string-like."""
        if isinstance(raw_leg_id, int):
            return raw_leg_id
        if isinstance(raw_leg_id, str):
            try:
                return int(raw_leg_id)
            except ValueError:
                return None
        return None

    @staticmethod
    def _parse_order_timestamp(raw_timestamp: object) -> Optional[datetime]:
        """Parse Schwab timestamps, tolerating 'Z' and compact timezone offsets."""
        if raw_timestamp is None:
            return None

        value = str(raw_timestamp).strip()
        if not value:
            return None
        if value.endswith("Z"):
            value = f"{value[:-1]}+00:00"
        if len(value) >= 5 and value[-5] in {"+", "-"} and value[-3] != ":":
            value = f"{value[:-2]}:{value[-2:]}"

        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=EASTERN_TZ)
        return parsed

    @classmethod
    def _normalize_fill_key(cls, symbol: str, multiplier: float) -> str:
        """Normalize fill symbol keys across Schwab symbol formats."""
        normalized = str(symbol).strip().upper()
        if multiplier >= 100:
            option_key = cls._option_symbol_key(normalized)
            if option_key:
                return option_key
        return normalized

    @classmethod
    def _extract_order_terminal_time(cls, order: dict) -> datetime:
        """Best-effort terminal timestamp extraction from an order payload."""
        candidates = [
            order.get("closeTime"),
            order.get("cancelTime"),
            order.get("enteredTime"),
        ]

        for activity in order.get("orderActivityCollection", []):
            if not isinstance(activity, dict):
                continue
            candidates.append(activity.get("executionTime"))
            candidates.append(activity.get("time"))

        for candidate in candidates:
            parsed = cls._parse_order_timestamp(candidate)
            if parsed is not None:
                return parsed

        return cls._now_eastern()

    def _is_order_stale(
        self,
        order: dict,
        *,
        order_id: str,
        side: str,
        stale_minutes: int,
    ) -> bool:
        """Return true when an entry/exit order has exceeded stale threshold."""
        if not self.live_ledger:
            return False

        pending = self.live_ledger.get_position_by_order_id(order_id, side=side)
        ledger_time_raw = None
        if pending:
            ledger_time_raw = (
                pending.get("entry_order_time")
                if side == "entry"
                else pending.get("last_reconciled")
            )
        submitted_at = self._parse_order_timestamp(ledger_time_raw)
        if submitted_at is None:
            submitted_at = self._parse_order_timestamp(order.get("enteredTime"))
        if submitted_at is None:
            submitted_at = self._extract_order_terminal_time(order)

        now_et = self._now_eastern()
        elapsed = now_et - submitted_at.astimezone(EASTERN_TZ)
        return elapsed >= timedelta(minutes=max(1, stale_minutes))

    def _cancel_stale_live_order(self, order_id: str, *, side: str) -> None:
        """Cancel a stale live order and reconcile ledger state immediately."""
        if self.is_paper or not self.live_ledger:
            return

        try:
            self.schwab.cancel_order(order_id)
            logger.warning("Canceled stale live %s order %s.", side, order_id)
            if side == "entry":
                self.live_ledger.reconcile_entry_order(order_id, status="CANCELED")
            else:
                self.live_ledger.reconcile_exit_order(order_id, status="CANCELED")
            self._alert(
                level="WARNING",
                title="Stale live order canceled",
                message=f"{side} order {order_id} canceled after stale timeout",
            )
        except Exception as exc:
            logger.warning("Failed to cancel stale %s order %s: %s", side, order_id, exc)
            self._alert(
                level="ERROR",
                title="Failed stale-order cancel",
                message=f"{side} order {order_id}: {exc}",
            )

    @classmethod
    def _extract_filled_contracts(cls, order: dict, fill_summary: Optional[dict]) -> float:
        """Extract total filled contracts from order payload."""
        raw_filled = safe_float(order.get("filledQuantity"), 0.0)
        if raw_filled > 0:
            return raw_filled
        if fill_summary and fill_summary.get("contracts", 0) > 0:
            return safe_float(fill_summary.get("contracts"), 0.0)
        return 0.0

    @classmethod
    def _summarize_order_fill(cls, order: dict) -> Optional[dict]:
        """Summarize filled order cashflow into contracts and per-contract price."""
        fills = cls._extract_order_fills([order])
        if not fills:
            return None

        net_cash = 0.0
        contracts = 0.0
        for fill in fills:
            instruction = str(fill.get("instruction", "")).upper()
            direction = 1.0 if instruction.startswith("SELL") else -1.0
            quantity = safe_float(fill.get("quantity"), 0.0)
            price = safe_float(fill.get("price"), 0.0)
            multiplier = safe_float(fill.get("multiplier"), 1.0)
            contracts = max(contracts, quantity)
            net_cash += direction * quantity * price * multiplier

        if contracts <= 0:
            return None

        per_contract = abs(net_cash) / (contracts * 100.0)
        return {
            "contracts": contracts,
            "net_cash": net_cash,
            "per_contract": round(per_contract, 2),
        }

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
            if not self.risk_manager.can_open_more_positions():
                logger.info(
                    "Max open positions reached (%d). Stopping entry scan.",
                    self.config.risk.max_open_positions,
                )
                break

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
                    if not self.risk_manager.can_open_more_positions():
                        break

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
        if analysis is None:
            logger.warning("Signal %s on %s missing analysis. Skipping.", signal.strategy, signal.symbol)
            return False
        effective_max_loss = self.risk_manager.effective_max_loss_per_contract(signal)

        logger.info(
            "EXECUTING TRADE: %s on %s | %d contracts | "
            "Credit: $%.2f | Max loss(proxy): $%.2f | POP: %.1f%% | Score: %.1f",
            signal.strategy, signal.symbol, signal.quantity,
            analysis.credit, effective_max_loss,
            analysis.probability_of_profit * 100, analysis.score,
        )

        details = self._build_position_details(analysis)

        if self.is_paper:
            result = self.paper_trader.execute_open(
                strategy=signal.strategy,
                symbol=signal.symbol,
                credit=analysis.credit,
                max_loss=effective_max_loss,
                quantity=signal.quantity,
                details=details,
            )
            logger.info("Paper trade opened: %s", result)
            self.risk_manager.register_open_position(
                symbol=signal.symbol,
                max_loss_per_contract=effective_max_loss,
                quantity=signal.quantity,
                strategy=signal.strategy,
            )
            return True

        opened = self._execute_live_entry(
            signal,
            details=details,
            max_loss_per_contract=effective_max_loss,
        )
        if opened:
            self.risk_manager.register_open_position(
                symbol=signal.symbol,
                max_loss_per_contract=effective_max_loss,
                quantity=signal.quantity,
                strategy=signal.strategy,
            )
        return opened

    @staticmethod
    def _build_position_details(analysis) -> dict:
        """Serialize strategy analysis fields used for lifecycle management."""
        return {
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
        if self.news_scanner:
            try:
                context["news"] = self.news_scanner.build_context(signal.symbol)
            except Exception as exc:
                logger.warning(
                    "Failed to fetch news context for %s: %s",
                    signal.symbol,
                    exc,
                )

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

    def _execute_live_entry(
        self,
        signal: TradeSignal,
        details: Optional[dict] = None,
        max_loss_per_contract: Optional[float] = None,
    ) -> bool:
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
            if self.live_ledger and analysis is not None:
                order_id = str(result.get("order_id", "")).strip()
                if order_id:
                    self.live_ledger.register_entry_order(
                        strategy=signal.strategy,
                        symbol=signal.symbol,
                        quantity=quantity,
                        max_loss=max_loss_per_contract if max_loss_per_contract is not None else analysis.max_loss,
                        entry_credit=analysis.credit,
                        details=details or self._build_position_details(analysis),
                        entry_order_id=order_id,
                        entry_order_status=str(result.get("status", "PLACED")),
                    )
                else:
                    logger.warning(
                        "Live order missing broker order_id; recording as immediately open."
                    )
                    self.live_ledger.register_entry_order(
                        strategy=signal.strategy,
                        symbol=signal.symbol,
                        quantity=quantity,
                        max_loss=max_loss_per_contract if max_loss_per_contract is not None else analysis.max_loss,
                        entry_credit=analysis.credit,
                        details=details or self._build_position_details(analysis),
                        entry_order_id="",
                        entry_order_status="FILLED",
                        opened_at=self._now_eastern().isoformat(),
                    )
            return True

        except Exception as e:
            logger.error("Failed to place live order: %s", e)
            logger.debug(traceback.format_exc())
            self._alert(
                level="ERROR",
                title="Live entry order failed",
                message=str(e),
                context={"strategy": signal.strategy, "symbol": signal.symbol},
            )
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
            closed = self._execute_live_exit(signal)
            if not closed:
                logger.warning(
                    "Failed to submit live close order for %s.",
                    signal.position_id,
                )

    def _execute_live_exit(self, signal: TradeSignal) -> bool:
        """Execute a live exit order for a tracked strategy position."""
        if not self.live_ledger or not signal.position_id:
            return False

        tracked = self.live_ledger.get_position(signal.position_id)
        if not tracked:
            logger.warning("Live position %s not found in ledger.", signal.position_id)
            return False

        if str(tracked.get("status", "")).lower() != "open":
            logger.info(
                "Skipping live exit for %s; current status is %s.",
                signal.position_id,
                tracked.get("status"),
            )
            return False

        details = tracked.get("details", {})
        expiration = self._extract_expiration_key(
            details.get("expiration", tracked.get("expiration", ""))
        )
        if not expiration:
            logger.warning("Tracked live position %s missing expiration.", signal.position_id)
            return False

        quantity = max(1, int(tracked.get("quantity", 1)))
        strategy = tracked.get("strategy", "")
        symbol = tracked.get("symbol", signal.symbol)
        debit_limit = self._resolve_live_close_debit(tracked)

        try:
            if strategy == "bull_put_spread":
                order = self.schwab.build_bull_put_spread_close(
                    symbol=symbol,
                    expiration=expiration,
                    short_strike=float(details.get("short_strike", 0.0)),
                    long_strike=float(details.get("long_strike", 0.0)),
                    quantity=quantity,
                    price=debit_limit,
                )
            elif strategy == "bear_call_spread":
                order = self.schwab.build_bear_call_spread_close(
                    symbol=symbol,
                    expiration=expiration,
                    short_strike=float(details.get("short_strike", 0.0)),
                    long_strike=float(details.get("long_strike", 0.0)),
                    quantity=quantity,
                    price=debit_limit,
                )
            elif strategy == "covered_call":
                order = self.schwab.build_covered_call_close(
                    symbol=symbol,
                    expiration=expiration,
                    short_strike=float(details.get("short_strike", 0.0)),
                    quantity=quantity,
                    price=debit_limit,
                )
            elif strategy == "iron_condor":
                order = self.schwab.build_iron_condor_close(
                    symbol=symbol,
                    expiration=expiration,
                    put_long_strike=float(details.get("put_long_strike", 0.0)),
                    put_short_strike=float(details.get("put_short_strike", 0.0)),
                    call_short_strike=float(details.get("call_short_strike", 0.0)),
                    call_long_strike=float(details.get("call_long_strike", 0.0)),
                    quantity=quantity,
                    price=debit_limit,
                )
            else:
                logger.warning("Live exit not implemented for strategy: %s", strategy)
                return False

            result = self.schwab.place_order(order)
            logger.info("LIVE exit order placed: %s", result)

            order_id = str(result.get("order_id", "")).strip()
            if not order_id:
                logger.warning(
                    "Live exit order for %s missing broker order_id.",
                    signal.position_id,
                )
                return False

            self.live_ledger.register_exit_order(
                position_id=signal.position_id,
                exit_order_id=order_id,
                reason=signal.reason,
            )
            return True
        except Exception as exc:
            logger.error("Failed to place live close order: %s", exc)
            logger.debug(traceback.format_exc())
            self._alert(
                level="ERROR",
                title="Live close order failed",
                message=str(exc),
                context={"strategy": strategy, "symbol": symbol, "position_id": signal.position_id},
            )
            return False

    @staticmethod
    def _resolve_live_close_debit(position: dict) -> float:
        """Choose a positive close debit limit from latest mark or entry credit."""
        mark = safe_float(position.get("current_value"), 0.0)
        if mark > 0:
            return round(mark, 2)

        entry_credit = safe_float(position.get("entry_credit"), 0.0)
        if entry_credit > 0:
            return round(max(0.01, entry_credit), 2)

        return 0.05

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
        """Return strategy-normalized positions for risk checks and exits."""
        if self.is_paper:
            return self.paper_trader.get_positions() if self.paper_trader else []

        if not self.live_ledger:
            return []

        broker_positions = self._get_broker_positions()
        if broker_positions is not None:
            open_option_symbols = self._collect_broker_option_symbols(broker_positions)
            self.live_ledger.close_missing_from_broker(
                open_strategy_symbols=open_option_symbols,
                position_symbol_resolver=self._resolve_live_option_symbols,
                close_metadata_resolver=self._resolve_external_close_metadata,
            )

        tracked = self.live_ledger.list_positions(
            statuses={"opening", "open", "closing"}
        )
        self._refresh_live_position_values(tracked)
        return tracked

    def _bootstrap_live_ledger_from_broker(self) -> int:
        """Import existing broker option positions into live ledger once."""
        if self.is_paper or not self.live_ledger:
            return 0

        broker_positions = self._get_broker_positions()
        if not broker_positions:
            return 0

        option_legs: list[dict] = []
        for pos in broker_positions:
            if not isinstance(pos, dict):
                continue
            instrument = pos.get("instrument", {})
            if not isinstance(instrument, dict):
                continue
            if str(instrument.get("assetType", "")).upper() != "OPTION":
                continue

            raw_symbol = str(instrument.get("symbol", "")).strip()
            parsed = self._parse_option_symbol(raw_symbol)
            if not parsed:
                continue

            long_qty = safe_float(pos.get("longQuantity"), 0.0)
            short_qty = safe_float(pos.get("shortQuantity"), 0.0)
            net_qty = long_qty - short_qty
            if abs(net_qty) < 1e-9:
                continue

            option_legs.append(
                {
                    "raw_symbol": raw_symbol,
                    "underlying": parsed["underlying"],
                    "expiration": parsed["expiration"],
                    "contract_type": parsed["contract_type"],
                    "strike": parsed["strike"],
                    "net_qty": net_qty,
                }
            )

        if not option_legs:
            return 0

        tracked_sets = [
            self._resolve_live_option_symbols(pos)
            for pos in self.live_ledger.list_positions(statuses={"opening", "open", "closing"})
        ]

        grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
        for leg in option_legs:
            qty_contracts = max(1, int(round(abs(leg["net_qty"]))))
            key = (leg["underlying"], leg["expiration"], qty_contracts)
            grouped[key].append(leg)

        imported = 0
        for (underlying, expiration, qty), legs in grouped.items():
            inferred = self._infer_bootstrap_position(
                underlying=underlying,
                expiration=expiration,
                quantity=qty,
                legs=legs,
            )
            if not inferred:
                continue

            symbols = inferred["symbols"]
            if not symbols:
                continue
            if any(symbols.issubset(existing) for existing in tracked_sets if existing):
                continue

            self.live_ledger.register_entry_order(
                strategy=inferred["strategy"],
                symbol=underlying,
                quantity=qty,
                max_loss=inferred["max_loss"],
                entry_credit=inferred["entry_credit"],
                details=inferred["details"],
                entry_order_id="",
                entry_order_status="FILLED",
                opened_at=self._now_eastern().isoformat(),
            )
            tracked_sets.append(symbols)
            imported += 1

        return imported

    def _infer_bootstrap_position(
        self,
        *,
        underlying: str,
        expiration: str,
        quantity: int,
        legs: list[dict],
    ) -> Optional[dict]:
        """Infer strategy metadata for pre-existing broker option positions."""
        short_puts = sorted(
            [l for l in legs if l["contract_type"] == "P" and l["net_qty"] < 0],
            key=lambda x: x["strike"],
        )
        long_puts = sorted(
            [l for l in legs if l["contract_type"] == "P" and l["net_qty"] > 0],
            key=lambda x: x["strike"],
        )
        short_calls = sorted(
            [l for l in legs if l["contract_type"] == "C" and l["net_qty"] < 0],
            key=lambda x: x["strike"],
        )
        long_calls = sorted(
            [l for l in legs if l["contract_type"] == "C" and l["net_qty"] > 0],
            key=lambda x: x["strike"],
        )

        def _symbols(details: dict, strategy: str) -> set[str]:
            sample = {
                "strategy": strategy,
                "symbol": underlying,
                "details": details,
                "expiration": expiration,
            }
            return self._resolve_live_option_symbols(sample)

        if short_puts and long_puts and short_calls and long_calls:
            put_short = short_puts[-1]["strike"]
            put_long_candidates = [l["strike"] for l in long_puts if l["strike"] < put_short]
            call_short = short_calls[0]["strike"]
            call_long_candidates = [l["strike"] for l in long_calls if l["strike"] > call_short]
            if put_long_candidates and call_long_candidates:
                put_long = max(put_long_candidates)
                call_long = min(call_long_candidates)
                width = max(put_short - put_long, call_long - call_short)
                details = {
                    "expiration": expiration,
                    "put_short_strike": put_short,
                    "put_long_strike": put_long,
                    "call_short_strike": call_short,
                    "call_long_strike": call_long,
                    "bootstrap_import": True,
                }
                return {
                    "strategy": "iron_condor",
                    "max_loss": round(max(width, 0.0), 2),
                    "entry_credit": 0.0,
                    "details": details,
                    "symbols": _symbols(details, "iron_condor"),
                }

        if short_puts and long_puts:
            put_short = short_puts[-1]["strike"]
            put_long_candidates = [l["strike"] for l in long_puts if l["strike"] < put_short]
            if put_long_candidates:
                put_long = max(put_long_candidates)
                width = put_short - put_long
                details = {
                    "expiration": expiration,
                    "short_strike": put_short,
                    "long_strike": put_long,
                    "bootstrap_import": True,
                }
                return {
                    "strategy": "bull_put_spread",
                    "max_loss": round(max(width, 0.0), 2),
                    "entry_credit": 0.0,
                    "details": details,
                    "symbols": _symbols(details, "bull_put_spread"),
                }

        if short_calls and long_calls:
            call_short = short_calls[0]["strike"]
            call_long_candidates = [l["strike"] for l in long_calls if l["strike"] > call_short]
            if call_long_candidates:
                call_long = min(call_long_candidates)
                width = call_long - call_short
                details = {
                    "expiration": expiration,
                    "short_strike": call_short,
                    "long_strike": call_long,
                    "bootstrap_import": True,
                }
                return {
                    "strategy": "bear_call_spread",
                    "max_loss": round(max(width, 0.0), 2),
                    "entry_credit": 0.0,
                    "details": details,
                    "symbols": _symbols(details, "bear_call_spread"),
                }

        if short_calls and not long_calls and not short_puts and not long_puts:
            call_short = short_calls[0]["strike"]
            notional_proxy = max(0.0, call_short) * (
                self.config.risk.covered_call_notional_risk_pct / 100.0
            )
            details = {
                "expiration": expiration,
                "short_strike": call_short,
                "bootstrap_import": True,
            }
            return {
                "strategy": "covered_call",
                "max_loss": round(max(notional_proxy, 0.25), 2),
                "entry_credit": 0.0,
                "details": details,
                "symbols": _symbols(details, "covered_call"),
            }

        return None

    def _get_broker_positions(self) -> Optional[list]:
        """Fetch raw live broker positions."""
        try:
            positions = self.schwab.get_positions()
            if isinstance(positions, list):
                return positions
            return []
        except Exception as e:
            logger.error("Failed to fetch broker positions: %s", e)
            return None

    @staticmethod
    def _collect_broker_option_symbols(positions: list) -> set[str]:
        """Collect canonical option keys with non-zero net quantity from broker payload."""
        symbols: set[str] = set()
        for pos in positions:
            if not isinstance(pos, dict):
                continue
            instrument = pos.get("instrument", {})
            if not isinstance(instrument, dict):
                continue
            if str(instrument.get("assetType", "")).upper() != "OPTION":
                continue

            symbol = str(instrument.get("symbol", "")).strip()
            if not symbol:
                continue
            key = TradingBot._option_symbol_key(symbol)
            if not key:
                continue

            long_qty = safe_float(pos.get("longQuantity"), 0.0)
            short_qty = safe_float(pos.get("shortQuantity"), 0.0)
            if abs(long_qty - short_qty) <= 1e-9:
                continue
            symbols.add(key)

        return symbols

    def _resolve_live_option_symbols(self, position: dict) -> set[str]:
        """Build canonical option keys for a tracked strategy position."""
        strategy = str(position.get("strategy", ""))
        symbol = str(position.get("symbol", "")).strip()
        details = position.get("details", {})
        expiration = self._extract_expiration_key(
            details.get("expiration", position.get("expiration", ""))
        )
        if not symbol or not expiration:
            return set()

        def build(contract_type: str, strike_key: str) -> Optional[str]:
            strike = details.get(strike_key)
            if strike in (None, ""):
                return None
            try:
                return self._build_option_key(
                    underlying=symbol,
                    expiration=expiration,
                    contract_type=contract_type,
                    strike=float(strike),
                )
            except Exception:
                return None

        if strategy == "bull_put_spread":
            return {
                s
                for s in (
                    build("P", "short_strike"),
                    build("P", "long_strike"),
                )
                if s
            }
        if strategy == "bear_call_spread":
            return {
                s
                for s in (
                    build("C", "short_strike"),
                    build("C", "long_strike"),
                )
                if s
            }
        if strategy == "covered_call":
            leg = build("C", "short_strike")
            return {leg} if leg else set()
        if strategy == "iron_condor":
            return {
                s
                for s in (
                    build("P", "put_short_strike"),
                    build("P", "put_long_strike"),
                    build("C", "call_short_strike"),
                    build("C", "call_long_strike"),
                )
                if s
            }

        return set()

    @staticmethod
    def _build_option_key(
        *, underlying: str, expiration: str, contract_type: str, strike: float
    ) -> str:
        """Build a canonical option key for cross-format matching."""
        normalized_exp = TradingBot._extract_expiration_key(expiration)
        normalized_cp = "C" if str(contract_type).upper().startswith("C") else "P"
        return f"{str(underlying).upper()}|{normalized_exp}|{normalized_cp}|{float(strike):.3f}"

    @staticmethod
    def _parse_option_symbol(raw_symbol: str) -> Optional[dict]:
        """Parse OCC compact or underscore option symbols into components."""
        if not raw_symbol:
            return None
        compact = str(raw_symbol).strip().upper().replace(" ", "")
        if not compact:
            return None

        match = OCC_COMPACT_PATTERN.match(compact)
        if match:
            underlying, yy, mm, dd, cp, strike_int = match.groups()
            try:
                expiration = datetime.strptime(f"20{yy}-{mm}-{dd}", "%Y-%m-%d").date()
            except ValueError:
                return None
            strike = int(strike_int) / 1000.0
            return {
                "underlying": underlying,
                "expiration": expiration.isoformat(),
                "contract_type": cp,
                "strike": strike,
            }

        match = UNDERSCORE_OPTION_PATTERN.match(compact)
        if match:
            underlying, mm, dd, yy, cp, strike_raw = match.groups()
            try:
                expiration = datetime.strptime(f"20{yy}-{mm}-{dd}", "%Y-%m-%d").date()
            except ValueError:
                return None
            return {
                "underlying": underlying,
                "expiration": expiration.isoformat(),
                "contract_type": cp,
                "strike": float(strike_raw),
            }

        return None

    @classmethod
    def _option_symbol_key(cls, raw_symbol: str) -> str:
        """Normalize any supported option symbol into a canonical key."""
        if "|" in str(raw_symbol):
            return str(raw_symbol).strip().upper()
        parsed = cls._parse_option_symbol(raw_symbol)
        if not parsed:
            return ""
        return cls._build_option_key(
            underlying=parsed["underlying"],
            expiration=parsed["expiration"],
            contract_type=parsed["contract_type"],
            strike=parsed["strike"],
        )

    def _refresh_live_position_values(self, positions: list[dict]) -> None:
        """Update tracked live positions with current mark and DTE values."""
        if not positions or not self.live_ledger:
            return

        chain_cache: dict[str, dict] = {}
        for position in positions:
            position_id = position.get("position_id")
            symbol = position.get("symbol")
            if not position_id or not symbol:
                continue

            status = str(position.get("status", "")).lower()
            if status not in {"open", "closing"}:
                continue

            if symbol not in chain_cache:
                chain_data, _ = self._get_chain_data(symbol)
                chain_cache[symbol] = chain_data

            chain_data = chain_cache.get(symbol, {})
            if not chain_data:
                continue

            mark = self._estimate_paper_position_value(position, chain_data)
            dte_remaining = position.get("dte_remaining")
            expiration = self._extract_expiration_key(
                (position.get("details") or {}).get("expiration", position.get("expiration", ""))
            )
            if expiration:
                try:
                    exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
                    dte_remaining = (exp_date - self._now_eastern().date()).days
                    position["dte_remaining"] = dte_remaining
                except ValueError:
                    pass

            if mark is not None:
                position["current_value"] = mark

            self.live_ledger.update_position_quote(
                position_id,
                current_value=mark,
                dte_remaining=dte_remaining if isinstance(dte_remaining, int) else None,
            )

    def _resolve_external_close_metadata(self, position: dict, symbols: set[str]) -> dict:
        """Estimate close metadata for positions closed outside bot-managed orders."""
        metadata: dict = {"exit_order_status": "EXTERNAL", "exit_reason": "external_close"}
        if not symbols:
            return metadata

        try:
            orders = self.schwab.get_orders(days_back=60)
        except Exception:
            return metadata
        if not isinstance(orders, list):
            return metadata

        open_date = self._parse_order_timestamp(position.get("open_date"))
        relevant_orders: list[dict] = []
        exit_reason = "external_close"
        for order in orders:
            if not isinstance(order, dict):
                continue

            order_symbols: set[str] = set()
            for leg in order.get("orderLegCollection", []):
                if not isinstance(leg, dict):
                    continue
                instrument = leg.get("instrument", {})
                if not isinstance(instrument, dict):
                    continue
                symbol = str(instrument.get("symbol", "")).strip()
                if not symbol:
                    continue
                key = self._option_symbol_key(symbol)
                order_symbols.add(key or symbol.upper())

            if not order_symbols.intersection(symbols):
                continue

            order_type = str(order.get("orderType", "")).upper()
            if order_type == "EXERCISE":
                exit_reason = "exercise_or_assignment"
            relevant_orders.append(order)

        if not relevant_orders:
            return metadata

        close_cash = 0.0
        close_contracts = 0.0
        latest_time: Optional[datetime] = None

        for fill in self._extract_order_fills(relevant_orders):
            fill_symbol = str(fill.get("key", "")).strip()
            if fill_symbol not in symbols:
                continue

            fill_time = fill.get("time")
            if isinstance(fill_time, datetime):
                if open_date and fill_time < open_date:
                    continue
                if latest_time is None or fill_time > latest_time:
                    latest_time = fill_time

            instruction = str(fill.get("instruction", "")).upper()
            if instruction in OPEN_LONG_INSTRUCTIONS or instruction in OPEN_SHORT_INSTRUCTIONS:
                continue

            direction = 1.0 if instruction.startswith("SELL") else -1.0
            quantity = safe_float(fill.get("quantity"), 0.0)
            price = safe_float(fill.get("price"), 0.0)
            multiplier = safe_float(fill.get("multiplier"), 1.0)
            close_cash += direction * quantity * price * multiplier
            close_contracts = max(close_contracts, quantity)

        quantity = max(1, safe_int(position.get("quantity", 1), 1))
        entry_credit = max(0.0, safe_float(position.get("entry_credit"), 0.0))
        realized = (entry_credit * quantity * 100.0) + close_cash
        close_value = (
            abs(close_cash) / (max(1.0, close_contracts) * 100.0)
            if close_contracts > 0
            else max(0.0, safe_float(position.get("current_value"), 0.0))
        )

        metadata.update(
            {
                "exit_reason": exit_reason,
                "close_value": round(close_value, 2),
                "realized_pnl": round(realized, 2),
            }
        )
        if latest_time is not None:
            metadata["close_date"] = latest_time.isoformat()

        return metadata

    def _get_live_equity_shares(self, symbol: str) -> int:
        """Return net long shares for a symbol from live account positions."""
        total = 0.0
        for pos in self._get_broker_positions() or []:
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

        if not self._is_market_open_now():
            logger.info("Market is closed. Skipping scan.")
            return

        self.scan_and_trade()

    def _scheduled_position_check(self) -> None:
        """Position monitoring wrapper."""
        if not self._is_market_open_now() and not self.is_paper:
            self._reconcile_live_orders()
            return

        try:
            self._update_portfolio_state()
            self._check_exits()
        except Exception as e:
            logger.error("Error during position check: %s", e)
            self._alert(
                level="ERROR",
                title="Position check failed",
                message=str(e),
            )

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
            return

        ledger_summary = (
            self.live_ledger.summary(today_iso=self._now_eastern().date().isoformat())
            if self.live_ledger
            else {}
        )
        balance = 0.0
        try:
            balance = self.schwab.get_account_balance()
        except Exception as exc:
            logger.warning("Failed to fetch live balance for report: %s", exc)

        logger.info("=" * 60)
        logger.info("LIVE DAILY PERFORMANCE REPORT")
        logger.info("=" * 60)
        logger.info("Balance: $%s", f"{balance:,.2f}")
        logger.info("Open strategy positions: %d", ledger_summary.get("open", 0))
        logger.info("Pending entries: %d", ledger_summary.get("opening", 0))
        logger.info("Pending exits: %d", ledger_summary.get("closing", 0))
        logger.info("Closed today: %d", ledger_summary.get("closed_today", 0))
        logger.info(
            "Ledger realized P/L today: $%s",
            f"{ledger_summary.get('realized_pnl_today', 0.0):,.2f}",
        )
        logger.info(
            "Order-history daily P/L: $%s",
            f"{self._compute_live_daily_pnl():,.2f}",
        )
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

        # Run an initial scan immediately when the market is open.
        if self.is_paper or self._is_market_open_now():
            logger.info("Running initial scan...")
            self.scan_and_trade()
        else:
            logger.info("Skipping initial scan because market is closed.")

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
        self._daily_report()

    def stop(self) -> None:
        """Stop the bot gracefully."""
        self._running = False
        logger.info("Bot stop requested.")
