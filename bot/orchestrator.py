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
import copy
from dataclasses import fields, is_dataclass
import json
import logging
from pathlib import Path
import re
import signal
import time
import traceback
import uuid
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import schedule

from bot.alerts import AlertManager
from bot.analysis import SpreadAnalysis
from bot.config import BotConfig, load_config
from bot.data_store import dump_json, load_json
from bot.dashboard import generate_dashboard
from bot.econ_calendar import EconomicCalendar
from bot.file_security import atomic_write_private
from bot.hedger import PortfolioHedger
from bot.llm_strategist import LLMStrategist
from bot.live_trade_ledger import LiveTradeLedger
from bot.schwab_client import SchwabClient
from bot.llm_advisor import LLMAdvisor
from bot.news_scanner import NewsScanner
from bot.number_utils import safe_float, safe_int
from bot.options_flow import OptionsFlowAnalyzer
from bot.paper_trader import PaperTrader
from bot.adjustments import AdjustmentEngine
from bot.regime_detector import (
    BULL_TREND,
    BEAR_TREND,
    HIGH_VOL_CHOP,
    LOW_VOL_GRIND,
    CRASH_CRISIS,
    CRASH_CRIISIS,
    MEAN_REVERSION,
    MarketRegimeDetector,
    RegimeState,
)
from bot.roll_manager import RollManager
from bot.risk_manager import RiskManager
from bot.market_scanner import MarketScanner, SECTOR_ETF_BY_SYMBOL
from bot.iv_history import IVHistory
from bot.technicals import TechnicalAnalyzer
from bot.vol_surface import VolSurfaceAnalyzer
from bot.strategies.broken_wing_butterfly import BrokenWingButterflyStrategy
from bot.strategies.calendar_spreads import CalendarSpreadStrategy
from bot.strategies.credit_spreads import CreditSpreadStrategy
from bot.strategies.earnings_vol_crush import EarningsVolCrushStrategy
from bot.strategies.iron_condors import IronCondorStrategy
from bot.strategies.naked_puts import NakedPutStrategy
from bot.strategies.covered_calls import CoveredCallStrategy
from bot.strategies.strangles import StranglesStrategy
from bot.strategies.base import TradeSignal

logger = logging.getLogger(__name__)

LIVE_SUPPORTED_ENTRY_STRATEGIES = {
    "credit_spreads",
    "covered_calls",
    "iron_condors",
    "naked_puts",
    "calendar_spreads",
}
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
HEDGE_COSTS_PATH = Path("bot/data/hedge_costs.json")
AUDIT_LOG_PATH = Path("bot/data/audit_log.jsonl")


class TradingBot:
    """Fully automated options trading bot."""

    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or load_config()
        self.is_paper = self.config.trading_mode == "paper"
        self._running = False
        self._market_open_cache: Optional[tuple[datetime, bool]] = None
        self._live_bootstrap_done = False
        self._base_risk_config = copy.deepcopy(self.config.risk)
        self._active_account: Optional[dict] = None
        self._equity_history: list[dict] = []
        self._last_heartbeat_time: Optional[datetime] = None
        self._last_stream_retry_time: Optional[datetime] = None
        self._dashboard_generated_date: Optional[str] = None
        self._signal_handlers_ready = False
        self._cycle_size_scalar: float = 1.0
        self.iv_history = IVHistory()
        self.current_regime_state = RegimeState(
            regime=LOW_VOL_GRIND,
            confidence=0.5,
        )
        self._chain_history_cache: dict[str, dict] = {}
        self._stream_option_symbols: set[str] = set()
        self._last_stream_exit_check: dict[str, datetime] = {}
        self._strategy_pause_until: dict[str, datetime] = {}
        self._symbol_pause_until: dict[str, datetime] = {}
        self._service_degradation: dict[str, bool] = {
            "llm_down": False,
            "news_down": False,
            "scanner_down": False,
            "stream_down": False,
            "rule_only_mode": False,
        }
        self._strategy_loss_streaks: dict[str, int] = defaultdict(int)
        self._symbol_loss_streaks: dict[str, int] = defaultdict(int)
        self._portfolio_halt_until: Optional[datetime] = None
        self._api_health_window: deque = deque(maxlen=256)
        self._llm_timeout_streak: int = 0

        self._log_config_overrides()

        # Initialize components
        self.risk_manager = RiskManager(self.config.risk)
        self.risk_manager.set_sizing_config(self.config.sizing)
        self.paper_trader = PaperTrader() if self.is_paper else None
        self.live_ledger = None if self.is_paper else LiveTradeLedger()
        self.alerts = AlertManager(self.config.alerts)
        self.technicals = TechnicalAnalyzer()
        self.llm_advisor: Optional[LLMAdvisor] = None
        if self.config.llm.enabled:
            self.llm_advisor = LLMAdvisor(self.config.llm)
        self.news_scanner: Optional[NewsScanner] = None
        if self.config.news.enabled:
            self.news_scanner = NewsScanner(self.config.news)
        self.regime_detector: Optional[MarketRegimeDetector] = None
        if self.config.regime.enabled:
            self.regime_detector = MarketRegimeDetector(
                get_price_history=lambda symbol, days: self.schwab.get_price_history(symbol, days),
                get_quote=lambda symbol: self.schwab.get_quote(symbol),
                get_option_chain=lambda symbol: self.schwab.get_option_chain(symbol),
                config=vars(self.config.regime),
            )
        self.vol_surface_analyzer: Optional[VolSurfaceAnalyzer] = None
        if self.config.vol_surface.enabled:
            self.vol_surface_analyzer = VolSurfaceAnalyzer(self.iv_history)
        self.econ_calendar: Optional[EconomicCalendar] = None
        if self.config.econ_calendar.enabled:
            self.econ_calendar = EconomicCalendar(
                cache_path=self.config.econ_calendar.cache_file,
                refresh_days=self.config.econ_calendar.refresh_days,
                policy={
                    "high": self.config.econ_calendar.high_severity_policy,
                    "medium": self.config.econ_calendar.medium_severity_policy,
                    "low": self.config.econ_calendar.low_severity_policy,
                },
            )
        self.options_flow_analyzer: Optional[OptionsFlowAnalyzer] = None
        if self.config.options_flow.enabled:
            self.options_flow_analyzer = OptionsFlowAnalyzer(
                unusual_volume_multiple=self.config.options_flow.unusual_volume_multiple
            )
        self.roll_manager = RollManager(vars(self.config.rolling))
        self.adjustment_engine = AdjustmentEngine(vars(self.config.adjustments))
        self.hedger = PortfolioHedger(vars(self.config.hedging))
        self.llm_strategist: Optional[LLMStrategist] = None
        if self.config.llm_strategist.enabled:
            self.llm_strategist = LLMStrategist(self.config.llm_strategist)

        # Market data is required in both paper and live modes.
        self.schwab = SchwabClient(self.config.schwab)
        self.risk_manager.set_price_history_provider(self.schwab.get_price_history)
        self._configure_active_account()

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
        if self.config.naked_puts.enabled:
            self.strategies.append(
                NakedPutStrategy(vars(self.config.naked_puts))
            )
        if self.config.calendar_spreads.enabled:
            self.strategies.append(
                CalendarSpreadStrategy(vars(self.config.calendar_spreads))
            )
        if self.config.strangles.enabled:
            self.strategies.append(
                StranglesStrategy(vars(self.config.strangles))
            )
        if self.config.broken_wing_butterfly.enabled:
            self.strategies.append(
                BrokenWingButterflyStrategy(vars(self.config.broken_wing_butterfly))
            )
        if self.config.earnings_vol_crush.enabled:
            self.strategies.append(
                EarningsVolCrushStrategy(vars(self.config.earnings_vol_crush))
            )

        self._apply_exit_overrides_to_strategies()
        self._base_strategy_min_credit = {
            strategy.name: float(strategy.config.get("min_credit_pct", 0.0))
            for strategy in self.strategies
        }
        self.circuit_state = {
            "regime": "normal",
            "vix": None,
            "halt_entries": False,
            "consecutive_loss_pause_until": None,
            "weekly_loss_pause_until": None,
        }
        self._breaker_alert_flags = {
            "crisis": False,
            "consecutive_loss": False,
            "weekly_loss": False,
        }

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

    def _log_config_overrides(self) -> None:
        """Log user overrides relative to BotConfig defaults."""
        try:
            baseline = BotConfig()
            diffs: list[str] = []
            self._collect_config_diffs("", self.config, baseline, diffs)
            if diffs:
                logger.info("Config overrides: %s", ", ".join(diffs))
        except Exception as exc:
            logger.debug("Could not compute config override diff: %s", exc)

    def _collect_config_diffs(self, path: str, current, default, diffs: list[str]) -> None:
        """Recursively collect changed config leaf values."""
        if is_dataclass(current) and is_dataclass(default):
            for field in fields(current):
                key = field.name
                self._collect_config_diffs(
                    f"{path}.{key}" if path else key,
                    getattr(current, key),
                    getattr(default, key),
                    diffs,
                )
            return

        if isinstance(current, dict) and isinstance(default, dict):
            keys = sorted(set(current.keys()) | set(default.keys()))
            for key in keys:
                self._collect_config_diffs(
                    f"{path}.{key}" if path else str(key),
                    current.get(key),
                    default.get(key),
                    diffs,
                )
            return

        if isinstance(current, list) and isinstance(default, list):
            if current != default:
                diffs.append(
                    f"{path}={self._safe_config_value(current)} "
                    f"(default: {self._safe_config_value(default)})"
                )
            return

        if current != default:
            diffs.append(
                f"{path}={self._safe_config_value(current)} "
                f"(default: {self._safe_config_value(default)})"
            )

    @staticmethod
    def _safe_config_value(value):
        text = str(value)
        lowered = text.lower()
        if any(token in lowered for token in ("secret", "token", "hash", "api_key", "webhook")):
            return "<redacted>"
        return value

    def _configure_active_account(self) -> None:
        """Select active account/profile for live mode from config accounts list."""
        if self.is_paper:
            return

        configured = [
            {
                "name": str(account.name).strip(),
                "hash": str(account.hash).strip(),
                "risk_profile": str(account.risk_profile).strip() or "moderate",
            }
            for account in (self.config.schwab.accounts or [])
            if str(getattr(account, "hash", "")).strip()
        ]
        if not configured:
            try:
                configured = self.schwab.configured_accounts()
            except Exception as exc:
                logger.debug("Could not fetch configured accounts from Schwab client: %s", exc)
                configured = []
        if not configured:
            return

        selected = configured[0]
        configured_hash = str(self.config.schwab.account_hash or "").strip()
        if configured_hash:
            for account in configured:
                if str(account.get("hash", "")).strip() == configured_hash:
                    selected = account
                    break
        self._active_account = selected

        account_hash = str(selected.get("hash", "")).strip()
        if account_hash:
            self.schwab.select_account(account_hash)

        profile = str(selected.get("risk_profile", "moderate")).strip().lower()
        self._apply_risk_profile(profile)
        logger.info(
            "Active account profile: %s (...%s) risk=%s",
            selected.get("name") or "account",
            account_hash[-4:] if account_hash else "auto",
            profile,
        )

    def _apply_risk_profile(self, profile: str) -> None:
        """Apply account risk-profile multipliers to runtime risk config."""
        self.risk_manager.config = copy.deepcopy(self._base_risk_config)
        named_profiles = self.config.risk_profiles if isinstance(self.config.risk_profiles, dict) else {}
        profile_key = str(profile or "").strip().lower()
        preset = named_profiles.get(profile_key)
        if preset:
            self.risk_manager.config.max_portfolio_risk_pct = float(preset.max_portfolio_risk_pct)
            self.risk_manager.config.max_position_risk_pct = float(preset.max_position_risk_pct)
            self.risk_manager.config.max_open_positions = int(preset.max_open_positions)
            self.risk_manager.config.max_daily_loss_pct = float(preset.max_daily_loss_pct)
            return

        # Backward-compatible fallback for unknown profile names.
        if profile == "aggressive":
            self.risk_manager.config.max_open_positions = max(
                1, int(round(self.risk_manager.config.max_open_positions * 1.5))
            )
            self.risk_manager.config.max_position_risk_pct *= 1.25
            self.risk_manager.config.max_portfolio_risk_pct *= 1.25
        elif profile == "conservative":
            self.risk_manager.config.max_open_positions = max(
                1, int(round(self.risk_manager.config.max_open_positions * 0.7))
            )
            self.risk_manager.config.max_position_risk_pct *= 0.70
            self.risk_manager.config.max_portfolio_risk_pct *= 0.70

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

    def _append_audit_event(
        self,
        *,
        event_type: str,
        details: dict,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Append a structured lifecycle event to the JSONL audit trail."""
        record = {
            "timestamp": self._now_eastern().isoformat(),
            "event_type": str(event_type),
            "correlation_id": correlation_id or str(uuid.uuid4().hex[:12]),
            "details": details if isinstance(details, dict) else {"value": str(details)},
        }
        try:
            existing = ""
            if AUDIT_LOG_PATH.exists():
                existing = AUDIT_LOG_PATH.read_text(encoding="utf-8")
            line = json.dumps(record, separators=(",", ":"), default=str)
            payload = f"{existing}{line}\n"
            atomic_write_private(
                AUDIT_LOG_PATH,
                payload,
                label="audit trail file",
            )
        except Exception as exc:
            logger.debug("Failed to append audit event %s: %s", event_type, exc)

    def _setup_streaming(self) -> None:
        """Try to enable Schwab streaming; fall back to polling on failure."""
        try:
            connected = self.schwab.start_streaming()
        except Exception as exc:
            connected = False
            logger.warning("Streaming setup failed: %s", exc)

        self._service_degradation["stream_down"] = not connected
        if not connected:
            self._stream_option_symbols.clear()
            if self.config.degradation.fallback_polling_on_stream_failure:
                logger.warning("Streaming unavailable, continuing with polling mode.")
            return

        self._stream_option_symbols.clear()
        # Account activity stream is the most useful low-volume subscription.
        self.schwab.stream_account_activity(self._on_stream_account_activity)

        # Stream open-position underlyings for tighter exit responsiveness.
        open_symbols = sorted(
            {
                str(position.get("symbol", "")).upper()
                for position in self.risk_manager.portfolio.open_positions
                if position.get("symbol")
            }
        )
        if open_symbols:
            self.schwab.stream_quotes(open_symbols, self._on_stream_quote)

    def _on_stream_quote(self, message) -> None:  # pragma: no cover - callback path
        logger.debug("Quote stream update: %s", message)
        self._process_stream_exit_check(message)

    def _on_stream_option(self, message) -> None:  # pragma: no cover - callback path
        logger.debug("Option stream update: %s", message)
        self._process_stream_exit_check(message)

    @staticmethod
    def _on_stream_account_activity(message) -> None:  # pragma: no cover - callback path
        logger.info("Account activity stream event: %s", message)

    def _process_stream_exit_check(self, message) -> None:
        """Run low-latency exit checks for affected symbols from stream updates."""
        if self._service_degradation.get("stream_down"):
            return
        symbol, price = self._extract_stream_symbol_price(message)
        if not symbol:
            return
        now_et = self._now_eastern()
        last = self._last_stream_exit_check.get(symbol)
        if last and (now_et - last) < timedelta(seconds=5):
            return
        self._last_stream_exit_check[symbol] = now_et

        if self.is_paper:
            positions = [
                p for p in self.paper_trader.get_positions()
                if str(p.get("status", "open")).lower() == "open" and str(p.get("symbol", "")).upper() == symbol
            ]
        else:
            positions = [
                p for p in self._get_tracked_positions()
                if str(p.get("status", "open")).lower() == "open" and str(p.get("symbol", "")).upper() == symbol
            ]
        if not positions:
            return

        for position in positions:
            if price > 0:
                position["underlying_price"] = price
        chain_data, _ = self._get_chain_data(symbol)
        if chain_data:
            for position in positions:
                mark = self._estimate_paper_position_value(position, chain_data)
                if mark is not None:
                    position["current_value"] = mark

        signals: list[TradeSignal] = []
        for strategy in self.strategies:
            try:
                signals.extend(strategy.check_exits(positions, self.schwab))
            except Exception:
                continue

        for signal in signals:
            if signal.action == "roll":
                self._execute_roll(signal)
            else:
                self._execute_exit(signal)

    @staticmethod
    def _extract_stream_symbol_price(message) -> tuple[str, float]:
        if not isinstance(message, dict):
            return "", 0.0
        symbol = str(
            message.get("symbol")
            or message.get("key")
            or message.get("serviceSymbol")
            or ""
        ).upper().strip()
        price = safe_float(
            message.get("lastPrice", message.get("mark", message.get("price", 0.0))),
            0.0,
        )
        return symbol, price

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
            self._setup_streaming()
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
                reconciled = self._startup_reconcile_positions()
                self._live_bootstrap_done = True
                if imported or reconciled:
                    logger.info(
                        "Live startup reconciliation complete: imported=%d reconciled=%d",
                        imported,
                        reconciled,
                    )
            except Exception as exc:
                logger.error("Failed to bootstrap live ledger: %s", exc)
                logger.debug(traceback.format_exc())
                self._alert(
                    level="ERROR",
                    title="Live ledger bootstrap failed",
                    message=str(exc),
                )
        self._setup_streaming()

    def _startup_reconcile_positions(self) -> int:
        """Reconcile local ledger positions against broker positions on startup."""
        if self.is_paper or not self.live_ledger:
            return 0
        broker_positions = self._get_broker_positions()
        if broker_positions is None:
            return 0

        open_broker_keys = self._collect_broker_option_symbols(broker_positions)
        ledger_positions = self.live_ledger.list_positions(
            statuses={"opening", "open", "closing"},
            copy_items=False,
        )
        reconciled = 0

        # Phantom positions: tracked locally but no longer at broker.
        phantom_closed = self.live_ledger.close_missing_from_broker(
            open_strategy_symbols=open_broker_keys,
            position_symbol_resolver=self._resolve_live_option_symbols,
            close_metadata_resolver=lambda _position, _symbols: {
                "exit_order_status": "EXTERNAL",
                "exit_reason": "not found at broker on startup reconciliation",
                "close_date": self._now_eastern().isoformat(),
            },
        )
        reconciled += int(phantom_closed)

        # Orphaned broker positions: broker has option legs that ledger does not track.
        tracked_symbols: set[str] = set()
        for position in ledger_positions:
            tracked_symbols.update(self._resolve_live_option_symbols(position))
        orphan_symbols = sorted(open_broker_keys - tracked_symbols)
        if orphan_symbols:
            groups: dict[tuple[str, str], list[str]] = defaultdict(list)
            for key in orphan_symbols:
                parts = str(key).split("|")
                if len(parts) != 4:
                    continue
                groups[(parts[0], parts[1])].append(key)
            for (underlying, expiration), symbols in groups.items():
                self.live_ledger.register_entry_order(
                    strategy="unknown_external",
                    symbol=underlying,
                    quantity=1,
                    max_loss=0.0,
                    entry_credit=0.0,
                    details={
                        "expiration": expiration,
                        "bootstrap_reconcile": True,
                        "orphaned_symbols": symbols,
                    },
                    entry_order_id="",
                    entry_order_status="FILLED",
                    opened_at=self._now_eastern().isoformat(),
                )
                reconciled += 1

        if reconciled:
            logger.info(
                "Startup position reconciliation actions: %d",
                reconciled,
            )
        return reconciled

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
                "Live execution currently supports only: "
                + ", ".join(sorted(LIVE_SUPPORTED_ENTRY_STRATEGIES))
                + ". "
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
        if self.config.naked_puts.enabled:
            enabled.add("naked_puts")
        if self.config.calendar_spreads.enabled:
            enabled.add("calendar_spreads")
        if self.config.strangles.enabled:
            enabled.add("strangles")
        if self.config.broken_wing_butterfly.enabled:
            enabled.add("broken_wing_butterfly")
        if self.config.earnings_vol_crush.enabled:
            enabled.add("earnings_vol_crush")
        return enabled

    def _apply_exit_overrides_to_strategies(self) -> None:
        """Propagate global adaptive-exit config into individual strategy configs."""
        overrides = {
            "adaptive_targets": bool(self.config.exits.adaptive_targets),
            "trailing_stop_enabled": bool(self.config.exits.trailing_stop_enabled),
            "trailing_stop_activation_pct": float(self.config.exits.trailing_stop_activation_pct),
            "trailing_stop_floor_pct": float(self.config.exits.trailing_stop_floor_pct),
        }
        for strategy in self.strategies:
            strategy.config.update(overrides)

    def _entries_allowed(self) -> bool:
        """Return True when no circuit breaker currently blocks new entries."""
        now = self._now_eastern()
        if self.circuit_state.get("halt_entries"):
            return False
        if self._portfolio_halt_until and now < self._portfolio_halt_until:
            return False

        cons_until = self.circuit_state.get("consecutive_loss_pause_until")
        if cons_until:
            try:
                cons_dt = datetime.fromisoformat(str(cons_until))
                if cons_dt.tzinfo is None:
                    cons_dt = cons_dt.replace(tzinfo=EASTERN_TZ)
                if now < cons_dt:
                    return False
            except ValueError:
                pass

        weekly_until = self.circuit_state.get("weekly_loss_pause_until")
        if weekly_until:
            try:
                weekly_dt = datetime.fromisoformat(str(weekly_until))
                if weekly_dt.tzinfo is None:
                    weekly_dt = weekly_dt.replace(tzinfo=EASTERN_TZ)
                if now < weekly_dt:
                    return False
            except ValueError:
                pass

        return True

    def _update_market_regime(self) -> None:
        """Refresh VIX-based volatility regime and risk throttles."""
        previous_regime = str(self.circuit_state.get("regime", "normal"))
        if self.regime_detector is not None:
            try:
                state = self.regime_detector.detect()
                self.current_regime_state = state
                regime_key = str(state.regime)
                self.circuit_state["regime"] = regime_key
                self.circuit_state["regime_confidence"] = round(float(state.confidence), 4)
                if regime_key in {CRASH_CRISIS, CRASH_CRIISIS}:
                    self.circuit_state["halt_entries"] = True
                self._apply_regime_adjustments(regime_key)
            except Exception as exc:
                logger.warning("Regime detector failed, falling back to VIX-only regime: %s", exc)

        vix_value = None
        for symbol in ("$VIX", "^VIX", "VIX"):
            try:
                quote = self.schwab.get_quote(symbol)
                self._record_api_health(True)
            except Exception:
                self._record_api_health(False)
                continue
            ref = quote.get("quote", quote) if isinstance(quote, dict) else {}
            value = safe_float(ref.get("lastPrice", ref.get("mark", 0.0)))
            if value > 0:
                vix_value = value
                break

        if vix_value is None:
            return

        self.circuit_state["vix"] = round(vix_value, 2)
        if vix_value > 35:
            regime = "crisis"
            self.circuit_state["halt_entries"] = True
            if not self._breaker_alert_flags["crisis"]:
                logger.warning(
                    "CIRCUIT BREAKER: VIX=%.2f, halting new entries",
                    vix_value,
                )
                self._alert(
                    level="ERROR",
                    title="CIRCUIT BREAKER",
                    message=f"VIX={vix_value:.2f}, halting new entries",
                )
                self._breaker_alert_flags["crisis"] = True
        elif vix_value >= 25:
            regime = "elevated"
            self.circuit_state["halt_entries"] = False
            self._breaker_alert_flags["crisis"] = False
        elif vix_value >= 15:
            regime = "normal"
            self.circuit_state["halt_entries"] = False
            self._breaker_alert_flags["crisis"] = False
        else:
            regime = "low_vol"
            self.circuit_state["halt_entries"] = False
            self._breaker_alert_flags["crisis"] = False

        self.circuit_state["regime"] = regime
        self._apply_regime_adjustments(regime)
        if str(regime) != previous_regime:
            self.alerts.regime_change(
                f"Regime shifted from {previous_regime} to {regime}",
                context={
                    "vix": self.circuit_state.get("vix"),
                    "confidence": self.circuit_state.get("regime_confidence"),
                },
            )

    def _apply_regime_adjustments(self, regime: str) -> None:
        """Adjust risk/entry thresholds based on volatility regime."""
        # Reset to base profile values before applying overlays.
        profile = (
            str((self._active_account or {}).get("risk_profile", "moderate")).strip().lower()
            if self._active_account
            else "moderate"
        )
        self._apply_risk_profile(profile)

        # Reset min-credit overlays before regime-specific scaling.
        for strategy in self.strategies:
            base_credit = self._base_strategy_min_credit.get(strategy.name)
            if base_credit is None:
                continue
            strategy.config["min_credit_pct"] = base_credit

        regime_key = str(regime).upper()
        if regime_key in {"ELEVATED", HIGH_VOL_CHOP}:
            self.risk_manager.config.max_open_positions = max(
                1, int(round(self.risk_manager.config.max_open_positions * 0.70))
            )
            for strategy in self.strategies:
                base_credit = self._base_strategy_min_credit.get(strategy.name)
                if base_credit is None:
                    continue
                strategy.config["min_credit_pct"] = round(base_credit * 1.20, 4)

        if regime_key in {CRASH_CRISIS, CRASH_CRIISIS}:
            self.risk_manager.config.max_open_positions = max(
                1, int(round(self.risk_manager.config.max_open_positions * 0.50))
            )

        if self.current_regime_state and self.current_regime_state.recommended_strategy_weights:
            weights = self.current_regime_state.recommended_strategy_weights
            for strategy in self.strategies:
                strategy.config["regime_weight"] = float(weights.get(strategy.name, 1.0))

    def _update_loss_breakers(self) -> None:
        """Apply consecutive-loss and weekly-loss entry pauses."""
        now = self._now_eastern()
        closed = self._recent_closed_trades()
        if not closed:
            return

        recent_three = closed[-3:]
        if len(recent_three) == 3:
            max_loss_hits = 0
            for trade in recent_three:
                pnl = float(trade.get("pnl", 0.0))
                max_loss = abs(float(trade.get("max_loss", 0.0)) * max(1, int(trade.get("quantity", 1))) * 100.0)
                if max_loss > 0 and pnl <= (-0.95 * max_loss):
                    max_loss_hits += 1
            if max_loss_hits == 3:
                pause_until = now + timedelta(hours=24)
                existing = self.circuit_state.get("consecutive_loss_pause_until")
                existing_dt = None
                if existing:
                    try:
                        existing_dt = datetime.fromisoformat(str(existing))
                    except ValueError:
                        existing_dt = None
                if not existing_dt or now >= existing_dt:
                    self.circuit_state["consecutive_loss_pause_until"] = pause_until.isoformat()
                    if not self._breaker_alert_flags["consecutive_loss"]:
                        self._alert(
                            level="ERROR",
                            title="Consecutive loss breaker",
                            message=f"Pausing entries until {pause_until.isoformat()}",
                        )
                        self._breaker_alert_flags["consecutive_loss"] = True
        else:
            self._breaker_alert_flags["consecutive_loss"] = False

        week_start = now.date() - timedelta(days=now.weekday())
        weekly_loss = 0.0
        for trade in closed:
            close_date_raw = str(trade.get("close_date", ""))
            close_date = close_date_raw.split("T", 1)[0]
            try:
                closed_day = datetime.strptime(close_date, "%Y-%m-%d").date()
            except ValueError:
                continue
            if closed_day >= week_start:
                weekly_loss += float(trade.get("pnl", 0.0))

        balance = max(1.0, float(self.risk_manager.portfolio.account_balance or 0.0))
        weekly_limit = -0.05 * balance
        if weekly_loss <= weekly_limit:
            days_until_monday = (7 - now.weekday()) % 7 or 7
            pause_until = datetime.combine(
                now.date() + timedelta(days=days_until_monday),
                datetime.min.time(),
                tzinfo=EASTERN_TZ,
            )
            existing = self.circuit_state.get("weekly_loss_pause_until")
            existing_dt = None
            if existing:
                try:
                    existing_dt = datetime.fromisoformat(str(existing))
                except ValueError:
                    existing_dt = None
            if not existing_dt or now >= existing_dt:
                self.circuit_state["weekly_loss_pause_until"] = pause_until.isoformat()
                if not self._breaker_alert_flags["weekly_loss"]:
                    self._alert(
                        level="ERROR",
                        title="Weekly loss breaker",
                        message=f"Weekly loss {weekly_loss:,.2f} breached 5% threshold. "
                                f"Entries paused until {pause_until.isoformat()}",
                    )
                    self._breaker_alert_flags["weekly_loss"] = True
        else:
            self._breaker_alert_flags["weekly_loss"] = False

    def _update_advanced_breakers(self) -> None:
        """Refresh strategy/symbol/API/LLM/drawdown circuit-breaker states."""
        self._refresh_strategy_and_symbol_breakers()
        self._apply_drawdown_breaker()
        self._apply_api_health_breaker()
        self._apply_llm_health_breaker()

    def _refresh_strategy_and_symbol_breakers(self) -> None:
        now = self._now_eastern()
        closed = self._recent_closed_trades()
        if not closed:
            return

        strategy_limit = int(self.config.circuit_breakers.strategy_loss_streak_limit)
        strategy_cooldown = int(self.config.circuit_breakers.strategy_cooldown_hours)
        symbol_limit = int(self.config.circuit_breakers.symbol_loss_streak_limit)
        symbol_cooldown_days = int(self.config.circuit_breakers.symbol_blacklist_days)

        streak_by_strategy: dict[str, int] = defaultdict(int)
        streak_by_symbol: dict[str, int] = defaultdict(int)

        for trade in sorted(closed, key=lambda item: str(item.get("close_date", ""))):
            pnl = float(trade.get("pnl", 0.0) or 0.0)
            strategy = self._strategy_group(str(trade.get("strategy", "")).strip().lower())
            symbol = str(trade.get("symbol", "")).strip().upper()

            if strategy:
                streak_by_strategy[strategy] = (
                    streak_by_strategy.get(strategy, 0) + 1 if pnl < 0 else 0
                )
            if symbol:
                streak_by_symbol[symbol] = (
                    streak_by_symbol.get(symbol, 0) + 1 if pnl < 0 else 0
                )

        self._strategy_loss_streaks = streak_by_strategy
        self._symbol_loss_streaks = streak_by_symbol

        for strategy, streak in streak_by_strategy.items():
            if streak < strategy_limit:
                continue
            pause_until = now + timedelta(hours=strategy_cooldown)
            existing = self._strategy_pause_until.get(strategy)
            if existing and existing > now:
                continue
            self._strategy_pause_until[strategy] = pause_until
            logger.warning(
                "Strategy breaker triggered: %s streak=%d paused_until=%s",
                strategy,
                streak,
                pause_until.isoformat(),
            )
            self._alert(
                level="WARNING",
                title="Strategy circuit breaker",
                message=f"{strategy} paused until {pause_until.isoformat()}",
                context={"streak": streak},
            )

        for symbol, streak in streak_by_symbol.items():
            if streak < symbol_limit:
                continue
            pause_until = now + timedelta(days=symbol_cooldown_days)
            existing = self._symbol_pause_until.get(symbol)
            if existing and existing > now:
                continue
            self._symbol_pause_until[symbol] = pause_until
            logger.warning(
                "Symbol breaker triggered: %s streak=%d paused_until=%s",
                symbol,
                streak,
                pause_until.isoformat(),
            )
            self._alert(
                level="WARNING",
                title="Symbol circuit breaker",
                message=f"{symbol} blacklisted until {pause_until.isoformat()}",
                context={"streak": streak},
            )

        self._strategy_pause_until = {
            strategy: until
            for strategy, until in self._strategy_pause_until.items()
            if until > now
        }
        self._symbol_pause_until = {
            symbol: until
            for symbol, until in self._symbol_pause_until.items()
            if until > now
        }

    def _apply_drawdown_breaker(self) -> None:
        now = self._now_eastern()
        threshold = float(self.config.circuit_breakers.portfolio_drawdown_halt_pct) / 100.0
        if threshold <= 0:
            return
        drawdown = self.risk_manager._current_drawdown()
        if drawdown >= threshold:
            pause_until = now + timedelta(hours=int(self.config.circuit_breakers.portfolio_halt_hours))
            if not self._portfolio_halt_until or now >= self._portfolio_halt_until:
                self._portfolio_halt_until = pause_until
                logger.warning(
                    "Portfolio drawdown breaker triggered: %.2f%% until %s",
                    drawdown * 100.0,
                    pause_until.isoformat(),
                )
                self._alert(
                    level="ERROR",
                    title="Portfolio drawdown breaker",
                    message=f"Drawdown {drawdown * 100.0:.2f}% — entries paused until {pause_until.isoformat()}",
                )

    def _apply_api_health_breaker(self) -> None:
        now = self._now_eastern()
        window_minutes = int(self.config.circuit_breakers.api_window_minutes)
        threshold = float(self.config.circuit_breakers.api_error_rate_threshold)
        cutoff = now - timedelta(minutes=window_minutes)
        while self._api_health_window and self._api_health_window[0][0] < cutoff:
            self._api_health_window.popleft()
        if len(self._api_health_window) < 5:
            return
        failures = sum(1 for _, ok in self._api_health_window if not ok)
        error_rate = failures / max(1, len(self._api_health_window))
        if error_rate > threshold:
            self.circuit_state["halt_entries"] = True
            self.circuit_state["api_health_halt"] = True
            self._service_degradation["scanner_down"] = True
            logger.warning(
                "API health breaker triggered: error_rate=%.1f%% over last %d calls",
                error_rate * 100.0,
                len(self._api_health_window),
            )
            self._alert(
                level="ERROR",
                title="API health breaker",
                message=f"Schwab API error rate {error_rate * 100.0:.1f}% exceeds threshold",
            )
        elif self.circuit_state.get("api_health_halt"):
            self.circuit_state["api_health_halt"] = False

    def _apply_llm_health_breaker(self) -> None:
        threshold = int(self.config.circuit_breakers.llm_timeout_streak)
        if self._llm_timeout_streak < threshold:
            return
        if self._service_degradation.get("rule_only_mode"):
            return
        self._service_degradation["rule_only_mode"] = True
        logger.warning(
            "LLM health breaker triggered: timeout streak=%d. Falling back to rule-only mode.",
            self._llm_timeout_streak,
        )
        self._alert(
            level="WARNING",
            title="LLM degraded mode",
            message="LLM provider timed out repeatedly. Trading will continue in rule-only mode.",
        )

    def _record_api_health(self, success: bool) -> None:
        self._api_health_window.append((self._now_eastern(), bool(success)))

    def _is_strategy_paused(self, strategy_name: str) -> bool:
        if not strategy_name:
            return False
        strategy_key = self._strategy_group(str(strategy_name).lower())
        until = self._strategy_pause_until.get(strategy_key)
        if not until:
            return False
        return self._now_eastern() < until

    def _is_symbol_paused(self, symbol: str) -> bool:
        if not symbol:
            return False
        until = self._symbol_pause_until.get(str(symbol).upper())
        if not until:
            return False
        return self._now_eastern() < until

    @staticmethod
    def _strategy_group(strategy_name: str) -> str:
        raw = str(strategy_name or "").strip().lower()
        if raw in {"bull_put_spread", "bear_call_spread", "credit_spreads"}:
            return "credit_spreads"
        if raw in {"iron_condor", "iron_condors"}:
            return "iron_condors"
        if raw in {"covered_call", "covered_calls"}:
            return "covered_calls"
        if raw in {"naked_put", "naked_puts"}:
            return "naked_puts"
        if raw in {"calendar_spread", "calendar_spreads"}:
            return "calendar_spreads"
        return raw

    def _apply_portfolio_strategist(self) -> None:
        """Apply optional high-level LLM strategist directives once per cycle."""
        if not self.llm_strategist:
            return
        for key in [name for name in self._service_degradation if name.startswith("skip_sector:")]:
            self._service_degradation.pop(key, None)
        context = {
            "portfolio_greeks": self.risk_manager.get_portfolio_greeks(),
            "open_positions": len(self.risk_manager.portfolio.open_positions),
            "sector_concentration": self.risk_manager.portfolio.sector_risk,
            "regime": self.circuit_state.get("regime"),
            "regime_confidence": self.circuit_state.get("regime_confidence"),
            "loss_breakers": {
                "consecutive": self.circuit_state.get("consecutive_loss_pause_until"),
                "weekly": self.circuit_state.get("weekly_loss_pause_until"),
            },
            "recent_trades": self._recent_closed_trades()[-20:],
        }
        if self.econ_calendar:
            context["economic_events"] = self.econ_calendar.context(days=21)
        try:
            directives = self.llm_strategist.review_portfolio(context)
        except Exception as exc:
            logger.warning("LLM strategist failed: %s", exc)
            self._service_degradation["llm_down"] = True
            return

        if not directives:
            return

        for directive in directives:
            if directive.action == "scale_size":
                scalar = float(
                    directive.payload.get(
                        "factor",
                        directive.payload.get("scalar", 0.8),
                    )
                    or 0.8
                )
                scalar = max(
                    0.5,
                    min(1.5, scalar),
                )
                self._cycle_size_scalar *= scalar
                logger.info("LLM strategist applied size scalar %.2f (%s)", scalar, directive.reason)
            elif directive.action == "reduce_delta":
                self.risk_manager.config.max_portfolio_delta_abs = max(
                    10.0,
                    float(self.risk_manager.config.max_portfolio_delta_abs) * 0.85,
                )
                target_count = max(1, safe_int(directive.payload.get("count"), 2))
                direction = str(directive.payload.get("direction", "")).strip().lower()
                if direction not in {"positive", "negative"}:
                    direction = "positive" if float(self.risk_manager.portfolio.net_delta or 0.0) >= 0 else "negative"

                candidates: list[tuple[float, dict, float]] = []
                for position in self._get_tracked_positions():
                    if str(position.get("status", "open")).lower() != "open":
                        continue
                    details = position.get("details", {}) if isinstance(position.get("details"), dict) else {}
                    qty = max(1, safe_int(position.get("quantity"), 1))
                    pos_delta = safe_float(details.get("net_delta", position.get("net_delta", 0.0)), 0.0) * qty
                    if direction == "positive" and pos_delta <= 0:
                        continue
                    if direction == "negative" and pos_delta >= 0:
                        continue
                    candidates.append((abs(pos_delta), position, pos_delta))
                candidates.sort(key=lambda row: row[0], reverse=True)
                for _, position, pos_delta in candidates[:target_count]:
                    self._execute_exit(
                        TradeSignal(
                            action="close",
                            strategy=str(position.get("strategy", "")),
                            symbol=str(position.get("symbol", "")),
                            position_id=position.get("position_id"),
                            reason=f"LLM strategist delta reduction ({directive.reason})",
                            quantity=max(1, safe_int(position.get("quantity"), 1)),
                        )
                    )
                    logger.info(
                        "LLM strategist closed delta-heavy position %s delta=%.2f",
                        position.get("position_id"),
                        pos_delta,
                    )
            elif directive.action == "skip_sector":
                sector = str(directive.payload.get("sector", "")).strip()
                if sector:
                    self._service_degradation[f"skip_sector:{sector}"] = True
                    logger.info("LLM strategist skip-sector directive: %s", sector)
            elif directive.action == "close_long_dte":
                threshold = int(
                    directive.payload.get(
                        "max_dte",
                        directive.payload.get("dte_gt", 30),
                    )
                    or 30
                )
                for position in self._get_tracked_positions():
                    if int(position.get("dte_remaining", 0) or 0) > threshold:
                        self._execute_exit(
                            TradeSignal(
                                action="close",
                                strategy=str(position.get("strategy", "")),
                                symbol=str(position.get("symbol", "")),
                                position_id=position.get("position_id"),
                                reason=f"LLM strategist directive: {directive.reason}",
                                quantity=max(1, int(position.get("quantity", 1))),
                            )
                        )

    def _apply_hedging_layer(self) -> None:
        """Create optional hedge recommendations and optionally execute."""
        account_value = float(self.risk_manager.portfolio.account_balance or 0.0)
        actions = self.hedger.propose(
            account_value=account_value,
            net_delta=float(self.risk_manager.portfolio.net_delta or 0.0),
            sector_exposure=self.risk_manager.portfolio.sector_risk,
            regime=str(self.circuit_state.get("regime", "normal")),
        )
        if not actions:
            return
        max_monthly_cost = account_value * (float(self.config.hedging.max_hedge_cost_pct) / 100.0)
        spent_this_month = self._monthly_hedge_cost()
        remaining_budget = max(0.0, max_monthly_cost - spent_this_month)
        for action in actions:
            if action.estimated_cost > remaining_budget:
                logger.info(
                    "Skipping hedge %s %s: monthly hedge budget exhausted (remaining %.2f).",
                    action.symbol,
                    action.direction,
                    remaining_budget,
                )
                continue
            remaining_budget -= action.estimated_cost
            logger.info(
                "HEDGE recommendation: %s %s x%d est_cost=%.2f reason=%s",
                action.symbol,
                action.direction,
                action.quantity,
                action.estimated_cost,
                action.reason,
            )
            executed = False
            if self.config.hedging.auto_execute:
                executed = self._execute_hedge_action(action)
                level = "INFO" if executed else "WARNING"
                self._alert(
                    level=level,
                    title="Hedge action executed" if executed else "Hedge action failed",
                    message=f"{action.symbol} {action.direction} x{action.quantity}",
                    context={"estimated_cost": action.estimated_cost, "reason": action.reason},
                )
            self._record_hedge_action(action, executed=executed)

    def _execute_hedge_action(self, action) -> bool:
        """Execute an approved hedge recommendation as a dedicated hedge position."""
        option = self._select_hedge_option(
            symbol=str(action.symbol),
            direction=str(action.direction),
            min_dte=30,
            max_dte=45,
        )
        if not option:
            return False

        quantity = max(1, int(action.quantity))
        debit = max(0.01, safe_float(option.get("mid"), safe_float(action.estimated_cost, 0.01)))
        details = {
            "expiration": option.get("expiration"),
            "dte": safe_int(option.get("dte"), 0),
            "short_strike": safe_float(option.get("strike"), 0.0),
            "long_strike": safe_float(option.get("strike"), 0.0),
            "hedge_type": action.direction,
            "hedge_reason": action.reason,
            "is_hedge": True,
        }
        analysis = SpreadAnalysis(
            symbol=str(action.symbol),
            strategy="hedge",
            expiration=str(option.get("expiration", "")),
            dte=safe_int(option.get("dte"), 0),
            short_strike=safe_float(option.get("strike"), 0.0),
            long_strike=safe_float(option.get("strike"), 0.0),
            credit=debit,
            max_loss=debit,
            probability_of_profit=0.50,
            score=60.0,
        )
        signal = TradeSignal(
            action="open",
            strategy="hedge",
            symbol=str(action.symbol),
            analysis=analysis,
            quantity=quantity,
            metadata={"is_hedge": True, "position_details": details},
        )

        approved, _ = self.risk_manager.approve_trade(signal)
        if not approved:
            return False
        if not self._review_entry_with_llm(signal):
            return False

        if self.is_paper:
            result = self.paper_trader.execute_open(
                strategy="hedge",
                symbol=signal.symbol,
                credit=-debit,
                max_loss=debit,
                quantity=quantity,
                details=details,
            )
            status = str(result.get("status", "")).upper()
            if status != "FILLED":
                return False
            self.risk_manager.register_open_position(
                symbol=signal.symbol,
                max_loss_per_contract=debit,
                quantity=quantity,
                strategy="hedge",
                greeks={
                    "net_delta": safe_float(option.get("delta"), 0.0),
                    "net_theta": safe_float(option.get("theta"), 0.0),
                    "net_gamma": safe_float(option.get("gamma"), 0.0),
                    "net_vega": safe_float(option.get("vega"), 0.0),
                },
            )
            return True

        order_factory = lambda price: self.schwab.build_long_option_open(
            symbol=signal.symbol,
            expiration=str(option.get("expiration", "")),
            contract_type=str(option.get("contract_type", "P")),
            strike=safe_float(option.get("strike"), 0.0),
            quantity=quantity,
            price=price,
        )
        result = self.schwab.place_order_with_ladder(
            order_factory=order_factory,
            midpoint_price=debit,
            spread_width=max(0.05, debit * 0.30),
            side="debit",
            step_timeout_seconds=self.config.execution.entry_step_timeout_seconds,
            max_attempts=self.config.execution.max_ladder_attempts,
            shifts=list(self.config.execution.entry_ladder_shifts),
            total_timeout_seconds=300,
        )
        status = str(result.get("status", "")).upper()
        if status in {"CANCELED", "REJECTED", "EXPIRED"}:
            return False
        order_id = str(result.get("order_id", "")).strip()
        if self.live_ledger and order_id:
            self.live_ledger.register_entry_order(
                strategy="hedge",
                symbol=signal.symbol,
                quantity=quantity,
                max_loss=debit,
                entry_credit=-debit,
                details=details,
                entry_order_id=order_id,
                entry_order_status=status or "PLACED",
            )
        return bool(order_id)

    def _select_hedge_option(
        self,
        *,
        symbol: str,
        direction: str,
        min_dte: int,
        max_dte: int,
    ) -> Optional[dict]:
        """Select a liquid OTM hedge contract from current chain data."""
        chain_data, underlying = self._get_chain_data(symbol)
        if not chain_data or underlying <= 0:
            return None
        direction_key = str(direction).lower()
        is_call = direction_key == "buy_call"
        contract_type = "C" if is_call else "P"
        exp_map = chain_data.get("calls" if is_call else "puts", {})
        options: list[dict] = []
        for expiration, rows in exp_map.items():
            if not isinstance(rows, list) or not rows:
                continue
            dte = safe_int(rows[0].get("dte"), 0)
            if dte < int(min_dte) or dte > int(max_dte):
                continue
            for row in rows:
                strike = safe_float(row.get("strike"), 0.0)
                if strike <= 0:
                    continue
                if is_call and strike <= underlying:
                    continue
                if not is_call and strike >= underlying:
                    continue
                candidate = dict(row)
                candidate["expiration"] = expiration
                candidate["contract_type"] = contract_type
                options.append(candidate)
        if not options:
            return None
        options.sort(
            key=lambda row: (
                abs(abs(safe_float(row.get("delta"), 0.0)) - 0.20),
                -safe_float(row.get("volume"), 0.0),
            )
        )
        return options[0]

    def _monthly_hedge_cost(self) -> float:
        """Return aggregate hedge cost booked in the current month."""
        payload = load_json(HEDGE_COSTS_PATH, {"entries": []})
        entries = payload.get("entries", []) if isinstance(payload, dict) else []
        if not isinstance(entries, list):
            return 0.0
        month_key = self._now_eastern().strftime("%Y-%m")
        total = 0.0
        for item in entries:
            if not isinstance(item, dict):
                continue
            timestamp = str(item.get("timestamp", ""))
            if not timestamp.startswith(month_key):
                continue
            total += safe_float(item.get("estimated_cost"), 0.0)
        return round(total, 4)

    def _record_hedge_action(self, action, *, executed: bool) -> None:
        """Persist hedge recommendations/executions for P&L accounting."""
        payload = load_json(HEDGE_COSTS_PATH, {"entries": []})
        if not isinstance(payload, dict):
            payload = {"entries": []}
        entries = payload.get("entries")
        if not isinstance(entries, list):
            entries = []
            payload["entries"] = entries

        entries.append(
            {
                "timestamp": self._now_eastern().isoformat(),
                "symbol": str(action.symbol),
                "direction": str(action.direction),
                "quantity": int(action.quantity),
                "estimated_cost": round(safe_float(action.estimated_cost, 0.0), 4),
                "reason": str(action.reason),
                "executed": bool(executed),
            }
        )
        payload["entries"] = entries[-5000:]
        dump_json(HEDGE_COSTS_PATH, payload)

    def _recent_closed_trades(self) -> list[dict]:
        """Return closed trades from paper or live tracking."""
        if self.is_paper and self.paper_trader:
            closed = []
            for trade in self.paper_trader.closed_trades:
                if not isinstance(trade, dict):
                    continue
                closed.append(
                    {
                        "strategy": trade.get("strategy", ""),
                        "symbol": trade.get("symbol", ""),
                        "close_date": trade.get("close_date", ""),
                        "pnl": float(trade.get("pnl", 0.0) or 0.0),
                        "max_loss": float(trade.get("max_loss", 0.0) or 0.0),
                        "quantity": int(trade.get("closed_quantity", trade.get("quantity", 1)) or 1),
                    }
                )
            return sorted(closed, key=lambda t: str(t.get("close_date", "")))

        if not self.live_ledger:
            return []

        closed_positions = self.live_ledger.list_positions(
            statuses={"closed", "closed_external", "rolled"}
        )
        normalized = []
        for pos in closed_positions:
            if not isinstance(pos, dict):
                continue
            normalized.append(
                {
                    "strategy": pos.get("strategy", ""),
                    "symbol": pos.get("symbol", ""),
                    "close_date": pos.get("close_date", ""),
                    "pnl": float(pos.get("realized_pnl", 0.0) or 0.0),
                    "max_loss": float(pos.get("max_loss", 0.0) or 0.0),
                    "quantity": int(pos.get("quantity", 1) or 1),
                }
            )
        return sorted(normalized, key=lambda t: str(t.get("close_date", "")))

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
            self._cycle_size_scalar = 1.0
            # Update portfolio state
            self._update_portfolio_state()

            # Check exits on existing positions first
            self._check_exits()

            # Update circuit breakers/regime state before considering new entries.
            self._update_market_regime()
            self._update_loss_breakers()
            self._update_advanced_breakers()
            self._apply_portfolio_strategist()
            self._apply_hedging_layer()

            # Scan for new entries
            if self._entries_allowed():
                self._scan_for_entries()
            else:
                logger.warning("Entry scanning paused by circuit breaker state.")

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
            try:
                balance = self.schwab.get_account_balance()
                self._record_api_health(True)
            except Exception:
                self._record_api_health(False)
                raise
            positions = self._get_tracked_positions()
            daily_pnl = self._compute_live_daily_pnl()

        self.risk_manager.update_portfolio(balance, positions, daily_pnl)
        self.risk_manager.update_trade_history(self._recent_closed_trades())
        drawdown_pct = self.risk_manager._current_drawdown() * 100.0
        self.alerts.drawdown_alert(
            drawdown_pct,
            context={"balance": round(float(balance), 2), "open_positions": len(positions)},
        )
        logger.info(
            "Portfolio: Balance=$%s | Open positions: %d | Daily P/L: $%.2f",
            f"{balance:,.2f}", len(positions), daily_pnl,
        )
        self._equity_history.append(
            {
                "date": self._now_eastern().date().isoformat(),
                "equity": round(float(balance), 2),
            }
        )
        self._equity_history = self._equity_history[-365:]

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
                    self._service_degradation["scanner_down"] = False
                    logger.info(
                        "Scanner found %d top tickers: %s",
                        len(targets), ", ".join(targets[:10]),
                    )
                    return targets
                logger.warning("Scanner returned no results. Falling back to watchlist.")
                self._service_degradation["scanner_down"] = True
            except Exception as e:
                logger.error("Scanner failed: %s. Falling back to watchlist.", e)
                self._service_degradation["scanner_down"] = True

            if not self.config.degradation.fallback_watchlist_on_scanner_failure:
                return []

        return self.config.watchlist

    def _active_skip_sectors(self) -> set[str]:
        sectors: set[str] = set()
        for key, enabled in self._service_degradation.items():
            if not enabled:
                continue
            if not str(key).startswith("skip_sector:"):
                continue
            sector = str(key).split(":", 1)[-1].strip().lower()
            if sector:
                sectors.add(sector)
        return sectors

    def _is_symbol_sector_skipped(self, symbol: str) -> bool:
        skip_sectors = self._active_skip_sectors()
        if not skip_sectors:
            return False
        sector = str(self.risk_manager.sector_map.get(str(symbol).upper(), "")).strip().lower()
        return bool(sector and sector in skip_sectors)

    def _subscribe_option_stream_for_symbol(self, symbol: str, chain_data: dict, underlying_price: float) -> None:
        """Subscribe level-1 option stream for near-ATM contracts of top scan symbols."""
        if underlying_price <= 0:
            return
        if not self.schwab.streaming_connected():
            return

        calls = chain_data.get("calls", {}) if isinstance(chain_data, dict) else {}
        puts = chain_data.get("puts", {}) if isinstance(chain_data, dict) else {}
        if not isinstance(calls, dict) or not isinstance(puts, dict):
            return

        expirations = sorted(set(calls.keys()) & set(puts.keys()))
        if not expirations:
            return

        selected_exp = None
        selected_distance = float("inf")
        for expiration in expirations:
            exp_calls = calls.get(expiration, [])
            if not exp_calls:
                continue
            dte = safe_int(exp_calls[0].get("dte"), 0)
            if dte <= 0:
                continue
            distance = abs(dte - 30)
            if distance < selected_distance:
                selected_exp = expiration
                selected_distance = distance
        if not selected_exp:
            return

        exp_calls = calls.get(selected_exp, [])
        exp_puts = puts.get(selected_exp, [])
        if not exp_calls or not exp_puts:
            return

        def _atm_option(options: list[dict]) -> Optional[dict]:
            best = None
            best_distance = float("inf")
            for row in options:
                strike = safe_float(row.get("strike"), 0.0)
                if strike <= 0:
                    continue
                distance = abs(strike - underlying_price)
                if distance < best_distance:
                    best = row
                    best_distance = distance
            return best

        atm_call = _atm_option(exp_calls)
        atm_put = _atm_option(exp_puts)
        symbols = []
        for option in (atm_call, atm_put):
            if not isinstance(option, dict):
                continue
            option_symbol = str(option.get("symbol", "")).strip()
            if option_symbol:
                symbols.append(option_symbol)
        if not symbols:
            return

        new_symbols = sorted(set(symbols) - self._stream_option_symbols)
        if not new_symbols:
            return
        if self.schwab.stream_option_level_one(new_symbols, self._on_stream_option):
            self._stream_option_symbols.update(new_symbols)

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

            if self._is_symbol_paused(symbol):
                logger.info("Skipping %s due to symbol circuit breaker pause.", symbol)
                continue
            if self._is_symbol_sector_skipped(symbol):
                logger.info("Skipping %s due to strategist sector skip directive.", symbol)
                continue

            logger.info("Scanning %s...", symbol)

            try:
                chain_data, underlying_price = self._get_chain_data(symbol)
                if not chain_data or underlying_price <= 0:
                    logger.warning("No chain data for %s, skipping.", symbol)
                    continue
                self._subscribe_option_stream_for_symbol(symbol, chain_data, underlying_price)

                # Run each strategy
                all_signals = []
                technical_context = None
                try:
                    technical_context = self.technicals.get_context(symbol, self.schwab)
                except Exception as exc:
                    logger.debug("Technical context unavailable for %s: %s", symbol, exc)
                market_context = self._build_market_context(symbol, chain_data)
                regime_weights = market_context.get("regime_weights", {}) or {}
                position_size_scalar = float(market_context.get("position_size_scalar", 1.0) or 1.0)
                position_size_scalar *= max(0.5, min(1.5, float(self._cycle_size_scalar or 1.0)))
                for strategy in self.strategies:
                    if self._is_strategy_paused(strategy.name):
                        logger.info(
                            "Skipping strategy %s on %s (strategy pause active).",
                            strategy.name,
                            symbol,
                        )
                        continue
                    regime_weight = 1.0
                    if isinstance(regime_weights, dict):
                        raw_weight = regime_weights.get(strategy.name, 1.0)
                        regime_weight = float(1.0 if raw_weight is None else raw_weight)
                    if regime_weight <= 0.0:
                        logger.info(
                            "Skipping strategy %s on %s due to regime weight %.2f",
                            strategy.name,
                            symbol,
                            regime_weight,
                        )
                        continue
                    try:
                        signals = strategy.scan_for_entries(
                            symbol,
                            chain_data,
                            underlying_price,
                            technical_context=technical_context,
                            market_context=market_context,
                        )
                    except Exception as exc:
                        if self.config.degradation.continue_on_strategy_errors:
                            logger.error(
                                "Strategy %s failed on %s; continuing with others: %s",
                                strategy.name,
                                symbol,
                                exc,
                            )
                            logger.debug(traceback.format_exc())
                            continue
                        raise
                    if technical_context is not None:
                        tech_payload = technical_context.to_dict()
                        for signal in signals:
                            signal.metadata["technical_context"] = tech_payload
                    for signal in signals:
                        correlation_id = str(signal.metadata.get("correlation_id", "")).strip() or uuid.uuid4().hex[:12]
                        signal.metadata["correlation_id"] = correlation_id
                        signal.metadata.setdefault("iv_rank", market_context.get("iv_rank"))
                        signal.metadata.setdefault("regime", market_context.get("regime"))
                        signal.metadata.setdefault("vol_surface", market_context.get("vol_surface", {}))
                        signal.metadata.setdefault("options_flow", market_context.get("options_flow", {}))
                        signal.metadata.setdefault("economic_events", market_context.get("economic_events", {}))
                        if signal.analysis is not None:
                            original = float(signal.analysis.score or 0.0)
                            adjusted = max(0.0, min(100.0, original * regime_weight))
                            signal.analysis.score = adjusted
                            signal.metadata["regime_score_weight"] = regime_weight
                            signal.metadata["original_score"] = original
                            logger.info(
                                "Regime-adjusted score %s %s: %.1f -> %.1f (w=%.2f)",
                                strategy.name,
                                signal.symbol,
                                original,
                                adjusted,
                                regime_weight,
                            )
                        self._append_audit_event(
                            event_type="signal_generated",
                            correlation_id=correlation_id,
                            details={
                                "symbol": signal.symbol,
                                "strategy": signal.strategy,
                                "score": float(signal.analysis.score if signal.analysis else 0.0),
                                "regime": signal.metadata.get("regime"),
                                "vol_surface": signal.metadata.get("vol_surface", {}),
                                "options_flow": signal.metadata.get("options_flow", {}),
                            },
                        )
                        signal.size_multiplier = max(
                            0.1,
                            float(signal.size_multiplier or 1.0) * max(0.5, min(1.5, position_size_scalar)),
                        )
                    all_signals.extend(self._filter_signals_by_context(signals, market_context))

                if not all_signals:
                    logger.info("No opportunities found on %s.", symbol)
                    continue

                if self.news_scanner:
                    try:
                        policy = self.news_scanner.trade_direction_policy(symbol)
                    except Exception as exc:
                        logger.debug("News policy unavailable for %s: %s", symbol, exc)
                        policy = {"block_all": False, "allow_bull_put": True, "allow_bear_call": True}

                    if policy.get("block_all"):
                        logger.info(
                            "Skipping %s: %s",
                            symbol,
                            policy.get("reason") or "news event risk",
                        )
                        continue

                    filtered = []
                    for signal in all_signals:
                        if signal.strategy == "bull_put_spread" and not policy.get("allow_bull_put", True):
                            continue
                        if signal.strategy == "bear_call_spread" and not policy.get("allow_bear_call", True):
                            continue
                        filtered.append(signal)
                    all_signals = filtered

                if not all_signals:
                    logger.info("No news-compliant opportunities found on %s.", symbol)
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

    def _filter_signals_by_context(
        self,
        signals: list[TradeSignal],
        market_context: Optional[dict],
    ) -> list[TradeSignal]:
        """Apply regime/vol-surface/flow overlays to candidate entry signals."""
        if not signals:
            return []

        context = market_context or {}
        regime = str(context.get("regime", "")).upper()
        vol_surface = context.get("vol_surface", {}) if isinstance(context.get("vol_surface"), dict) else {}
        flow = context.get("options_flow", {}) if isinstance(context.get("options_flow"), dict) else {}
        vol_flags = vol_surface.get("flags", {}) if isinstance(vol_surface.get("flags"), dict) else {}
        flow_bias = str(flow.get("directional_bias", "neutral")).strip().lower()

        premium_selling = {
            "bull_put_spread",
            "bear_call_spread",
            "iron_condor",
            "covered_call",
            "naked_put",
            "short_strangle",
            "short_straddle",
            "earnings_vol_crush",
        }

        filtered: list[TradeSignal] = []
        for signal in signals:
            if signal.action != "open" or signal.analysis is None:
                filtered.append(signal)
                continue

            strategy_name = str(signal.strategy or "").lower()
            dte = int(getattr(signal.analysis, "dte", 0) or 0)
            width = self._signal_width(signal)
            size_multiplier = float(signal.size_multiplier or 1.0)

            if regime in {"CRASH", "CRISIS", CRASH_CRISIS, CRASH_CRIISIS} and strategy_name in premium_selling:
                only_allowed = strategy_name in {"bull_put_spread", "bear_call_spread", "iron_condor"} and dte <= 14 and width >= 8.0
                if not only_allowed:
                    continue
                size_multiplier *= 0.60

            if regime == BULL_TREND and strategy_name == "bear_call_spread":
                continue
            if regime == BEAR_TREND and strategy_name == "bull_put_spread":
                continue

            if regime == HIGH_VOL_CHOP:
                if strategy_name in {"iron_condor", "short_strangle", "short_straddle"}:
                    size_multiplier *= 1.15
                elif strategy_name == "calendar_spread":
                    size_multiplier *= 0.85
            elif regime == LOW_VOL_GRIND:
                if strategy_name in premium_selling:
                    size_multiplier *= 0.75
                if strategy_name == "calendar_spread":
                    size_multiplier *= 1.20
            elif regime == MEAN_REVERSION and strategy_name in {"bull_put_spread", "bear_call_spread"}:
                tech = signal.metadata.get("technical_context", {})
                zscore = safe_float((tech or {}).get("return_5d_zscore"), 0.0)
                if abs(zscore) >= 2.0:
                    signal.analysis.score = min(100.0, float(signal.analysis.score or 0.0) + 8.0)

            if self.vol_surface_analyzer and vol_surface:
                if (
                    strategy_name in {"bull_put_spread", "bear_call_spread", "naked_put", "iron_condor"}
                    and bool(self.config.vol_surface.require_positive_vol_risk_premium)
                    and not bool(vol_flags.get("positive_vol_risk_premium", False))
                ):
                    continue

                if strategy_name == "calendar_spread" and not bool(vol_flags.get("front_richer_than_back", False)):
                    continue

                if strategy_name == "iron_condor":
                    vol_of_vol = safe_float(vol_surface.get("vol_of_vol"), 0.0)
                    if vol_of_vol > float(self.config.vol_surface.max_vol_of_vol_for_condors):
                        continue

            if flow_bias == "bearish" and strategy_name == "bull_put_spread":
                size_multiplier *= 0.80
            elif flow_bias == "bullish" and strategy_name == "bear_call_spread":
                size_multiplier *= 0.80

            signal.size_multiplier = max(0.1, min(2.0, size_multiplier))
            filtered.append(signal)

        return filtered

    @staticmethod
    def _signal_width(signal: TradeSignal) -> float:
        """Approximate strategy width used for crash-regime gating."""
        analysis = signal.analysis
        if analysis is None:
            return 0.0
        if signal.strategy == "iron_condor":
            put_width = abs(float(analysis.put_short_strike or 0.0) - float(analysis.put_long_strike or 0.0))
            call_width = abs(float(analysis.call_long_strike or 0.0) - float(analysis.call_short_strike or 0.0))
            return max(put_width, call_width)
        return abs(float(analysis.long_strike or 0.0) - float(analysis.short_strike or 0.0))

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

        all_exit_signals: list[TradeSignal] = []
        for strategy in self.strategies:
            exit_signals = strategy.check_exits(positions, self.schwab)
            all_exit_signals.extend(exit_signals)

        covered_positions = {
            str(signal.position_id)
            for signal in all_exit_signals
            if signal.position_id
        }
        adjustment_jobs: list[tuple[dict, object]] = []
        regime = str(self.circuit_state.get("regime", "normal"))

        if bool(self.config.rolling.enabled):
            for position in positions:
                if str(position.get("status", "open")).lower() != "open":
                    continue
                position_id = str(position.get("position_id", ""))
                if not position_id or position_id in covered_positions:
                    continue
                decision = self.roll_manager.evaluate(position, regime=regime)
                if not decision.should_roll:
                    continue
                all_exit_signals.append(
                    TradeSignal(
                        action="roll",
                        strategy=str(position.get("strategy", "")),
                        symbol=str(position.get("symbol", "")),
                        position_id=position_id,
                        quantity=max(1, int(position.get("quantity", 1))),
                        reason=decision.reason,
                        metadata={
                            "roll_type": decision.roll_type,
                            "min_credit_required": decision.min_credit_required,
                        },
                    )
                )
                covered_positions.add(position_id)

        if bool(self.config.adjustments.enabled):
            for position in positions:
                if str(position.get("status", "open")).lower() != "open":
                    continue
                position_id = str(position.get("position_id", ""))
                if not position_id or position_id in covered_positions:
                    continue

                details = position.get("details", {}) if isinstance(position.get("details"), dict) else {}
                entry_iv = safe_float(details.get("entry_iv"), 0.0)
                current_iv = safe_float(position.get("iv_rank", details.get("iv_rank", 0.0)), 0.0)
                iv_change = 0.0
                if entry_iv > 0:
                    iv_change = (current_iv - entry_iv) / max(entry_iv, 1e-9)
                plan = self.adjustment_engine.evaluate(
                    position=position,
                    regime=regime,
                    iv_change_since_entry=iv_change,
                )
                if plan.action == "none":
                    continue
                adjustment_jobs.append((position, plan))
                covered_positions.add(position_id)

        self._persist_runtime_exit_state(positions)

        for signal in all_exit_signals:
            if signal.action == "roll":
                self._execute_roll(signal)
                continue
            self._execute_exit(signal)

        for position, plan in adjustment_jobs:
            self._execute_adjustment_plan(position, plan)

    def _persist_runtime_exit_state(self, positions: list[dict]) -> None:
        """Persist mutable trailing/exit fields updated by strategy exit checks."""
        if self.is_paper:
            # Paper positions are in-memory references; persist opportunistically.
            if self.paper_trader:
                self.paper_trader._save_state()
            return
        if not self.live_ledger:
            return

        for position in positions:
            position_id = str(position.get("position_id", "")).strip()
            if not position_id:
                continue
            fields = {}
            detail_fields = {}
            if "trailing_stop_high" in position:
                fields["trailing_stop_high"] = safe_float(position.get("trailing_stop_high"), 0.0)
            details = position.get("details", {}) if isinstance(position.get("details"), dict) else {}
            for key in ("trailing_stop_high", "max_profit_pct_seen"):
                if key in details:
                    detail_fields[key] = safe_float(details.get(key), 0.0)
            if fields or detail_fields:
                self.live_ledger.update_position_metadata(
                    position_id,
                    fields=fields if fields else None,
                    detail_fields=detail_fields if detail_fields else None,
                )

    def _refresh_paper_position_values(self) -> None:
        """Refresh paper position marks using the latest option-chain mids."""
        if not self.paper_trader:
            return

        positions = self.paper_trader.get_positions()
        if not positions:
            return

        chain_cache: dict[str, dict] = {}
        position_marks: dict[str, float] = {}
        position_meta: dict[str, dict] = {}

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

            position_meta[position_id] = {
                "underlying_price": float(chain_data.get("underlying_price", 0.0) or 0.0)
            }

            mark = self._estimate_paper_position_value(position, chain_data)
            if mark is not None:
                position_marks[position_id] = mark

        if position_marks or position_meta:
            self.paper_trader.update_position_values(position_marks, position_meta=position_meta)

    def _estimate_paper_position_value(
        self, position: dict, chain_data: dict
    ) -> Optional[float]:
        """Estimate current debit-to-close for a paper position."""
        details = position.get("details", {})
        strategy = position.get("strategy", "")

        if strategy == "calendar_spread":
            strike = details.get("strike", details.get("short_strike"))
            front_exp = self._extract_expiration_key(details.get("front_expiration", details.get("expiration", "")))
            back_exp = self._extract_expiration_key(details.get("back_expiration", ""))
            if not front_exp or not back_exp or strike is None:
                return None

            front_calls = chain_data.get("calls", {}).get(front_exp, [])
            back_calls = chain_data.get("calls", {}).get(back_exp, [])
            front_mid = self._find_option_mid(front_calls, strike)
            back_mid = self._find_option_mid(back_calls, strike)
            if front_mid is None or back_mid is None:
                return None
            # Mark represented as negative to align debit strategy cashflow accounting.
            return round(-(back_mid - front_mid), 2)

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

        if strategy == "naked_put":
            short_mid = self._find_option_mid(puts, details.get("short_strike"))
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
        correlation_id = str(signal.metadata.get("correlation_id", "")).strip() or uuid.uuid4().hex[:12]
        signal.metadata["correlation_id"] = correlation_id
        if self.econ_calendar and signal.analysis is not None:
            allowed, econ_reason = self.econ_calendar.adjust_signal(signal)
            if not allowed:
                logger.info(
                    "Trade skipped by economic calendar policy: %s %s on %s — %s",
                    signal.strategy,
                    signal.action,
                    signal.symbol,
                    econ_reason,
                )
                return False
            if econ_reason:
                signal.metadata["economic_calendar_adjustment"] = econ_reason

        # Calculate position size
        quantity = self.risk_manager.calculate_position_size(signal)
        size_multiplier = max(0.1, float(signal.size_multiplier or 1.0))
        signal.quantity = max(1, int(round(quantity * size_multiplier)))

        # Risk check
        approved, reason = self.risk_manager.approve_trade(signal)
        if not approved:
            self._append_audit_event(
                event_type="risk_check",
                correlation_id=correlation_id,
                details={
                    "symbol": signal.symbol,
                    "strategy": signal.strategy,
                    "approved": False,
                    "reason": reason,
                },
            )
            logger.info(
                "Trade REJECTED by risk manager: %s %s on %s — %s",
                signal.strategy, signal.action, signal.symbol, reason,
            )
            return False
        self._append_audit_event(
            event_type="risk_check",
            correlation_id=correlation_id,
            details={
                "symbol": signal.symbol,
                "strategy": signal.strategy,
                "approved": True,
                "reason": reason,
            },
        )

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
        extra_details = signal.metadata.get("position_details")
        if isinstance(extra_details, dict):
            details.update(extra_details)
        details.setdefault("regime", signal.metadata.get("regime"))
        details.setdefault("iv_rank", signal.metadata.get("iv_rank"))
        details.setdefault("entry_time", self._now_eastern().isoformat())

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
            self._append_audit_event(
                event_type="order_filled",
                correlation_id=correlation_id,
                details={
                    "mode": "paper",
                    "symbol": signal.symbol,
                    "strategy": signal.strategy,
                    "quantity": signal.quantity,
                    "entry_credit": analysis.credit,
                    "position_id": result.get("position_id"),
                },
            )
            review_id = signal.metadata.get("llm_review_id")
            position_id = result.get("position_id")
            if position_id:
                signal.metadata["paper_position_id"] = str(position_id)
            if self.llm_advisor and review_id and position_id:
                self.llm_advisor.bind_position(str(review_id), str(position_id))
            self.risk_manager.register_open_position(
                symbol=signal.symbol,
                max_loss_per_contract=effective_max_loss,
                quantity=signal.quantity,
                strategy=signal.strategy,
                greeks={
                    "net_delta": getattr(analysis, "net_delta", 0.0),
                    "net_theta": getattr(analysis, "net_theta", 0.0),
                    "net_gamma": getattr(analysis, "net_gamma", 0.0),
                    "net_vega": getattr(analysis, "net_vega", 0.0),
                },
            )
            self.alerts.trade_opened(
                f"{signal.strategy} {signal.symbol} x{signal.quantity} credit={analysis.credit:.2f}",
                context={
                    "mode": "paper",
                    "position_id": position_id,
                    "regime": signal.metadata.get("regime"),
                },
            )
            return True

        opened = self._execute_live_entry(
            signal,
            details=details,
            max_loss_per_contract=effective_max_loss,
        )
        if opened:
            self._append_audit_event(
                event_type="order_placed",
                correlation_id=correlation_id,
                details={
                    "mode": "live",
                    "symbol": signal.symbol,
                    "strategy": signal.strategy,
                    "quantity": signal.quantity,
                    "position_id": signal.metadata.get("live_position_id"),
                },
            )
            review_id = signal.metadata.get("llm_review_id")
            live_position_id = signal.metadata.get("live_position_id")
            if self.llm_advisor and review_id and live_position_id:
                self.llm_advisor.bind_position(str(review_id), str(live_position_id))
            self.risk_manager.register_open_position(
                symbol=signal.symbol,
                max_loss_per_contract=effective_max_loss,
                quantity=signal.quantity,
                strategy=signal.strategy,
                greeks={
                    "net_delta": getattr(analysis, "net_delta", 0.0),
                    "net_theta": getattr(analysis, "net_theta", 0.0),
                    "net_gamma": getattr(analysis, "net_gamma", 0.0),
                    "net_vega": getattr(analysis, "net_vega", 0.0),
                },
            )
            self.alerts.trade_opened(
                f"{signal.strategy} {signal.symbol} x{signal.quantity} credit={analysis.credit:.2f}",
                context={
                    "mode": "live",
                    "position_id": signal.metadata.get("live_position_id"),
                    "regime": signal.metadata.get("regime"),
                },
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
            "net_delta": getattr(analysis, "net_delta", 0.0),
            "net_theta": getattr(analysis, "net_theta", 0.0),
            "net_gamma": getattr(analysis, "net_gamma", 0.0),
            "net_vega": getattr(analysis, "net_vega", 0.0),
        }

    def _review_entry_with_llm(self, signal: TradeSignal) -> bool:
        """Optionally review an entry with the configured LLM advisor."""
        if not self.llm_advisor or signal.action != "open":
            return True
        if self._service_degradation.get("rule_only_mode"):
            return True

        earnings_proximity = None
        if signal.analysis is not None:
            in_window, earnings_date = self.risk_manager.earnings_calendar.earnings_within_window(
                signal.symbol,
                signal.analysis.expiration,
            )
            if in_window:
                earnings_proximity = {
                    "in_window": True,
                    "earnings_date": earnings_date,
                    "expiration": signal.analysis.expiration,
                }
            else:
                earnings_proximity = {"in_window": False}

        sector_performance = signal.metadata.get("sector_performance")
        if not isinstance(sector_performance, dict) or not sector_performance:
            sector_performance = self._sector_performance_context(signal.symbol)

        context = {
            "trading_mode": self.config.trading_mode,
            "account_balance": self.risk_manager.portfolio.account_balance,
            "open_positions": len(self.risk_manager.portfolio.open_positions),
            "daily_pnl": self.risk_manager.portfolio.daily_pnl,
            "deployed_risk": self.risk_manager.portfolio.total_risk_deployed,
            "technical_context": signal.metadata.get("technical_context", {}),
            "iv_rank": signal.metadata.get("iv_rank"),
            "iv_percentile": signal.metadata.get("iv_rank"),
            "vol_surface": signal.metadata.get("vol_surface", {}),
            "options_flow": signal.metadata.get("options_flow", {}),
            "earnings_proximity": earnings_proximity,
            "economic_events": signal.metadata.get("economic_events", {}),
            "portfolio_exposure": {
                "total_delta": self.risk_manager.portfolio.net_delta,
                "total_theta": self.risk_manager.portfolio.net_theta,
                "total_gamma": self.risk_manager.portfolio.net_gamma,
                "total_vega": self.risk_manager.portfolio.net_vega,
                "sector_concentration": self.risk_manager.portfolio.sector_risk,
                "var": self.risk_manager.get_var_metrics(),
            },
            "sector_performance": sector_performance,
            "regime": self.circuit_state.get("regime", "normal"),
            "regime_confidence": self.circuit_state.get("regime_confidence"),
        }
        if self.news_scanner:
            try:
                news = self.news_scanner.build_context(
                    signal.symbol,
                    macro_events=context.get("economic_events", {}),
                )
                self._service_degradation["news_down"] = False
                context["news"] = news
                context["news_sentiment_summary"] = {
                    "symbol_sentiment": news.get("symbol_sentiment", 0.0),
                    "market_sentiment": news.get("market_sentiment", 0.0),
                    "dominant_market_topics": news.get("dominant_market_topics", []),
                }
            except Exception as exc:
                self._service_degradation["news_down"] = True
                logger.warning(
                    "Failed to fetch news context for %s: %s",
                    signal.symbol,
                    exc,
                )

        try:
            decision = self.llm_advisor.review_trade(signal, context)
            self._llm_timeout_streak = 0
            self._service_degradation["llm_down"] = False
        except Exception as e:
            if "timeout" in str(e).lower():
                self._llm_timeout_streak += 1
            if self.config.llm.mode == "blocking":
                logger.error("LLM review failed in blocking mode: %s", e)
                return False

            logger.warning("LLM review failed in advisory mode: %s", e)
            self._service_degradation["llm_down"] = True
            if (
                self.config.degradation.rule_only_on_llm_failures
                and self._llm_timeout_streak >= int(self.config.circuit_breakers.llm_timeout_streak)
            ):
                self._service_degradation["rule_only_mode"] = True
            self._append_audit_event(
                event_type="llm_review",
                correlation_id=str(signal.metadata.get("correlation_id", "")) or None,
                details={
                    "symbol": signal.symbol,
                    "strategy": signal.strategy,
                    "error": str(e),
                    "llm_down": True,
                },
            )
            return True

        signal.metadata["llm_review_id"] = decision.review_id
        self._append_audit_event(
            event_type="llm_review",
            correlation_id=str(signal.metadata.get("correlation_id", "")) or None,
            details={
                "symbol": signal.symbol,
                "strategy": signal.strategy,
                "verdict": decision.verdict,
                "confidence": decision.confidence_pct,
                "reasoning": decision.reasoning,
            },
        )

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
                decision.reasoning,
            )
            return False

        if not decision.approve:
            if self.config.llm.mode == "blocking":
                logger.info(
                    "Trade REJECTED by LLM: %s %s on %s | %s",
                    signal.strategy,
                    signal.action,
                    signal.symbol,
                    decision.reasoning,
                )
                return False

            logger.warning(
                "LLM flagged trade but advisory mode allows execution: %s",
                decision.reasoning,
            )
        else:
            logger.info(
                "LLM %s trade: %s (confidence %.1f%%)",
                decision.verdict,
                decision.reasoning,
                decision.confidence_pct,
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
        if analysis is None:
            return False
        quantity = signal.quantity

        try:
            order_factory = None
            spread_proxy = max(0.05, float(analysis.credit) * 0.20)
            if signal.strategy == "bull_put_spread":
                spread_proxy = max(
                    0.05,
                    abs(float(analysis.short_strike) - float(analysis.long_strike)) * 0.10,
                )
                order_factory = lambda price: self.schwab.build_bull_put_spread(
                    symbol=signal.symbol,
                    expiration=analysis.expiration,
                    short_strike=analysis.short_strike,
                    long_strike=analysis.long_strike,
                    quantity=quantity,
                    price=price,
                )
            elif signal.strategy == "bear_call_spread":
                spread_proxy = max(
                    0.05,
                    abs(float(analysis.short_strike) - float(analysis.long_strike)) * 0.10,
                )
                order_factory = lambda price: self.schwab.build_bear_call_spread(
                    symbol=signal.symbol,
                    expiration=analysis.expiration,
                    short_strike=analysis.short_strike,
                    long_strike=analysis.long_strike,
                    quantity=quantity,
                    price=price,
                )
            elif signal.strategy == "iron_condor":
                spread_proxy = max(
                    0.05,
                    max(
                        abs(float(analysis.put_short_strike) - float(analysis.put_long_strike)),
                        abs(float(analysis.call_long_strike) - float(analysis.call_short_strike)),
                    )
                    * 0.10,
                )
                order_factory = lambda price: self.schwab.build_iron_condor(
                    symbol=signal.symbol,
                    expiration=analysis.expiration,
                    put_long_strike=analysis.put_long_strike,
                    put_short_strike=analysis.put_short_strike,
                    call_short_strike=analysis.call_short_strike,
                    call_long_strike=analysis.call_long_strike,
                    quantity=quantity,
                    price=price,
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

                order_factory = lambda price: self.schwab.build_covered_call_open(
                    symbol=signal.symbol,
                    expiration=analysis.expiration,
                    short_strike=analysis.short_strike,
                    quantity=quantity,
                    price=price,
                )
            else:
                logger.warning("Live execution not implemented for: %s", signal.strategy)
                return False

            result = self.schwab.place_order_with_ladder(
                order_factory=order_factory,
                midpoint_price=float(analysis.credit),
                spread_width=spread_proxy,
                side="credit",
                step_timeout_seconds=self.config.execution.entry_step_timeout_seconds,
                max_attempts=self.config.execution.max_ladder_attempts,
                shifts=list(self.config.execution.entry_ladder_shifts),
                total_timeout_seconds=300,
            )
            logger.info("LIVE order placed: %s", result)
            status = str(result.get("status", "")).upper()
            if status in {"CANCELED", "REJECTED", "EXPIRED"}:
                return False

            self._record_execution_quality(
                symbol=signal.symbol,
                strategy=signal.strategy,
                side="entry",
                midpoint=float(result.get("midpoint_price", analysis.credit)),
                fill_price=float(result.get("fill_price", result.get("requested_price", analysis.credit))),
                status=status,
            )
            if self.live_ledger and analysis is not None:
                order_id = str(result.get("order_id", "")).strip()
                if order_id:
                    live_position_id = self.live_ledger.register_entry_order(
                        strategy=signal.strategy,
                        symbol=signal.symbol,
                        quantity=quantity,
                        max_loss=max_loss_per_contract if max_loss_per_contract is not None else analysis.max_loss,
                        entry_credit=analysis.credit,
                        details=details or self._build_position_details(analysis),
                        entry_order_id=order_id,
                        entry_order_status=str(result.get("status", "PLACED")),
                    )
                    signal.metadata["live_position_id"] = live_position_id
                else:
                    logger.warning(
                        "Live order missing broker order_id; recording as immediately open."
                    )
                    live_position_id = self.live_ledger.register_entry_order(
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
                    signal.metadata["live_position_id"] = live_position_id
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
        correlation_id = str(signal.metadata.get("correlation_id", "")).strip() or str(signal.position_id or "")
        logger.info(
            "CLOSING POSITION: %s | Reason: %s",
            signal.position_id, signal.reason,
        )

        if self.is_paper:
            result = self.paper_trader.execute_close(
                position_id=signal.position_id,
                reason=signal.reason,
                quantity=signal.quantity,
            )
            logger.info("Paper close result: %s", result)
            self.alerts.trade_closed(
                f"{signal.strategy} {signal.symbol} qty={signal.quantity} reason={signal.reason}",
                context={
                    "mode": "paper",
                    "position_id": signal.position_id,
                    "pnl": result.get("pnl"),
                    "status": result.get("status"),
                },
            )
            if self.llm_advisor and result.get("status") == "FILLED":
                self.llm_advisor.record_outcome(
                    str(signal.position_id),
                    float(result.get("pnl", 0.0) or 0.0),
                )
            self._append_audit_event(
                event_type="position_exit",
                correlation_id=correlation_id or None,
                details={
                    "mode": "paper",
                    "position_id": signal.position_id,
                    "symbol": signal.symbol,
                    "strategy": signal.strategy,
                    "reason": signal.reason,
                    "pnl": result.get("pnl"),
                    "status": result.get("status"),
                },
            )
        else:
            closed = self._execute_live_exit(signal)
            if not closed:
                logger.warning(
                    "Failed to submit live close order for %s.",
                    signal.position_id,
                )
            else:
                self.alerts.trade_closed(
                    f"{signal.strategy} {signal.symbol} qty={signal.quantity} reason={signal.reason}",
                    context={"mode": "live", "position_id": signal.position_id},
                )
                self._append_audit_event(
                    event_type="position_exit",
                    correlation_id=correlation_id or None,
                    details={
                        "mode": "live",
                        "position_id": signal.position_id,
                        "symbol": signal.symbol,
                        "strategy": signal.strategy,
                        "reason": signal.reason,
                        "status": "PLACED",
                    },
                )

    def _execute_roll(self, signal: TradeSignal) -> None:
        """Handle a roll action as close-then-open with same strategy family."""
        if not signal.position_id:
            return

        tracked_positions = self._get_tracked_positions()
        source_position = next(
            (
                item
                for item in tracked_positions
                if str(item.get("position_id", "")) == str(signal.position_id)
            ),
            None,
        )
        if not source_position:
            logger.warning("Roll skipped: source position %s not found", signal.position_id)
            return

        details = source_position.get("details", {}) if isinstance(source_position.get("details"), dict) else {}
        roll_count = int(details.get("roll_count", 0) or 0)
        if roll_count >= int(self.config.rolling.max_rolls_per_position):
            logger.info("Roll skipped: max rolls reached for %s", signal.position_id)
            return

        regime = str(self.circuit_state.get("regime", "normal"))
        decision = self.roll_manager.evaluate(source_position, regime=regime)
        min_credit_required = float(signal.metadata.get("min_credit_required", 0.0) or 0.0)
        if decision.should_roll:
            min_credit_required = max(min_credit_required, float(decision.min_credit_required))
        elif not signal.metadata.get("force_roll"):
            logger.info("Roll skipped for %s: %s", signal.position_id, decision.reason)
            return

        replacement = self._find_roll_replacement_signal(
            source_position=source_position,
            strategy_name=signal.strategy,
            symbol=signal.symbol,
            min_credit_required=min_credit_required,
        )
        if replacement is None:
            logger.info(
                "Roll skipped for %s: no qualified replacement position found.",
                signal.position_id,
            )
            return

        logger.info("ROLLING POSITION: %s | %s", signal.position_id, signal.reason)
        close_signal = TradeSignal(
            action="close",
            strategy=signal.strategy,
            symbol=signal.symbol,
            position_id=signal.position_id,
            reason=signal.reason,
            quantity=signal.quantity,
            metadata={"roll_close": True},
        )

        rolled = False
        new_position_id: Optional[str] = None
        if self.is_paper:
            close_result = self.paper_trader.execute_close(
                position_id=str(signal.position_id),
                reason=f"Roll close: {signal.reason}",
                quantity=signal.quantity,
            )
            if str(close_result.get("status", "")).upper() != "FILLED":
                logger.warning("Roll close failed for %s in paper mode.", signal.position_id)
                return
            rolled = self._try_execute_entry(replacement)
            if rolled:
                new_position_id = str(replacement.metadata.get("paper_position_id", "")).strip() or None
        else:
            if not self.live_ledger or not self.live_ledger.get_position(str(signal.position_id)):
                # Compatibility path for tests/mocks where live ledger is absent.
                self._execute_exit(close_signal)
                rolled = self._try_execute_entry(replacement)
            else:
                submitted = self._execute_live_exit(close_signal)
                if not submitted:
                    logger.warning("Roll close order was not submitted for %s.", signal.position_id)
                    return
                if not self._wait_for_live_position_close(str(signal.position_id), timeout_seconds=330):
                    logger.warning("Roll close did not fill in time for %s.", signal.position_id)
                    return
                rolled = self._try_execute_entry(replacement)
                if rolled:
                    new_position_id = str(replacement.metadata.get("live_position_id", "")).strip() or None

        if not rolled:
            logger.warning("Roll replacement entry failed for %s.", signal.position_id)
            return

        self._annotate_roll_linkage(
            source_position_id=str(signal.position_id),
            source_symbol=str(signal.symbol),
            target_position_id=new_position_id,
        )

    def _find_roll_replacement_signal(
        self,
        *,
        source_position: dict,
        strategy_name: str,
        symbol: str,
        min_credit_required: float,
    ) -> Optional[TradeSignal]:
        """Find the best roll replacement signal in a later expiration cycle."""
        chain_data, underlying_price = self._get_chain_data(symbol)
        if not chain_data or underlying_price <= 0:
            return None

        technical_context = None
        try:
            technical_context = self.technicals.get_context(symbol, self.schwab)
        except Exception:
            technical_context = None

        market_context = self._build_market_context(symbol, chain_data)
        market_context["roll_context"] = True
        current_dte = int(source_position.get("dte_remaining", 0) or 0)
        source_details = source_position.get("details", {}) if isinstance(source_position.get("details"), dict) else {}
        source_delta = safe_float(source_details.get("net_delta", source_position.get("net_delta", 0.0)), 0.0)
        candidates: list[TradeSignal] = []

        for strategy in self.strategies:
            strategy_signals = strategy.scan_for_entries(
                symbol,
                chain_data,
                underlying_price,
                technical_context=technical_context,
                market_context=market_context,
            )
            for candidate in strategy_signals:
                if candidate.strategy != strategy_name or not candidate.analysis:
                    continue
                if int(candidate.analysis.dte or 0) <= current_dte:
                    continue
                if float(candidate.analysis.credit or 0.0) < min_credit_required:
                    continue
                candidate_delta = safe_float(getattr(candidate.analysis, "net_delta", 0.0), 0.0)
                if abs(candidate_delta - source_delta) > 0.25:
                    continue
                self.roll_manager.annotate_roll_metadata(source_position, candidate)
                candidate.metadata.setdefault("rolled_from_position_id", source_position.get("position_id"))
                candidate.metadata.setdefault(
                    "roll_count",
                    int(source_details.get("roll_count", 0) or 0) + 1,
                )
                candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                int(item.analysis.dte or 9999),
                -(float(item.analysis.score or 0.0)),
            ),
        )
        return candidates[0] if candidates else None

    def _wait_for_live_position_close(self, position_id: str, *, timeout_seconds: int = 300) -> bool:
        """Wait for a live position to reach a terminal post-exit status."""
        deadline = time.time() + max(5, int(timeout_seconds))
        while time.time() < deadline:
            self._reconcile_live_orders()
            if not self.live_ledger:
                return False
            position = self.live_ledger.get_position(position_id)
            if not position:
                return False
            status = str(position.get("status", "")).lower()
            if status in {"closed", "closed_external", "rolled"}:
                return True
            if status in {"canceled", "rejected", "expired"}:
                return False
            time.sleep(3)
        return False

    def _annotate_roll_linkage(
        self,
        *,
        source_position_id: str,
        source_symbol: str,
        target_position_id: Optional[str],
    ) -> None:
        """Persist roll linkage metadata for paper/live ledgers and analytics."""
        if self.is_paper and self.paper_trader:
            for trade in reversed(self.paper_trader.closed_trades):
                if str(trade.get("position_id", "")) != source_position_id:
                    continue
                trade["status"] = "rolled"
                trade["rolled_to_position_id"] = target_position_id
                details = trade.get("details") if isinstance(trade.get("details"), dict) else {}
                details["roll_status"] = "rolled"
                if target_position_id:
                    details["rolled_to"] = target_position_id
                trade["details"] = details
                break
            self.paper_trader._save_state()
            return

        if self.live_ledger:
            self.live_ledger.mark_position_rolled(
                source_position_id=source_position_id,
                rolled_to_position_id=target_position_id,
            )

    def _execute_adjustment_plan(self, position: dict, plan) -> None:
        """Execute an adjustment action emitted by the adjustment engine."""
        action = str(getattr(plan, "action", "none")).lower()
        if action == "none":
            return

        position_id = str(position.get("position_id", ""))
        if not position_id:
            return
        logger.info(
            "Executing adjustment %s for %s (%s)",
            action,
            position_id,
            getattr(plan, "reason", ""),
        )

        if action == "roll_tested_side":
            self._execute_roll(
                TradeSignal(
                    action="roll",
                    strategy=str(position.get("strategy", "")),
                    symbol=str(position.get("symbol", "")),
                    position_id=position_id,
                    reason=f"Adjustment roll: {getattr(plan, 'reason', '')}",
                    quantity=max(1, safe_int(position.get("quantity"), 1)),
                    metadata={"force_roll": True, "roll_type": "defensive"},
                )
            )
            self._bump_adjustment_state(position_id=position_id, additional_cost=0.0, action=action)
            return

        chain_data, _ = self._get_chain_data(str(position.get("symbol", "")))
        if not chain_data:
            return

        if action == "add_wing":
            wing = self._select_adjustment_wing(position=position, chain_data=chain_data)
            if wing is None:
                return
            if not self._review_adjustment_with_llm(position, "add_wing", wing["cost"], note=wing["reason"]):
                return
            executed = self._execute_adjustment_long_option(
                symbol=str(position.get("symbol", "")),
                expiration=wing["expiration"],
                contract_type=wing["contract_type"],
                strike=wing["strike"],
                debit=wing["cost"],
                quantity=1,
                reason=f"Adjustment add_wing for {position_id}",
            )
            if executed:
                self._bump_adjustment_state(
                    position_id=position_id,
                    additional_cost=wing["cost"] * 100.0,
                    action=action,
                    extra_details={"last_wing_strike": wing["strike"]},
                )
            return

        if action == "add_hedge":
            hedge = self._select_adjustment_hedge(position=position, chain_data=chain_data)
            if hedge is None:
                return
            if not self._review_adjustment_with_llm(position, "add_hedge", hedge["cost"], note=hedge["reason"]):
                return
            executed = self._execute_adjustment_debit_spread(
                symbol=str(position.get("symbol", "")),
                expiration=hedge["expiration"],
                contract_type=hedge["contract_type"],
                long_strike=hedge["long_strike"],
                short_strike=hedge["short_strike"],
                debit=hedge["cost"],
                quantity=1,
                reason=f"Adjustment add_hedge for {position_id}",
            )
            if executed:
                self._bump_adjustment_state(
                    position_id=position_id,
                    additional_cost=hedge["cost"] * 100.0,
                    action=action,
                    extra_details={"last_hedge": hedge["contract_type"]},
                )

    def _review_adjustment_with_llm(self, position: dict, action: str, cost: float, *, note: str = "") -> bool:
        """Run LLM trade advisor over adjustment proposals before execution."""
        if not self.llm_advisor:
            return True
        details = position.get("details", {}) if isinstance(position.get("details"), dict) else {}
        analysis = SpreadAnalysis(
            symbol=str(position.get("symbol", "")),
            strategy=f"adjustment_{action}",
            expiration=str(details.get("expiration", position.get("expiration", ""))),
            dte=safe_int(position.get("dte_remaining"), 0),
            short_strike=safe_float(details.get("short_strike", details.get("put_short_strike", 0.0)), 0.0),
            long_strike=safe_float(details.get("long_strike", details.get("put_long_strike", 0.0)), 0.0),
            credit=max(0.01, float(cost)),
            max_loss=max(0.01, float(cost)),
            probability_of_profit=0.5,
            score=55.0,
        )
        signal = TradeSignal(
            action="open",
            strategy=f"adjustment_{action}",
            symbol=str(position.get("symbol", "")),
            analysis=analysis,
            quantity=1,
            metadata={"adjustment": True, "note": note},
        )
        return self._review_entry_with_llm(signal)

    def _select_adjustment_wing(self, *, position: dict, chain_data: dict) -> Optional[dict]:
        details = position.get("details", {}) if isinstance(position.get("details"), dict) else {}
        expiration = self._extract_expiration_key(details.get("expiration", position.get("expiration", "")))
        if not expiration:
            return None

        puts = chain_data.get("puts", {}).get(expiration, [])
        calls = chain_data.get("calls", {}).get(expiration, [])
        underlying = safe_float(position.get("underlying_price"), 0.0)
        put_short = safe_float(details.get("put_short_strike", details.get("short_strike", 0.0)), 0.0)
        call_short = safe_float(details.get("call_short_strike", 0.0), 0.0)
        is_put_test = put_short > 0 and (underlying <= (put_short * 1.01) or call_short <= 0)

        if is_put_test and puts:
            current_long = safe_float(details.get("put_long_strike", details.get("long_strike", 0.0)), 0.0)
            width = abs(put_short - current_long) if put_short > 0 and current_long > 0 else 5.0
            target_strike = max(0.01, current_long - max(1.0, width))
            option = self._closest_strike_option(puts, target_strike)
            if not option:
                return None
            return {
                "expiration": expiration,
                "contract_type": "P",
                "strike": safe_float(option.get("strike"), target_strike),
                "cost": max(0.01, safe_float(option.get("mid"), 0.0)),
                "reason": "put side tested",
            }

        if calls:
            current_long = safe_float(details.get("call_long_strike", details.get("long_strike", 0.0)), 0.0)
            width = abs(current_long - call_short) if call_short > 0 and current_long > 0 else 5.0
            target_strike = max(0.01, current_long + max(1.0, width))
            option = self._closest_strike_option(calls, target_strike)
            if not option:
                return None
            return {
                "expiration": expiration,
                "contract_type": "C",
                "strike": safe_float(option.get("strike"), target_strike),
                "cost": max(0.01, safe_float(option.get("mid"), 0.0)),
                "reason": "call side tested",
            }
        return None

    def _select_adjustment_hedge(self, *, position: dict, chain_data: dict) -> Optional[dict]:
        details = position.get("details", {}) if isinstance(position.get("details"), dict) else {}
        expiration = self._extract_expiration_key(details.get("expiration", position.get("expiration", "")))
        if not expiration:
            return None
        puts = chain_data.get("puts", {}).get(expiration, [])
        calls = chain_data.get("calls", {}).get(expiration, [])
        underlying = safe_float(position.get("underlying_price"), 0.0)
        put_short = safe_float(details.get("put_short_strike", details.get("short_strike", 0.0)), 0.0)
        call_short = safe_float(details.get("call_short_strike", 0.0), 0.0)
        is_put_test = put_short > 0 and (underlying <= (put_short * 1.01) or call_short <= 0)

        if is_put_test and len(puts) >= 2:
            long_leg = self._find_option_by_abs_delta(puts, 0.35)
            short_leg = self._find_option_by_abs_delta(puts, 0.20)
            if not long_leg or not short_leg:
                return None
            if safe_float(long_leg.get("strike"), 0.0) <= safe_float(short_leg.get("strike"), 0.0):
                return None
            debit = safe_float(long_leg.get("mid"), 0.0) - safe_float(short_leg.get("mid"), 0.0)
            if debit <= 0:
                return None
            return {
                "expiration": expiration,
                "contract_type": "P",
                "long_strike": safe_float(long_leg.get("strike"), 0.0),
                "short_strike": safe_float(short_leg.get("strike"), 0.0),
                "cost": debit,
                "reason": "put side tested",
            }

        if len(calls) >= 2:
            long_leg = self._find_option_by_abs_delta(calls, 0.35)
            short_leg = self._find_option_by_abs_delta(calls, 0.20)
            if not long_leg or not short_leg:
                return None
            if safe_float(long_leg.get("strike"), 0.0) >= safe_float(short_leg.get("strike"), 0.0):
                return None
            debit = safe_float(long_leg.get("mid"), 0.0) - safe_float(short_leg.get("mid"), 0.0)
            if debit <= 0:
                return None
            return {
                "expiration": expiration,
                "contract_type": "C",
                "long_strike": safe_float(long_leg.get("strike"), 0.0),
                "short_strike": safe_float(short_leg.get("strike"), 0.0),
                "cost": debit,
                "reason": "call side tested",
            }
        return None

    @staticmethod
    def _closest_strike_option(options: list[dict], target_strike: float) -> Optional[dict]:
        best = None
        best_dist = float("inf")
        for option in options:
            strike = safe_float(option.get("strike"), 0.0)
            if strike <= 0:
                continue
            dist = abs(strike - target_strike)
            if dist < best_dist:
                best = option
                best_dist = dist
        return best

    @staticmethod
    def _find_option_by_abs_delta(options: list[dict], target: float) -> Optional[dict]:
        best = None
        best_diff = float("inf")
        for option in options:
            delta = abs(safe_float(option.get("delta"), 0.0))
            if delta <= 0:
                continue
            diff = abs(delta - target)
            if diff < best_diff:
                best = option
                best_diff = diff
        return best

    def _execute_adjustment_long_option(
        self,
        *,
        symbol: str,
        expiration: str,
        contract_type: str,
        strike: float,
        debit: float,
        quantity: int,
        reason: str,
    ) -> bool:
        debit = max(0.01, float(debit))
        quantity = max(1, int(quantity))
        if self.is_paper:
            self.paper_trader.balance -= debit * quantity * 100.0
            self.paper_trader.orders.append(
                {
                    "order_id": f"paper_adj_{int(time.time())}",
                    "type": "adjustment",
                    "symbol": symbol,
                    "debit": round(debit, 4),
                    "quantity": quantity,
                    "reason": reason,
                    "timestamp": self._now_eastern().isoformat(),
                    "status": "FILLED",
                }
            )
            self.paper_trader._save_state()
            return True

        order_factory = lambda price: self.schwab.build_long_option_open(
            symbol=symbol,
            expiration=expiration,
            contract_type=contract_type,
            strike=strike,
            quantity=quantity,
            price=price,
        )
        result = self.schwab.place_order_with_ladder(
            order_factory=order_factory,
            midpoint_price=debit,
            spread_width=max(0.05, debit * 0.30),
            side="debit",
            step_timeout_seconds=self.config.execution.exit_step_timeout_seconds,
            max_attempts=self.config.execution.max_ladder_attempts,
            shifts=list(self.config.execution.exit_ladder_shifts),
            total_timeout_seconds=180,
        )
        return str(result.get("status", "")).upper() not in {"CANCELED", "REJECTED", "EXPIRED"}

    def _execute_adjustment_debit_spread(
        self,
        *,
        symbol: str,
        expiration: str,
        contract_type: str,
        long_strike: float,
        short_strike: float,
        debit: float,
        quantity: int,
        reason: str,
    ) -> bool:
        debit = max(0.01, float(debit))
        quantity = max(1, int(quantity))
        if self.is_paper:
            self.paper_trader.balance -= debit * quantity * 100.0
            self.paper_trader.orders.append(
                {
                    "order_id": f"paper_adj_spread_{int(time.time())}",
                    "type": "adjustment",
                    "symbol": symbol,
                    "debit": round(debit, 4),
                    "quantity": quantity,
                    "reason": reason,
                    "timestamp": self._now_eastern().isoformat(),
                    "status": "FILLED",
                }
            )
            self.paper_trader._save_state()
            return True

        order_factory = lambda price: self.schwab.build_debit_spread_open(
            symbol=symbol,
            expiration=expiration,
            contract_type=contract_type,
            long_strike=long_strike,
            short_strike=short_strike,
            quantity=quantity,
            price=price,
        )
        result = self.schwab.place_order_with_ladder(
            order_factory=order_factory,
            midpoint_price=debit,
            spread_width=max(0.05, abs(long_strike - short_strike) * 0.10),
            side="debit",
            step_timeout_seconds=self.config.execution.exit_step_timeout_seconds,
            max_attempts=self.config.execution.max_ladder_attempts,
            shifts=list(self.config.execution.exit_ladder_shifts),
            total_timeout_seconds=180,
        )
        return str(result.get("status", "")).upper() not in {"CANCELED", "REJECTED", "EXPIRED"}

    def _bump_adjustment_state(
        self,
        *,
        position_id: str,
        additional_cost: float,
        action: str,
        extra_details: Optional[dict] = None,
    ) -> None:
        """Increment adjustment counters and accumulated costs for a position."""
        details_update = {
            "last_adjustment": action,
        }
        if extra_details:
            details_update.update(extra_details)

        if self.is_paper and self.paper_trader:
            for pos in self.paper_trader.positions:
                if str(pos.get("position_id", "")) != position_id:
                    continue
                details = pos.get("details") if isinstance(pos.get("details"), dict) else {}
                details["adjustment_count"] = safe_int(details.get("adjustment_count"), 0) + 1
                details["adjustment_cost"] = round(
                    safe_float(details.get("adjustment_cost"), 0.0) + safe_float(additional_cost, 0.0),
                    4,
                )
                details.update(details_update)
                pos["details"] = details
                break
            self.paper_trader._save_state()
            return

        if self.live_ledger:
            tracked = self.live_ledger.get_position(position_id)
            if not tracked:
                return
            details = tracked.get("details") if isinstance(tracked.get("details"), dict) else {}
            next_count = safe_int(details.get("adjustment_count"), 0) + 1
            next_cost = round(
                safe_float(details.get("adjustment_cost"), 0.0) + safe_float(additional_cost, 0.0),
                4,
            )
            details_update["adjustment_count"] = next_count
            details_update["adjustment_cost"] = next_cost
            self.live_ledger.update_position_metadata(
                position_id,
                detail_fields=details_update,
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
        if signal.quantity > 0:
            quantity = max(1, min(quantity, int(signal.quantity)))
        strategy = tracked.get("strategy", "")
        symbol = tracked.get("symbol", signal.symbol)
        debit_limit = self._resolve_live_close_debit(tracked)

        try:
            order_factory = None
            spread_proxy = max(0.05, debit_limit * 0.20)
            if strategy == "bull_put_spread":
                order_factory = lambda price: self.schwab.build_bull_put_spread_close(
                    symbol=symbol,
                    expiration=expiration,
                    short_strike=float(details.get("short_strike", 0.0)),
                    long_strike=float(details.get("long_strike", 0.0)),
                    quantity=quantity,
                    price=price,
                )
            elif strategy == "bear_call_spread":
                order_factory = lambda price: self.schwab.build_bear_call_spread_close(
                    symbol=symbol,
                    expiration=expiration,
                    short_strike=float(details.get("short_strike", 0.0)),
                    long_strike=float(details.get("long_strike", 0.0)),
                    quantity=quantity,
                    price=price,
                )
            elif strategy == "covered_call":
                order_factory = lambda price: self.schwab.build_covered_call_close(
                    symbol=symbol,
                    expiration=expiration,
                    short_strike=float(details.get("short_strike", 0.0)),
                    quantity=quantity,
                    price=price,
                )
            elif strategy == "iron_condor":
                order_factory = lambda price: self.schwab.build_iron_condor_close(
                    symbol=symbol,
                    expiration=expiration,
                    put_long_strike=float(details.get("put_long_strike", 0.0)),
                    put_short_strike=float(details.get("put_short_strike", 0.0)),
                    call_short_strike=float(details.get("call_short_strike", 0.0)),
                    call_long_strike=float(details.get("call_long_strike", 0.0)),
                    quantity=quantity,
                    price=price,
                )
            else:
                logger.warning("Live exit not implemented for strategy: %s", strategy)
                return False

            result = self.schwab.place_order_with_ladder(
                order_factory=order_factory,
                midpoint_price=debit_limit,
                spread_width=spread_proxy,
                side="debit",
                step_timeout_seconds=self.config.execution.exit_step_timeout_seconds,
                max_attempts=self.config.execution.max_ladder_attempts,
                shifts=list(self.config.execution.exit_ladder_shifts),
                total_timeout_seconds=180,
            )
            logger.info("LIVE exit order placed: %s", result)
            status = str(result.get("status", "")).upper()
            if status in {"CANCELED", "REJECTED", "EXPIRED"}:
                return False

            self._record_execution_quality(
                symbol=symbol,
                strategy=strategy,
                side="exit",
                midpoint=float(result.get("midpoint_price", debit_limit)),
                fill_price=float(result.get("fill_price", result.get("requested_price", debit_limit))),
                status=status,
            )

            order_id = str(result.get("order_id", "")).strip()
            if not order_id:
                logger.warning(
                    "Live exit order for %s missing broker order_id.",
                    signal.position_id,
                )
                return False

            tracked_quantity = max(1, int(tracked.get("quantity", 1)))
            if quantity < tracked_quantity:
                self.live_ledger.register_exit_order(
                    position_id=signal.position_id,
                    exit_order_id=order_id,
                    reason=signal.reason,
                    quantity=quantity,
                )
            else:
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

    def _sector_performance_context(self, symbol: str) -> dict:
        """Compute symbol/sector/SPY relative-strength context for LLM decisions."""
        symbol_key = str(symbol or "").upper().strip()
        if not symbol_key:
            return {}

        sector_etf = self._sector_etf_for_symbol(symbol_key)
        symbol_return = self._one_month_return(symbol_key)
        sector_return = self._one_month_return(sector_etf)
        spy_return = self._one_month_return("SPY")

        return {
            "symbol": symbol_key,
            "sector_etf": sector_etf,
            "symbol_1m_return": round(symbol_return, 6),
            "sector_1m_return": round(sector_return, 6),
            "spy_1m_return": round(spy_return, 6),
            "sector_vs_spy": round(sector_return - spy_return, 6),
            "symbol_vs_sector": round(symbol_return - sector_return, 6),
        }

    def _one_month_return(self, symbol: str) -> float:
        """Approximate one-month return from daily closes."""
        try:
            bars = self.schwab.get_price_history(symbol, days=40)
        except Exception:
            return 0.0
        if not isinstance(bars, list):
            return 0.0

        closes = [
            safe_float(row.get("close", 0.0), 0.0)
            for row in bars
            if isinstance(row, dict)
        ]
        closes = [value for value in closes if value > 0]
        if len(closes) < 2:
            return 0.0

        return (closes[-1] / closes[0]) - 1.0

    @staticmethod
    def _sector_etf_for_symbol(symbol: str) -> str:
        symbol_key = str(symbol).upper().strip()
        for etf, members in SECTOR_ETF_BY_SYMBOL.items():
            if symbol_key in members:
                return etf
        return "SPY"

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

    def _record_execution_quality(
        self,
        *,
        symbol: str,
        strategy: str,
        side: str,
        midpoint: float,
        fill_price: float,
        status: str,
    ) -> None:
        """Persist per-trade execution quality/slippage stats."""
        path = Path("bot/data/execution_quality.json")
        payload = load_json(path, {"fills": [], "meta": {}})
        if not isinstance(payload, dict):
            payload = {"fills": [], "meta": {}}
        fills = payload.get("fills")
        if not isinstance(fills, list):
            fills = []
            payload["fills"] = fills

        midpoint_value = float(midpoint or 0.0)
        fill_value = float(fill_price or midpoint_value)
        if side == "entry":
            slippage = midpoint_value - fill_value
        else:
            slippage = fill_value - midpoint_value

        fills.append(
            {
                "timestamp": self._now_eastern().isoformat(),
                "symbol": symbol,
                "strategy": strategy,
                "side": side,
                "midpoint": round(midpoint_value, 4),
                "fill_price": round(fill_value, 4),
                "slippage": round(slippage, 4),
                "status": status,
            }
        )
        payload["fills"] = fills[-5000:]

        week_key = self._now_eastern().strftime("%Y-W%W")
        week_fills = []
        for item in payload["fills"]:
            ts_raw = str(item.get("timestamp", ""))
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts.strftime("%Y-W%W") == week_key:
                week_fills.append(item)
        if week_fills:
            avg = sum(float(item.get("slippage", 0.0) or 0.0) for item in week_fills) / len(week_fills)
            meta = payload.get("meta")
            if not isinstance(meta, dict):
                meta = {}
                payload["meta"] = meta
            if meta.get("last_logged_week") != week_key:
                logger.info(
                    "Weekly average slippage: %.4f across %d fills",
                    avg,
                    len(week_fills),
                )
                meta["last_logged_week"] = week_key
                meta["last_week_avg_slippage"] = round(avg, 4)

        dump_json(path, payload)

    # ── Data Fetching ────────────────────────────────────────────────

    def _build_market_context(self, symbol: str, chain_data: dict) -> dict:
        """Build lightweight shared market context for strategy scans."""
        iv = self._average_chain_iv(chain_data)
        iv_rank = self.iv_history.update_and_rank(symbol, iv) if iv > 0 else 0.0
        context = {
            "iv_rank": iv_rank,
            "regime": self.circuit_state.get("regime", "normal"),
            "vix": self.circuit_state.get("vix"),
            "account_balance": float(self.risk_manager.portfolio.account_balance or 0.0),
            "service_degradation": dict(self._service_degradation),
        }
        if self.current_regime_state:
            context["regime_confidence"] = self.current_regime_state.confidence
            context["regime_weights"] = self.current_regime_state.recommended_strategy_weights
            context["position_size_scalar"] = self.current_regime_state.recommended_position_size_scalar

        if self.vol_surface_analyzer:
            try:
                price_history = self.schwab.get_price_history(symbol, days=90)
                vol_ctx = self.vol_surface_analyzer.analyze(
                    symbol=symbol,
                    chain_data=chain_data,
                    price_history=price_history,
                )
                context["vol_surface"] = vol_ctx.to_dict()
            except Exception as exc:
                logger.debug("Vol-surface analysis failed for %s: %s", symbol, exc)

        if self.options_flow_analyzer:
            try:
                flow_ctx = self.options_flow_analyzer.analyze(
                    symbol=symbol,
                    chain_data=chain_data,
                    previous_chain_data=self._chain_history_cache.get(symbol.upper()),
                )
                context["options_flow"] = flow_ctx.to_dict()
            except Exception as exc:
                logger.debug("Options-flow analysis failed for %s: %s", symbol, exc)

        if self.econ_calendar:
            try:
                context["economic_events"] = self.econ_calendar.context(days=30)
            except Exception as exc:
                logger.debug("Economic-calendar context failed for %s: %s", symbol, exc)

        self._chain_history_cache[symbol.upper()] = chain_data
        return context

    @staticmethod
    def _average_chain_iv(chain_data: dict) -> float:
        values: list[float] = []
        for side in ("calls", "puts"):
            exp_map = chain_data.get(side, {})
            if not isinstance(exp_map, dict):
                continue
            for options in exp_map.values():
                for option in options or []:
                    if not isinstance(option, dict):
                        continue
                    iv = safe_float(option.get("iv", 0.0), 0.0)
                    if iv > 0:
                        values.append(iv)
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _get_chain_data(self, symbol: str) -> tuple[dict, float]:
        """Fetch and parse options chain data for a symbol."""
        try:
            raw_chain = self.schwab.get_option_chain(symbol)
            self._record_api_health(True)
            parsed = SchwabClient.parse_option_chain(raw_chain)
            return parsed, parsed.get("underlying_price", 0.0)
        except Exception as e:
            self._record_api_health(False)
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
            underlying_price = float(chain_data.get("underlying_price", 0.0) or 0.0)
            position["underlying_price"] = underlying_price

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
                underlying_price=underlying_price,
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
        schedule.every().day.at("16:10", tz="America/New_York").do(self._scheduled_dashboard)
        logger.info("Scheduled dashboard generation at 16:10 ET")

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
            self.alerts.daily_summary(
                "Paper daily summary",
                context={
                    "balance": summary.get("balance"),
                    "total_trades": summary.get("total_trades"),
                    "total_pnl": summary.get("total_pnl"),
                    "regime": self.circuit_state.get("regime", "normal"),
                },
            )
            if self._now_eastern().weekday() == 4:
                self.alerts.weekly_summary(
                    "Paper weekly summary",
                    context={
                        "balance": summary.get("balance"),
                        "return_pct": summary.get("return_pct"),
                    },
                )
            if self.llm_advisor:
                self.llm_advisor.log_weekly_hit_rate()
            self._auto_generate_dashboard_if_due(force=True)
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
        self.alerts.daily_summary(
            "Live daily summary",
            context={
                "balance": balance,
                "open_positions": ledger_summary.get("open", 0),
                "realized_pnl_today": ledger_summary.get("realized_pnl_today", 0.0),
                "regime": self.circuit_state.get("regime", "normal"),
            },
        )
        if self._now_eastern().weekday() == 4:
            self.alerts.weekly_summary(
                "Live weekly summary",
                context={
                    "balance": balance,
                    "daily_pnl": self._compute_live_daily_pnl(),
                },
            )
        if self.llm_advisor:
            self.llm_advisor.log_weekly_hit_rate()
        self._auto_generate_dashboard_if_due(force=True)

    def _scheduled_dashboard(self) -> None:
        """Scheduled dashboard generation wrapper."""
        self._auto_generate_dashboard_if_due(force=True)

    def _auto_generate_dashboard_if_due(self, *, force: bool = False) -> None:
        """Generate dashboard once per day after close or when explicitly forced."""
        now_et = self._now_eastern()
        today_iso = now_et.date().isoformat()
        if self._dashboard_generated_date == today_iso:
            return
        if not force and now_et.hour < 16:
            return
        try:
            output = self.generate_dashboard()
            self._dashboard_generated_date = today_iso
            logger.info("End-of-day dashboard generated: %s", output)
        except Exception as exc:
            logger.error("Dashboard generation failed: %s", exc)

    def _setup_signal_handlers(self) -> None:
        """Register SIGTERM/SIGINT handlers for graceful stop."""
        if self._signal_handlers_ready:
            return
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._handle_shutdown)
            except Exception as exc:
                logger.debug("Could not register signal handler for %s: %s", sig, exc)
        self._signal_handlers_ready = True

    def _handle_shutdown(self, signum, frame) -> None:  # pragma: no cover - invoked by OS signals
        """Gracefully stop loop and cancel pending live orders."""
        logger.info("Shutdown signal received: %s", signum)
        self._running = False

        if not self.is_paper and self.live_ledger:
            order_ids = []
            try:
                order_ids.extend(self.live_ledger.pending_entry_order_ids())
                order_ids.extend(self.live_ledger.pending_exit_order_ids())
            except Exception as exc:
                logger.warning("Failed to enumerate pending order IDs on shutdown: %s", exc)

            for order_id in order_ids:
                try:
                    self.schwab.cancel_order(order_id)
                except Exception as exc:
                    logger.warning("Failed to cancel pending order %s during shutdown: %s", order_id, exc)

            if hasattr(self.live_ledger, "save"):
                try:
                    self.live_ledger.save()
                except Exception as exc:
                    logger.warning("Failed to persist live ledger on shutdown: %s", exc)

        logger.info("Shutdown complete")

    def _maybe_log_heartbeat(self) -> None:
        """Emit a periodic liveness heartbeat in long-running mode."""
        now_et = self._now_eastern()
        if self._last_heartbeat_time and (now_et - self._last_heartbeat_time) < timedelta(minutes=5):
            return

        positions = len(self.risk_manager.portfolio.open_positions)
        balance = float(self.risk_manager.portfolio.account_balance or 0.0)
        logger.info(
            "Heartbeat: %d positions, balance=$%s, mode=%s, regime=%s",
            positions,
            f"{balance:,.2f}",
            self.config.trading_mode,
            self.circuit_state.get("regime", "normal"),
        )
        self._last_heartbeat_time = now_et

    def _maintain_streaming(self) -> None:
        """Re-attempt streaming on disconnect while polling fallback remains active."""
        if not self.config.degradation.fallback_polling_on_stream_failure:
            return
        if self.schwab.streaming_connected():
            self._service_degradation["stream_down"] = False
            return
        self._service_degradation["stream_down"] = True
        now_et = self._now_eastern()
        if self._last_stream_retry_time and (now_et - self._last_stream_retry_time) < timedelta(minutes=5):
            return
        self._last_stream_retry_time = now_et
        self._setup_streaming()

    def generate_dashboard(self) -> str:
        """Generate dashboard HTML from current paper/live state."""
        closed_trades = self._recent_closed_trades()
        strategy_breakdown: dict[str, dict] = {}
        regime_breakdown: dict[str, dict] = {}
        monthly: dict[str, float] = {}
        daily_calendar: dict[str, float] = {}
        winners: list[dict] = []
        losers: list[dict] = []
        rolled_credits: list[float] = []

        source_positions = []
        if self.is_paper and self.paper_trader:
            source_positions = self.paper_trader.get_positions()
            closed_source = self.paper_trader.closed_trades
        else:
            source_positions = self.live_ledger.list_positions(statuses={"open", "opening", "closing"}) if self.live_ledger else []
            closed_source = self.live_ledger.list_positions(statuses={"closed", "closed_external", "rolled"}) if self.live_ledger else []

        for trade in closed_source:
            pnl = float(trade.get("pnl", trade.get("realized_pnl", 0.0)) or 0.0)
            strategy = str(trade.get("strategy", "unknown"))
            close_date = str(trade.get("close_date", ""))
            month = close_date[:7] if len(close_date) >= 7 else "unknown"
            monthly[month] = monthly.get(month, 0.0) + pnl
            if close_date:
                day_key = close_date[:10]
                if len(day_key) == 10:
                    daily_calendar[day_key] = daily_calendar.get(day_key, 0.0) + pnl

            entry = strategy_breakdown.setdefault(
                strategy,
                {"wins": 0, "count": 0, "total_pnl": 0.0, "sum_wins": 0.0, "sum_losses": 0.0, "losses": 0},
            )
            entry["count"] += 1
            if pnl > 0:
                entry["wins"] += 1
                entry["sum_wins"] += pnl
            elif pnl < 0:
                entry["losses"] += 1
                entry["sum_losses"] += pnl
            entry["total_pnl"] += pnl

            details = trade.get("details", {}) if isinstance(trade.get("details"), dict) else {}
            regime_key = str(trade.get("regime", details.get("regime", "unknown")) or "unknown")
            regime_entry = regime_breakdown.setdefault(
                regime_key,
                {"count": 0, "wins": 0, "total_pnl": 0.0},
            )
            regime_entry["count"] += 1
            if pnl > 0:
                regime_entry["wins"] += 1
            regime_entry["total_pnl"] += pnl

            row = {"symbol": trade.get("symbol", ""), "pnl": pnl}
            (winners if pnl >= 0 else losers).append(row)
            if str(trade.get("status", "")).lower() == "rolled" or str(details.get("roll_status", "")).lower() == "rolled":
                rolled_credits.append(safe_float(trade.get("entry_credit"), 0.0))

        strategy_view = {}
        for strategy, stats in strategy_breakdown.items():
            count = max(1, int(stats["count"]))
            strategy_view[strategy] = {
                "win_rate": (stats["wins"] / count) * 100.0,
                "avg_pnl": stats["total_pnl"] / count,
                "avg_profit": (stats["sum_wins"] / max(1, int(stats["wins"]))),
                "avg_loss": (stats["sum_losses"] / max(1, int(stats["losses"]))),
                "total_pnl": stats["total_pnl"],
            }

        regime_view = {}
        for regime, stats in regime_breakdown.items():
            count = max(1, int(stats["count"]))
            regime_view[regime] = {
                "trades": int(stats["count"]),
                "win_rate": (stats["wins"] / count) * 100.0,
                "total_pnl": stats["total_pnl"],
            }

        risk_metrics = self._compute_equity_risk_metrics(self._equity_history)
        sector_total = sum(self.risk_manager.portfolio.sector_risk.values()) or 1.0
        sector_exposure = {
            sector: round(value / sector_total * 100.0, 2)
            for sector, value in self.risk_manager.portfolio.sector_risk.items()
        }
        open_positions_table = []
        for position in source_positions:
            if not isinstance(position, dict):
                continue
            entry_credit = safe_float(position.get("entry_credit"), 0.0)
            current_value = safe_float(position.get("current_value"), 0.0)
            quantity = max(1, safe_int(position.get("quantity"), 1))
            pnl = (entry_credit - current_value) * quantity * 100.0
            details = position.get("details", {}) if isinstance(position.get("details"), dict) else {}
            open_positions_table.append(
                {
                    "symbol": position.get("symbol", ""),
                    "strategy": position.get("strategy", ""),
                    "quantity": quantity,
                    "dte_remaining": position.get("dte_remaining", details.get("dte", "")),
                    "pnl": round(pnl, 2),
                    "delta": safe_float(details.get("net_delta", position.get("net_delta", 0.0)), 0.0),
                    "theta": safe_float(details.get("net_theta", position.get("net_theta", 0.0)), 0.0),
                    "gamma": safe_float(details.get("net_gamma", position.get("net_gamma", 0.0)), 0.0),
                    "vega": safe_float(details.get("net_vega", position.get("net_vega", 0.0)), 0.0),
                }
            )

        trade_journal_payload = load_json(Path(self.config.llm.journal_file), {"entries": []})
        trade_journal = trade_journal_payload.get("entries", []) if isinstance(trade_journal_payload, dict) else []
        if not isinstance(trade_journal, list):
            trade_journal = []
        hedge_payload = load_json(HEDGE_COSTS_PATH, {"entries": []})
        hedge_entries = hedge_payload.get("entries", []) if isinstance(hedge_payload, dict) else []
        if not isinstance(hedge_entries, list):
            hedge_entries = []
        month_key = self._now_eastern().strftime("%Y-%m")
        month_cost = 0.0
        lifetime_cost = 0.0
        for entry in hedge_entries:
            if not isinstance(entry, dict):
                continue
            if not bool(entry.get("executed", False)):
                continue
            cost = safe_float(entry.get("estimated_cost"), 0.0)
            lifetime_cost += cost
            if str(entry.get("timestamp", "")).startswith(month_key):
                month_cost += cost

        payload = {
            "equity_curve": self._equity_history,
            "monthly_pnl": monthly,
            "daily_pnl_calendar": daily_calendar,
            "strategy_breakdown": strategy_view,
            "regime_performance": regime_view,
            "top_winners": sorted(winners, key=lambda row: row["pnl"], reverse=True)[:5],
            "top_losers": sorted(losers, key=lambda row: row["pnl"])[:5],
            "risk_metrics": risk_metrics,
            "portfolio_greeks": self.risk_manager.get_portfolio_greeks(),
            "sector_exposure": sector_exposure,
            "circuit_breakers": self.circuit_state,
            "regime_state": {
                "regime": self.circuit_state.get("regime", "normal"),
                "confidence": self.circuit_state.get("regime_confidence", 0.0),
            },
            "service_degradation": dict(self._service_degradation),
            "correlation_matrix": self.risk_manager.get_correlation_matrix(),
            "var_metrics": self.risk_manager.get_var_metrics(),
            "open_positions_table": open_positions_table,
            "trade_journal": trade_journal[-10:],
            "hedge_costs": {
                "month_to_date": round(month_cost, 2),
                "lifetime": round(lifetime_cost, 2),
                "executed_count": sum(
                    1
                    for item in hedge_entries
                    if isinstance(item, dict) and bool(item.get("executed", False))
                ),
            },
            "roll_metrics": {
                "rolled_count": len(rolled_credits),
                "avg_roll_credit_captured": round(
                    (sum(rolled_credits) / len(rolled_credits)) if rolled_credits else 0.0,
                    4,
                ),
            },
            "portfolio_halt_until": self._portfolio_halt_until.isoformat() if self._portfolio_halt_until else None,
            "closed_trades": len(closed_trades),
            "open_positions": len(source_positions),
        }
        return generate_dashboard(payload)

    @staticmethod
    def _compute_equity_risk_metrics(equity_history: list[dict]) -> dict:
        if not equity_history:
            return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "current_drawdown": 0.0}

        values = [float(point.get("equity", 0.0) or 0.0) for point in equity_history if point.get("equity") is not None]
        if len(values) < 2:
            return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "current_drawdown": 0.0}

        returns = []
        for prev, current in zip(values[:-1], values[1:]):
            if prev > 0:
                returns.append((current / prev) - 1.0)
        if not returns:
            return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "current_drawdown": 0.0}

        mean_ret = sum(returns) / len(returns)
        variance = sum((value - mean_ret) ** 2 for value in returns) / max(1, (len(returns) - 1))
        std = variance ** 0.5
        downside = [value for value in returns if value < 0]
        downside_std = (
            (sum(value ** 2 for value in downside) / max(1, len(downside))) ** 0.5
            if downside
            else 0.0
        )

        peak = values[0]
        max_drawdown = 0.0
        for value in values:
            peak = max(peak, value)
            if peak > 0:
                max_drawdown = min(max_drawdown, (value - peak) / peak)
        current_drawdown = 0.0
        if peak > 0:
            current_drawdown = (values[-1] - peak) / peak

        sharpe = (mean_ret / std * (252 ** 0.5)) if std > 0 else 0.0
        sortino = (mean_ret / downside_std * (252 ** 0.5)) if downside_std > 0 else 0.0
        return {
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "max_drawdown": abs(round(max_drawdown, 6)),
            "current_drawdown": abs(round(current_drawdown, 6)),
        }

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

        self._setup_signal_handlers()
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
                self._maybe_log_heartbeat()
                self._maintain_streaming()
                self._auto_generate_dashboard_if_due()
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
