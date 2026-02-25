"""Configuration management for the trading bot."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
SECRET_PLACEHOLDER_MARKERS = ("your_", "_here", "changeme")


@dataclass
class SchwabAccountConfig:
    name: str = ""
    hash: str = ""
    risk_profile: str = "moderate"


@dataclass
class SchwabConfig:
    app_key: str = ""
    app_secret: str = ""
    callback_url: str = "https://127.0.0.1:8182/callback"
    token_path: str = "./token.json"
    account_hash: str = ""
    account_index: int = -1
    accounts: list[SchwabAccountConfig] = field(default_factory=list)


@dataclass
class CreditSpreadConfig:
    enabled: bool = True
    direction: str = "both"
    min_dte: int = 20
    max_dte: int = 45
    short_delta: float = 0.30
    spread_width: int = 5
    min_credit_pct: float = 0.30
    profit_target_pct: float = 0.50
    stop_loss_pct: float = 2.0


@dataclass
class IronCondorConfig:
    enabled: bool = False
    min_dte: int = 25
    max_dte: int = 50
    short_delta: float = 0.16
    spread_width: int = 5
    min_credit_pct: float = 0.30
    profit_target_pct: float = 0.50
    stop_loss_pct: float = 2.0


@dataclass
class CoveredCallConfig:
    enabled: bool = False
    min_dte: int = 20
    max_dte: int = 45
    short_delta: float = 0.30
    tickers: list = field(default_factory=list)


@dataclass
class ScannerConfig:
    enabled: bool = True
    # Max number of tickers to trade from scan results
    max_scan_results: int = 20
    # Cache scan results for this many seconds (avoid re-scanning too fast)
    cache_seconds: int = 1800
    # Price range for underlying
    min_price: float = 10.0
    max_price: float = 0  # 0 = no upper limit
    # Minimum underlying daily volume
    min_underlying_volume: int = 500_000
    # Include movers from Schwab API
    include_movers: bool = True
    # Pause between ticker requests during scans (helps avoid rate limits)
    request_pause_seconds: float = 0.10
    # Retry/backoff controls for transient API errors
    error_backoff_seconds: float = 1.0
    max_retry_attempts: int = 2
    max_consecutive_errors: int = 20
    # Optional hard cap on scanned universe size (0 = disabled)
    max_symbols_per_scan: int = 0
    # Optional symbol blacklist never scanned/traded
    blacklist: list = field(default_factory=list)
    # Relative contribution of movers API ranking boost
    movers_weight: float = 0.15
    # Relative contribution of sector-rotation boost
    sector_rotation_weight: float = 0.15
    # Relative contribution of relative-strength boost
    relative_strength_weight: float = 0.20
    # Relative contribution of option-liquidity boost
    options_liquidity_weight: float = 0.15


@dataclass
class RiskConfig:
    max_portfolio_risk_pct: float = 5.0
    max_position_risk_pct: float = 2.0
    max_open_positions: int = 10
    max_positions_per_symbol: int = 2
    min_account_balance: float = 5000.0
    max_daily_loss_pct: float = 3.0
    covered_call_notional_risk_pct: float = 20.0
    # Portfolio greek/correlation risk controls
    max_portfolio_delta_abs: float = 50.0
    max_portfolio_vega_pct_of_account: float = 0.5
    max_sector_risk_pct: float = 40.0
    correlation_lookback_days: int = 60
    correlation_threshold: float = 0.8


@dataclass
class ScheduleConfig:
    scan_times: list = field(default_factory=lambda: ["09:45", "11:00", "14:00"])
    position_check_interval: int = 15
    trading_days: list = field(
        default_factory=lambda: [
            "monday", "tuesday", "wednesday", "thursday", "friday"
        ]
    )


@dataclass
class ExecutionConfig:
    cancel_stale_orders: bool = True
    stale_order_minutes: int = 20
    include_partials_in_ledger: bool = True
    entry_step_timeout_seconds: int = 90
    exit_step_timeout_seconds: int = 60
    max_ladder_attempts: int = 3
    entry_ladder_shifts: list = field(default_factory=lambda: [0.0, 0.25, 0.50])
    exit_ladder_shifts: list = field(default_factory=lambda: [0.0, 0.25, 0.50])


@dataclass
class LLMConfig:
    enabled: bool = False
    # "ollama" for local models, "openai" for cloud models
    provider: str = "openai"
    model: str = "gpt-5.2-pro"
    base_url: str = "http://127.0.0.1:11434"
    mode: str = "advisory"  # "advisory" or "blocking"
    risk_style: str = "moderate"  # "conservative" | "moderate" | "aggressive"
    timeout_seconds: int = 20
    temperature: float = 0.1
    min_confidence: float = 0.55
    track_record_file: str = "bot/data/llm_track_record.json"


@dataclass
class NewsConfig:
    enabled: bool = True
    provider: str = "google_rss"
    cache_seconds: int = 900
    request_timeout_seconds: int = 10
    max_symbol_headlines: int = 8
    max_market_headlines: int = 15
    include_symbol_news: bool = True
    include_market_news: bool = True
    finnhub_api_key: str = ""
    llm_sentiment_enabled: bool = True
    llm_sentiment_cache_seconds: int = 1800
    market_queries: list = field(
        default_factory=lambda: [
            "stock market",
            "federal reserve interest rates",
            "inflation report",
            "earnings season",
            "options market volatility",
        ]
    )


@dataclass
class AlertsConfig:
    enabled: bool = False
    webhook_url: str = ""
    min_level: str = "ERROR"
    timeout_seconds: int = 5
    require_in_live: bool = True


@dataclass
class BotConfig:
    trading_mode: str = "paper"
    schwab: SchwabConfig = field(default_factory=SchwabConfig)
    credit_spreads: CreditSpreadConfig = field(default_factory=CreditSpreadConfig)
    iron_condors: IronCondorConfig = field(default_factory=IronCondorConfig)
    covered_calls: CoveredCallConfig = field(default_factory=CoveredCallConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    watchlist: list = field(
        default_factory=lambda: ["SPY", "QQQ", "IWM", "AAPL", "MSFT"]
    )
    risk: RiskConfig = field(default_factory=RiskConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    log_level: str = "INFO"
    log_file: str = "logs/tradingbot.log"
    log_max_bytes: int = 10_485_760
    log_backup_count: int = 5


def load_config(config_path: str = "config.yaml") -> BotConfig:
    """Load configuration from YAML file and environment variables."""
    load_dotenv()

    cfg = BotConfig()

    # Load YAML config
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            raw = yaml.safe_load(f)
        if raw:
            _apply_yaml(cfg, raw)

    # Override with environment variables (credentials always from env)
    cfg.schwab.app_key = os.getenv("SCHWAB_APP_KEY", cfg.schwab.app_key)
    cfg.schwab.app_secret = os.getenv("SCHWAB_APP_SECRET", cfg.schwab.app_secret)
    cfg.schwab.callback_url = os.getenv(
        "SCHWAB_CALLBACK_URL", cfg.schwab.callback_url
    )
    cfg.schwab.token_path = os.getenv("SCHWAB_TOKEN_PATH", cfg.schwab.token_path)
    cfg.schwab.account_hash = os.getenv(
        "SCHWAB_ACCOUNT_HASH", cfg.schwab.account_hash
    )
    cfg.schwab.account_index = _env_int(
        "SCHWAB_ACCOUNT_INDEX", cfg.schwab.account_index, minimum=-1
    )

    env_mode = os.getenv("TRADING_MODE")
    if env_mode:
        cfg.trading_mode = env_mode

    cfg.llm.enabled = _env_bool("LLM_ENABLED", cfg.llm.enabled)
    cfg.llm.provider = os.getenv("LLM_PROVIDER", cfg.llm.provider)
    cfg.llm.model = os.getenv("LLM_MODEL", cfg.llm.model)
    cfg.llm.base_url = os.getenv("LLM_BASE_URL", cfg.llm.base_url)
    cfg.llm.mode = os.getenv("LLM_MODE", cfg.llm.mode)
    cfg.llm.risk_style = os.getenv("LLM_RISK_STYLE", cfg.llm.risk_style)
    cfg.llm.timeout_seconds = _env_int(
        "LLM_TIMEOUT_SECONDS", cfg.llm.timeout_seconds, minimum=1
    )
    cfg.llm.temperature = _env_float(
        "LLM_TEMPERATURE", cfg.llm.temperature, minimum=0.0, maximum=2.0
    )
    cfg.llm.min_confidence = _env_float(
        "LLM_MIN_CONFIDENCE", cfg.llm.min_confidence, minimum=0.0, maximum=1.0
    )
    cfg.llm.track_record_file = os.getenv(
        "LLM_TRACK_RECORD_FILE",
        cfg.llm.track_record_file,
    )
    cfg.news.enabled = _env_bool("NEWS_ENABLED", cfg.news.enabled)
    cfg.news.cache_seconds = _env_int(
        "NEWS_CACHE_SECONDS", cfg.news.cache_seconds, minimum=0
    )
    cfg.news.request_timeout_seconds = _env_int(
        "NEWS_REQUEST_TIMEOUT_SECONDS", cfg.news.request_timeout_seconds, minimum=1
    )
    cfg.news.max_symbol_headlines = _env_int(
        "NEWS_MAX_SYMBOL_HEADLINES", cfg.news.max_symbol_headlines, minimum=1
    )
    cfg.news.max_market_headlines = _env_int(
        "NEWS_MAX_MARKET_HEADLINES", cfg.news.max_market_headlines, minimum=1
    )
    cfg.news.include_symbol_news = _env_bool(
        "NEWS_INCLUDE_SYMBOL_NEWS", cfg.news.include_symbol_news
    )
    cfg.news.include_market_news = _env_bool(
        "NEWS_INCLUDE_MARKET_NEWS", cfg.news.include_market_news
    )
    cfg.news.finnhub_api_key = os.getenv(
        "FINNHUB_API_KEY",
        cfg.news.finnhub_api_key,
    )
    cfg.news.llm_sentiment_enabled = _env_bool(
        "NEWS_LLM_SENTIMENT_ENABLED",
        cfg.news.llm_sentiment_enabled,
    )
    cfg.news.llm_sentiment_cache_seconds = _env_int(
        "NEWS_LLM_SENTIMENT_CACHE_SECONDS",
        cfg.news.llm_sentiment_cache_seconds,
        minimum=0,
    )
    cfg.execution.cancel_stale_orders = _env_bool(
        "EXECUTION_CANCEL_STALE_ORDERS", cfg.execution.cancel_stale_orders
    )
    cfg.execution.stale_order_minutes = _env_int(
        "EXECUTION_STALE_ORDER_MINUTES", cfg.execution.stale_order_minutes, minimum=1
    )
    cfg.execution.include_partials_in_ledger = _env_bool(
        "EXECUTION_INCLUDE_PARTIALS_IN_LEDGER",
        cfg.execution.include_partials_in_ledger,
    )
    cfg.alerts.enabled = _env_bool("ALERTS_ENABLED", cfg.alerts.enabled)
    cfg.alerts.webhook_url = os.getenv("ALERTS_WEBHOOK_URL", cfg.alerts.webhook_url)
    cfg.alerts.min_level = os.getenv("ALERTS_MIN_LEVEL", cfg.alerts.min_level)
    cfg.alerts.timeout_seconds = _env_int(
        "ALERTS_TIMEOUT_SECONDS", cfg.alerts.timeout_seconds, minimum=1
    )
    cfg.alerts.require_in_live = _env_bool(
        "ALERTS_REQUIRE_IN_LIVE", cfg.alerts.require_in_live
    )
    cfg.log_max_bytes = _env_int("LOG_MAX_BYTES", cfg.log_max_bytes, minimum=1024)
    cfg.log_backup_count = _env_int(
        "LOG_BACKUP_COUNT", cfg.log_backup_count, minimum=1
    )

    _normalize_config(cfg)

    return cfg


def _apply_yaml(cfg: BotConfig, raw: dict) -> None:
    """Apply raw YAML dict onto the BotConfig."""
    cfg.trading_mode = raw.get("trading_mode", cfg.trading_mode)
    cfg.watchlist = raw.get("watchlist", cfg.watchlist)

    schwab = raw.get("schwab", {})
    if schwab:
        for key, val in schwab.items():
            if hasattr(cfg.schwab, key):
                setattr(cfg.schwab, key, val)

    accounts = raw.get("accounts", [])
    if isinstance(accounts, list):
        normalized_accounts: list[SchwabAccountConfig] = []
        for account in accounts:
            if not isinstance(account, dict):
                continue
            normalized_accounts.append(
                SchwabAccountConfig(
                    name=str(account.get("name", "")).strip(),
                    hash=str(account.get("hash", "")).strip(),
                    risk_profile=str(account.get("risk_profile", "moderate")).strip(),
                )
            )
        if normalized_accounts:
            cfg.schwab.accounts = normalized_accounts

    strats = raw.get("strategies", {})

    cs = strats.get("credit_spreads", {})
    if cs:
        for key, val in cs.items():
            if hasattr(cfg.credit_spreads, key):
                setattr(cfg.credit_spreads, key, val)

    ic = strats.get("iron_condors", {})
    if ic:
        for key, val in ic.items():
            if hasattr(cfg.iron_condors, key):
                setattr(cfg.iron_condors, key, val)

    cc = strats.get("covered_calls", {})
    if cc:
        for key, val in cc.items():
            if hasattr(cfg.covered_calls, key):
                setattr(cfg.covered_calls, key, val)

    scanner = raw.get("scanner", {})
    if scanner:
        for key, val in scanner.items():
            if hasattr(cfg.scanner, key):
                setattr(cfg.scanner, key, val)

    risk = raw.get("risk", {})
    if risk:
        for key, val in risk.items():
            if hasattr(cfg.risk, key):
                setattr(cfg.risk, key, val)

    sched = raw.get("schedule", {})
    if sched:
        for key, val in sched.items():
            if hasattr(cfg.schedule, key):
                setattr(cfg.schedule, key, val)

    execution = raw.get("execution", {})
    if execution:
        for key, val in execution.items():
            if hasattr(cfg.execution, key):
                setattr(cfg.execution, key, val)

    llm = raw.get("llm", {})
    if llm:
        for key, val in llm.items():
            if hasattr(cfg.llm, key):
                setattr(cfg.llm, key, val)

    news = raw.get("news", {})
    if news:
        for key, val in news.items():
            if hasattr(cfg.news, key):
                setattr(cfg.news, key, val)

    alerts = raw.get("alerts", {})
    if alerts:
        for key, val in alerts.items():
            if hasattr(cfg.alerts, key):
                setattr(cfg.alerts, key, val)

    log_cfg = raw.get("logging", {})
    if log_cfg:
        cfg.log_level = log_cfg.get("level", cfg.log_level)
        cfg.log_file = log_cfg.get("file", cfg.log_file)
        cfg.log_max_bytes = log_cfg.get("max_bytes", cfg.log_max_bytes)
        cfg.log_backup_count = log_cfg.get("backup_count", cfg.log_backup_count)


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(
    name: str,
    default: int,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Parse an integer environment variable with optional bounds."""
    value = os.getenv(name)
    if value is None:
        return default

    try:
        parsed = int(value.strip())
    except ValueError:
        logger.warning("Invalid integer for %s=%r. Using default %r.", name, value, default)
        return default

    if minimum is not None and parsed < minimum:
        return minimum
    if maximum is not None and parsed > maximum:
        return maximum
    return parsed


def _env_float(
    name: str,
    default: float,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Parse a float environment variable with optional bounds."""
    value = os.getenv(name)
    if value is None:
        return default

    try:
        parsed = float(value.strip())
    except ValueError:
        logger.warning("Invalid float for %s=%r. Using default %r.", name, value, default)
        return default

    if minimum is not None and parsed < minimum:
        return minimum
    if maximum is not None and parsed > maximum:
        return maximum
    return parsed


def _normalize_config(cfg: BotConfig) -> None:
    """Normalize string enum-like fields and sanitize basic list inputs."""
    cfg.schwab.app_key = _sanitize_secret(
        cfg.schwab.app_key, field_name="SCHWAB_APP_KEY"
    )
    cfg.schwab.app_secret = _sanitize_secret(
        cfg.schwab.app_secret, field_name="SCHWAB_APP_SECRET"
    )
    cfg.schwab.account_hash = _sanitize_secret(
        cfg.schwab.account_hash, field_name="SCHWAB_ACCOUNT_HASH"
    )

    cfg.trading_mode = _normalize_choice(
        cfg.trading_mode,
        allowed={"paper", "live"},
        default="paper",
        field_name="trading_mode",
    )

    cfg.llm.provider = _normalize_choice(
        cfg.llm.provider,
        allowed={"ollama", "openai", "anthropic"},
        default="ollama",
        field_name="llm.provider",
    )
    cfg.llm.mode = _normalize_choice(
        cfg.llm.mode,
        allowed={"advisory", "blocking"},
        default="advisory",
        field_name="llm.mode",
    )
    cfg.llm.risk_style = _normalize_choice(
        cfg.llm.risk_style,
        allowed={"conservative", "moderate", "aggressive"},
        default="moderate",
        field_name="llm.risk_style",
    )

    cfg.log_level = _normalize_choice(
        cfg.log_level,
        allowed={"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
        default="INFO",
        field_name="logging.level",
        transform=str.upper,
    )

    cfg.watchlist = _normalize_symbol_list(cfg.watchlist, default=["SPY"])
    cfg.covered_calls.tickers = _normalize_symbol_list(
        cfg.covered_calls.tickers, default=[]
    )
    cfg.risk.covered_call_notional_risk_pct = max(
        0.0, min(100.0, float(cfg.risk.covered_call_notional_risk_pct))
    )

    cfg.schedule.trading_days = _normalize_trading_days(cfg.schedule.trading_days)
    cfg.execution.stale_order_minutes = max(1, int(cfg.execution.stale_order_minutes))
    cfg.execution.cancel_stale_orders = bool(cfg.execution.cancel_stale_orders)
    cfg.execution.include_partials_in_ledger = bool(cfg.execution.include_partials_in_ledger)
    cfg.scanner.request_pause_seconds = max(0.0, float(cfg.scanner.request_pause_seconds))
    cfg.scanner.error_backoff_seconds = max(0.1, float(cfg.scanner.error_backoff_seconds))
    cfg.scanner.max_retry_attempts = max(0, int(cfg.scanner.max_retry_attempts))
    cfg.scanner.max_consecutive_errors = max(1, int(cfg.scanner.max_consecutive_errors))
    cfg.scanner.max_symbols_per_scan = max(0, int(cfg.scanner.max_symbols_per_scan))
    cfg.scanner.blacklist = _normalize_symbol_list(cfg.scanner.blacklist, default=[])
    cfg.scanner.movers_weight = max(0.0, min(1.0, float(cfg.scanner.movers_weight)))
    cfg.scanner.sector_rotation_weight = max(0.0, min(1.0, float(cfg.scanner.sector_rotation_weight)))
    cfg.scanner.relative_strength_weight = max(0.0, min(1.0, float(cfg.scanner.relative_strength_weight)))
    cfg.scanner.options_liquidity_weight = max(0.0, min(1.0, float(cfg.scanner.options_liquidity_weight)))
    cfg.llm.timeout_seconds = max(1, int(cfg.llm.timeout_seconds))
    cfg.llm.temperature = max(0.0, min(2.0, float(cfg.llm.temperature)))
    cfg.llm.min_confidence = max(0.0, min(1.0, float(cfg.llm.min_confidence)))
    cfg.llm.track_record_file = str(cfg.llm.track_record_file or "bot/data/llm_track_record.json")
    if cfg.llm.provider == "anthropic":
        if not str(cfg.llm.model).strip() or str(cfg.llm.model).strip().lower().startswith("gpt-"):
            cfg.llm.model = "claude-sonnet-4-20250514"

    cfg.news.provider = _normalize_choice(
        cfg.news.provider,
        allowed={"google_rss"},
        default="google_rss",
        field_name="news.provider",
    )
    cfg.news.cache_seconds = max(0, int(cfg.news.cache_seconds))
    cfg.news.request_timeout_seconds = max(1, int(cfg.news.request_timeout_seconds))
    cfg.news.max_symbol_headlines = max(1, int(cfg.news.max_symbol_headlines))
    cfg.news.max_market_headlines = max(1, int(cfg.news.max_market_headlines))
    cfg.news.include_symbol_news = bool(cfg.news.include_symbol_news)
    cfg.news.include_market_news = bool(cfg.news.include_market_news)
    cfg.news.finnhub_api_key = _sanitize_secret(
        cfg.news.finnhub_api_key, field_name="FINNHUB_API_KEY"
    )
    cfg.news.llm_sentiment_enabled = bool(cfg.news.llm_sentiment_enabled)
    cfg.news.llm_sentiment_cache_seconds = max(0, int(cfg.news.llm_sentiment_cache_seconds))
    cfg.news.market_queries = _normalize_string_list(
        cfg.news.market_queries,
        default=[
            "stock market",
            "federal reserve interest rates",
            "inflation report",
            "earnings season",
            "options market volatility",
        ],
    )
    cfg.alerts.min_level = _normalize_choice(
        cfg.alerts.min_level,
        allowed={"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
        default="ERROR",
        field_name="alerts.min_level",
        transform=str.upper,
    )
    cfg.alerts.timeout_seconds = max(1, int(cfg.alerts.timeout_seconds))
    cfg.alerts.require_in_live = bool(cfg.alerts.require_in_live)
    cfg.log_max_bytes = max(1024, int(cfg.log_max_bytes))
    cfg.log_backup_count = max(1, int(cfg.log_backup_count))
    cfg.execution.entry_step_timeout_seconds = max(10, int(cfg.execution.entry_step_timeout_seconds))
    cfg.execution.exit_step_timeout_seconds = max(10, int(cfg.execution.exit_step_timeout_seconds))
    cfg.execution.max_ladder_attempts = max(1, int(cfg.execution.max_ladder_attempts))
    cfg.execution.entry_ladder_shifts = _normalize_float_list(
        cfg.execution.entry_ladder_shifts,
        default=[0.0, 0.25, 0.50],
    )
    cfg.execution.exit_ladder_shifts = _normalize_float_list(
        cfg.execution.exit_ladder_shifts,
        default=[0.0, 0.25, 0.50],
    )
    cfg.risk.max_portfolio_delta_abs = max(0.0, float(cfg.risk.max_portfolio_delta_abs))
    cfg.risk.max_portfolio_vega_pct_of_account = max(0.0, float(cfg.risk.max_portfolio_vega_pct_of_account))
    cfg.risk.max_sector_risk_pct = max(0.0, min(100.0, float(cfg.risk.max_sector_risk_pct)))
    cfg.risk.correlation_lookback_days = max(20, int(cfg.risk.correlation_lookback_days))
    cfg.risk.correlation_threshold = max(0.0, min(1.0, float(cfg.risk.correlation_threshold)))
    cfg.schwab.accounts = _normalize_accounts(cfg.schwab.accounts)


def _normalize_choice(
    value: object,
    allowed: set[str],
    default: str,
    field_name: str,
    transform=lambda s: s.lower(),
) -> str:
    """Normalize an enum-like string field and validate allowed values."""
    normalized = transform(str(value).strip()) if value is not None else ""
    if normalized in allowed:
        return normalized

    logger.warning(
        "Invalid %s=%r. Falling back to %r.",
        field_name,
        value,
        default,
    )
    return default


def _normalize_symbol_list(value: object, default: list[str]) -> list[str]:
    """Normalize a list of ticker strings."""
    if not isinstance(value, list):
        return default

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in value:
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized or default


def _normalize_trading_days(value: object) -> list[str]:
    """Normalize configured trading days to lowercase weekday names."""
    valid = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    if not isinstance(value, list):
        return ["monday", "tuesday", "wednesday", "thursday", "friday"]

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in value:
        day = str(raw).strip().lower()
        if day in valid and day not in seen:
            seen.add(day)
            normalized.append(day)

    return normalized or ["monday", "tuesday", "wednesday", "thursday", "friday"]


def _normalize_string_list(value: object, default: list[str]) -> list[str]:
    """Normalize a list of non-empty strings with de-duplication."""
    if not isinstance(value, list):
        return default

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in value:
        text = str(raw).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)

    return normalized or default


def _normalize_float_list(value: object, default: list[float]) -> list[float]:
    """Normalize a small list of bounded float values."""
    if not isinstance(value, list):
        return list(default)

    normalized: list[float] = []
    for raw in value:
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            continue
        normalized.append(max(0.0, min(1.0, parsed)))

    return normalized or list(default)


def _normalize_accounts(value: object) -> list[SchwabAccountConfig]:
    """Normalize configured multi-account definitions."""
    if not isinstance(value, list):
        return []

    normalized: list[SchwabAccountConfig] = []
    seen_hashes: set[str] = set()
    for raw in value:
        if isinstance(raw, SchwabAccountConfig):
            account = raw
        elif isinstance(raw, dict):
            account = SchwabAccountConfig(
                name=str(raw.get("name", "")).strip(),
                hash=str(raw.get("hash", "")).strip(),
                risk_profile=str(raw.get("risk_profile", "moderate")).strip(),
            )
        else:
            continue

        account_hash = _sanitize_secret(account.hash, field_name="accounts.hash")
        if not account_hash or account_hash in seen_hashes:
            continue
        seen_hashes.add(account_hash)
        normalized.append(
            SchwabAccountConfig(
                name=account.name or account_hash[-4:],
                hash=account_hash,
                risk_profile=_normalize_choice(
                    account.risk_profile,
                    allowed={"conservative", "moderate", "aggressive"},
                    default="moderate",
                    field_name=f"accounts[{account.name or account_hash[-4:]}].risk_profile",
                ),
            )
        )

    return normalized


def _sanitize_secret(value: object, field_name: str) -> str:
    """Normalize sensitive string values and clear common placeholder values."""
    text = str(value or "").strip()
    if not text:
        return ""

    lowered = text.lower()
    if any(marker in lowered for marker in SECRET_PLACEHOLDER_MARKERS):
        logger.warning(
            "Ignoring placeholder value for %s. Set a real secret before live use.",
            field_name,
        )
        return ""

    return text
