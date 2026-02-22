"""Configuration management for the trading bot."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class SchwabConfig:
    app_key: str = ""
    app_secret: str = ""
    callback_url: str = "https://127.0.0.1:8182/callback"
    token_path: str = "./token.json"
    account_hash: str = ""


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
    enabled: bool = True
    min_dte: int = 25
    max_dte: int = 50
    short_delta: float = 0.16
    spread_width: int = 5
    min_credit_pct: float = 0.30
    profit_target_pct: float = 0.50
    stop_loss_pct: float = 2.0


@dataclass
class CoveredCallConfig:
    enabled: bool = True
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


@dataclass
class RiskConfig:
    max_portfolio_risk_pct: float = 5.0
    max_position_risk_pct: float = 2.0
    max_open_positions: int = 10
    max_positions_per_symbol: int = 2
    min_account_balance: float = 5000.0
    max_daily_loss_pct: float = 3.0


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
    llm: LLMConfig = field(default_factory=LLMConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    log_level: str = "INFO"
    log_file: str = "logs/tradingbot.log"


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

    _normalize_config(cfg)

    return cfg


def _apply_yaml(cfg: BotConfig, raw: dict) -> None:
    """Apply raw YAML dict onto the BotConfig."""
    cfg.trading_mode = raw.get("trading_mode", cfg.trading_mode)
    cfg.watchlist = raw.get("watchlist", cfg.watchlist)

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

    log_cfg = raw.get("logging", {})
    if log_cfg:
        cfg.log_level = log_cfg.get("level", cfg.log_level)
        cfg.log_file = log_cfg.get("file", cfg.log_file)


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
    cfg.trading_mode = _normalize_choice(
        cfg.trading_mode,
        allowed={"paper", "live"},
        default="paper",
        field_name="trading_mode",
    )

    cfg.llm.provider = _normalize_choice(
        cfg.llm.provider,
        allowed={"ollama", "openai"},
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

    cfg.schedule.trading_days = _normalize_trading_days(cfg.schedule.trading_days)
    cfg.llm.timeout_seconds = max(1, int(cfg.llm.timeout_seconds))
    cfg.llm.temperature = max(0.0, min(2.0, float(cfg.llm.temperature)))
    cfg.llm.min_confidence = max(0.0, min(1.0, float(cfg.llm.min_confidence)))

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
