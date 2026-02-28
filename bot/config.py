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
class NakedPutConfig:
    enabled: bool = False
    min_dte: int = 25
    max_dte: int = 45
    short_delta: float = 0.22
    profit_target_pct: float = 0.50
    exit_dte: int = 21


@dataclass
class CalendarSpreadConfig:
    enabled: bool = False
    front_min_dte: int = 20
    front_max_dte: int = 30
    back_min_dte: int = 50
    back_max_dte: int = 60
    profit_target_pct: float = 0.25
    exit_dte: int = 7


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
    max_open_positions: int = 25
    max_positions_per_symbol: int = 5
    min_account_balance: float = 5000.0
    max_daily_loss_pct: float = 15.0
    covered_call_notional_risk_pct: float = 20.0
    # Portfolio greek/correlation risk controls
    max_portfolio_delta_abs: float = 50.0
    max_portfolio_vega_pct_of_account: float = 0.5
    max_sector_risk_pct: float = 80.0
    correlation_lookback_days: int = 60
    correlation_threshold: float = 0.95
    max_portfolio_correlation: float = 0.95
    var_enabled: bool = False
    var_lookback_days: int = 60
    var_limit_pct_95: float = 3.0
    var_limit_pct_99: float = 5.0
    correlated_loss_threshold: int = 3
    correlated_loss_pct: float = 0.50
    correlated_loss_cooldown_hours: int = 2
    gamma_week_tight_stop: float = 1.5
    expiration_day_close_pct: float = 0.03


@dataclass
class ScheduleConfig:
    scan_times: list = field(default_factory=lambda: ["09:45", "11:00", "14:00"])
    position_check_interval: int = 15
    trading_days: list = field(
        default_factory=lambda: ["monday", "tuesday", "wednesday", "thursday", "friday"]
    )


@dataclass
class ExecutionConfig:
    cancel_stale_orders: bool = True
    stale_order_minutes: int = 20
    include_partials_in_ledger: bool = True
    partial_fill_handling: bool = True
    market_move_cancel_pct: float = 1.0
    block_first_minutes: int = 15
    block_last_minutes: int = 10
    entry_step_timeout_seconds: int = 45
    exit_step_timeout_seconds: int = 60
    max_ladder_attempts: int = 4
    smart_ladder_enabled: bool = True
    ladder_width_fractions: list = field(
        default_factory=lambda: [0.0, 0.10, 0.25, 0.40]
    )
    ladder_step_timeouts_seconds: list = field(default_factory=lambda: [45, 45, 45, 30])
    entry_ladder_shifts: list = field(default_factory=lambda: [0.0, 0.25, 0.50])
    exit_ladder_shifts: list = field(default_factory=lambda: [0.0, 0.25, 0.50])


@dataclass
class SignalRankingConfig:
    enabled: bool = True
    weight_score: float = 0.30
    weight_pop: float = 0.25
    weight_credit: float = 0.10
    weight_vol_premium: float = 0.10
    ml_score_weight: float = 0.25
    top_ranked_to_log: int = 5


@dataclass
class MLScorerConfig:
    enabled: bool = True
    min_training_trades: int = 30
    retrain_day: str = "sunday"
    retrain_time: str = "18:00"
    feature_importance_log: bool = True
    closed_trades_file: str = "bot/data/closed_trades.json"
    model_file: str = "bot/data/ml_model.json"
    feature_importance_file: str = "bot/data/ml_feature_importance.json"


@dataclass
class ThetaHarvestConfig:
    enabled: bool = True
    satisfied_pct: float = 0.80
    underperform_pct: float = 0.30
    cutoff_hour: int = 14


@dataclass
class CorrelationMonitorConfig:
    enabled: bool = True
    lookback_days: int = 20
    crisis_threshold: float = 0.95
    stress_threshold: float = 0.85


@dataclass
class ExecutionTimingConfig:
    adaptive: bool = True
    min_fills_per_bucket: int = 20
    analysis_file: str = "bot/data/execution_timing_analysis.json"


@dataclass
class TerminalUIConfig:
    enabled: bool = False
    refresh_rate: float = 0.5
    max_activity_events: int = 50
    show_rejected_trades: bool = True
    compact_mode: bool = False


@dataclass
class RLPromptOptimizerConfig:
    enabled: bool = False
    min_trades_for_pattern: int = 8
    loss_rate_threshold: float = 0.60
    max_rules: int = 25
    rolling_window_size: int = 100
    learned_rules_file: str = "bot/data/learned_rules.json"
    audit_log_file: str = "bot/data/audit_log.jsonl"


@dataclass
class AltDataConfig:
    gex_enabled: bool = True
    dark_pool_proxy_enabled: bool = True
    social_sentiment_enabled: bool = True
    social_sentiment_cache_minutes: int = 30
    social_sentiment_model: str = "gemini-3.1-pro-thinking-preview"


@dataclass
class ExecutionAlgoConfig:
    enabled: bool = False
    algo_type: str = "smart_ladder"  # smart_ladder | twap | iceberg | adaptive
    twap_slices: int = 4
    twap_window_seconds: int = 300
    iceberg_visible_qty: int = 1
    adaptive_spread_pause_threshold: float = 1.5
    adaptive_spread_accelerate_threshold: float = 0.8


@dataclass
class StrategySandboxConfig:
    enabled: bool = False
    min_failing_score: float = 40.0
    consecutive_fail_cycles: int = 3
    backtest_days: int = 30
    min_sharpe: float = 0.5
    max_drawdown_pct: float = 15.0
    deployment_days: int = 5
    sizing_scalar: float = 0.5
    state_file: str = "bot/data/strategy_sandbox_state.json"
    active_strategy: dict = field(default_factory=dict)


@dataclass
class ScalingConfig:
    enabled: bool = False
    scale_in_delay_minutes: int = 60
    scale_in_max_adds: int = 2
    partial_exit_pct: float = 0.40
    partial_exit_size: float = 0.50


@dataclass
class WalkForwardConfig:
    enabled: bool = False
    train_days: int = 60
    test_days: int = 20
    step_days: int = 20


@dataclass
class MonteCarloConfig:
    enabled: bool = True
    simulations: int = 10_000
    var_limit_pct: float = 3.0


@dataclass
class ReconciliationConfig:
    interval_minutes: int = 30
    auto_import: bool = True


@dataclass
class MultiTimeframeConfig:
    enabled: bool = True
    min_agreement: int = 2


@dataclass
class CooldownConfig:
    graduated: bool = True
    level_1_losses: int = 2
    level_1_reduction: float = 0.25
    level_2_losses: int = 3
    level_2_reduction: float = 0.50


@dataclass
class LLMConfig:
    enabled: bool = False
    # "ollama" for local models, "google"/"openai"/"anthropic" for cloud models
    provider: str = "google"
    model: str = "gemini-3.1-pro-thinking-preview"
    base_url: str = "http://127.0.0.1:11434"
    mode: str = "advisory"  # "advisory" or "blocking"
    risk_style: str = "aggressive"  # "conservative" | "moderate" | "aggressive"
    timeout_seconds: int = 20
    temperature: float = 0.1
    min_confidence: float = 0.10
    track_record_file: str = "bot/data/llm_track_record.json"
    reasoning_effort: str = "xhigh"
    text_verbosity: str = "low"
    max_output_tokens: int = 500
    chat_fallback_model: str = "gemini-3.1-flash-thinking-preview"
    ensemble_enabled: bool = False
    ensemble_models: list = field(default_factory=lambda: ["google:gemini-3.1-pro-thinking-preview"])
    ensemble_agreement_threshold: float = 0.66
    multi_turn_enabled: bool = True
    multi_turn_confidence_threshold: float = 70.0
    adversarial_review_enabled: bool = True
    adversarial_loss_threshold_pct: float = 0.50
    journal_enabled: bool = False
    journal_file: str = "bot/data/trade_journal.json"
    journal_context_entries: int = 20
    explanations_file: str = "bot/data/trade_explanations.json"


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
    llm_model: str = "gemini-3.1-pro-thinking-preview"
    llm_reasoning_effort: str = "xhigh"
    llm_text_verbosity: str = "low"
    llm_max_output_tokens: int = 400
    llm_chat_fallback_model: str = "gemini-3.1-flash-thinking-preview"
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
    trade_notifications: bool = False
    daily_summary: bool = False
    regime_changes: bool = False
    weekly_summary: bool = False
    drawdown_thresholds: list = field(default_factory=lambda: [1.0, 2.0, 3.0])
    webhook_format: str = "generic"  # generic | slack | discord


@dataclass
class RegimeConfig:
    enabled: bool = False
    min_confidence: float = 0.55
    uncertain_size_scalar: float = 0.80
    cache_seconds: int = 1800
    history_file: str = "bot/data/regime_history.json"


@dataclass
class VolSurfaceConfig:
    enabled: bool = False
    require_positive_vol_risk_premium: bool = True
    max_vol_of_vol_for_condors: float = 0.20


@dataclass
class EconCalendarConfig:
    enabled: bool = False
    refresh_days: int = 1
    cache_file: str = "bot/data/econ_calendar.json"
    high_severity_policy: str = "reduce_size"  # skip | reduce_size | widen | none
    medium_severity_policy: str = "widen"
    low_severity_policy: str = "none"


@dataclass
class FlowConfig:
    enabled: bool = False
    unusual_volume_multiple: float = 5.0
    include_in_llm_context: bool = True


@dataclass
class RollingConfig:
    enabled: bool = False
    min_dte_trigger: int = 7
    min_credit_for_roll: float = 0.15
    max_rolls_per_position: int = 2
    allow_defensive_rolls: bool = True


@dataclass
class ExitsConfig:
    adaptive_targets: bool = True
    trailing_stop_enabled: bool = False
    trailing_stop_activation_pct: float = 0.25
    trailing_stop_floor_pct: float = 0.10


@dataclass
class AdjustmentsConfig:
    enabled: bool = False
    delta_test_threshold: float = 0.50
    min_dte_remaining: int = 7
    max_adjustments_per_position: int = 2


@dataclass
class SizingConfig:
    method: str = "fixed"  # fixed | kelly
    kelly_fraction: float = 0.5
    kelly_min_trades: int = 30
    drawdown_decay_threshold: float = 0.05
    equity_curve_scaling: bool = True
    equity_curve_lookback: int = 20
    max_scale_up: float = 1.25
    max_scale_down: float = 0.50


@dataclass
class StrategyAllocationConfig:
    enabled: bool = True
    lookback_trades: int = 30
    min_sharpe_for_boost: float = 1.5
    cold_start_penalty: float = 0.75
    cold_start_window_days: int = 60
    cold_start_min_trades: int = 10


@dataclass
class GreeksBudgetConfig:
    enabled: bool = True
    reduce_size_to_fit: bool = True
    limits: dict = field(
        default_factory=lambda: {
            "BULL_TREND": {
                "delta_min": -50.0,
                "delta_max": 80.0,
                "vega_min": -200.0,
                "vega_max": 100.0,
            },
            "BEAR_TREND": {
                "delta_min": -80.0,
                "delta_max": 30.0,
                "vega_min": -100.0,
                "vega_max": 200.0,
            },
            "HIGH_VOL_CHOP": {
                "delta_min": -30.0,
                "delta_max": 30.0,
                "vega_min": -300.0,
                "vega_max": -50.0,
            },
            "LOW_VOL_GRIND": {
                "delta_min": -40.0,
                "delta_max": 60.0,
                "vega_min": -150.0,
                "vega_max": 50.0,
            },
            "CRASH/CRISIS": {
                "delta_min": -20.0,
                "delta_max": 10.0,
                "vega_min": 0.0,
                "vega_max": 500.0,
            },
            "NORMAL": {
                "delta_min": -50.0,
                "delta_max": 50.0,
                "vega_min": -200.0,
                "vega_max": 200.0,
            },
        }
    )


@dataclass
class HedgingConfig:
    enabled: bool = False
    delta_hedge_trigger: float = 50.0
    tail_risk_enabled: bool = False
    max_hedge_cost_pct: float = 1.0
    auto_execute: bool = False


@dataclass
class LLMStrategistConfig:
    enabled: bool = False
    provider: str = "google"
    model: str = "gemini-3.1-pro-thinking-preview"
    timeout_seconds: int = 20
    max_directives: int = 3


@dataclass
class CircuitBreakerConfig:
    strategy_loss_streak_limit: int = 5
    strategy_cooldown_hours: int = 24
    symbol_loss_streak_limit: int = 2
    symbol_blacklist_days: int = 7
    portfolio_drawdown_halt_pct: float = 5.0
    portfolio_halt_hours: int = 48
    api_error_rate_threshold: float = 0.30
    api_window_minutes: int = 10
    llm_timeout_streak: int = 3


@dataclass
class DegradationConfig:
    enabled: bool = True
    fallback_watchlist_on_scanner_failure: bool = True
    fallback_polling_on_stream_failure: bool = True
    continue_on_strategy_errors: bool = True
    rule_only_on_llm_failures: bool = True


@dataclass
class StranglesConfig:
    enabled: bool = False
    min_iv_rank: float = 70.0
    short_delta: float = 0.16
    min_account_balance: float = 25000.0
    allow_straddles_on: list = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])
    min_dte: int = 20
    max_dte: int = 45
    profit_target_pct: float = 0.50
    stop_loss_multiple: float = 2.0


@dataclass
class BrokenWingButterflyConfig:
    enabled: bool = False
    min_dte: int = 20
    max_dte: int = 45
    short_delta: float = 0.30
    near_wing_width: float = 5.0
    far_wing_width: float = 10.0
    min_credit: float = 0.10


@dataclass
class EarningsVolCrushConfig:
    enabled: bool = False
    min_dte: int = 1
    max_dte: int = 3
    max_position_risk_pct: float = 0.5
    min_iv_rank: float = 75.0
    wing_width: float = 10.0
    earnings_moves_file: str = "bot/data/earnings_moves.json"


def _default_risk_profiles() -> dict[str, RiskConfig]:
    """Default named risk profiles used for multi-account and runtime overlays."""
    return {
        "conservative": RiskConfig(
            max_portfolio_risk_pct=3.0,
            max_position_risk_pct=1.0,
            max_open_positions=5,
            max_daily_loss_pct=2.0,
        ),
        "moderate": RiskConfig(
            max_portfolio_risk_pct=5.0,
            max_position_risk_pct=2.0,
            max_open_positions=10,
            max_daily_loss_pct=3.0,
        ),
        "aggressive": RiskConfig(
            max_portfolio_risk_pct=8.0,
            max_position_risk_pct=3.0,
            max_open_positions=15,
            max_daily_loss_pct=5.0,
        ),
    }


@dataclass
class BotConfig:
    trading_mode: str = "paper"
    schwab: SchwabConfig = field(default_factory=SchwabConfig)
    credit_spreads: CreditSpreadConfig = field(default_factory=CreditSpreadConfig)
    iron_condors: IronCondorConfig = field(default_factory=IronCondorConfig)
    covered_calls: CoveredCallConfig = field(default_factory=CoveredCallConfig)
    naked_puts: NakedPutConfig = field(default_factory=NakedPutConfig)
    calendar_spreads: CalendarSpreadConfig = field(default_factory=CalendarSpreadConfig)
    strangles: StranglesConfig = field(default_factory=StranglesConfig)
    broken_wing_butterfly: BrokenWingButterflyConfig = field(
        default_factory=BrokenWingButterflyConfig
    )
    earnings_vol_crush: EarningsVolCrushConfig = field(
        default_factory=EarningsVolCrushConfig
    )
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    watchlist: list = field(
        default_factory=lambda: ["SPY", "QQQ", "IWM", "AAPL", "MSFT"]
    )
    risk: RiskConfig = field(default_factory=RiskConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    strategy_allocation: StrategyAllocationConfig = field(
        default_factory=StrategyAllocationConfig
    )
    greeks_budget: GreeksBudgetConfig = field(default_factory=GreeksBudgetConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    vol_surface: VolSurfaceConfig = field(default_factory=VolSurfaceConfig)
    econ_calendar: EconCalendarConfig = field(default_factory=EconCalendarConfig)
    options_flow: FlowConfig = field(default_factory=FlowConfig)
    rolling: RollingConfig = field(default_factory=RollingConfig)
    exits: ExitsConfig = field(default_factory=ExitsConfig)
    adjustments: AdjustmentsConfig = field(default_factory=AdjustmentsConfig)
    hedging: HedgingConfig = field(default_factory=HedgingConfig)
    llm_strategist: LLMStrategistConfig = field(default_factory=LLMStrategistConfig)
    circuit_breakers: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    signal_ranking: SignalRankingConfig = field(default_factory=SignalRankingConfig)
    ml_scorer: MLScorerConfig = field(default_factory=MLScorerConfig)
    theta_harvest: ThetaHarvestConfig = field(default_factory=ThetaHarvestConfig)
    correlation_monitor: CorrelationMonitorConfig = field(
        default_factory=CorrelationMonitorConfig
    )
    execution_timing: ExecutionTimingConfig = field(
        default_factory=ExecutionTimingConfig
    )
    rl_prompt_optimizer: RLPromptOptimizerConfig = field(
        default_factory=RLPromptOptimizerConfig
    )
    alt_data: AltDataConfig = field(default_factory=AltDataConfig)
    execution_algos: ExecutionAlgoConfig = field(default_factory=ExecutionAlgoConfig)
    strategy_sandbox: StrategySandboxConfig = field(
        default_factory=StrategySandboxConfig
    )
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    reconciliation: ReconciliationConfig = field(default_factory=ReconciliationConfig)
    multi_timeframe: MultiTimeframeConfig = field(default_factory=MultiTimeframeConfig)
    cooldown: CooldownConfig = field(default_factory=CooldownConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    terminal_ui: TerminalUIConfig = field(default_factory=TerminalUIConfig)
    max_signals_per_symbol_per_strategy: int = 2
    risk_profiles: dict[str, RiskConfig] = field(default_factory=_default_risk_profiles)
    log_level: str = "INFO"
    log_file: str = "logs/tradingbot.log"
    log_max_bytes: int = 10_485_760
    log_backup_count: int = 5


@dataclass
class ConfigValidationReport:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.failed) == 0


def load_config(config_path: str = "config.yaml") -> BotConfig:
    """Load configuration from YAML file and environment variables."""
    cfg = BotConfig()
    config_base_dir = Path(__file__).resolve().parents[1]

    # Load YAML config
    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        # Try CWD first, fall back to package root (so it works from any directory)
        if not config_file.exists():
            fallback = config_base_dir / config_file
            if fallback.exists():
                config_file = fallback
    if config_file.exists():
        with open(config_file) as f:
            raw = yaml.safe_load(f)
        if raw:
            _apply_yaml(cfg, raw)
        config_base_dir = config_file.resolve().parent

    # Load .env from config directory (not CWD) so credentials are found
    # regardless of which directory the bot is launched from.
    for env_name in (".env.local", ".env"):
        env_path = config_base_dir / env_name
        try:
            if env_path.exists():
                load_dotenv(env_path, override=True)
                break
        except PermissionError:
            logger.warning(f"Permission denied reading {env_path}, skipping.")
            continue
    else:
        try:
            load_dotenv()  # Fall back to CWD-based search
        except PermissionError:
            logger.warning("Permission denied reading CWD .env, skipping.")

    # Override with environment variables (credentials always from env)
    cfg.schwab.app_key = os.getenv("SCHWAB_APP_KEY", cfg.schwab.app_key)
    cfg.schwab.app_secret = os.getenv("SCHWAB_APP_SECRET", cfg.schwab.app_secret)
    cfg.schwab.callback_url = os.getenv("SCHWAB_CALLBACK_URL", cfg.schwab.callback_url)
    cfg.schwab.token_path = os.getenv("SCHWAB_TOKEN_PATH", cfg.schwab.token_path)
    token_path = Path(cfg.schwab.token_path).expanduser()
    if not token_path.is_absolute():
        token_path = (config_base_dir / token_path).resolve()
    cfg.schwab.token_path = str(token_path)
    cfg.schwab.account_hash = os.getenv("SCHWAB_ACCOUNT_HASH", cfg.schwab.account_hash)
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
    cfg.llm.reasoning_effort = os.getenv(
        "LLM_REASONING_EFFORT",
        cfg.llm.reasoning_effort,
    )
    cfg.llm.text_verbosity = os.getenv(
        "LLM_TEXT_VERBOSITY",
        cfg.llm.text_verbosity,
    )
    cfg.llm.max_output_tokens = _env_int(
        "LLM_MAX_OUTPUT_TOKENS",
        cfg.llm.max_output_tokens,
        minimum=64,
    )
    cfg.llm.chat_fallback_model = os.getenv(
        "LLM_CHAT_FALLBACK_MODEL",
        cfg.llm.chat_fallback_model,
    )
    cfg.llm.ensemble_enabled = _env_bool(
        "LLM_ENSEMBLE_ENABLED",
        cfg.llm.ensemble_enabled,
    )
    env_ensemble_models = os.getenv("LLM_ENSEMBLE_MODELS")
    if env_ensemble_models:
        cfg.llm.ensemble_models = [
            item.strip() for item in env_ensemble_models.split(",") if item.strip()
        ]
    cfg.llm.ensemble_agreement_threshold = _env_float(
        "LLM_ENSEMBLE_AGREEMENT_THRESHOLD",
        cfg.llm.ensemble_agreement_threshold,
        minimum=0.0,
        maximum=1.0,
    )
    cfg.llm.multi_turn_enabled = _env_bool(
        "LLM_MULTI_TURN_ENABLED",
        cfg.llm.multi_turn_enabled,
    )
    cfg.llm.multi_turn_confidence_threshold = _env_float(
        "LLM_MULTI_TURN_CONFIDENCE_THRESHOLD",
        cfg.llm.multi_turn_confidence_threshold,
        minimum=0.0,
        maximum=100.0,
    )
    cfg.llm.adversarial_review_enabled = _env_bool(
        "LLM_ADVERSARIAL_REVIEW_ENABLED",
        cfg.llm.adversarial_review_enabled,
    )
    cfg.llm.adversarial_loss_threshold_pct = _env_float(
        "LLM_ADVERSARIAL_LOSS_THRESHOLD_PCT",
        cfg.llm.adversarial_loss_threshold_pct,
        minimum=0.0,
        maximum=1.0,
    )
    cfg.llm.journal_enabled = _env_bool(
        "LLM_JOURNAL_ENABLED",
        cfg.llm.journal_enabled,
    )
    cfg.llm.journal_file = os.getenv("LLM_JOURNAL_FILE", cfg.llm.journal_file)
    cfg.llm.journal_context_entries = _env_int(
        "LLM_JOURNAL_CONTEXT_ENTRIES",
        cfg.llm.journal_context_entries,
        minimum=1,
    )
    cfg.llm.explanations_file = os.getenv(
        "LLM_EXPLANATIONS_FILE",
        cfg.llm.explanations_file,
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
    cfg.news.llm_model = os.getenv(
        "NEWS_LLM_MODEL",
        cfg.news.llm_model,
    )
    cfg.news.llm_reasoning_effort = os.getenv(
        "NEWS_LLM_REASONING_EFFORT",
        cfg.news.llm_reasoning_effort,
    )
    cfg.news.llm_text_verbosity = os.getenv(
        "NEWS_LLM_TEXT_VERBOSITY",
        cfg.news.llm_text_verbosity,
    )
    cfg.news.llm_max_output_tokens = _env_int(
        "NEWS_LLM_MAX_OUTPUT_TOKENS",
        cfg.news.llm_max_output_tokens,
        minimum=64,
    )
    cfg.news.llm_chat_fallback_model = os.getenv(
        "NEWS_LLM_CHAT_FALLBACK_MODEL",
        cfg.news.llm_chat_fallback_model,
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
    cfg.rl_prompt_optimizer.enabled = _env_bool(
        "RL_PROMPT_OPTIMIZER_ENABLED",
        cfg.rl_prompt_optimizer.enabled,
    )
    cfg.rl_prompt_optimizer.min_trades_for_pattern = _env_int(
        "RL_PROMPT_OPTIMIZER_MIN_TRADES_FOR_PATTERN",
        cfg.rl_prompt_optimizer.min_trades_for_pattern,
        minimum=1,
    )
    cfg.rl_prompt_optimizer.loss_rate_threshold = _env_float(
        "RL_PROMPT_OPTIMIZER_LOSS_RATE_THRESHOLD",
        cfg.rl_prompt_optimizer.loss_rate_threshold,
        minimum=0.0,
        maximum=1.0,
    )
    cfg.rl_prompt_optimizer.max_rules = _env_int(
        "RL_PROMPT_OPTIMIZER_MAX_RULES",
        cfg.rl_prompt_optimizer.max_rules,
        minimum=1,
    )
    cfg.rl_prompt_optimizer.rolling_window_size = _env_int(
        "RL_PROMPT_OPTIMIZER_ROLLING_WINDOW_SIZE",
        cfg.rl_prompt_optimizer.rolling_window_size,
        minimum=5,
    )
    cfg.alt_data.gex_enabled = _env_bool(
        "ALT_DATA_GEX_ENABLED",
        cfg.alt_data.gex_enabled,
    )
    cfg.alt_data.dark_pool_proxy_enabled = _env_bool(
        "ALT_DATA_DARK_POOL_PROXY_ENABLED",
        cfg.alt_data.dark_pool_proxy_enabled,
    )
    cfg.alt_data.social_sentiment_enabled = _env_bool(
        "ALT_DATA_SOCIAL_SENTIMENT_ENABLED",
        cfg.alt_data.social_sentiment_enabled,
    )
    cfg.alt_data.social_sentiment_cache_minutes = _env_int(
        "ALT_DATA_SOCIAL_SENTIMENT_CACHE_MINUTES",
        cfg.alt_data.social_sentiment_cache_minutes,
        minimum=1,
    )
    cfg.alt_data.social_sentiment_model = os.getenv(
        "ALT_DATA_SOCIAL_SENTIMENT_MODEL",
        cfg.alt_data.social_sentiment_model,
    )
    cfg.execution_algos.enabled = _env_bool(
        "EXECUTION_ALGOS_ENABLED",
        cfg.execution_algos.enabled,
    )
    cfg.execution_algos.algo_type = os.getenv(
        "EXECUTION_ALGO_TYPE",
        cfg.execution_algos.algo_type,
    )
    cfg.execution_algos.twap_slices = _env_int(
        "EXECUTION_ALGO_TWAP_SLICES",
        cfg.execution_algos.twap_slices,
        minimum=1,
    )
    cfg.execution_algos.twap_window_seconds = _env_int(
        "EXECUTION_ALGO_TWAP_WINDOW_SECONDS",
        cfg.execution_algos.twap_window_seconds,
        minimum=10,
    )
    cfg.execution_algos.iceberg_visible_qty = _env_int(
        "EXECUTION_ALGO_ICEBERG_VISIBLE_QTY",
        cfg.execution_algos.iceberg_visible_qty,
        minimum=1,
    )
    cfg.execution_algos.adaptive_spread_pause_threshold = _env_float(
        "EXECUTION_ALGO_ADAPTIVE_PAUSE_THRESHOLD",
        cfg.execution_algos.adaptive_spread_pause_threshold,
        minimum=1.0,
    )
    cfg.execution_algos.adaptive_spread_accelerate_threshold = _env_float(
        "EXECUTION_ALGO_ADAPTIVE_ACCELERATE_THRESHOLD",
        cfg.execution_algos.adaptive_spread_accelerate_threshold,
        minimum=0.1,
        maximum=1.0,
    )
    cfg.strategy_sandbox.enabled = _env_bool(
        "STRATEGY_SANDBOX_ENABLED",
        cfg.strategy_sandbox.enabled,
    )
    cfg.strategy_sandbox.min_failing_score = _env_float(
        "STRATEGY_SANDBOX_MIN_FAILING_SCORE",
        cfg.strategy_sandbox.min_failing_score,
        minimum=0.0,
        maximum=100.0,
    )
    cfg.strategy_sandbox.consecutive_fail_cycles = _env_int(
        "STRATEGY_SANDBOX_CONSECUTIVE_FAIL_CYCLES",
        cfg.strategy_sandbox.consecutive_fail_cycles,
        minimum=1,
    )
    cfg.strategy_sandbox.backtest_days = _env_int(
        "STRATEGY_SANDBOX_BACKTEST_DAYS",
        cfg.strategy_sandbox.backtest_days,
        minimum=5,
    )
    cfg.strategy_sandbox.min_sharpe = _env_float(
        "STRATEGY_SANDBOX_MIN_SHARPE",
        cfg.strategy_sandbox.min_sharpe,
    )
    cfg.strategy_sandbox.max_drawdown_pct = _env_float(
        "STRATEGY_SANDBOX_MAX_DRAWDOWN_PCT",
        cfg.strategy_sandbox.max_drawdown_pct,
        minimum=1.0,
    )
    cfg.strategy_sandbox.deployment_days = _env_int(
        "STRATEGY_SANDBOX_DEPLOYMENT_DAYS",
        cfg.strategy_sandbox.deployment_days,
        minimum=1,
    )
    cfg.strategy_sandbox.sizing_scalar = _env_float(
        "STRATEGY_SANDBOX_SIZING_SCALAR",
        cfg.strategy_sandbox.sizing_scalar,
        minimum=0.1,
        maximum=1.0,
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
    cfg.alerts.trade_notifications = _env_bool(
        "ALERTS_TRADE_NOTIFICATIONS",
        cfg.alerts.trade_notifications,
    )
    cfg.alerts.daily_summary = _env_bool(
        "ALERTS_DAILY_SUMMARY",
        cfg.alerts.daily_summary,
    )
    cfg.alerts.regime_changes = _env_bool(
        "ALERTS_REGIME_CHANGES",
        cfg.alerts.regime_changes,
    )
    cfg.alerts.weekly_summary = _env_bool(
        "ALERTS_WEEKLY_SUMMARY",
        cfg.alerts.weekly_summary,
    )
    cfg.terminal_ui.enabled = _env_bool(
        "TERMINAL_UI_ENABLED",
        cfg.terminal_ui.enabled,
    )
    cfg.terminal_ui.refresh_rate = _env_float(
        "TERMINAL_UI_REFRESH_RATE",
        cfg.terminal_ui.refresh_rate,
        minimum=0.1,
        maximum=10.0,
    )
    cfg.terminal_ui.max_activity_events = _env_int(
        "TERMINAL_UI_MAX_ACTIVITY_EVENTS",
        cfg.terminal_ui.max_activity_events,
        minimum=10,
    )
    cfg.terminal_ui.show_rejected_trades = _env_bool(
        "TERMINAL_UI_SHOW_REJECTED_TRADES",
        cfg.terminal_ui.show_rejected_trades,
    )
    cfg.terminal_ui.compact_mode = _env_bool(
        "TERMINAL_UI_COMPACT_MODE",
        cfg.terminal_ui.compact_mode,
    )
    cfg.log_max_bytes = _env_int("LOG_MAX_BYTES", cfg.log_max_bytes, minimum=1024)
    cfg.log_backup_count = _env_int("LOG_BACKUP_COUNT", cfg.log_backup_count, minimum=1)

    _normalize_config(cfg)

    return cfg


def validate_config(cfg: BotConfig) -> ConfigValidationReport:
    """Validate cross-field config constraints before trading starts."""
    report = ConfigValidationReport()
    if not isinstance(cfg, BotConfig):
        report.failed.append("Config object is invalid.")
        return report

    if float(cfg.risk.max_portfolio_risk_pct) < float(cfg.risk.max_position_risk_pct):
        report.failed.append(
            "risk.max_portfolio_risk_pct must be >= risk.max_position_risk_pct."
        )
    else:
        report.passed.append("risk budget hierarchy")

    if str(cfg.sizing.method).strip().lower() == "kelly":
        kelly = float(cfg.sizing.kelly_fraction)
        if kelly < 0.10 or kelly > 1.0:
            report.failed.append(
                "sizing.kelly_fraction must be between 0.10 and 1.0 when sizing.method=kelly."
            )
        else:
            report.passed.append("kelly sizing bounds")
    else:
        report.passed.append("sizing method fixed")

    invalid_models: list[str] = []
    for model_ref in list(cfg.llm.ensemble_models or []):
        provider, sep, model = str(model_ref).partition(":")
        provider = provider.strip().lower()
        if provider == "gemini":
            provider = "google"
        model = model.strip()
        if (
            not sep
            or provider not in {"openai", "anthropic", "ollama", "google"}
            or not model
        ):
            invalid_models.append(str(model_ref))
    if invalid_models:
        report.failed.append(
            "llm.ensemble_models must use provider:model format with provider in "
            "{openai, anthropic, ollama, google}. Invalid: " + ", ".join(invalid_models)
        )
    else:
        report.passed.append("ensemble model format")

    if bool(cfg.hedging.enabled):
        hedge_budget = float(cfg.hedging.max_hedge_cost_pct)
        portfolio_budget = float(cfg.risk.max_portfolio_risk_pct)
        if hedge_budget > portfolio_budget:
            report.failed.append(
                "hedging.max_hedge_cost_pct cannot exceed risk.max_portfolio_risk_pct."
            )
        else:
            report.passed.append("hedging budget vs risk budget")
    else:
        report.passed.append("hedging disabled")

    overlap_pairs = _strategy_dte_overlaps(cfg)
    if overlap_pairs and int(cfg.risk.max_positions_per_symbol) <= 1:
        joined = ", ".join(overlap_pairs)
        report.warnings.append(
            "Overlapping strategy DTE windows detected but max_positions_per_symbol=1: "
            + joined
        )
    else:
        report.passed.append("strategy DTE overlap check")

    return report


def format_validation_report(report: ConfigValidationReport) -> str:
    """Render a human-readable validation report."""
    lines = [
        "Configuration validation report:",
        f"  Passed: {len(report.passed)}",
        f"  Warnings: {len(report.warnings)}",
        f"  Failed: {len(report.failed)}",
    ]
    if report.passed:
        lines.append("  Passed checks:")
        lines.extend(f"    - {item}" for item in report.passed)
    if report.warnings:
        lines.append("  Warnings:")
        lines.extend(f"    - {item}" for item in report.warnings)
    if report.failed:
        lines.append("  Failures:")
        lines.extend(f"    - {item}" for item in report.failed)
    return "\n".join(lines)


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

    np = strats.get("naked_puts", {})
    if np:
        for key, val in np.items():
            if hasattr(cfg.naked_puts, key):
                setattr(cfg.naked_puts, key, val)

    cal = strats.get("calendar_spreads", {})
    if cal:
        for key, val in cal.items():
            if hasattr(cfg.calendar_spreads, key):
                setattr(cfg.calendar_spreads, key, val)

    strangles = strats.get("strangles", {})
    if strangles:
        for key, val in strangles.items():
            if hasattr(cfg.strangles, key):
                setattr(cfg.strangles, key, val)

    bwb = strats.get("broken_wing_butterfly", {})
    if bwb:
        for key, val in bwb.items():
            if hasattr(cfg.broken_wing_butterfly, key):
                setattr(cfg.broken_wing_butterfly, key, val)

    crush = strats.get("earnings_vol_crush", {})
    if crush:
        for key, val in crush.items():
            if hasattr(cfg.earnings_vol_crush, key):
                setattr(cfg.earnings_vol_crush, key, val)
    sandbox = strats.get("sandbox", {})
    if isinstance(sandbox, dict):
        cfg.strategy_sandbox.active_strategy = dict(sandbox)

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

    sizing = raw.get("sizing", {})
    if sizing:
        for key, val in sizing.items():
            if hasattr(cfg.sizing, key):
                setattr(cfg.sizing, key, val)
    strategy_allocation = raw.get("strategy_allocation", {})
    if strategy_allocation:
        for key, val in strategy_allocation.items():
            if hasattr(cfg.strategy_allocation, key):
                setattr(cfg.strategy_allocation, key, val)
    greeks_budget = raw.get("greeks_budget", {})
    if greeks_budget:
        for key, val in greeks_budget.items():
            if hasattr(cfg.greeks_budget, key):
                setattr(cfg.greeks_budget, key, val)

    regime = raw.get("regime", {})
    if regime:
        for key, val in regime.items():
            if hasattr(cfg.regime, key):
                setattr(cfg.regime, key, val)

    vol_surface = raw.get("vol_surface", {})
    if vol_surface:
        for key, val in vol_surface.items():
            if hasattr(cfg.vol_surface, key):
                setattr(cfg.vol_surface, key, val)

    econ = raw.get("econ_calendar", {})
    if econ:
        for key, val in econ.items():
            if hasattr(cfg.econ_calendar, key):
                setattr(cfg.econ_calendar, key, val)

    flow = raw.get("options_flow", {})
    if flow:
        for key, val in flow.items():
            if hasattr(cfg.options_flow, key):
                setattr(cfg.options_flow, key, val)

    rolling = raw.get("rolling", {})
    if rolling:
        for key, val in rolling.items():
            if hasattr(cfg.rolling, key):
                setattr(cfg.rolling, key, val)

    exits = raw.get("exits", {})
    if exits:
        for key, val in exits.items():
            if hasattr(cfg.exits, key):
                setattr(cfg.exits, key, val)

    adjustments = raw.get("adjustments", {})
    if adjustments:
        for key, val in adjustments.items():
            if hasattr(cfg.adjustments, key):
                setattr(cfg.adjustments, key, val)

    hedging = raw.get("hedging", {})
    if hedging:
        for key, val in hedging.items():
            if hasattr(cfg.hedging, key):
                setattr(cfg.hedging, key, val)

    strategist = raw.get("llm_strategist", {})
    if strategist:
        for key, val in strategist.items():
            if hasattr(cfg.llm_strategist, key):
                setattr(cfg.llm_strategist, key, val)

    breakers = raw.get("circuit_breakers", {})
    if breakers:
        for key, val in breakers.items():
            if hasattr(cfg.circuit_breakers, key):
                setattr(cfg.circuit_breakers, key, val)

    degradation = raw.get("degradation", {})
    if degradation:
        for key, val in degradation.items():
            if hasattr(cfg.degradation, key):
                setattr(cfg.degradation, key, val)

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

    signal_ranking = raw.get("signal_ranking", {})
    if signal_ranking:
        for key, val in signal_ranking.items():
            if hasattr(cfg.signal_ranking, key):
                setattr(cfg.signal_ranking, key, val)
    ml_scorer = raw.get("ml_scorer", {})
    if ml_scorer:
        for key, val in ml_scorer.items():
            if hasattr(cfg.ml_scorer, key):
                setattr(cfg.ml_scorer, key, val)
    theta_harvest = raw.get("theta_harvest", {})
    if theta_harvest:
        for key, val in theta_harvest.items():
            if hasattr(cfg.theta_harvest, key):
                setattr(cfg.theta_harvest, key, val)
    correlation_monitor = raw.get("correlation_monitor", {})
    if correlation_monitor:
        for key, val in correlation_monitor.items():
            if hasattr(cfg.correlation_monitor, key):
                setattr(cfg.correlation_monitor, key, val)
    execution_timing = raw.get("execution_timing", {})
    if execution_timing:
        for key, val in execution_timing.items():
            if hasattr(cfg.execution_timing, key):
                setattr(cfg.execution_timing, key, val)
    rl_prompt_optimizer = raw.get("rl_prompt_optimizer", {})
    if rl_prompt_optimizer:
        for key, val in rl_prompt_optimizer.items():
            if hasattr(cfg.rl_prompt_optimizer, key):
                setattr(cfg.rl_prompt_optimizer, key, val)
    alt_data = raw.get("alt_data", {})
    if alt_data:
        for key, val in alt_data.items():
            if hasattr(cfg.alt_data, key):
                setattr(cfg.alt_data, key, val)
    execution_algos = raw.get("execution_algos", {})
    if execution_algos:
        for key, val in execution_algos.items():
            if hasattr(cfg.execution_algos, key):
                setattr(cfg.execution_algos, key, val)
    strategy_sandbox = raw.get("strategy_sandbox", {})
    if strategy_sandbox:
        for key, val in strategy_sandbox.items():
            if hasattr(cfg.strategy_sandbox, key):
                setattr(cfg.strategy_sandbox, key, val)
    scaling = raw.get("scaling", {})
    if scaling:
        for key, val in scaling.items():
            if hasattr(cfg.scaling, key):
                setattr(cfg.scaling, key, val)
    walkforward = raw.get("walkforward", {})
    if walkforward:
        for key, val in walkforward.items():
            if hasattr(cfg.walkforward, key):
                setattr(cfg.walkforward, key, val)
    monte_carlo = raw.get("monte_carlo", {})
    if monte_carlo:
        for key, val in monte_carlo.items():
            if hasattr(cfg.monte_carlo, key):
                setattr(cfg.monte_carlo, key, val)
    reconciliation = raw.get("reconciliation", {})
    if reconciliation:
        for key, val in reconciliation.items():
            if hasattr(cfg.reconciliation, key):
                setattr(cfg.reconciliation, key, val)
    multi_timeframe = raw.get("multi_timeframe", {})
    if multi_timeframe:
        for key, val in multi_timeframe.items():
            if hasattr(cfg.multi_timeframe, key):
                setattr(cfg.multi_timeframe, key, val)
    cooldown = raw.get("cooldown", {})
    if cooldown:
        for key, val in cooldown.items():
            if hasattr(cfg.cooldown, key):
                setattr(cfg.cooldown, key, val)

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

    terminal_ui = raw.get("terminal_ui", {})
    if terminal_ui:
        for key, val in terminal_ui.items():
            if hasattr(cfg.terminal_ui, key):
                setattr(cfg.terminal_ui, key, val)

    if "max_signals_per_symbol_per_strategy" in raw:
        cfg.max_signals_per_symbol_per_strategy = raw.get(
            "max_signals_per_symbol_per_strategy",
            cfg.max_signals_per_symbol_per_strategy,
        )

    risk_profiles = raw.get("risk_profiles", {})
    if isinstance(risk_profiles, dict) and risk_profiles:
        normalized: dict[str, RiskConfig] = {}
        for name, values in risk_profiles.items():
            if not isinstance(values, dict):
                continue
            base = RiskConfig()
            for key, val in values.items():
                if hasattr(base, key):
                    setattr(base, key, val)
            normalized[str(name).strip().lower()] = base
        if normalized:
            cfg.risk_profiles = normalized

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
        logger.warning(
            "Invalid integer for %s=%r. Using default %r.", name, value, default
        )
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
        logger.warning(
            "Invalid float for %s=%r. Using default %r.", name, value, default
        )
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
        allowed={"ollama", "openai", "anthropic", "google", "gemini"},
        default="google",
        field_name="llm.provider",
    )
    if cfg.llm.provider == "gemini":
        cfg.llm.provider = "google"
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
    cfg.llm.reasoning_effort = _normalize_choice(
        cfg.llm.reasoning_effort,
        allowed={"none", "low", "medium", "high", "xhigh"},
        default="xhigh",
        field_name="llm.reasoning_effort",
    )
    cfg.llm.text_verbosity = _normalize_choice(
        cfg.llm.text_verbosity,
        allowed={"low", "medium", "high"},
        default="low",
        field_name="llm.text_verbosity",
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
    cfg.execution.stale_order_minutes = max(1, int(cfg.execution.stale_order_minutes))
    cfg.execution.cancel_stale_orders = bool(cfg.execution.cancel_stale_orders)
    cfg.execution.include_partials_in_ledger = bool(
        cfg.execution.include_partials_in_ledger
    )
    cfg.execution.partial_fill_handling = bool(cfg.execution.partial_fill_handling)
    cfg.execution.market_move_cancel_pct = max(
        0.0, float(cfg.execution.market_move_cancel_pct)
    )
    cfg.execution.block_first_minutes = max(0, int(cfg.execution.block_first_minutes))
    cfg.execution.block_last_minutes = max(0, int(cfg.execution.block_last_minutes))
    cfg.scanner.request_pause_seconds = max(
        0.0, float(cfg.scanner.request_pause_seconds)
    )
    cfg.scanner.error_backoff_seconds = max(
        0.1, float(cfg.scanner.error_backoff_seconds)
    )
    cfg.scanner.max_retry_attempts = max(0, int(cfg.scanner.max_retry_attempts))
    cfg.scanner.max_consecutive_errors = max(1, int(cfg.scanner.max_consecutive_errors))
    cfg.scanner.max_symbols_per_scan = max(0, int(cfg.scanner.max_symbols_per_scan))
    cfg.scanner.blacklist = _normalize_symbol_list(cfg.scanner.blacklist, default=[])
    cfg.scanner.movers_weight = max(0.0, min(1.0, float(cfg.scanner.movers_weight)))
    cfg.scanner.sector_rotation_weight = max(
        0.0, min(1.0, float(cfg.scanner.sector_rotation_weight))
    )
    cfg.scanner.relative_strength_weight = max(
        0.0, min(1.0, float(cfg.scanner.relative_strength_weight))
    )
    cfg.scanner.options_liquidity_weight = max(
        0.0, min(1.0, float(cfg.scanner.options_liquidity_weight))
    )
    cfg.llm.timeout_seconds = max(1, int(cfg.llm.timeout_seconds))
    cfg.llm.temperature = max(0.0, min(2.0, float(cfg.llm.temperature)))
    cfg.llm.min_confidence = max(0.0, min(1.0, float(cfg.llm.min_confidence)))
    cfg.llm.max_output_tokens = max(64, int(cfg.llm.max_output_tokens))
    cfg.llm.track_record_file = str(
        cfg.llm.track_record_file or "bot/data/llm_track_record.json"
    )
    llm_chat_fallback = str(cfg.llm.chat_fallback_model or "").strip()
    if llm_chat_fallback:
        cfg.llm.chat_fallback_model = llm_chat_fallback
    elif cfg.llm.provider == "google":
        cfg.llm.chat_fallback_model = "gemini-3.1-flash-thinking-preview"
    elif cfg.llm.provider == "openai":
        cfg.llm.chat_fallback_model = "gpt-4.1"
    else:
        cfg.llm.chat_fallback_model = ""
    cfg.llm.ensemble_enabled = bool(cfg.llm.ensemble_enabled)
    cfg.llm.ensemble_agreement_threshold = max(
        0.0,
        min(1.0, float(cfg.llm.ensemble_agreement_threshold)),
    )
    cfg.llm.ensemble_models = _normalize_string_list(
        cfg.llm.ensemble_models,
        default=["google:gemini-3.1-pro-thinking-preview"],
    )
    cfg.llm.multi_turn_enabled = bool(cfg.llm.multi_turn_enabled)
    cfg.llm.multi_turn_confidence_threshold = max(
        50.0,
        min(100.0, float(cfg.llm.multi_turn_confidence_threshold)),
    )
    cfg.llm.adversarial_review_enabled = bool(cfg.llm.adversarial_review_enabled)
    cfg.llm.adversarial_loss_threshold_pct = max(
        0.0,
        min(1.0, float(cfg.llm.adversarial_loss_threshold_pct)),
    )
    cfg.llm.journal_enabled = bool(cfg.llm.journal_enabled)
    cfg.llm.journal_file = str(cfg.llm.journal_file or "bot/data/trade_journal.json")
    cfg.llm.journal_context_entries = max(1, int(cfg.llm.journal_context_entries))
    cfg.llm.explanations_file = str(
        cfg.llm.explanations_file or "bot/data/trade_explanations.json"
    )
    if cfg.llm.provider == "anthropic":
        model_key = str(cfg.llm.model).strip().lower()
        if (
            not model_key
            or model_key.startswith("gpt-")
            or model_key.startswith("gemini-")
        ):
            cfg.llm.model = "claude-sonnet-4-20250514"
    elif cfg.llm.provider == "openai":
        model_key = str(cfg.llm.model).strip().lower()
        if (
            not model_key
            or model_key.startswith("gemini-")
            or model_key.startswith("claude-")
        ):
            cfg.llm.model = "gpt-4.1"
    elif cfg.llm.provider == "google":
        model_key = str(cfg.llm.model).strip().lower()
        if (
            not model_key
            or model_key.startswith("gpt-")
            or model_key.startswith("claude-")
        ):
            cfg.llm.model = "gemini-3.1-pro-thinking-preview"

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
    cfg.news.llm_sentiment_cache_seconds = max(
        0, int(cfg.news.llm_sentiment_cache_seconds)
    )
    cfg.news.llm_model = str(cfg.news.llm_model or "gemini-3.1-pro-thinking-preview").strip()
    cfg.news.llm_reasoning_effort = _normalize_choice(
        cfg.news.llm_reasoning_effort,
        allowed={"none", "low", "medium", "high", "xhigh"},
        default="xhigh",
        field_name="news.llm_reasoning_effort",
    )
    cfg.news.llm_text_verbosity = _normalize_choice(
        cfg.news.llm_text_verbosity,
        allowed={"low", "medium", "high"},
        default="low",
        field_name="news.llm_text_verbosity",
    )
    cfg.news.llm_max_output_tokens = max(64, int(cfg.news.llm_max_output_tokens))
    news_chat_fallback = str(cfg.news.llm_chat_fallback_model or "").strip()
    cfg.news.llm_chat_fallback_model = news_chat_fallback or "gemini-3.1-flash-thinking-preview"
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
    cfg.alerts.trade_notifications = bool(cfg.alerts.trade_notifications)
    cfg.alerts.daily_summary = bool(cfg.alerts.daily_summary)
    cfg.alerts.regime_changes = bool(cfg.alerts.regime_changes)
    cfg.alerts.weekly_summary = bool(cfg.alerts.weekly_summary)
    cfg.alerts.webhook_format = _normalize_choice(
        cfg.alerts.webhook_format,
        allowed={"generic", "slack", "discord"},
        default="generic",
        field_name="alerts.webhook_format",
    )
    cfg.alerts.drawdown_thresholds = _normalize_positive_float_list(
        cfg.alerts.drawdown_thresholds,
        default=[1.0, 2.0, 3.0],
    )
    cfg.terminal_ui.enabled = bool(cfg.terminal_ui.enabled)
    cfg.terminal_ui.refresh_rate = max(
        0.1,
        min(10.0, float(cfg.terminal_ui.refresh_rate)),
    )
    cfg.terminal_ui.max_activity_events = max(
        10,
        int(cfg.terminal_ui.max_activity_events),
    )
    cfg.terminal_ui.show_rejected_trades = bool(cfg.terminal_ui.show_rejected_trades)
    cfg.terminal_ui.compact_mode = bool(cfg.terminal_ui.compact_mode)
    cfg.log_max_bytes = max(1024, int(cfg.log_max_bytes))
    cfg.log_backup_count = max(1, int(cfg.log_backup_count))
    cfg.execution.entry_step_timeout_seconds = max(
        10, int(cfg.execution.entry_step_timeout_seconds)
    )
    cfg.execution.exit_step_timeout_seconds = max(
        10, int(cfg.execution.exit_step_timeout_seconds)
    )
    cfg.execution.max_ladder_attempts = max(1, int(cfg.execution.max_ladder_attempts))
    cfg.execution.smart_ladder_enabled = bool(cfg.execution.smart_ladder_enabled)
    cfg.execution.ladder_width_fractions = _normalize_float_list(
        cfg.execution.ladder_width_fractions,
        default=[0.0, 0.10, 0.25, 0.40],
    )
    cfg.execution.ladder_step_timeouts_seconds = _normalize_int_list(
        cfg.execution.ladder_step_timeouts_seconds,
        default=[45, 45, 45, 30],
        minimum=5,
    )
    cfg.execution.entry_ladder_shifts = _normalize_float_list(
        cfg.execution.entry_ladder_shifts,
        default=[0.0, 0.25, 0.50],
    )
    cfg.execution.exit_ladder_shifts = _normalize_float_list(
        cfg.execution.exit_ladder_shifts,
        default=[0.0, 0.25, 0.50],
    )
    cfg.signal_ranking.enabled = bool(cfg.signal_ranking.enabled)
    cfg.signal_ranking.weight_score = max(
        0.0, min(1.0, float(cfg.signal_ranking.weight_score))
    )
    cfg.signal_ranking.weight_pop = max(
        0.0, min(1.0, float(cfg.signal_ranking.weight_pop))
    )
    cfg.signal_ranking.weight_credit = max(
        0.0, min(1.0, float(cfg.signal_ranking.weight_credit))
    )
    cfg.signal_ranking.weight_vol_premium = max(
        0.0, min(1.0, float(cfg.signal_ranking.weight_vol_premium))
    )
    cfg.signal_ranking.ml_score_weight = max(
        0.0, min(1.0, float(cfg.signal_ranking.ml_score_weight))
    )
    cfg.signal_ranking.top_ranked_to_log = max(
        1, int(cfg.signal_ranking.top_ranked_to_log)
    )
    total_rank_weight = (
        cfg.signal_ranking.weight_score
        + cfg.signal_ranking.weight_pop
        + cfg.signal_ranking.weight_credit
        + cfg.signal_ranking.weight_vol_premium
        + cfg.signal_ranking.ml_score_weight
    )
    if total_rank_weight <= 0:
        cfg.signal_ranking.weight_score = 0.30
        cfg.signal_ranking.weight_pop = 0.25
        cfg.signal_ranking.weight_credit = 0.10
        cfg.signal_ranking.weight_vol_premium = 0.10
        cfg.signal_ranking.ml_score_weight = 0.25
    cfg.ml_scorer.enabled = bool(cfg.ml_scorer.enabled)
    cfg.ml_scorer.min_training_trades = max(1, int(cfg.ml_scorer.min_training_trades))
    cfg.ml_scorer.retrain_day = _normalize_choice(
        cfg.ml_scorer.retrain_day,
        allowed={
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        },
        default="sunday",
        field_name="ml_scorer.retrain_day",
    )
    cfg.ml_scorer.retrain_time = (
        str(cfg.ml_scorer.retrain_time or "18:00").strip() or "18:00"
    )
    cfg.ml_scorer.feature_importance_log = bool(cfg.ml_scorer.feature_importance_log)
    cfg.ml_scorer.closed_trades_file = str(
        cfg.ml_scorer.closed_trades_file or "bot/data/closed_trades.json"
    )
    cfg.ml_scorer.model_file = str(cfg.ml_scorer.model_file or "bot/data/ml_model.json")
    cfg.ml_scorer.feature_importance_file = str(
        cfg.ml_scorer.feature_importance_file or "bot/data/ml_feature_importance.json"
    )
    cfg.theta_harvest.enabled = bool(cfg.theta_harvest.enabled)
    cfg.theta_harvest.satisfied_pct = max(
        0.0, min(1.0, float(cfg.theta_harvest.satisfied_pct))
    )
    cfg.theta_harvest.underperform_pct = max(
        0.0, min(1.0, float(cfg.theta_harvest.underperform_pct))
    )
    cfg.theta_harvest.cutoff_hour = max(0, min(23, int(cfg.theta_harvest.cutoff_hour)))
    cfg.correlation_monitor.enabled = bool(cfg.correlation_monitor.enabled)
    cfg.correlation_monitor.lookback_days = max(
        10, int(cfg.correlation_monitor.lookback_days)
    )
    cfg.correlation_monitor.crisis_threshold = max(
        0.0, min(1.0, float(cfg.correlation_monitor.crisis_threshold))
    )
    cfg.correlation_monitor.stress_threshold = max(
        0.0, min(1.0, float(cfg.correlation_monitor.stress_threshold))
    )
    cfg.execution_timing.adaptive = bool(cfg.execution_timing.adaptive)
    cfg.execution_timing.min_fills_per_bucket = max(
        1, int(cfg.execution_timing.min_fills_per_bucket)
    )
    cfg.execution_timing.analysis_file = str(
        cfg.execution_timing.analysis_file or "bot/data/execution_timing_analysis.json"
    )
    cfg.rl_prompt_optimizer.enabled = bool(cfg.rl_prompt_optimizer.enabled)
    cfg.rl_prompt_optimizer.min_trades_for_pattern = max(
        3, int(cfg.rl_prompt_optimizer.min_trades_for_pattern)
    )
    cfg.rl_prompt_optimizer.loss_rate_threshold = max(
        0.5,
        min(0.99, float(cfg.rl_prompt_optimizer.loss_rate_threshold)),
    )
    cfg.rl_prompt_optimizer.max_rules = max(1, int(cfg.rl_prompt_optimizer.max_rules))
    cfg.rl_prompt_optimizer.rolling_window_size = max(
        20, int(cfg.rl_prompt_optimizer.rolling_window_size)
    )
    cfg.rl_prompt_optimizer.learned_rules_file = str(
        cfg.rl_prompt_optimizer.learned_rules_file or "bot/data/learned_rules.json"
    )
    cfg.rl_prompt_optimizer.audit_log_file = str(
        cfg.rl_prompt_optimizer.audit_log_file or "bot/data/audit_log.jsonl"
    )
    cfg.alt_data.gex_enabled = bool(cfg.alt_data.gex_enabled)
    cfg.alt_data.dark_pool_proxy_enabled = bool(cfg.alt_data.dark_pool_proxy_enabled)
    cfg.alt_data.social_sentiment_enabled = bool(cfg.alt_data.social_sentiment_enabled)
    cfg.alt_data.social_sentiment_cache_minutes = max(
        1,
        int(cfg.alt_data.social_sentiment_cache_minutes),
    )
    cfg.alt_data.social_sentiment_model = str(
        cfg.alt_data.social_sentiment_model or "gemini-3.1-pro-thinking-preview"
    ).strip()
    cfg.execution_algos.enabled = bool(cfg.execution_algos.enabled)
    cfg.execution_algos.algo_type = _normalize_choice(
        cfg.execution_algos.algo_type,
        allowed={"smart_ladder", "twap", "iceberg", "adaptive"},
        default="smart_ladder",
        field_name="execution_algos.algo_type",
    )
    cfg.execution_algos.twap_slices = max(1, int(cfg.execution_algos.twap_slices))
    cfg.execution_algos.twap_window_seconds = max(
        10, int(cfg.execution_algos.twap_window_seconds)
    )
    cfg.execution_algos.iceberg_visible_qty = max(
        1, int(cfg.execution_algos.iceberg_visible_qty)
    )
    cfg.execution_algos.adaptive_spread_pause_threshold = max(
        1.0,
        float(cfg.execution_algos.adaptive_spread_pause_threshold),
    )
    cfg.execution_algos.adaptive_spread_accelerate_threshold = max(
        0.1,
        min(1.0, float(cfg.execution_algos.adaptive_spread_accelerate_threshold)),
    )
    cfg.strategy_sandbox.enabled = bool(cfg.strategy_sandbox.enabled)
    cfg.strategy_sandbox.min_failing_score = max(
        0.0,
        min(100.0, float(cfg.strategy_sandbox.min_failing_score)),
    )
    cfg.strategy_sandbox.consecutive_fail_cycles = max(
        1,
        int(cfg.strategy_sandbox.consecutive_fail_cycles),
    )
    cfg.strategy_sandbox.backtest_days = max(5, int(cfg.strategy_sandbox.backtest_days))
    cfg.strategy_sandbox.min_sharpe = float(cfg.strategy_sandbox.min_sharpe)
    cfg.strategy_sandbox.max_drawdown_pct = max(
        1.0, float(cfg.strategy_sandbox.max_drawdown_pct)
    )
    cfg.strategy_sandbox.deployment_days = max(
        1, int(cfg.strategy_sandbox.deployment_days)
    )
    cfg.strategy_sandbox.sizing_scalar = max(
        0.1,
        min(1.0, float(cfg.strategy_sandbox.sizing_scalar)),
    )
    cfg.strategy_sandbox.state_file = str(
        cfg.strategy_sandbox.state_file or "bot/data/strategy_sandbox_state.json"
    )
    if not isinstance(cfg.strategy_sandbox.active_strategy, dict):
        cfg.strategy_sandbox.active_strategy = {}
    cfg.scaling.enabled = bool(cfg.scaling.enabled)
    cfg.scaling.scale_in_delay_minutes = max(1, int(cfg.scaling.scale_in_delay_minutes))
    cfg.scaling.scale_in_max_adds = max(0, int(cfg.scaling.scale_in_max_adds))
    cfg.scaling.partial_exit_pct = max(
        0.0, min(1.0, float(cfg.scaling.partial_exit_pct))
    )
    cfg.scaling.partial_exit_size = max(
        0.0, min(1.0, float(cfg.scaling.partial_exit_size))
    )
    cfg.walkforward.enabled = bool(cfg.walkforward.enabled)
    cfg.walkforward.train_days = max(10, int(cfg.walkforward.train_days))
    cfg.walkforward.test_days = max(5, int(cfg.walkforward.test_days))
    cfg.walkforward.step_days = max(1, int(cfg.walkforward.step_days))
    cfg.monte_carlo.enabled = bool(cfg.monte_carlo.enabled)
    cfg.monte_carlo.simulations = max(100, int(cfg.monte_carlo.simulations))
    cfg.monte_carlo.var_limit_pct = max(0.0, float(cfg.monte_carlo.var_limit_pct))
    cfg.reconciliation.interval_minutes = max(
        5, int(cfg.reconciliation.interval_minutes)
    )
    cfg.reconciliation.auto_import = bool(cfg.reconciliation.auto_import)
    cfg.multi_timeframe.enabled = bool(cfg.multi_timeframe.enabled)
    cfg.multi_timeframe.min_agreement = max(
        1, min(3, int(cfg.multi_timeframe.min_agreement))
    )
    cfg.cooldown.graduated = bool(cfg.cooldown.graduated)
    cfg.cooldown.level_1_losses = max(1, int(cfg.cooldown.level_1_losses))
    cfg.cooldown.level_1_reduction = max(
        0.0, min(0.95, float(cfg.cooldown.level_1_reduction))
    )
    cfg.cooldown.level_2_losses = max(
        cfg.cooldown.level_1_losses, int(cfg.cooldown.level_2_losses)
    )
    cfg.cooldown.level_2_reduction = max(
        cfg.cooldown.level_1_reduction,
        min(0.95, float(cfg.cooldown.level_2_reduction)),
    )
    cfg.max_signals_per_symbol_per_strategy = max(
        1, int(cfg.max_signals_per_symbol_per_strategy)
    )
    cfg.risk = _normalize_risk_config(cfg.risk)
    cfg.risk_profiles = _normalize_risk_profiles(cfg.risk_profiles)
    cfg.schwab.accounts = _normalize_accounts(cfg.schwab.accounts)
    cfg.sizing.method = _normalize_choice(
        cfg.sizing.method,
        allowed={"fixed", "kelly"},
        default="fixed",
        field_name="sizing.method",
    )
    cfg.sizing.kelly_fraction = max(0.0, min(1.0, float(cfg.sizing.kelly_fraction)))
    cfg.sizing.kelly_min_trades = max(1, int(cfg.sizing.kelly_min_trades))
    cfg.sizing.drawdown_decay_threshold = max(
        0.0, min(1.0, float(cfg.sizing.drawdown_decay_threshold))
    )
    cfg.sizing.equity_curve_scaling = bool(cfg.sizing.equity_curve_scaling)
    cfg.sizing.equity_curve_lookback = max(5, int(cfg.sizing.equity_curve_lookback))
    cfg.sizing.max_scale_up = max(1.0, float(cfg.sizing.max_scale_up))
    cfg.sizing.max_scale_down = max(0.1, min(1.0, float(cfg.sizing.max_scale_down)))
    cfg.strategy_allocation.enabled = bool(cfg.strategy_allocation.enabled)
    cfg.strategy_allocation.lookback_trades = max(
        5, int(cfg.strategy_allocation.lookback_trades)
    )
    cfg.strategy_allocation.min_sharpe_for_boost = float(
        cfg.strategy_allocation.min_sharpe_for_boost
    )
    cfg.strategy_allocation.cold_start_penalty = max(
        0.1,
        min(1.0, float(cfg.strategy_allocation.cold_start_penalty)),
    )
    cfg.strategy_allocation.cold_start_window_days = max(
        7, int(cfg.strategy_allocation.cold_start_window_days)
    )
    cfg.strategy_allocation.cold_start_min_trades = max(
        1, int(cfg.strategy_allocation.cold_start_min_trades)
    )
    cfg.greeks_budget.enabled = bool(cfg.greeks_budget.enabled)
    cfg.greeks_budget.reduce_size_to_fit = bool(cfg.greeks_budget.reduce_size_to_fit)
    cfg.greeks_budget.limits = _normalize_greeks_budget_limits(cfg.greeks_budget.limits)
    cfg.regime.min_confidence = max(0.0, min(1.0, float(cfg.regime.min_confidence)))
    cfg.regime.uncertain_size_scalar = max(
        0.1, min(2.0, float(cfg.regime.uncertain_size_scalar))
    )
    cfg.vol_surface.max_vol_of_vol_for_condors = max(
        0.0, float(cfg.vol_surface.max_vol_of_vol_for_condors)
    )
    cfg.econ_calendar.refresh_days = max(1, int(cfg.econ_calendar.refresh_days))
    cfg.econ_calendar.high_severity_policy = _normalize_choice(
        cfg.econ_calendar.high_severity_policy,
        allowed={"skip", "reduce_size", "widen", "none"},
        default="reduce_size",
        field_name="econ_calendar.high_severity_policy",
    )
    cfg.econ_calendar.medium_severity_policy = _normalize_choice(
        cfg.econ_calendar.medium_severity_policy,
        allowed={"skip", "reduce_size", "widen", "none"},
        default="widen",
        field_name="econ_calendar.medium_severity_policy",
    )
    cfg.econ_calendar.low_severity_policy = _normalize_choice(
        cfg.econ_calendar.low_severity_policy,
        allowed={"skip", "reduce_size", "widen", "none"},
        default="none",
        field_name="econ_calendar.low_severity_policy",
    )
    cfg.options_flow.unusual_volume_multiple = max(
        1.0, float(cfg.options_flow.unusual_volume_multiple)
    )
    cfg.rolling.min_dte_trigger = max(1, int(cfg.rolling.min_dte_trigger))
    cfg.rolling.min_credit_for_roll = max(0.0, float(cfg.rolling.min_credit_for_roll))
    cfg.rolling.max_rolls_per_position = max(0, int(cfg.rolling.max_rolls_per_position))
    cfg.exits.trailing_stop_activation_pct = max(
        0.0, min(1.0, float(cfg.exits.trailing_stop_activation_pct))
    )
    cfg.exits.trailing_stop_floor_pct = max(
        0.0, min(1.0, float(cfg.exits.trailing_stop_floor_pct))
    )
    cfg.adjustments.delta_test_threshold = max(
        0.0, min(1.0, float(cfg.adjustments.delta_test_threshold))
    )
    cfg.adjustments.min_dte_remaining = max(0, int(cfg.adjustments.min_dte_remaining))
    cfg.adjustments.max_adjustments_per_position = max(
        0, int(cfg.adjustments.max_adjustments_per_position)
    )
    cfg.hedging.delta_hedge_trigger = max(0.0, float(cfg.hedging.delta_hedge_trigger))
    cfg.hedging.max_hedge_cost_pct = max(0.0, float(cfg.hedging.max_hedge_cost_pct))
    cfg.llm_strategist.provider = _normalize_choice(
        cfg.llm_strategist.provider,
        allowed={"openai", "anthropic", "ollama", "google", "gemini"},
        default="google",
        field_name="llm_strategist.provider",
    )
    if cfg.llm_strategist.provider == "gemini":
        cfg.llm_strategist.provider = "google"
    if cfg.llm_strategist.provider == "anthropic":
        model_key = str(cfg.llm_strategist.model).strip().lower()
        if (
            not model_key
            or model_key.startswith("gpt-")
            or model_key.startswith("gemini-")
        ):
            cfg.llm_strategist.model = "claude-sonnet-4-20250514"
    elif cfg.llm_strategist.provider == "openai":
        model_key = str(cfg.llm_strategist.model).strip().lower()
        if (
            not model_key
            or model_key.startswith("gemini-")
            or model_key.startswith("claude-")
        ):
            cfg.llm_strategist.model = "gpt-4o"
    elif cfg.llm_strategist.provider == "google":
        model_key = str(cfg.llm_strategist.model).strip().lower()
        if (
            not model_key
            or model_key.startswith("gpt-")
            or model_key.startswith("claude-")
        ):
            cfg.llm_strategist.model = "gemini-3.1-pro-thinking-preview"
    cfg.llm_strategist.timeout_seconds = max(1, int(cfg.llm_strategist.timeout_seconds))
    cfg.llm_strategist.max_directives = max(1, int(cfg.llm_strategist.max_directives))
    cfg.circuit_breakers.strategy_loss_streak_limit = max(
        1, int(cfg.circuit_breakers.strategy_loss_streak_limit)
    )
    cfg.circuit_breakers.strategy_cooldown_hours = max(
        1, int(cfg.circuit_breakers.strategy_cooldown_hours)
    )
    cfg.circuit_breakers.symbol_loss_streak_limit = max(
        1, int(cfg.circuit_breakers.symbol_loss_streak_limit)
    )
    cfg.circuit_breakers.symbol_blacklist_days = max(
        1, int(cfg.circuit_breakers.symbol_blacklist_days)
    )
    cfg.circuit_breakers.portfolio_drawdown_halt_pct = max(
        0.0, float(cfg.circuit_breakers.portfolio_drawdown_halt_pct)
    )
    cfg.circuit_breakers.portfolio_halt_hours = max(
        1, int(cfg.circuit_breakers.portfolio_halt_hours)
    )
    cfg.circuit_breakers.api_error_rate_threshold = max(
        0.0, min(1.0, float(cfg.circuit_breakers.api_error_rate_threshold))
    )
    cfg.circuit_breakers.api_window_minutes = max(
        1, int(cfg.circuit_breakers.api_window_minutes)
    )
    cfg.circuit_breakers.llm_timeout_streak = max(
        1, int(cfg.circuit_breakers.llm_timeout_streak)
    )
    cfg.strangles.allow_straddles_on = _normalize_symbol_list(
        cfg.strangles.allow_straddles_on,
        default=["SPY", "QQQ", "IWM"],
    )
    cfg.strangles.short_delta = max(0.01, min(0.49, float(cfg.strangles.short_delta)))
    cfg.strangles.min_iv_rank = max(0.0, min(100.0, float(cfg.strangles.min_iv_rank)))
    cfg.strangles.min_dte = max(1, int(cfg.strangles.min_dte))
    cfg.strangles.max_dte = max(cfg.strangles.min_dte, int(cfg.strangles.max_dte))
    cfg.broken_wing_butterfly.min_dte = max(1, int(cfg.broken_wing_butterfly.min_dte))
    cfg.broken_wing_butterfly.max_dte = max(
        cfg.broken_wing_butterfly.min_dte, int(cfg.broken_wing_butterfly.max_dte)
    )
    cfg.broken_wing_butterfly.short_delta = max(
        0.01, min(0.49, float(cfg.broken_wing_butterfly.short_delta))
    )
    cfg.broken_wing_butterfly.near_wing_width = max(
        0.5, float(cfg.broken_wing_butterfly.near_wing_width)
    )
    cfg.broken_wing_butterfly.far_wing_width = max(
        cfg.broken_wing_butterfly.near_wing_width,
        float(cfg.broken_wing_butterfly.far_wing_width),
    )
    cfg.broken_wing_butterfly.min_credit = max(
        0.0, float(cfg.broken_wing_butterfly.min_credit)
    )
    cfg.earnings_vol_crush.min_dte = max(0, int(cfg.earnings_vol_crush.min_dte))
    cfg.earnings_vol_crush.max_dte = max(
        cfg.earnings_vol_crush.min_dte, int(cfg.earnings_vol_crush.max_dte)
    )
    cfg.earnings_vol_crush.max_position_risk_pct = max(
        0.0, float(cfg.earnings_vol_crush.max_position_risk_pct)
    )
    cfg.earnings_vol_crush.min_iv_rank = max(
        0.0, min(100.0, float(cfg.earnings_vol_crush.min_iv_rank))
    )
    cfg.earnings_vol_crush.wing_width = max(
        0.5, float(cfg.earnings_vol_crush.wing_width)
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
    valid = {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    }
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


def _normalize_positive_float_list(value: object, default: list[float]) -> list[float]:
    """Normalize a list of positive float values."""
    if not isinstance(value, list):
        return list(default)

    normalized: list[float] = []
    for raw in value:
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            normalized.append(parsed)
    return normalized or list(default)


def _normalize_greeks_budget_limits(value: object) -> dict:
    """Normalize per-regime Greeks budget limits."""
    defaults = GreeksBudgetConfig().limits
    normalized: dict[str, dict[str, float]] = {
        str(regime): {
            "delta_min": float(row.get("delta_min", -1e9)),
            "delta_max": float(row.get("delta_max", 1e9)),
            "vega_min": float(row.get("vega_min", -1e9)),
            "vega_max": float(row.get("vega_max", 1e9)),
        }
        for regime, row in defaults.items()
        if isinstance(row, dict)
    }

    if not isinstance(value, dict):
        return normalized

    def _regime_key(raw: object) -> str:
        key = str(raw or "").strip().upper()
        if key in {"CRASH", "CRISIS", "CRASH_CRISIS"}:
            return "CRASH/CRISIS"
        return key or "NORMAL"

    for raw_regime, raw_limits in value.items():
        if not isinstance(raw_limits, dict):
            continue
        regime = _regime_key(raw_regime)
        row = normalized.get(regime, normalized.get("NORMAL", {}).copy())
        if not isinstance(row, dict):
            row = {}
        for field_name in ("delta_min", "delta_max", "vega_min", "vega_max"):
            raw_val = raw_limits.get(field_name, row.get(field_name))
            try:
                row[field_name] = float(raw_val) if raw_val is not None else 0.0
            except (TypeError, ValueError):
                row[field_name] = float(row.get(field_name, 0.0))
        if row["delta_min"] > row["delta_max"]:
            row["delta_min"], row["delta_max"] = row["delta_max"], row["delta_min"]
        if row["vega_min"] > row["vega_max"]:
            row["vega_min"], row["vega_max"] = row["vega_max"], row["vega_min"]
        normalized[regime] = row

    return normalized


def _normalize_int_list(
    value: object,
    default: list[int],
    *,
    minimum: int = 0,
) -> list[int]:
    """Normalize a list of bounded integer values."""
    if not isinstance(value, list):
        return list(default)

    normalized: list[int] = []
    for raw in value:
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            continue
        normalized.append(max(minimum, parsed))
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


def _normalize_risk_config(risk_cfg: RiskConfig) -> RiskConfig:
    """Clamp and sanitize risk configuration values."""
    if not isinstance(risk_cfg, RiskConfig):
        risk_cfg = RiskConfig()
    risk_cfg.max_portfolio_risk_pct = max(0.0, float(risk_cfg.max_portfolio_risk_pct))
    risk_cfg.max_position_risk_pct = max(0.0, float(risk_cfg.max_position_risk_pct))
    risk_cfg.max_open_positions = max(1, int(risk_cfg.max_open_positions))
    risk_cfg.max_positions_per_symbol = max(1, int(risk_cfg.max_positions_per_symbol))
    risk_cfg.min_account_balance = max(0.0, float(risk_cfg.min_account_balance))
    risk_cfg.max_daily_loss_pct = max(0.0, float(risk_cfg.max_daily_loss_pct))
    risk_cfg.covered_call_notional_risk_pct = max(
        0.0, min(100.0, float(risk_cfg.covered_call_notional_risk_pct))
    )
    risk_cfg.max_portfolio_delta_abs = max(0.0, float(risk_cfg.max_portfolio_delta_abs))
    risk_cfg.max_portfolio_vega_pct_of_account = max(
        0.0, float(risk_cfg.max_portfolio_vega_pct_of_account)
    )
    risk_cfg.max_sector_risk_pct = max(
        0.0, min(100.0, float(risk_cfg.max_sector_risk_pct))
    )
    risk_cfg.correlation_lookback_days = max(
        20, int(risk_cfg.correlation_lookback_days)
    )
    risk_cfg.correlation_threshold = max(
        0.0, min(1.0, float(risk_cfg.correlation_threshold))
    )
    risk_cfg.max_portfolio_correlation = max(
        0.0, min(1.0, float(risk_cfg.max_portfolio_correlation))
    )
    risk_cfg.var_enabled = bool(risk_cfg.var_enabled)
    risk_cfg.var_lookback_days = max(20, int(risk_cfg.var_lookback_days))
    risk_cfg.var_limit_pct_95 = max(0.0, float(risk_cfg.var_limit_pct_95))
    risk_cfg.var_limit_pct_99 = max(0.0, float(risk_cfg.var_limit_pct_99))
    risk_cfg.correlated_loss_threshold = max(1, int(risk_cfg.correlated_loss_threshold))
    risk_cfg.correlated_loss_pct = max(
        0.0, min(1.0, float(risk_cfg.correlated_loss_pct))
    )
    risk_cfg.correlated_loss_cooldown_hours = max(
        1, int(risk_cfg.correlated_loss_cooldown_hours)
    )
    risk_cfg.gamma_week_tight_stop = max(1.0, float(risk_cfg.gamma_week_tight_stop))
    risk_cfg.expiration_day_close_pct = max(
        0.0, min(0.25, float(risk_cfg.expiration_day_close_pct))
    )
    return risk_cfg


def _normalize_risk_profiles(value: object) -> dict[str, RiskConfig]:
    """Normalize optional named risk profiles."""
    if not isinstance(value, dict):
        return _default_risk_profiles()

    normalized: dict[str, RiskConfig] = {}
    for raw_name, raw_cfg in value.items():
        name = str(raw_name).strip().lower()
        if not name:
            continue
        if isinstance(raw_cfg, RiskConfig):
            normalized[name] = _normalize_risk_config(raw_cfg)
            continue
        if not isinstance(raw_cfg, dict):
            continue
        candidate = RiskConfig()
        for key, val in raw_cfg.items():
            if hasattr(candidate, key):
                setattr(candidate, key, val)
        normalized[name] = _normalize_risk_config(candidate)

    if not normalized:
        return _default_risk_profiles()
    return normalized


def _strategy_dte_overlaps(cfg: BotConfig) -> list[str]:
    """Return overlapping enabled strategy DTE windows."""
    windows: dict[str, tuple[int, int]] = {}
    if bool(cfg.credit_spreads.enabled):
        windows["credit_spreads"] = (
            int(cfg.credit_spreads.min_dte),
            int(cfg.credit_spreads.max_dte),
        )
    if bool(cfg.iron_condors.enabled):
        windows["iron_condors"] = (
            int(cfg.iron_condors.min_dte),
            int(cfg.iron_condors.max_dte),
        )
    if bool(cfg.naked_puts.enabled):
        windows["naked_puts"] = (
            int(cfg.naked_puts.min_dte),
            int(cfg.naked_puts.max_dte),
        )
    if bool(cfg.calendar_spreads.enabled):
        windows["calendar_spreads"] = (
            int(cfg.calendar_spreads.front_min_dte),
            int(cfg.calendar_spreads.front_max_dte),
        )
    if bool(cfg.strangles.enabled):
        windows["strangles"] = (
            int(cfg.strangles.min_dte),
            int(cfg.strangles.max_dte),
        )
    if bool(cfg.broken_wing_butterfly.enabled):
        windows["broken_wing_butterfly"] = (
            int(cfg.broken_wing_butterfly.min_dte),
            int(cfg.broken_wing_butterfly.max_dte),
        )
    if bool(cfg.earnings_vol_crush.enabled):
        windows["earnings_vol_crush"] = (
            int(cfg.earnings_vol_crush.min_dte),
            int(cfg.earnings_vol_crush.max_dte),
        )

    names = sorted(windows.keys())
    overlaps: list[str] = []
    for idx, left in enumerate(names):
        left_min, left_max = windows[left]
        for right in names[idx + 1 :]:
            right_min, right_max = windows[right]
            if max(left_min, right_min) <= min(left_max, right_max):
                overlaps.append(
                    f"{left}({left_min}-{left_max})~{right}({right_min}-{right_max})"
                )
    return overlaps


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
