"""Base class for all trading strategies."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from bot.analysis import SpreadAnalysis

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """A standard signal representation used to open, close, or roll a trade.
    
    This object serves as the common currency between the Strategy (which generates it), 
    the RiskManager (which validates and sizes it), the LLMAdvisor (which approves it), 
    and the Orchestrator (which executes it).
    
    Other AI agents should look for these fields:
    - action: Must be 'open', 'close', or 'roll'
    - strategy: Name of the originating strategy (e.g. 'credit_spreads')
    - analysis: SpreadAnalysis containing Greeks, probability, and theoretical risk/reward
    - metadata: Free-form dict for debugging or LLM context (e.g. why it was picked)
    """

    action: str  # "open", "close", or "roll"
    strategy: str
    symbol: str
    analysis: Optional[SpreadAnalysis] = None
    # For closing trades
    position_id: Optional[str] = None
    reason: str = ""
    # Order details — set by strategy
    order_spec: object = None
    quantity: int = 1
    size_multiplier: float = 1.0
    metadata: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies in the bot.
    
    When creating a new strategy, inherit from this class and implement:
    1. scan_for_entries: Iterate over option chains and return valid 'open' TradeSignals.
    2. check_exits: Iterate over open positions and return valid 'close' or 'roll' TradeSignals.
    """

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"strategy.{name}")

    @abstractmethod
    def scan_for_entries(
        self,
        symbol: str,
        chain_data: dict,
        underlying_price: float,
        technical_context=None,
        market_context: Optional[dict] = None,
    ) -> list[TradeSignal]:
        """Scan an options chain for new entry opportunities.

        Args:
            symbol: Ticker symbol (e.g., 'SPY')
            chain_data: Raw JSON payload from Schwab options chain API
            underlying_price: Current market price of the underlying asset
            technical_context: Optional indicators (RSI, MACD) from TechnicalAnalyzer
            market_context: Optional dictionary containing VIX, regime, or news sentiment

        Returns:
            A list of TradeSignals populated with 'open' actions and SpreadAnalysis data.
        """

    @abstractmethod
    def check_exits(self, positions: list, market_client) -> list[TradeSignal]:
        """Check existing positions for exit signals based on P&L, DTE, or technicals.

        Args:
            positions: List of dictionary records representing currently open trades
            market_client: Reference to the SchwabClient or PaperTrader for fetching latest quotes

        Returns:
            A list of TradeSignals populated with 'close' actions.
        """

    def meets_minimum_quality(self, analysis: SpreadAnalysis) -> bool:
        """Check if a spread analysis meets minimum quality thresholds."""
        min_credit_pct = self.config.get("min_credit_pct", 0.02)
        min_score = self.config.get("min_score", 5.0)
        min_pop = self.config.get("min_pop", 0.10)

        if analysis.credit <= 0:
            self.logger.debug("Rejected: zero or negative credit")
            return False

        if analysis.credit_pct_of_width < min_credit_pct:
            self.logger.debug(
                "Rejected: credit %% %.1f%% < min %.1f%%",
                analysis.credit_pct_of_width * 100,
                min_credit_pct * 100,
            )
            return False

        if analysis.probability_of_profit < min_pop:
            self.logger.debug(
                "Rejected: POP %.1f%% < %.0f%%",
                analysis.probability_of_profit * 100,
                min_pop * 100,
            )
            return False

        if analysis.score < min_score:
            self.logger.debug(
                "Rejected: score %.1f < min %.1f", analysis.score, min_score
            )
            return False

        return True
