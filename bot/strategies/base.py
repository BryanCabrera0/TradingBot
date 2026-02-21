"""Base class for all trading strategies."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from bot.analysis import SpreadAnalysis

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """A signal to open or close a trade."""
    action: str  # "open" or "close"
    strategy: str
    symbol: str
    analysis: Optional[SpreadAnalysis] = None
    # For closing trades
    position_id: Optional[str] = None
    reason: str = ""
    # Order details — set by strategy
    order_spec: object = None
    quantity: int = 1


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"strategy.{name}")

    @abstractmethod
    def scan_for_entries(
        self, symbol: str, chain_data: dict, underlying_price: float
    ) -> list[TradeSignal]:
        """Scan an options chain for new entry opportunities.

        Returns a list of TradeSignals for positions to open.
        """

    @abstractmethod
    def check_exits(
        self, positions: list, market_client
    ) -> list[TradeSignal]:
        """Check existing positions for exit signals.

        Returns a list of TradeSignals for positions to close.
        """

    def meets_minimum_quality(self, analysis: SpreadAnalysis) -> bool:
        """Check if a spread analysis meets minimum quality thresholds."""
        min_credit_pct = self.config.get("min_credit_pct", 0.25)
        min_score = 40.0

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

        if analysis.probability_of_profit < 0.50:
            self.logger.debug(
                "Rejected: POP %.1f%% < 50%%",
                analysis.probability_of_profit * 100,
            )
            return False

        if analysis.score < min_score:
            self.logger.debug(
                "Rejected: score %.1f < min %.1f", analysis.score, min_score
            )
            return False

        return True
