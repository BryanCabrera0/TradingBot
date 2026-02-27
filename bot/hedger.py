"""Portfolio-level hedging suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HedgeAction:
    symbol: str
    instrument: str
    direction: str
    quantity: int
    estimated_cost: float
    reason: str


class PortfolioHedger:
    """Compute minimal hedge actions when aggregate risk exceeds thresholds."""

    def __init__(self, config: dict):
        self.config = config

    def propose(
        self,
        *,
        account_value: float,
        net_delta: float,
        sector_exposure: dict,
        regime: str = "normal",
        existing_hedge_symbols: set | None = None,
    ) -> list[HedgeAction]:
        if not bool(self.config.get("enabled", False)):
            return []

        _existing = existing_hedge_symbols or set()
        actions: list[HedgeAction] = []
        delta_trigger = float(self.config.get("delta_hedge_trigger", 50.0))
        if abs(net_delta) > delta_trigger:
            # Skip if we already have an active delta hedge for SPY
            if "SPY" in _existing:
                return []
            excess = abs(net_delta) - delta_trigger
            qty = max(1, int(round(excess / 20.0)))
            direction = "buy_put" if net_delta > 0 else "sell_call_spread"
            actions.append(
                HedgeAction(
                    symbol="SPY",
                    instrument="options",
                    direction=direction,
                    quantity=qty,
                    estimated_cost=round(min(account_value * 0.0015, qty * 50.0), 2),
                    reason=f"Net delta {net_delta:.2f} exceeds trigger {delta_trigger:.2f}",
                )
            )

        if bool(self.config.get("tail_risk_enabled", False)) and regime.upper() in {"LOW_VOL_GRIND", "BULL_TREND"}:
            actions.append(
                HedgeAction(
                    symbol="VIX",
                    instrument="options",
                    direction="buy_call",
                    quantity=1,
                    estimated_cost=round(min(account_value * 0.0008, 75.0), 2),
                    reason="Tail-risk insurance in low-vol environment",
                )
            )

        largest_sector = _largest_sector(sector_exposure)
        if largest_sector and largest_sector[1] > 0.40:
            etf = _sector_to_etf(largest_sector[0])
            if etf:
                actions.append(
                    HedgeAction(
                        symbol=etf,
                        instrument="options",
                        direction="buy_put",
                        quantity=1,
                        estimated_cost=round(min(account_value * 0.001, 60.0), 2),
                        reason=f"Sector concentration hedge for {largest_sector[0]} ({largest_sector[1]:.1%})",
                    )
                )

        max_cost = float(self.config.get("max_hedge_cost_pct", 1.0)) / 100.0 * max(0.0, account_value)
        capped: list[HedgeAction] = []
        running = 0.0
        for action in actions:
            if running + action.estimated_cost > max_cost:
                continue
            capped.append(action)
            running += action.estimated_cost
        return capped


def _largest_sector(sector_exposure: dict) -> Optional[tuple[str, float]]:
    if not isinstance(sector_exposure, dict) or not sector_exposure:
        return None
    pairs = []
    for name, value in sector_exposure.items():
        try:
            pct = float(value)
        except (TypeError, ValueError):
            continue
        if pct > 1.0:
            pct /= 100.0
        pairs.append((str(name), pct))
    if not pairs:
        return None
    return max(pairs, key=lambda row: row[1])


def _sector_to_etf(sector: str) -> str:
    mapping = {
        "Information Technology": "XLK",
        "Financials": "XLF",
        "Health Care": "XLV",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Energy": "XLE",
        "Industrials": "XLI",
        "Materials": "XLB",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Communication Services": "XLC",
    }
    return mapping.get(str(sector), "")

