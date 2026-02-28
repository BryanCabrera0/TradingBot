"""Portfolio Monte Carlo risk simulation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from bot.number_utils import safe_float, safe_int


@dataclass
class MonteCarloResult:
    """Structured Monte Carlo output for portfolio risk controls."""

    simulations: int
    horizons: dict
    var99_pct_account: float
    high_risk: bool

    def to_dict(self) -> dict:
        return {
            "simulations": int(self.simulations),
            "horizons": dict(self.horizons),
            "var99_pct_account": float(self.var99_pct_account),
            "high_risk": bool(self.high_risk),
        }


class MonteCarloRiskEngine:
    """Simulate short-horizon portfolio P&L paths from Greeks + IV assumptions."""

    def __init__(
        self,
        *,
        simulations: int = 10_000,
        var_limit_pct: float = 3.0,
        random_seed: Optional[int] = None,
    ):
        self.simulations = max(100, int(simulations))
        self.var_limit_pct = max(0.0, float(var_limit_pct))
        self._rng = np.random.default_rng(random_seed)

    def simulate(
        self,
        positions: list[dict],
        *,
        account_balance: float,
        horizons: tuple[int, ...] = (1, 5, 21),
    ) -> MonteCarloResult:
        sims = max(100, int(self.simulations))
        balance = max(1.0, float(account_balance or 0.0))
        if not positions:
            zero_horizons = {
                f"{h}d": {
                    "mean_pnl": 0.0,
                    "var95": 0.0,
                    "var99": 0.0,
                    "cvar95": 0.0,
                    "expected_shortfall": 0.0,
                }
                for h in horizons
            }
            return MonteCarloResult(
                simulations=sims,
                horizons=zero_horizons,
                var99_pct_account=0.0,
                high_risk=False,
            )

        results: dict[str, dict] = {}
        max_var99 = 0.0
        for horizon in horizons:
            horizon_days = max(1, int(horizon))
            pnl = np.zeros(sims, dtype=float)
            for position in positions:
                pnl += self._simulate_position_pnl(position, horizon_days, sims)

            var95 = self._var_from_distribution(pnl, 0.95)
            var99 = self._var_from_distribution(pnl, 0.99)
            cvar95 = self._cvar_from_distribution(pnl, 0.95)
            max_var99 = max(max_var99, var99)
            results[f"{horizon_days}d"] = {
                "mean_pnl": round(float(np.mean(pnl)), 4),
                "var95": round(var95, 4),
                "var99": round(var99, 4),
                "cvar95": round(cvar95, 4),
                "expected_shortfall": round(cvar95, 4),
            }

        var99_pct = (max_var99 / balance) * 100.0
        return MonteCarloResult(
            simulations=sims,
            horizons=results,
            var99_pct_account=round(var99_pct, 4),
            high_risk=bool(var99_pct > self.var_limit_pct),
        )

    def _simulate_position_pnl(
        self, position: dict, horizon_days: int, sims: int
    ) -> np.ndarray:
        details = (
            position.get("details", {})
            if isinstance(position.get("details"), dict)
            else {}
        )
        quantity = max(1, safe_int(position.get("quantity"), 1))
        delta = safe_float(
            details.get("net_delta", position.get("net_delta", 0.0)), 0.0
        )
        gamma = safe_float(
            details.get("net_gamma", position.get("net_gamma", 0.0)), 0.0
        )
        theta = safe_float(
            details.get("net_theta", position.get("net_theta", 0.0)), 0.0
        )
        vega = safe_float(details.get("net_vega", position.get("net_vega", 0.0)), 0.0)
        rho = safe_float(details.get("net_rho", position.get("net_rho", 0.0)), 0.0)
        underlying = max(0.01, safe_float(position.get("underlying_price"), 100.0))
        iv_raw = safe_float(
            details.get("current_iv", details.get("entry_iv", details.get("iv", 25.0))),
            25.0,
        )
        iv = iv_raw / 100.0 if iv_raw > 2.0 else iv_raw
        iv = max(0.05, min(2.0, iv))
        vol_of_vol = safe_float(details.get("vol_of_vol"), 0.20)
        vol_of_vol = max(0.01, min(1.0, vol_of_vol))

        t = horizon_days / 252.0
        z_price = self._rng.standard_normal(sims)
        z_iv = self._rng.standard_normal(sims)
        price_change = underlying * (
            np.exp((-0.5 * iv * iv * t) + (iv * np.sqrt(t) * z_price)) - 1.0
        )
        iv_change = z_iv * vol_of_vol * np.sqrt(t)
        rate_shift = self._rng.standard_normal(sims) * 0.0025 * np.sqrt(t)

        pnl_per_contract = (
            (delta * price_change)
            + (0.5 * gamma * np.square(price_change))
            + (theta * horizon_days)
            + (vega * iv_change)
            + (rho * rate_shift * 100.0)
        )
        return pnl_per_contract * quantity * 100.0

    @staticmethod
    def _var_from_distribution(pnl: np.ndarray, confidence: float) -> float:
        if pnl.size == 0:
            return 0.0
        percentile = max(0.0, min(100.0, (1.0 - confidence) * 100.0))
        threshold = float(np.percentile(pnl, percentile))
        return max(0.0, -threshold)

    @staticmethod
    def _cvar_from_distribution(pnl: np.ndarray, confidence: float) -> float:
        if pnl.size == 0:
            return 0.0
        percentile = max(0.0, min(100.0, (1.0 - confidence) * 100.0))
        threshold = float(np.percentile(pnl, percentile))
        tail = pnl[pnl <= threshold]
        if tail.size == 0:
            return 0.0
        return max(0.0, -float(np.mean(tail)))
