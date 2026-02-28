"""Daily P&L attribution utilities for options portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from bot.data_store import dump_json, ensure_data_dir, load_json
from bot.number_utils import safe_float, safe_int

DEFAULT_ATTRIBUTION_PATH = Path("bot/data/pnl_attribution.json")


@dataclass
class AttributionConfig:
    path: str = str(DEFAULT_ATTRIBUTION_PATH)


class PnLAttributionEngine:
    """Compute and persist delta/gamma/theta/vega/rho P&L attribution."""

    def __init__(self, *, path: Optional[str] = None):
        self.path = Path(path or DEFAULT_ATTRIBUTION_PATH)
        ensure_data_dir(self.path.parent)

    def compute_attribution(
        self,
        positions: list,
        price_changes: dict,
        iv_changes: dict,
    ) -> dict:
        """Decompose position and portfolio P&L into Greeks-driven components."""
        rows = []
        totals = {
            "delta_pnl": 0.0,
            "gamma_pnl": 0.0,
            "theta_pnl": 0.0,
            "vega_pnl": 0.0,
            "rho_pnl": 0.0,
            "residual": 0.0,
            "total_pnl": 0.0,
        }

        for position in positions or []:
            if not isinstance(position, dict):
                continue
            details = (
                position.get("details", {})
                if isinstance(position.get("details"), dict)
                else {}
            )
            symbol = str(position.get("symbol", "")).upper()
            qty = max(1, safe_int(position.get("quantity"), 1))
            delta = (
                safe_float(
                    details.get("net_delta", position.get("net_delta", 0.0)), 0.0
                )
                * qty
            )
            gamma = (
                safe_float(
                    details.get("net_gamma", position.get("net_gamma", 0.0)), 0.0
                )
                * qty
            )
            theta = (
                safe_float(
                    details.get("net_theta", position.get("net_theta", 0.0)), 0.0
                )
                * qty
            )
            vega = (
                safe_float(details.get("net_vega", position.get("net_vega", 0.0)), 0.0)
                * qty
            )
            rho = (
                safe_float(details.get("net_rho", position.get("net_rho", 0.0)), 0.0)
                * qty
            )
            d_price = (
                safe_float(price_changes.get(symbol), 0.0)
                if isinstance(price_changes, dict)
                else 0.0
            )
            d_iv = (
                safe_float(iv_changes.get(symbol), 0.0)
                if isinstance(iv_changes, dict)
                else 0.0
            )
            days = safe_float(details.get("days_elapsed", 1.0), 1.0)
            rate_change = safe_float(details.get("rate_change", 0.0), 0.0)

            delta_pnl = delta * d_price
            gamma_pnl = 0.5 * gamma * (d_price**2)
            theta_pnl = theta * days
            vega_pnl = vega * d_iv
            rho_pnl = rho * rate_change

            entry_credit = safe_float(position.get("entry_credit"), 0.0)
            current_value = safe_float(position.get("current_value"), 0.0)
            actual_total = (entry_credit - current_value) * qty * 100.0
            explained = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl
            residual = actual_total - explained

            row = {
                "position_id": position.get("position_id", ""),
                "symbol": symbol,
                "strategy": position.get("strategy", ""),
                "quantity": qty,
                "delta_pnl": round(delta_pnl, 6),
                "gamma_pnl": round(gamma_pnl, 6),
                "theta_pnl": round(theta_pnl, 6),
                "vega_pnl": round(vega_pnl, 6),
                "rho_pnl": round(rho_pnl, 6),
                "residual": round(residual, 6),
                "total_pnl": round(actual_total, 6),
            }
            rows.append(row)
            for key in totals:
                totals[key] += safe_float(row.get(key), 0.0)

        return {
            "as_of": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "positions": rows,
            "portfolio": {key: round(value, 6) for key, value in totals.items()},
        }

    def record_daily_snapshot(self, day_iso: str, attribution: dict) -> None:
        """Append/update one daily attribution record keyed by ISO date."""
        payload = load_json(self.path, {"history": []})
        history = payload.get("history", []) if isinstance(payload, dict) else []
        if not isinstance(history, list):
            history = []
        filtered = [
            row
            for row in history
            if isinstance(row, dict) and str(row.get("date", "")) != str(day_iso)
        ]
        filtered.append(
            {
                "date": str(day_iso),
                "recorded_at": datetime.utcnow().replace(microsecond=0).isoformat()
                + "Z",
                "attribution": attribution if isinstance(attribution, dict) else {},
            }
        )
        filtered.sort(key=lambda item: str(item.get("date", "")))
        dump_json(self.path, {"history": filtered[-730:]})
