"""Rolling implied-volatility history storage and percentile utilities."""

from __future__ import annotations

import bisect
from datetime import date
from pathlib import Path

from bot.data_store import dump_json, ensure_data_dir, load_json

IV_HISTORY_PATH = Path("bot/data/iv_history.json")


class IVHistory:
    """Maintains per-symbol daily IV history for percentile/rank calculations."""

    def __init__(self, path: Path | str = IV_HISTORY_PATH):
        self.path = Path(path)
        ensure_data_dir(self.path.parent)

    def update_and_rank(self, symbol: str, iv_value: float, *, as_of: str | None = None) -> float:
        symbol_key = symbol.upper().strip()
        as_of_key = as_of or date.today().isoformat()
        state = load_json(self.path, {"symbols": {}})
        if not isinstance(state, dict):
            state = {"symbols": {}}
        symbols = state.get("symbols")
        if not isinstance(symbols, dict):
            symbols = {}
            state["symbols"] = symbols

        series = symbols.get(symbol_key, [])
        if not isinstance(series, list):
            series = []

        # Replace same-day value when present.
        updated = False
        for row in series:
            if not isinstance(row, dict):
                continue
            if str(row.get("date")) == as_of_key:
                row["iv"] = float(iv_value)
                updated = True
                break
        if not updated:
            series.append({"date": as_of_key, "iv": float(iv_value)})

        # Keep ~1 trading year history.
        series = sorted(
            [item for item in series if isinstance(item, dict) and "iv" in item and "date" in item],
            key=lambda item: str(item.get("date")),
        )[-252:]
        symbols[symbol_key] = series
        dump_json(self.path, state)

        return self.percentile_rank(symbol_key, iv_value, series_override=series)

    def percentile_rank(
        self,
        symbol: str,
        iv_value: float,
        *,
        series_override: list[dict] | None = None,
    ) -> float:
        symbol_key = symbol.upper().strip()
        if series_override is None:
            state = load_json(self.path, {"symbols": {}})
            symbols = state.get("symbols", {}) if isinstance(state, dict) else {}
            series = symbols.get(symbol_key, []) if isinstance(symbols, dict) else []
        else:
            series = series_override

        ivs = sorted(
            float(item.get("iv", 0.0))
            for item in series
            if isinstance(item, dict) and item.get("iv") is not None
        )
        if not ivs:
            return 50.0
        idx = bisect.bisect_right(ivs, float(iv_value))
        rank = (idx / len(ivs)) * 100.0
        return round(max(0.0, min(100.0, rank)), 2)
