"""Economic calendar integration for macro-event aware position sizing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from bot.data_store import dump_json, load_json

logger = logging.getLogger(__name__)

ECON_CACHE_PATH = Path("bot/data/econ_calendar.json")
ECON_STATIC_EVENTS_PATH = Path("bot/data/econ_events_2025_2026.json")

HIGH_IMPACT = {"FOMC", "CPI", "NFP", "GDP", "PCE"}


@dataclass
class EconEvent:
    name: str
    event_date: str
    severity: str = "high"  # low | medium | high
    country: str = "US"
    impact: str = "macro"

    @property
    def date_value(self) -> Optional[date]:
        try:
            return datetime.strptime(self.event_date, "%Y-%m-%d").date()
        except ValueError:
            return None


class EconomicCalendar:
    """Fetch/cache macro events and evaluate trade-entry policies."""

    def __init__(
        self,
        *,
        cache_path: Path | str = ECON_CACHE_PATH,
        refresh_days: int = 1,
        static_events: Optional[list[dict]] = None,
        policy: Optional[dict] = None,
    ):
        self.cache_path = Path(cache_path)
        self.refresh_days = max(1, int(refresh_days))
        self.static_events = static_events or _default_static_events()
        self.policy = policy or {
            "high": "reduce_size",   # skip | reduce_size | widen
            "medium": "widen",
            "low": "none",
        }

    def refresh(self) -> list[EconEvent]:
        payload = load_json(self.cache_path, {})
        today = date.today().isoformat()
        if isinstance(payload, dict) and payload.get("as_of") == today:
            return self._parse_events(payload.get("events", []))

        latest_static = _default_static_events()
        if latest_static:
            self.static_events = latest_static
        events = self._parse_events(self.static_events)
        out = {
            "as_of": today,
            "events": [event.__dict__ for event in events],
        }
        dump_json(self.cache_path, out)
        return events

    def upcoming_events(self, *, start: date, end: date) -> list[EconEvent]:
        events = self.refresh()
        out = []
        for event in events:
            value = event.date_value
            if value is None:
                continue
            if start <= value <= end:
                out.append(event)
        return sorted(out, key=lambda item: item.event_date)

    def policy_for_trade(
        self,
        *,
        expiration: str,
        as_of: Optional[date] = None,
    ) -> dict:
        """Return the strongest event policy that applies through expiration."""
        start = as_of or date.today()
        try:
            end = datetime.strptime(str(expiration).split("T", 1)[0], "%Y-%m-%d").date()
        except ValueError:
            return {"action": "none", "events": []}

        events = self.upcoming_events(start=start, end=end)
        if not events:
            return {"action": "none", "events": []}

        severity_rank = {"low": 0, "medium": 1, "high": 2}
        strongest = max(events, key=lambda ev: severity_rank.get(ev.severity, 0))
        action = self.policy.get(strongest.severity, "none")
        return {
            "action": action,
            "events": [event.__dict__ for event in events],
            "strongest": strongest.__dict__,
        }

    def adjust_signal(self, signal, *, as_of: Optional[date] = None) -> tuple[bool, str]:
        """Apply macro policy to a trade signal in-place."""
        analysis = getattr(signal, "analysis", None)
        if analysis is None:
            return True, ""
        decision = self.policy_for_trade(expiration=analysis.expiration, as_of=as_of)
        action = str(decision.get("action", "none")).lower()
        if action == "none":
            return True, ""
        if action == "skip":
            event = decision.get("strongest", {})
            return False, f"Macro event {event.get('name')} before expiration"
        if action == "reduce_size":
            signal.size_multiplier = max(0.1, float(signal.size_multiplier or 1.0) * 0.70)
            return True, "Reduced size due to macro event risk"
        if action == "widen" and analysis is not None:
            widened = _apply_widening(analysis)
            if widened:
                signal.size_multiplier = max(0.1, float(signal.size_multiplier or 1.0) * 0.85)
                return True, "Widened strikes and reduced size due to macro event risk"
            return True, "Macro event policy requested wider strikes"
        return True, ""

    def context(self, *, days: int = 14) -> dict:
        start = date.today()
        end = start + timedelta(days=max(1, int(days)))
        events = self.upcoming_events(start=start, end=end)
        return {
            "upcoming_macro_events": [event.__dict__ for event in events],
            "horizon_days": days,
        }

    @staticmethod
    def _parse_events(rows: list[dict]) -> list[EconEvent]:
        events = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip().upper()
            if not name:
                continue
            if name not in HIGH_IMPACT and str(row.get("impact", "")).lower() != "macro":
                continue
            events.append(
                EconEvent(
                    name=name,
                    event_date=str(row.get("event_date", "")),
                    severity=str(row.get("severity", "high")).lower(),
                    country=str(row.get("country", "US")),
                    impact=str(row.get("impact", "macro")),
                )
            )
        return events


def _default_static_events() -> list[dict]:
    payload = load_json(ECON_STATIC_EVENTS_PATH, [])
    rows = payload.get("events", []) if isinstance(payload, dict) else payload
    if isinstance(rows, list) and rows:
        return [row for row in rows if isinstance(row, dict)]
    return _comprehensive_static_events()


def refresh_static_calendar_file(path: Path | str = ECON_STATIC_EVENTS_PATH) -> str:
    """Regenerate the bundled 2025-2026 static US macro calendar file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of": date.today().isoformat(),
        "events": _comprehensive_static_events(),
    }
    dump_json(output_path, payload)
    return str(output_path)


def _comprehensive_static_events() -> list[dict]:
    # Approximate schedule of high-impact US macro events for 2025-2026.
    return [
        # 2025
        {"name": "NFP", "event_date": "2025-01-10", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2025-01-15", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2025-01-16", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2025-01-31", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2025-02-07", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2025-02-12", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2025-02-13", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "GDP", "event_date": "2025-02-27", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2025-02-28", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2025-03-07", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2025-03-12", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2025-03-13", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "FOMC", "event_date": "2025-03-19", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2025-03-28", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2025-04-04", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2025-04-10", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2025-04-11", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "GDP", "event_date": "2025-04-30", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2025-04-30", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2025-05-02", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2025-05-14", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2025-05-15", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "FOMC", "event_date": "2025-05-07", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2025-05-30", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2025-06-06", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2025-06-11", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2025-06-12", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "FOMC", "event_date": "2025-06-18", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2025-06-27", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2025-07-03", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2025-07-15", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2025-07-16", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "GDP", "event_date": "2025-07-30", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "FOMC", "event_date": "2025-07-30", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2025-07-31", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2025-08-01", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2025-08-13", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2025-08-14", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2025-08-29", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2025-09-05", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2025-09-11", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2025-09-12", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "FOMC", "event_date": "2025-09-17", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2025-09-26", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2025-10-03", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2025-10-15", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2025-10-16", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "GDP", "event_date": "2025-10-30", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2025-10-31", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2025-11-07", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "FOMC", "event_date": "2025-11-06", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2025-11-13", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2025-11-14", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2025-11-26", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2025-12-05", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2025-12-11", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2025-12-12", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "FOMC", "event_date": "2025-12-17", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2025-12-24", "severity": "high", "country": "US", "impact": "macro"},
        # 2026
        {"name": "NFP", "event_date": "2026-01-09", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2026-01-14", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2026-01-15", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2026-01-30", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2026-02-06", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2026-02-11", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2026-02-12", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "GDP", "event_date": "2026-02-26", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2026-02-27", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2026-03-06", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2026-03-12", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2026-03-13", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "FOMC", "event_date": "2026-03-18", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2026-03-27", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2026-04-03", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2026-04-10", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2026-04-13", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "GDP", "event_date": "2026-04-30", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2026-04-30", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2026-05-01", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2026-05-13", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2026-05-14", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "FOMC", "event_date": "2026-05-06", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2026-05-29", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2026-06-05", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2026-06-11", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2026-06-12", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "FOMC", "event_date": "2026-06-17", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2026-06-26", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2026-07-02", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2026-07-15", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2026-07-16", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "GDP", "event_date": "2026-07-30", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "FOMC", "event_date": "2026-07-29", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2026-07-31", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2026-08-07", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2026-08-12", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2026-08-13", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2026-08-28", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2026-09-04", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2026-09-10", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2026-09-11", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "FOMC", "event_date": "2026-09-16", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2026-09-25", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2026-10-02", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2026-10-14", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2026-10-15", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "GDP", "event_date": "2026-10-29", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2026-10-30", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2026-11-06", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "FOMC", "event_date": "2026-11-05", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2026-11-12", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2026-11-13", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2026-11-25", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "NFP", "event_date": "2026-12-04", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "CPI", "event_date": "2026-12-10", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PPI", "event_date": "2026-12-11", "severity": "medium", "country": "US", "impact": "macro"},
        {"name": "FOMC", "event_date": "2026-12-16", "severity": "high", "country": "US", "impact": "macro"},
        {"name": "PCE", "event_date": "2026-12-24", "severity": "high", "country": "US", "impact": "macro"},
    ]


def _apply_widening(analysis) -> bool:
    """Widen common spread wing fields by one existing wing-width."""
    changed = False
    try:
        short_strike = float(getattr(analysis, "short_strike", 0.0) or 0.0)
        long_strike = float(getattr(analysis, "long_strike", 0.0) or 0.0)
        if short_strike > 0 and long_strike > 0:
            width = abs(short_strike - long_strike)
            if width > 0:
                if long_strike < short_strike:
                    setattr(analysis, "long_strike", round(long_strike - width, 4))
                else:
                    setattr(analysis, "long_strike", round(long_strike + width, 4))
                changed = True
    except Exception:
        pass

    # Iron-condor style legs.
    try:
        put_short = float(getattr(analysis, "put_short_strike", 0.0) or 0.0)
        put_long = float(getattr(analysis, "put_long_strike", 0.0) or 0.0)
        if put_short > 0 and put_long > 0:
            put_width = abs(put_short - put_long)
            if put_width > 0:
                setattr(analysis, "put_long_strike", round(put_long - put_width, 4))
                changed = True
    except Exception:
        pass

    try:
        call_short = float(getattr(analysis, "call_short_strike", 0.0) or 0.0)
        call_long = float(getattr(analysis, "call_long_strike", 0.0) or 0.0)
        if call_short > 0 and call_long > 0:
            call_width = abs(call_long - call_short)
            if call_width > 0:
                setattr(analysis, "call_long_strike", round(call_long + call_width, 4))
                changed = True
    except Exception:
        pass

    return changed
