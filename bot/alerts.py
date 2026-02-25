"""Lightweight webhook-based alerting for runtime incidents."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

import requests

from bot.config import AlertsConfig

logger = logging.getLogger(__name__)

LEVEL_ORDER = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


class AlertManager:
    """Send operational alerts to an optional webhook endpoint."""

    def __init__(self, config: AlertsConfig):
        self.config = config
        self._threshold = LEVEL_ORDER.get(str(config.min_level).upper(), 40)

    def send(
        self,
        *,
        level: str,
        title: str,
        message: str,
        context: Optional[dict] = None,
    ) -> bool:
        """Send an alert if enabled and above threshold."""
        normalized = str(level).upper()
        rank = LEVEL_ORDER.get(normalized, 40)
        if rank < self._threshold:
            return False
        if not self.config.enabled:
            return False
        webhook = str(self.config.webhook_url or "").strip()
        if not webhook:
            return False

        timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        text = f"[TradingBot][{normalized}] {title}\n{message}"
        if context:
            context_text = json.dumps(context, default=str, separators=(",", ":"))
            text = f"{text}\ncontext={context_text}"

        payload = self._format_payload(
            text=text,
            level=normalized,
            title=title,
            timestamp=timestamp,
            context=context or {},
        )

        try:
            response = requests.post(
                webhook,
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Failed to send alert webhook: %s", exc)
            return False

    def trade_opened(self, message: str, *, context: Optional[dict] = None) -> bool:
        if not self.config.trade_notifications:
            return False
        return self.send(level="INFO", title="Trade Opened", message=message, context=context)

    def trade_closed(self, message: str, *, context: Optional[dict] = None) -> bool:
        if not self.config.trade_notifications:
            return False
        return self.send(level="INFO", title="Trade Closed", message=message, context=context)

    def daily_summary(self, message: str, *, context: Optional[dict] = None) -> bool:
        if not self.config.daily_summary:
            return False
        return self.send(level="INFO", title="Daily Summary", message=message, context=context)

    def weekly_summary(self, message: str, *, context: Optional[dict] = None) -> bool:
        if not self.config.weekly_summary:
            return False
        return self.send(level="INFO", title="Weekly Summary", message=message, context=context)

    def regime_change(self, message: str, *, context: Optional[dict] = None) -> bool:
        if not self.config.regime_changes:
            return False
        return self.send(level="WARNING", title="Regime Change", message=message, context=context)

    def risk_warning(self, message: str, *, context: Optional[dict] = None) -> bool:
        return self.send(level="WARNING", title="Risk Warning", message=message, context=context)

    def drawdown_alert(self, drawdown_pct: float, *, context: Optional[dict] = None) -> bool:
        thresholds = sorted(float(v) for v in (self.config.drawdown_thresholds or []))
        if not thresholds:
            return False
        matched = [value for value in thresholds if drawdown_pct >= value]
        if not matched:
            return False
        return self.send(
            level="WARNING",
            title="Drawdown Alert",
            message=f"Drawdown reached {drawdown_pct:.2f}% (threshold {max(matched):.2f}%)",
            context=context,
        )

    def _format_payload(
        self,
        *,
        text: str,
        level: str,
        title: str,
        timestamp: str,
        context: dict,
    ) -> dict:
        webhook_format = str(self.config.webhook_format or "generic").lower()
        if webhook_format == "slack":
            blocks = [
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*[{level}]* *{title}*"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": text}},
            ]
            if context:
                blocks.append(
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"```{json.dumps(context, default=str, indent=2)}```"},
                    }
                )
            return {"text": text, "blocks": blocks}

        if webhook_format == "discord":
            embed = {
                "title": title,
                "description": text,
                "timestamp": timestamp,
                "color": _discord_color(level),
                "fields": [
                    {"name": "Level", "value": level, "inline": True},
                ],
            }
            if context:
                embed["fields"].append(
                    {"name": "Context", "value": json.dumps(context, default=str)[:1000], "inline": False}
                )
            return {"embeds": [embed]}

        return {
            "text": text,
            "level": level,
            "title": title,
            "timestamp": timestamp,
            "source": "tradingbot",
            "context": context,
        }


def _discord_color(level: str) -> int:
    level_upper = str(level).upper()
    if level_upper in {"ERROR", "CRITICAL"}:
        return 0xE74C3C
    if level_upper == "WARNING":
        return 0xF39C12
    return 0x2ECC71
