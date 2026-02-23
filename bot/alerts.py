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

        payload = {
            "text": text,
            "level": normalized,
            "title": title,
            "timestamp": timestamp,
            "source": "tradingbot",
        }

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
