import unittest
from unittest import mock

from bot.alerts import AlertManager
from bot.config import AlertsConfig


class AlertsTests(unittest.TestCase):
    @mock.patch("bot.alerts.requests.post")
    def test_alert_manager_sends_when_enabled_and_above_threshold(self, mock_post) -> None:
        mock_response = mock.Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        manager = AlertManager(
            AlertsConfig(
                enabled=True,
                webhook_url="https://example.com/hook",
                min_level="ERROR",
                timeout_seconds=3,
            )
        )

        sent = manager.send(level="ERROR", title="Failure", message="oops", context={"x": 1})

        self.assertTrue(sent)
        mock_post.assert_called_once()

    @mock.patch("bot.alerts.requests.post")
    def test_alert_manager_skips_below_threshold(self, mock_post) -> None:
        manager = AlertManager(
            AlertsConfig(
                enabled=True,
                webhook_url="https://example.com/hook",
                min_level="ERROR",
                timeout_seconds=3,
            )
        )

        sent = manager.send(level="INFO", title="Info", message="skip")

        self.assertFalse(sent)
        mock_post.assert_not_called()

    @mock.patch("bot.alerts.requests.post")
    def test_slack_format_payload_contains_blocks(self, mock_post) -> None:
        mock_response = mock.Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        manager = AlertManager(
            AlertsConfig(
                enabled=True,
                webhook_url="https://example.com/hook",
                min_level="INFO",
                webhook_format="slack",
                daily_summary=True,
            )
        )
        manager.daily_summary("Summary", context={"pnl": 123.4})

        payload = mock_post.call_args.kwargs["json"]
        self.assertIn("blocks", payload)
        self.assertIsInstance(payload["blocks"], list)

    @mock.patch("bot.alerts.requests.post")
    def test_discord_format_payload_contains_embeds(self, mock_post) -> None:
        mock_response = mock.Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        manager = AlertManager(
            AlertsConfig(
                enabled=True,
                webhook_url="https://example.com/hook",
                min_level="INFO",
                webhook_format="discord",
                trade_notifications=True,
            )
        )
        manager.trade_opened("Opened", context={"symbol": "SPY"})

        payload = mock_post.call_args.kwargs["json"]
        self.assertIn("embeds", payload)
        self.assertIsInstance(payload["embeds"], list)

    def test_drawdown_alert_only_fires_at_threshold(self) -> None:
        manager = AlertManager(
            AlertsConfig(
                enabled=True,
                webhook_url="https://example.com/hook",
                drawdown_thresholds=[1.0, 2.0, 3.0],
            )
        )
        manager.send = mock.Mock(return_value=True)

        sent_small = manager.drawdown_alert(0.5)
        sent_large = manager.drawdown_alert(2.1)

        self.assertFalse(sent_small)
        self.assertTrue(sent_large)


if __name__ == "__main__":
    unittest.main()
