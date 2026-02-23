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


if __name__ == "__main__":
    unittest.main()
