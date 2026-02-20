import unittest
from unittest import mock

from bot.config import SchwabConfig
from bot.schwab_client import SchwabClient, _make_option_symbol


class SchwabClientParserTests(unittest.TestCase):
    def test_parse_option_chain_normalizes_expiration_dates(self) -> None:
        chain_data = {
            "underlyingPrice": 100.0,
            "callExpDateMap": {
                "2026-03-20:30": {
                    "100.0": [
                        {
                            "symbol": "SPY  260320C00100000",
                            "expirationDate": "2026-03-20T21:00:00+00:00",
                            "daysToExpiration": 30,
                            "bid": 1.1,
                            "ask": 1.3,
                        }
                    ]
                }
            },
            "putExpDateMap": {
                "2026-03-20:30": {
                    "95.0": [
                        {
                            "symbol": "SPY  260320P00095000",
                            "expirationDate": "2026-03-20T21:00:00+00:00",
                            "daysToExpiration": 30,
                            "bid": 0.9,
                            "ask": 1.0,
                        }
                    ]
                }
            },
        }

        parsed = SchwabClient.parse_option_chain(chain_data)
        call_contract = parsed["calls"]["2026-03-20"][0]

        self.assertEqual(call_contract["expiration"], "2026-03-20")

    def test_make_option_symbol_accepts_iso_expiration(self) -> None:
        symbol = _make_option_symbol("SPY", "2026-03-20T21:00:00+00:00", "C", 500.0)

        self.assertEqual(symbol, "SPY_032026C500")

    def test_resolve_account_hash_from_single_linked_account(self) -> None:
        client = SchwabClient(SchwabConfig())
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = [{"hashValue": "abc123xyz789"}]
        client._client = mock.Mock()
        client._client.get_account_numbers.return_value = response

        account_hash = client.resolve_account_hash(require_unique=True)

        self.assertEqual(account_hash, "abc123xyz789")

    def test_resolve_account_hash_raises_on_multiple_accounts(self) -> None:
        client = SchwabClient(SchwabConfig())
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = [
            {"hashValue": "firsthash001"},
            {"hashValue": "secondhash2"},
        ]
        client._client = mock.Mock()
        client._client.get_account_numbers.return_value = response

        with self.assertRaises(RuntimeError):
            client.resolve_account_hash(require_unique=True)


if __name__ == "__main__":
    unittest.main()
