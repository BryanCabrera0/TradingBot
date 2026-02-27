import asyncio
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from bot.config import SchwabConfig
from bot.schwab_client import (
    SchwabClient,
    _ladder_price,
    _make_option_symbol,
    _market_open_from_hours_payload,
)


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

    def test_parse_option_chain_handles_null_numeric_fields(self) -> None:
        chain_data = {
            "underlyingPrice": None,
            "callExpDateMap": {
                "2026-03-20:30": {
                    "100.0": [
                        {
                            "symbol": "SPY  260320C00100000",
                            "expirationDate": "2026-03-20T21:00:00+00:00",
                            "daysToExpiration": None,
                            "bid": None,
                            "ask": "1.25",
                            "delta": None,
                            "volatility": None,
                            "totalVolume": None,
                            "openInterest": None,
                        }
                    ]
                }
            },
        }

        parsed = SchwabClient.parse_option_chain(chain_data)
        contract = parsed["calls"]["2026-03-20"][0]

        self.assertEqual(parsed["underlying_price"], 0.0)
        self.assertEqual(contract["dte"], 0)
        self.assertEqual(contract["bid"], 0.0)
        self.assertEqual(contract["ask"], 1.25)
        self.assertEqual(contract["mid"], 0.62)
        self.assertEqual(contract["delta"], 0.0)
        self.assertEqual(contract["volume"], 0)
        self.assertEqual(contract["open_interest"], 0)

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

    def test_connect_rejects_symlink_token_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            real_token = Path(tmp_dir) / "token-real.json"
            real_token.write_text("{}", encoding="utf-8")
            symlink_token = Path(tmp_dir) / "token-link.json"
            symlink_token.symlink_to(real_token)

            client = SchwabClient(
                SchwabConfig(
                    token_path=str(symlink_token),
                    app_key="test-app",
                    app_secret="test-secret",
                )
            )

            with self.assertRaises(RuntimeError):
                client.connect()

    def test_market_hours_payload_reports_open_during_regular_session(self) -> None:
        payload = {
            "equity": {
                "EQ": {
                    "isOpen": True,
                    "sessionHours": {
                        "regularMarket": [
                            {
                                "start": "2026-02-23T14:30:00+0000",
                                "end": "2026-02-23T21:00:00+0000",
                            }
                        ]
                    },
                }
            }
        }
        now_dt = datetime.fromisoformat("2026-02-23T15:00:00+00:00")

        self.assertTrue(_market_open_from_hours_payload(payload, now_dt))

    def test_market_hours_payload_reports_closed_when_is_open_false(self) -> None:
        payload = {
            "equity": {
                "EQ": {
                    "isOpen": False,
                    "sessionHours": {"regularMarket": []},
                }
            }
        }
        now_dt = datetime.fromisoformat("2026-12-25T15:00:00+00:00")

        self.assertFalse(_market_open_from_hours_payload(payload, now_dt))

    def test_build_iron_condor_open_and_close_orders(self) -> None:
        client = SchwabClient(SchwabConfig())

        open_order = client.build_iron_condor(
            symbol="SPY",
            expiration="2026-03-20",
            put_long_strike=90,
            put_short_strike=95,
            call_short_strike=110,
            call_long_strike=115,
            quantity=1,
            price=1.2,
        ).build()
        close_order = client.build_iron_condor_close(
            symbol="SPY",
            expiration="2026-03-20",
            put_long_strike=90,
            put_short_strike=95,
            call_short_strike=110,
            call_long_strike=115,
            quantity=1,
            price=0.6,
        ).build()

        self.assertEqual(open_order["orderType"], "NET_CREDIT")
        self.assertEqual(close_order["orderType"], "NET_DEBIT")
        self.assertEqual(open_order["complexOrderStrategyType"], "IRON_CONDOR")
        self.assertEqual(close_order["complexOrderStrategyType"], "IRON_CONDOR")
        self.assertEqual(len(open_order["orderLegCollection"]), 4)
        self.assertEqual(len(close_order["orderLegCollection"]), 4)

    def test_ladder_price_moves_credit_and_debit_sides_correctly(self) -> None:
        self.assertEqual(_ladder_price(midpoint=1.0, spread=0.4, shift=0.25, side="credit"), 0.9)
        self.assertEqual(_ladder_price(midpoint=1.0, spread=0.4, shift=0.25, side="debit"), 1.1)

    def test_ladder_respects_total_timeout_budget(self) -> None:
        client = SchwabClient(SchwabConfig())
        client.place_order = mock.Mock(
            side_effect=[
                {"order_id": "A1", "status": "PLACED"},
                {"order_id": "A2", "status": "PLACED"},
                {"order_id": "A3", "status": "PLACED"},
            ]
        )
        timeouts = []

        def _fake_wait(order_id: str, timeout_seconds: int) -> dict:
            timeouts.append(timeout_seconds)
            return {"status": "WORKING"}

        client._wait_for_terminal_status = mock.Mock(side_effect=_fake_wait)
        client.cancel_order = mock.Mock()
        order_factory = mock.Mock(return_value={"type": "LIMIT"})

        with mock.patch(
            "bot.schwab_client.time.time",
            side_effect=[0.0, 0.0, 90.0, 180.0],
        ):
            result = client.place_order_with_ladder(
                order_factory=order_factory,
                midpoint_price=1.0,
                spread_width=0.4,
                side="credit",
                step_timeout_seconds=90,
                max_attempts=3,
                shifts=[0.0, 0.25, 0.5],
                total_timeout_seconds=300,
            )

        self.assertEqual(timeouts, [90, 90, 120])
        self.assertEqual(result["status"], "CANCELED")
        self.assertEqual(client.cancel_order.call_count, 3)

    def test_ladder_uses_step_timeouts_and_fills_on_third_attempt(self) -> None:
        client = SchwabClient(SchwabConfig())
        client.place_order = mock.Mock(
            side_effect=[
                {"order_id": "B1", "status": "PLACED"},
                {"order_id": "B2", "status": "PLACED"},
                {"order_id": "B3", "status": "PLACED"},
            ]
        )
        wait_calls = []

        def _fake_wait(order_id: str, timeout_seconds: int) -> dict:
            wait_calls.append((order_id, timeout_seconds))
            if order_id == "B3":
                return {
                    "status": "FILLED",
                    "orderActivityCollection": [
                        {"executionLegs": [{"price": 0.9}]}
                    ],
                }
            return {"status": "WORKING"}

        client._wait_for_terminal_status = mock.Mock(side_effect=_fake_wait)
        client.cancel_order = mock.Mock()
        order_factory = mock.Mock(return_value={"type": "LIMIT"})

        result = client.place_order_with_ladder(
            order_factory=order_factory,
            midpoint_price=1.0,
            spread_width=0.4,
            side="credit",
            step_timeout_seconds=90,
            step_timeouts=[45, 45, 30],
            max_attempts=4,
            shifts=[0.0, 0.10, 0.25, 0.40],
            total_timeout_seconds=300,
        )

        self.assertEqual(wait_calls, [("B1", 45), ("B2", 45), ("B3", 30)])
        self.assertEqual(result["status"], "FILLED")
        self.assertEqual(result["attempt"], 3)
        self.assertAlmostEqual(float(result["fill_price"]), 0.9, places=4)
        self.assertIn("fill_improvement_vs_mid", result)

    def test_start_streaming_and_quote_subscribe_support_async_methods(self) -> None:
        class _AsyncStreamClient:
            def __init__(self, _client, account_id: str):
                self.account_id = account_id
                self.logged_in = False
                self.handler = None
                self.subscribed_symbols = []

            async def login(self):
                self.logged_in = True

            def add_level_one_equity_handler(self, handler):
                self.handler = handler

            async def level_one_equity_subs(self, symbols):
                self.subscribed_symbols = list(symbols)

        client = SchwabClient(SchwabConfig())
        client._client = mock.Mock()
        client._account_hash = "hash123"
        handler = mock.Mock()

        with mock.patch("schwab.streaming.StreamClient", _AsyncStreamClient):
            self.assertTrue(client.start_streaming())
            self.assertTrue(client.stream_quotes(["spy"], handler))

        self.assertTrue(client._stream_client.logged_in)
        self.assertEqual(client._stream_client.account_id, "hash123")
        self.assertEqual(client._stream_client.subscribed_symbols, ["SPY"])
        self.assertIs(client._stream_client.handler, handler)

    def test_stop_streaming_supports_async_logout(self) -> None:
        class _AsyncLogoutStream:
            def __init__(self):
                self.logged_out = False

            async def logout(self):
                self.logged_out = True

        stream = _AsyncLogoutStream()
        client = SchwabClient(SchwabConfig())
        client._stream_client = stream
        client._stream_connected = True

        client.stop_streaming()

        self.assertTrue(stream.logged_out)
        self.assertFalse(client.streaming_connected())
        self.assertIsNone(client._stream_client)

    def test_stream_async_calls_run_on_single_event_loop(self) -> None:
        stream_ref = {"value": None}

        class _AsyncStreamClient:
            def __init__(self, _client, account_id: str):
                self.account_id = account_id
                self.loop_ids = []
                self.subscribed_symbols = []
                stream_ref["value"] = self

            async def login(self):
                self.loop_ids.append(id(asyncio.get_running_loop()))

            def add_level_one_equity_handler(self, _handler):
                return None

            async def level_one_equity_subs(self, symbols):
                self.loop_ids.append(id(asyncio.get_running_loop()))
                self.subscribed_symbols = list(symbols)

            async def logout(self):
                self.loop_ids.append(id(asyncio.get_running_loop()))

        client = SchwabClient(SchwabConfig())
        client._client = mock.Mock()
        client._account_hash = "hash456"

        with mock.patch("schwab.streaming.StreamClient", _AsyncStreamClient):
            self.assertTrue(client.start_streaming())
            self.assertTrue(client.stream_quotes(["spy"], mock.Mock()))
            client.stop_streaming()

        stream = stream_ref["value"]
        self.assertIsNotNone(stream)
        self.assertEqual(stream.subscribed_symbols, ["SPY"])
        self.assertEqual(len(set(stream.loop_ids)), 1)


if __name__ == "__main__":
    unittest.main()
