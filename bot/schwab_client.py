"""Schwab API client wrapper for authentication, market data, and order execution."""

import asyncio
import concurrent.futures
import inspect
import logging
import random
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import schwab  # type: ignore[import-untyped]
from schwab.orders.common import (  # type: ignore[import-untyped]
    ComplexOrderStrategyType,
    Duration,
    OptionInstruction,
    OrderStrategyType,
    OrderType,
    Session,
)
from schwab.orders.generic import OrderBuilder  # type: ignore[import-untyped]
from schwab.orders.options import (  # type: ignore[import-untyped]
    bear_call_vertical_close,
    bear_call_vertical_open,
    bull_put_vertical_close,
    bull_put_vertical_open,
    option_buy_to_close_limit,
    option_buy_to_close_market,
    option_sell_to_open_limit,
    option_sell_to_open_market,
)

from bot.config import SchwabConfig
from bot.file_security import tighten_file_permissions, validate_sensitive_file
from bot.number_utils import safe_float, safe_int

logger = logging.getLogger(__name__)
EASTERN_TZ = ZoneInfo("America/New_York")


class SchwabClient:
    """Wrapper around the schwab-py library for trading operations."""

    def __init__(self, config: SchwabConfig):
        self.config = config
        self._client = None
        self._account_hash = config.account_hash
        self._configured_accounts = list(config.accounts or [])
        self._stream_client = None
        self._stream_connected = False
        self._stream_last_error: str = ""
        self._stream_loop: Optional[asyncio.AbstractEventLoop] = None
        self._stream_loop_thread: Optional[threading.Thread] = None
        self._stream_loop_ready = threading.Event()
        self._stream_loop_lock = threading.Lock()
        self._stream_message_future: Optional[concurrent.futures.Future] = None

    def connect(self) -> None:
        """Authenticate and create the API client."""
        if self._client is not None:
            return

        token_path = Path(self.config.token_path).expanduser()
        token_path_str = str(token_path)

        try:
            validate_sensitive_file(
                token_path,
                label="Schwab token file",
                allow_missing=True,
            )
            try:
                if token_path.exists():
                    tighten_file_permissions(token_path, label="Schwab token file")
            except PermissionError:
                pass

            # ── Check token expiry BEFORE connecting ──
            self._check_token_age(token_path)

            self._client = schwab.auth.client_from_token_file(
                token_path_str,
                self.config.app_key,
                self.config.app_secret,
            )
            try:
                if token_path.exists():
                    tighten_file_permissions(token_path, label="Schwab token file")
            except PermissionError:
                pass
            logger.info("Authenticated via existing token file.")

        except FileNotFoundError:
            logger.info("No token file found. You need to run the initial auth flow.")
            logger.info(
                "Run: python3 -m bot.auth to complete browser-based authentication."
            )
            raise RuntimeError(
                "Token file not found. Run `python3 -m bot.auth` first to "
                "complete the OAuth flow."
            )

    @staticmethod
    def _check_token_age(token_path: Path) -> None:
        """Warn if the Schwab refresh token is nearing or past its 7-day expiry."""
        import json as _json

        try:
            raw = token_path.read_text()
            data = _json.loads(raw)
            creation_ts = data.get("creation_timestamp")
            if creation_ts is None:
                return
            created_at = datetime.fromtimestamp(float(creation_ts), tz=EASTERN_TZ)
            age = datetime.now(EASTERN_TZ) - created_at
            days_old = age.total_seconds() / 86400.0

            if days_old >= 7.0:
                msg = (
                    "\n"
                    "╭──────────────────────────────────────────────────────╮\n"
                    "│  ⚠  SCHWAB TOKEN EXPIRED                           │\n"
                    "│                                                     │\n"
                    "│  Your refresh token is {days:.1f} days old (>7 day  │\n"
                    "│  limit). API calls will fail.                       │\n"
                    "│                                                     │\n"
                    "│  Run:  python3 -m bot.auth                          │\n"
                    "╰──────────────────────────────────────────────────────╯\n"
                ).replace("{days:.1f}", f"{days_old:.1f}")
                print(msg)
                logger.error(
                    "Schwab refresh token EXPIRED (%.1f days old). "
                    "Run `python3 -m bot.auth` to re-authenticate.",
                    days_old,
                )
                raise RuntimeError(
                    f"Schwab refresh token EXPIRED ({days_old:.1f} days old). "
                    "Run `python3 -m bot.auth` to grant a new 7-day token."
                )
            elif days_old >= 6.0:
                hours_left = max(0.0, (7.0 - days_old) * 24.0)
                msg = (
                    "\n"
                    "╭──────────────────────────────────────────────────────╮\n"
                    "│  ⚠  SCHWAB TOKEN EXPIRING SOON                     │\n"
                    "│                                                     │\n"
                    "│  Your refresh token expires in ~{hours:.0f} hours.  │\n"
                    "│  Re-authenticate before it expires.                 │\n"
                    "│                                                     │\n"
                    "│  Run:  python3 -m bot.auth                          │\n"
                    "╰──────────────────────────────────────────────────────╯\n"
                ).replace("{hours:.0f}", f"{hours_left:.0f}")
                print(msg)
                logger.warning(
                    "Schwab refresh token expires in ~%.0f hours. "
                    "Run `python3 -m bot.auth` soon.",
                    hours_left,
                )
        except RuntimeError:
            raise
        except Exception as e:
            logger.debug("Minor error reading token age: %s", e)

    @property
    def client(self):
        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._client

    def _ensure_stream_loop(self) -> asyncio.AbstractEventLoop:
        """Ensure stream async operations run on a single dedicated loop."""
        with self._stream_loop_lock:
            if self._stream_loop and self._stream_loop.is_running():
                return self._stream_loop

            self._stream_loop_ready.clear()

            def _runner() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._stream_loop = loop
                self._stream_loop_ready.set()
                loop.run_forever()
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.close()

            self._stream_loop_thread = threading.Thread(
                target=_runner,
                name="SchwabStreamLoop",
                daemon=True,
            )
            self._stream_loop_thread.start()

        if not self._stream_loop_ready.wait(timeout=2.0):
            raise RuntimeError("Timed out starting Schwab streaming event loop.")
        if self._stream_loop is None:
            raise RuntimeError("Schwab streaming event loop failed to initialize.")
        return self._stream_loop

    def _stop_stream_loop(self) -> None:
        with self._stream_loop_lock:
            loop = self._stream_loop
            thread = self._stream_loop_thread
            self._stream_loop = None
            self._stream_loop_thread = None
            self._stream_loop_ready.clear()

        if loop and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _run_coroutine_sync(self, awaitable: Any, *, timeout_seconds: float = 15.0):
        """Run an awaitable on the dedicated stream loop and wait for result."""
        loop = self._ensure_stream_loop()
        future = asyncio.run_coroutine_threadsafe(awaitable, loop)
        try:
            return future.result(timeout=max(0.1, float(timeout_seconds)))
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise RuntimeError(
                "Timed out waiting for Schwab streaming operation."
            ) from exc

    def _run_in_stream_loop(self, func, *args, timeout_seconds: float = 15.0, **kwargs):
        """Run a synchronous callable on the dedicated stream loop thread."""
        if (
            self._stream_loop_thread
            and threading.current_thread() is self._stream_loop_thread
        ):
            return func(*args, **kwargs)

        loop = self._ensure_stream_loop()
        result_future: concurrent.futures.Future = concurrent.futures.Future()

        def _invoke() -> None:
            try:
                result_future.set_result(func(*args, **kwargs))
            except Exception as exc:
                result_future.set_exception(exc)

        loop.call_soon_threadsafe(_invoke)
        try:
            return result_future.result(timeout=max(0.1, float(timeout_seconds)))
        except concurrent.futures.TimeoutError as exc:
            result_future.cancel()
            raise RuntimeError(
                "Timed out waiting for Schwab stream loop callback."
            ) from exc

    async def _stream_message_loop(self) -> None:
        """Continuously process stream messages while connected."""
        while self._stream_connected and self._stream_client is not None:
            handle_message = getattr(self._stream_client, "handle_message", None)
            if not callable(handle_message):
                return
            try:
                result = handle_message()
                if inspect.isawaitable(result):
                    await result
                else:
                    await asyncio.sleep(0.25)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stream_last_error = str(exc)
                self._stream_connected = False
                logger.warning("Schwab streaming message loop stopped: %s", exc)
                return

    def _start_stream_message_loop(self) -> None:
        if not self._stream_client:
            return
        if self._stream_message_future and not self._stream_message_future.done():
            return
        handle_message = getattr(self._stream_client, "handle_message", None)
        if not callable(handle_message):
            return
        loop = self._ensure_stream_loop()
        self._stream_message_future = asyncio.run_coroutine_threadsafe(
            self._stream_message_loop(),
            loop,
        )

    def _stop_stream_message_loop(self) -> None:
        future = self._stream_message_future
        self._stream_message_future = None
        if not future or future.done():
            return
        future.cancel()
        try:
            future.result(timeout=1.0)
        except Exception:
            pass

    def _run_stream_callable(
        self, func, *args, timeout_seconds: float = 15.0, **kwargs
    ):
        result = self._run_in_stream_loop(
            func,
            *args,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        if inspect.isawaitable(result):
            return self._run_coroutine_sync(result, timeout_seconds=timeout_seconds)
        return result

    def start_streaming(self) -> bool:
        """Best-effort start of Schwab streaming; returns False on unsupported setups."""
        if self._stream_connected:
            return True
        try:
            from schwab.streaming import StreamClient  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on schwab-py extras
            self._stream_last_error = f"Streaming import unavailable: {exc}"
            logger.warning(self._stream_last_error)
            self._stream_connected = False
            return False

        try:  # pragma: no cover - runtime integration
            account_hash = self._require_account_hash()
            stream_client = self._run_in_stream_loop(
                StreamClient,
                self.client,
                account_id=account_hash,
                timeout_seconds=10.0,
            )
            self._run_stream_callable(stream_client.login, timeout_seconds=20.0)
            self._stream_client = stream_client
            self._stream_connected = True
            self._stream_last_error = ""
            self._start_stream_message_loop()
            logger.info("Schwab streaming connected.")
            return True
        except Exception as exc:
            self._stream_last_error = str(exc)
            self._stream_connected = False
            self._stream_client = None
            self._stop_stream_message_loop()
            self._stop_stream_loop()
            logger.warning(
                "Schwab streaming unavailable, using polling fallback: %s", exc
            )
            return False

    def stop_streaming(self) -> None:
        if not self._stream_client:
            self._stream_connected = False
            self._stop_stream_message_loop()
            self._stop_stream_loop()
            return
        self._stream_connected = False
        self._stop_stream_message_loop()
        try:  # pragma: no cover - runtime integration
            logout = getattr(self._stream_client, "logout", None)
            if callable(logout):
                self._run_stream_callable(logout, timeout_seconds=10.0)
        except Exception:
            pass
        self._stream_client = None
        self._stop_stream_loop()

    def streaming_connected(self) -> bool:
        return bool(self._stream_connected)

    def stream_quotes(self, symbols: list[str], handler) -> bool:
        """Subscribe to equity quote stream when streaming is available."""
        if not symbols:
            return False
        if not self._stream_connected and not self.start_streaming():
            return False
        try:  # pragma: no cover - runtime integration
            add_handler = getattr(
                self._stream_client, "add_level_one_equity_handler", None
            )
            subscribe = getattr(self._stream_client, "level_one_equity_subs", None)
            if callable(add_handler):
                self._run_stream_callable(add_handler, handler)
            if callable(subscribe):
                self._run_stream_callable(
                    subscribe,
                    [symbol.upper() for symbol in symbols],
                    timeout_seconds=20.0,
                )
            return True
        except Exception as exc:
            self._stream_last_error = str(exc)
            self._stream_connected = False
            logger.warning("Quote streaming subscription failed: %s", exc)
            return False

    def stream_option_level_one(self, option_symbols: list[str], handler) -> bool:
        """Subscribe to options level-1 stream when available."""
        if not option_symbols:
            return False
        if not self._stream_connected and not self.start_streaming():
            return False
        try:  # pragma: no cover - runtime integration
            add_handler = getattr(
                self._stream_client, "add_level_one_option_handler", None
            )
            subscribe = getattr(self._stream_client, "level_one_option_subs", None)
            if callable(add_handler):
                self._run_stream_callable(add_handler, handler)
            if callable(subscribe):
                self._run_stream_callable(
                    subscribe, option_symbols, timeout_seconds=20.0
                )
            return True
        except Exception as exc:
            self._stream_last_error = str(exc)
            self._stream_connected = False
            logger.warning("Option streaming subscription failed: %s", exc)
            return False

    def stream_account_activity(self, handler) -> bool:
        """Subscribe to account activity stream when available."""
        if not self._stream_connected and not self.start_streaming():
            return False
        try:  # pragma: no cover - runtime integration
            add_handler = getattr(
                self._stream_client, "add_account_activity_handler", None
            )
            subscribe = getattr(self._stream_client, "account_activity_sub", None)
            if callable(add_handler):
                self._run_stream_callable(add_handler, handler)
            if callable(subscribe):
                self._run_stream_callable(subscribe, timeout_seconds=20.0)
            return True
        except Exception as exc:
            self._stream_last_error = str(exc)
            self._stream_connected = False
            logger.warning("Account activity streaming subscription failed: %s", exc)
            return False

    def _retry_api_call(
        self,
        func,
        *args,
        max_retries: int = 3,
        base_delay: float = 2.0,
        **kwargs,
    ):
        """Retry Schwab API calls with exponential backoff and jitter."""
        retries = max(0, int(max_retries))
        delay_base = max(0.1, float(base_delay))
        last_exc = None

        for attempt in range(retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
                err_msg = str(exc)
                if "refresh_token_authentication_error" in err_msg:
                    logger.error(
                        "Schwab refresh token authentication failed. "
                        "Token may be expired or invalid."
                    )
                    raise RuntimeError(
                        "Schwab refresh token expired or invalid. "
                        "Run `tradingbot auth` (or `python3 main.py auth`) once to re-authenticate."
                    ) from exc

                if status_code == 401:
                    logger.error("Schwab API returned 401 Unauthorized.")
                    raise RuntimeError(
                        "Schwab API unauthorized (401). Token may be stale or account permissions changed. "
                        "Try `tradingbot auth` (or `python3 main.py auth`)."
                    ) from exc

                if status_code == 400:
                    logger.warning(
                        "Schwab API call failed with 400 Bad Request. Aborting retries for this symbol. (%s)",
                        err_msg,
                    )
                    raise

                if attempt >= retries:
                    raise

                delay = delay_base * (2**attempt) + random.uniform(0.0, 1.0)
                logger.warning(
                    "Schwab API call failed (%s) attempt %d/%d: %s. Retrying in %.2fs",
                    getattr(func, "__name__", "api_call"),
                    attempt + 1,
                    retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)

        if last_exc is not None:
            raise last_exc

    def _request_with_status(self, func, *args, **kwargs):
        """Execute request-like call and retry on failures until status is OK."""

        def _call():
            response = func(*args, **kwargs)
            response.raise_for_status()
            return response

        return self._retry_api_call(_call)

    # ── Account Info ──────────────────────────────────────────────────

    def resolve_account_hash(self, require_unique: bool = False) -> Optional[str]:
        """Resolve account hash from linked accounts when not configured."""
        if self._account_hash:
            return self._account_hash

        if self._configured_accounts:
            preferred = self._configured_accounts[0]
            candidate = str(getattr(preferred, "hash", "")).strip()
            if candidate:
                self._account_hash = candidate
                logger.info(
                    "Using configured Schwab account hash: %s",
                    _mask_hash(self._account_hash),
                )
                return self._account_hash

        account_hashes = self._fetch_account_hashes()
        if not account_hashes:
            if require_unique:
                raise RuntimeError("No linked Schwab accounts found for this token.")
            return None

        if self.config.account_index >= 0:
            index = int(self.config.account_index)
            if index >= len(account_hashes):
                raise RuntimeError(
                    f"SCHWAB_ACCOUNT_INDEX={index} is out of range "
                    f"(0..{len(account_hashes) - 1})."
                )
            self._account_hash = account_hashes[index]
            logger.info(
                "Using linked Schwab account hash by index %d: %s",
                index,
                _mask_hash(self._account_hash),
            )
            return self._account_hash

        if len(account_hashes) == 1:
            self._account_hash = account_hashes[0]
            logger.info(
                "Using linked Schwab account hash: %s",
                _mask_hash(self._account_hash),
            )
            return self._account_hash

        if require_unique:
            hashes_str = ", ".join(_mask_hash(h) for h in account_hashes)
            raise RuntimeError(
                "Multiple linked Schwab accounts detected. "
                f"Set SCHWAB_ACCOUNT_HASH to one of: {hashes_str}"
            )

        return None

    def configured_accounts(self) -> list[dict]:
        """Return configured account metadata from config.yaml."""
        out = []
        for account in self._configured_accounts:
            out.append(
                {
                    "name": str(getattr(account, "name", "")).strip(),
                    "hash": str(getattr(account, "hash", "")).strip(),
                    "risk_profile": str(
                        getattr(account, "risk_profile", "moderate")
                    ).strip(),
                }
            )
        return out

    def select_account(self, account_hash: str) -> None:
        """Switch the active account hash used for all account/order endpoints."""
        account_hash = str(account_hash).strip()
        if not account_hash:
            raise ValueError("account_hash is required")
        self._account_hash = account_hash

    def _require_account_hash(self) -> str:
        account_hash = self.resolve_account_hash(require_unique=True)
        if not account_hash:
            raise RuntimeError("Account hash is required but could not be resolved.")
        return account_hash

    def _fetch_account_hashes(self) -> list[str]:
        """Fetch linked account hashes from Schwab."""
        resp = self._request_with_status(self.client.get_account_numbers)
        return _extract_account_hashes(resp.json())

    def get_account(self) -> dict:
        """Get account details including balances and positions."""
        account_hash = self._require_account_hash()
        resp = self._request_with_status(
            self.client.get_account,
            account_hash,
            fields=[schwab.client.Client.Account.Fields.POSITIONS],
        )
        return resp.json()

    def get_account_balance(self) -> float:
        """Get total account liquidation value."""
        account = self.get_account()
        balance = (
            account.get("securitiesAccount", {})
            .get("currentBalances", {})
            .get("liquidationValue", 0.0)
        )
        return safe_float(balance)

    def get_positions(self) -> list:
        """Get all current positions."""
        account = self.get_account()
        return account.get("securitiesAccount", {}).get("positions", [])

    # ── Market Data ───────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> dict:
        """Get a real-time quote for a symbol."""
        resp = self._request_with_status(self.client.get_quote, symbol)
        data = resp.json()
        return data.get(symbol, data)

    def get_price(self, symbol: str) -> float:
        """Get the last price for a symbol."""
        quote = self.get_quote(symbol)
        ref = quote.get("quote", quote)
        return safe_float(ref.get("lastPrice", ref.get("mark", 0.0)))

    def get_price_history(self, symbol: str, days: int = 120) -> list[dict]:
        """Get daily OHLCV bars for the given symbol."""
        lookback_days = max(30, int(days))
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=lookback_days * 2)

        response = None
        if hasattr(self.client, "get_price_history_every_day"):
            response = self._request_with_status(
                self.client.get_price_history_every_day,
                symbol,
                start_datetime=start_dt,
                end_datetime=end_dt,
                need_extended_hours_data=False,
                need_previous_close=True,
            )
        elif hasattr(self.client, "get_price_history"):
            response = self._request_with_status(
                self.client.get_price_history,
                symbol,
                period_type=self.client.PriceHistory.PeriodType.YEAR,
                period=self.client.PriceHistory.Period.ONE_YEAR,
                frequency_type=self.client.PriceHistory.FrequencyType.DAILY,
                frequency=self.client.PriceHistory.Frequency.DAILY,
                need_extended_hours_data=False,
                need_previous_close=True,
            )
        else:
            raise RuntimeError(
                "Schwab client does not expose a price-history endpoint."
            )

        payload = response.json()
        candles = payload.get("candles", [])
        out: list[dict] = []
        for candle in candles[-lookback_days:]:
            if not isinstance(candle, dict):
                continue
            out.append(
                {
                    "datetime": candle.get("datetime"),
                    "open": safe_float(candle.get("open", 0.0)),
                    "high": safe_float(candle.get("high", 0.0)),
                    "low": safe_float(candle.get("low", 0.0)),
                    "close": safe_float(candle.get("close", 0.0)),
                    "volume": safe_int(candle.get("volume", 0)),
                }
            )
        return out

    def get_option_chain(
        self,
        symbol: str,
        contract_type: str = "ALL",
        strike_count: int = 20,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> dict:
        """Fetch the options chain for a symbol.

        Args:
            symbol: Underlying ticker.
            contract_type: "CALL", "PUT", or "ALL".
            strike_count: Number of strikes above/below ATM to include.
            from_date: Earliest expiration date to include.
            to_date: Latest expiration date to include.
        """
        ct_map = {
            "ALL": self.client.Options.ContractType.ALL,
            "CALL": self.client.Options.ContractType.CALL,
            "PUT": self.client.Options.ContractType.PUT,
        }

        if from_date is None:
            from_date = datetime.now() + timedelta(days=14)
        if to_date is None:
            to_date = datetime.now() + timedelta(days=60)

        resp = self._request_with_status(
            self.client.get_option_chain,
            symbol,
            contract_type=ct_map.get(contract_type, ct_map["ALL"]),
            strike_count=strike_count,
            from_date=from_date,
            to_date=to_date,
            include_underlying_quote=True,
        )
        return resp.json()

    # ── Options Chain Parsing ─────────────────────────────────────────

    @staticmethod
    def parse_option_chain(chain_data: dict) -> dict:
        """Parse raw option chain into a structured format.

        Returns dict with:
            underlying_price: float
            calls: {expiration_date: [option_dicts]}
            puts: {expiration_date: [option_dicts]}
        """
        underlying_price = safe_float(
            chain_data.get(
                "underlyingPrice", chain_data.get("underlying", {}).get("mark", 0)
            )
        )

        result: dict[str, Any] = {
            "underlying_price": underlying_price,
            "calls": {},
            "puts": {},
        }

        for exp_key, strikes in chain_data.get("callExpDateMap", {}).items():
            if not isinstance(strikes, dict):
                continue
            exp_date = exp_key.split(":")[0]
            options = []
            for strike_str, contracts in strikes.items():
                strike = safe_float(strike_str)
                for c in contracts or []:
                    if isinstance(c, dict):
                        options.append(_normalize_contract(c, strike, exp_date))
            result["calls"][exp_date] = sorted(options, key=lambda o: o["strike"])

        for exp_key, strikes in chain_data.get("putExpDateMap", {}).items():
            if not isinstance(strikes, dict):
                continue
            exp_date = exp_key.split(":")[0]
            options = []
            for strike_str, contracts in strikes.items():
                strike = safe_float(strike_str)
                for c in contracts or []:
                    if isinstance(c, dict):
                        options.append(_normalize_contract(c, strike, exp_date))
            result["puts"][exp_date] = sorted(options, key=lambda o: o["strike"])

        return result

    # ── Order Placement ───────────────────────────────────────────────

    def place_order(self, order_spec) -> dict:
        """Place an order and return the response."""
        account_hash = self._require_account_hash()
        resp = self._request_with_status(
            self.client.place_order, account_hash, order_spec
        )
        # Extract order ID from Location header
        order_id = None
        location = resp.headers.get("Location", "")
        if location:
            order_id = location.split("/")[-1]
        logger.info("Order placed. Order ID: %s", order_id)
        return {"order_id": order_id, "status": "PLACED"}

    def place_order_with_ladder(
        self,
        *,
        order_factory,
        midpoint_price: float,
        spread_width: float,
        side: str,
        step_timeout_seconds: int,
        step_timeouts: Optional[list[int]] = None,
        max_attempts: int = 4,
        shifts: Optional[list[float]] = None,
        total_timeout_seconds: Optional[int] = None,
    ) -> dict:
        """Place an order with a midpoint-to-natural price ladder."""
        if shifts is None or not shifts:
            shifts = [0.0, 0.10, 0.25, 0.40]
        shifts = shifts[:max_attempts]
        if not shifts:
            shifts = [0.0]

        midpoint = max(0.01, float(midpoint_price))
        spread = max(0.01, float(spread_width))
        last_result: dict[str, Any] = {"status": "REJECTED"}
        last_price = midpoint
        total_timeout = (
            max(5, int(total_timeout_seconds))
            if total_timeout_seconds is not None
            else None
        )
        start_clock = time.time()

        for attempt_index, shift in enumerate(shifts, start=1):
            timeout_from_steps = None
            if isinstance(step_timeouts, list) and step_timeouts:
                if attempt_index - 1 < len(step_timeouts):
                    timeout_from_steps = max(5, int(step_timeouts[attempt_index - 1]))
                else:
                    timeout_from_steps = max(5, int(step_timeouts[-1]))
            if total_timeout is not None:
                elapsed = max(0.0, time.time() - start_clock)
                remaining_total = total_timeout - elapsed
                if remaining_total <= 0:
                    break
                if timeout_from_steps is not None:
                    wait_timeout = min(timeout_from_steps, max(5, int(remaining_total)))
                elif attempt_index < len(shifts):
                    wait_timeout = min(
                        max(5, int(step_timeout_seconds)), max(5, int(remaining_total))
                    )
                else:
                    # Final attempt receives remaining time budget.
                    wait_timeout = max(5, int(remaining_total))
            else:
                wait_timeout = (
                    timeout_from_steps
                    if timeout_from_steps is not None
                    else max(5, int(step_timeout_seconds))
                )

            candidate_price = _ladder_price(
                midpoint=midpoint,
                spread=spread,
                shift=float(shift),
                side=side,
            )
            if attempt_index > 1:
                previous_order_id = str(last_result.get("order_id", "unknown"))
                logger.info(
                    "Order %s price improved from %.2f to %.2f, attempt %d/%d",
                    previous_order_id,
                    last_price,
                    candidate_price,
                    attempt_index,
                    len(shifts),
                )
            last_price = candidate_price

            order_spec = order_factory(candidate_price)
            placed = self.place_order(order_spec)
            order_id = str(placed.get("order_id", "")).strip()
            last_result = {
                **placed,
                "attempt": attempt_index,
                "requested_price": candidate_price,
                "midpoint_price": midpoint,
            }

            if not order_id:
                return last_result

            terminal = self._wait_for_terminal_status(
                order_id,
                timeout_seconds=wait_timeout,
            )
            if terminal.get("poll_error"):
                return last_result
            status = str(terminal.get("status", "")).upper()
            if status == "FILLED":
                fill_price = _extract_fill_price(terminal) or candidate_price
                fill_improvement_vs_mid = _fill_improvement_vs_mid(
                    midpoint=midpoint,
                    fill_price=fill_price,
                    side=side,
                )
                last_result.update(
                    {
                        "status": "FILLED",
                        "fill_price": fill_price,
                        "fill_improvement_vs_mid": round(fill_improvement_vs_mid, 4),
                        "order_id": order_id,
                    }
                )
                return last_result

            if status == "REJECTED":
                last_result.update({"status": "REJECTED", "order_id": order_id})
                return last_result

            if status in {"CANCELED", "EXPIRED"}:
                last_result.update(
                    {"status": status, "order_id": order_id}
                )
                continue

            try:
                self.cancel_order(order_id)
                last_result.update({"status": "CANCELED", "order_id": order_id})
            except Exception as exc:
                logger.debug("Failed to cancel stale order %s: %s", order_id, exc)

        return last_result

    def _wait_for_terminal_status(self, order_id: str, timeout_seconds: int) -> dict:
        """Poll order status until terminal status or timeout."""
        timeout_seconds = max(5, int(timeout_seconds))
        start = time.time()
        latest: dict[str, Any] = {"status": "UNKNOWN"}

        while (time.time() - start) < timeout_seconds:
            try:
                latest = self.get_order(order_id)
            except Exception as exc:
                logger.debug("Order poll failed for %s: %s", order_id, exc)
                latest["poll_error"] = True
                return latest

            status = str(latest.get("status", "")).upper()
            if status in {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}:
                return latest
            time.sleep(5)

        return latest

    def get_order(self, order_id: str) -> dict:
        """Get the status of an order."""
        account_hash = self._require_account_hash()
        resp = self._request_with_status(self.client.get_order, order_id, account_hash)
        return resp.json()

    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order."""
        account_hash = self._require_account_hash()
        self._request_with_status(self.client.cancel_order, order_id, account_hash)
        logger.info("Order %s cancelled.", order_id)

    def get_orders(self, days_back: int = 7) -> list:
        """Get recent orders."""
        account_hash = self._require_account_hash()
        from_time = datetime.now() - timedelta(days=days_back)
        to_time = datetime.now()
        resp = self._request_with_status(
            self.client.get_orders_for_account,
            account_hash,
            from_entered_datetime=from_time,
            to_entered_datetime=to_time,
        )
        return resp.json()

    def is_equity_market_open(self, now: Optional[datetime] = None) -> bool:
        """Return whether the equity market is open right now."""
        now_dt = now or datetime.now(EASTERN_TZ)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=EASTERN_TZ)

        resp = self.client.get_market_hours(
            [
                self.client.MarketHours.Market.EQUITY,
                self.client.MarketHours.Market.OPTION,
            ],
            date=now_dt.date(),
        )
        resp.raise_for_status()
        return _market_open_from_hours_payload(resp.json(), now_dt)

    # ── Spread Order Builders ─────────────────────────────────────────

    def build_bull_put_spread(
        self,
        symbol: str,
        expiration: str,
        short_strike: float,
        long_strike: float,
        quantity: int = 1,
        price: Optional[float] = None,
    ):
        """Build a bull put spread order (sell put, buy lower put)."""
        if price is None or price <= 0:
            raise ValueError("Bull put spread requires a positive net credit.")
        short_sym = _make_option_symbol(symbol, expiration, "P", short_strike)
        long_sym = _make_option_symbol(symbol, expiration, "P", long_strike)
        order = bull_put_vertical_open(long_sym, short_sym, quantity, price)
        order.set_duration(Duration.DAY)
        order.set_session(Session.NORMAL)
        return order

    def build_bear_call_spread(
        self,
        symbol: str,
        expiration: str,
        short_strike: float,
        long_strike: float,
        quantity: int = 1,
        price: Optional[float] = None,
    ):
        """Build a bear call spread order (sell call, buy higher call)."""
        if price is None or price <= 0:
            raise ValueError("Bear call spread requires a positive net credit.")
        short_sym = _make_option_symbol(symbol, expiration, "C", short_strike)
        long_sym = _make_option_symbol(symbol, expiration, "C", long_strike)
        order = bear_call_vertical_open(short_sym, long_sym, quantity, price)
        order.set_duration(Duration.DAY)
        order.set_session(Session.NORMAL)
        return order

    def build_bull_put_spread_close(
        self,
        symbol: str,
        expiration: str,
        short_strike: float,
        long_strike: float,
        quantity: int = 1,
        price: Optional[float] = None,
    ):
        """Build a bull put spread close order (buy back short, sell long)."""
        if price is None or price <= 0:
            raise ValueError("Bull put spread close requires a positive net debit.")
        short_sym = _make_option_symbol(symbol, expiration, "P", short_strike)
        long_sym = _make_option_symbol(symbol, expiration, "P", long_strike)
        order = bull_put_vertical_close(long_sym, short_sym, quantity, price)
        order.set_duration(Duration.DAY)
        order.set_session(Session.NORMAL)
        return order

    def build_bear_call_spread_close(
        self,
        symbol: str,
        expiration: str,
        short_strike: float,
        long_strike: float,
        quantity: int = 1,
        price: Optional[float] = None,
    ):
        """Build a bear call spread close order (buy back short, sell long)."""
        if price is None or price <= 0:
            raise ValueError("Bear call spread close requires a positive net debit.")
        short_sym = _make_option_symbol(symbol, expiration, "C", short_strike)
        long_sym = _make_option_symbol(symbol, expiration, "C", long_strike)
        order = bear_call_vertical_close(short_sym, long_sym, quantity, price)
        order.set_duration(Duration.DAY)
        order.set_session(Session.NORMAL)
        return order

    def build_covered_call_open(
        self,
        symbol: str,
        expiration: str,
        short_strike: float,
        quantity: int = 1,
        price: Optional[float] = None,
    ):
        """Build a covered-call entry order (sell call to open)."""
        option_sym = _make_option_symbol(symbol, expiration, "C", short_strike)
        if price is None or price <= 0:
            order = option_sell_to_open_market(option_sym, quantity)
        else:
            order = option_sell_to_open_limit(option_sym, quantity, price)
        order.set_duration(Duration.DAY)
        order.set_session(Session.NORMAL)
        return order

    def build_covered_call_close(
        self,
        symbol: str,
        expiration: str,
        short_strike: float,
        quantity: int = 1,
        price: Optional[float] = None,
    ):
        """Build a covered-call exit order (buy call to close)."""
        option_sym = _make_option_symbol(symbol, expiration, "C", short_strike)
        if price is None or price <= 0:
            order = option_buy_to_close_market(option_sym, quantity)
        else:
            order = option_buy_to_close_limit(option_sym, quantity, price)
        order.set_duration(Duration.DAY)
        order.set_session(Session.NORMAL)
        return order

    def build_long_option_open(
        self,
        *,
        symbol: str,
        expiration: str,
        contract_type: str,
        strike: float,
        quantity: int = 1,
        price: Optional[float] = None,
    ):
        """Build a single-leg long-option open order (debit)."""
        option_sym = _make_option_symbol(
            symbol, expiration, contract_type.upper(), strike
        )
        order = (
            OrderBuilder()
            .set_order_strategy_type(OrderStrategyType.SINGLE)
            .set_order_type(
                OrderType.MARKET if price is None or price <= 0 else OrderType.LIMIT
            )
            .set_quantity(max(1, int(quantity)))
            .add_option_leg(
                OptionInstruction.BUY_TO_OPEN, option_sym, max(1, int(quantity))
            )
        )
        if price is not None and price > 0:
            order.set_price(price)
        order.set_duration(Duration.DAY)
        order.set_session(Session.NORMAL)
        return order

    def build_debit_spread_open(
        self,
        *,
        symbol: str,
        expiration: str,
        contract_type: str,
        long_strike: float,
        short_strike: float,
        quantity: int = 1,
        price: Optional[float] = None,
    ):
        """Build a vertical debit spread order using generic legs."""
        ctype = contract_type.upper()
        if ctype not in {"P", "C"}:
            raise ValueError("contract_type must be 'P' or 'C'")
        if price is None or price <= 0:
            raise ValueError("Debit spread open requires a positive net debit.")

        long_sym = _make_option_symbol(symbol, expiration, ctype, long_strike)
        short_sym = _make_option_symbol(symbol, expiration, ctype, short_strike)

        order = (
            OrderBuilder()
            .set_order_type(OrderType.NET_DEBIT)
            .set_order_strategy_type(OrderStrategyType.SINGLE)
            .set_complex_order_strategy_type(ComplexOrderStrategyType.VERTICAL)
            .set_quantity(max(1, int(quantity)))
            .set_price(price)
            .add_option_leg(
                OptionInstruction.BUY_TO_OPEN, long_sym, max(1, int(quantity))
            )
            .add_option_leg(
                OptionInstruction.SELL_TO_OPEN, short_sym, max(1, int(quantity))
            )
        )
        order.set_duration(Duration.DAY)
        order.set_session(Session.NORMAL)
        return order

    def build_iron_condor(
        self,
        symbol: str,
        expiration: str,
        put_long_strike: float,
        put_short_strike: float,
        call_short_strike: float,
        call_long_strike: float,
        quantity: int = 1,
        price: Optional[float] = None,
    ):
        """Build an iron-condor entry order (net credit)."""
        if price is None or price <= 0:
            raise ValueError("Iron condor open requires a positive net credit.")

        put_long_sym = _make_option_symbol(symbol, expiration, "P", put_long_strike)
        put_short_sym = _make_option_symbol(symbol, expiration, "P", put_short_strike)
        call_short_sym = _make_option_symbol(symbol, expiration, "C", call_short_strike)
        call_long_sym = _make_option_symbol(symbol, expiration, "C", call_long_strike)

        order = (
            OrderBuilder()
            .set_order_type(OrderType.NET_CREDIT)
            .set_complex_order_strategy_type(ComplexOrderStrategyType.IRON_CONDOR)
            .set_order_strategy_type(OrderStrategyType.SINGLE)
            .set_quantity(quantity)
            .set_price(price)
            .add_option_leg(OptionInstruction.BUY_TO_OPEN, put_long_sym, quantity)
            .add_option_leg(OptionInstruction.SELL_TO_OPEN, put_short_sym, quantity)
            .add_option_leg(OptionInstruction.SELL_TO_OPEN, call_short_sym, quantity)
            .add_option_leg(OptionInstruction.BUY_TO_OPEN, call_long_sym, quantity)
        )
        order.set_duration(Duration.DAY)
        order.set_session(Session.NORMAL)
        return order

    def build_iron_condor_close(
        self,
        symbol: str,
        expiration: str,
        put_long_strike: float,
        put_short_strike: float,
        call_short_strike: float,
        call_long_strike: float,
        quantity: int = 1,
        price: Optional[float] = None,
    ):
        """Build an iron-condor exit order (net debit)."""
        if price is None or price <= 0:
            raise ValueError("Iron condor close requires a positive net debit.")

        put_long_sym = _make_option_symbol(symbol, expiration, "P", put_long_strike)
        put_short_sym = _make_option_symbol(symbol, expiration, "P", put_short_strike)
        call_short_sym = _make_option_symbol(symbol, expiration, "C", call_short_strike)
        call_long_sym = _make_option_symbol(symbol, expiration, "C", call_long_strike)

        order = (
            OrderBuilder()
            .set_order_type(OrderType.NET_DEBIT)
            .set_complex_order_strategy_type(ComplexOrderStrategyType.IRON_CONDOR)
            .set_order_strategy_type(OrderStrategyType.SINGLE)
            .set_quantity(quantity)
            .set_price(price)
            .add_option_leg(OptionInstruction.SELL_TO_CLOSE, put_long_sym, quantity)
            .add_option_leg(OptionInstruction.BUY_TO_CLOSE, put_short_sym, quantity)
            .add_option_leg(OptionInstruction.BUY_TO_CLOSE, call_short_sym, quantity)
            .add_option_leg(OptionInstruction.SELL_TO_CLOSE, call_long_sym, quantity)
        )
        order.set_duration(Duration.DAY)
        order.set_session(Session.NORMAL)
        return order


# ── Helpers ──────────────────────────────────────────────────────────


def _normalize_contract(contract: dict, strike: float, expiration_date: str) -> dict:
    """Normalize a raw options contract from the API."""
    bid = safe_float(contract.get("bid"))
    ask = safe_float(contract.get("ask"))
    return {
        "symbol": contract.get("symbol", ""),
        "description": contract.get("description", ""),
        "strike": strike,
        "expiration": _extract_expiration_date(
            contract.get("expirationDate"), expiration_date
        ),
        "dte": safe_int(contract.get("daysToExpiration", 0)),
        "bid": bid,
        "ask": ask,
        "mid": round((bid + ask) / 2, 2),
        "last": safe_float(contract.get("last", 0)),
        "volume": safe_int(contract.get("totalVolume", 0)),
        "open_interest": safe_int(contract.get("openInterest", 0)),
        "delta": safe_float(contract.get("delta", 0)),
        "gamma": safe_float(contract.get("gamma", 0)),
        "theta": safe_float(contract.get("theta", 0)),
        "vega": safe_float(contract.get("vega", 0)),
        "iv": safe_float(contract.get("volatility", 0)),
        "in_the_money": contract.get("inTheMoney", False),
        "contract_type": contract.get("putCall", ""),
    }


def _extract_expiration_date(raw_expiration: object, fallback: str) -> str:
    """Normalize an expiration value to YYYY-MM-DD."""
    if isinstance(raw_expiration, str):
        exp = raw_expiration.strip()
        if exp:
            if "T" in exp:
                exp = exp.split("T", 1)[0]
            if ":" in exp:
                exp = exp.split(":", 1)[0]
            return exp

    return fallback


def _make_option_symbol(
    underlying: str, expiration: str, put_call: str, strike: float
) -> str:
    """Create a Schwab-compatible option symbol in stable underscore format."""
    expiration_key = expiration.split("T", 1)[0].split(":", 1)[0]
    expiration_date = datetime.strptime(expiration_key, "%Y-%m-%d").date()
    strike_text = f"{float(strike):.3f}".rstrip("0").rstrip(".")
    return (
        f"{underlying.upper()}_{expiration_date:%m%d%y}{put_call.upper()}{strike_text}"
    )


def _extract_account_hashes(raw_accounts: object) -> list[str]:
    """Extract account hash values from get_account_numbers payload."""
    hashes: list[str] = []
    if isinstance(raw_accounts, list):
        for item in raw_accounts:
            if not isinstance(item, dict):
                continue
            hash_value = item.get("hashValue") or item.get("hash")
            if hash_value:
                hashes.append(str(hash_value))
    elif isinstance(raw_accounts, dict):
        accounts = raw_accounts.get("accounts", [])
        if isinstance(accounts, list):
            hashes.extend(_extract_account_hashes(accounts))
    return list(dict.fromkeys(hashes))


def _mask_hash(account_hash: str) -> str:
    """Mask account hash for logs."""
    if len(account_hash) <= 8:
        return "***"
    return f"{account_hash[:4]}...{account_hash[-4:]}"


def _market_open_from_hours_payload(payload: object, now_dt: datetime) -> bool:
    """Parse market-hours payload and determine if regular equity session is open."""
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=EASTERN_TZ)

    if not isinstance(payload, dict):
        return False

    market_blocks = []
    for market_key in ("equity", "option"):
        market_value = payload.get(market_key)
        if isinstance(market_value, dict):
            market_blocks.append(market_value)

    if not market_blocks:
        # Some payload variants may be nested one level deeper.
        for value in payload.values():
            if isinstance(value, dict):
                market_blocks.append(value)

    for market_block in market_blocks:
        for product in market_block.values():
            if not isinstance(product, dict):
                continue
            if not product.get("isOpen", False):
                continue

            session_hours = product.get("sessionHours", {})
            if not isinstance(session_hours, dict):
                return True

            regular = session_hours.get("regularMarket", [])
            if not isinstance(regular, list) or not regular:
                return True

            for session in regular:
                if not isinstance(session, dict):
                    continue
                start = _parse_market_time(session.get("start"))
                end = _parse_market_time(session.get("end"))
                if start is None or end is None:
                    continue
                if start <= now_dt <= end:
                    return True

    return False


def _parse_market_time(raw_time: object) -> Optional[datetime]:
    """Parse Schwab market-hours timestamps with compact timezone offsets."""
    if raw_time is None:
        return None

    value = str(raw_time).strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    if len(value) >= 5 and value[-5] in {"+", "-"} and value[-3] != ":":
        value = f"{value[:-2]}:{value[-2:]}"

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=EASTERN_TZ)
    return parsed


def _ladder_price(*, midpoint: float, spread: float, shift: float, side: str) -> float:
    """Shift midpoint toward natural price depending on credit/debit side."""
    side_key = str(side).lower().strip()
    if side_key == "credit":
        value = midpoint - (shift * spread)
    else:
        value = midpoint + (shift * spread)
    return round(max(0.01, value), 2)


def _fill_improvement_vs_mid(*, midpoint: float, fill_price: float, side: str) -> float:
    """Return positive numbers when fill is better than midpoint."""
    side_key = str(side).lower().strip()
    if side_key == "credit":
        return float(fill_price) - float(midpoint)
    return float(midpoint) - float(fill_price)


def _extract_fill_price(order: dict) -> Optional[float]:
    """Best-effort per-contract fill price extraction from order payload."""
    activities = order.get("orderActivityCollection", [])
    if not isinstance(activities, list):
        return None
    prices: list[float] = []
    for activity in activities:
        if not isinstance(activity, dict):
            continue
        for leg in activity.get("executionLegs", []) or []:
            if not isinstance(leg, dict):
                continue
            price = safe_float(leg.get("price"), 0.0)
            if price > 0:
                prices.append(price)
    if not prices:
        direct = safe_float(order.get("price"), 0.0)
        return direct if direct > 0 else None
    return round(float(sum(prices) / len(prices)), 4)
