"""Schwab API client wrapper for authentication, market data, and order execution."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import schwab
from schwab.orders.generic import OrderBuilder
from schwab.orders.options import (
    OptionSymbol,
    bear_call_vertical_close,
    bear_call_vertical_open,
    bull_put_vertical_close,
    bull_put_vertical_open,
    option_buy_to_close_limit,
    option_buy_to_close_market,
    option_sell_to_open_limit,
    option_sell_to_open_market,
)
from schwab.orders.common import (
    ComplexOrderStrategyType,
    Duration,
    OptionInstruction,
    OrderStrategyType,
    OrderType,
    Session,
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
            if token_path.exists():
                tighten_file_permissions(token_path, label="Schwab token file")
            self._client = schwab.auth.client_from_token_file(
                token_path_str,
                self.config.app_key,
                self.config.app_secret,
            )
            if token_path.exists():
                tighten_file_permissions(token_path, label="Schwab token file")
            logger.info("Authenticated via existing token file.")
        except FileNotFoundError:
            logger.info(
                "No token file found. You need to run the initial auth flow."
            )
            logger.info(
                "Run: python3 -m bot.auth to complete browser-based authentication."
            )
            raise RuntimeError(
                "Token file not found. Run `python3 -m bot.auth` first to "
                "complete the OAuth flow."
            )

    @property
    def client(self):
        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._client

    # ── Account Info ──────────────────────────────────────────────────

    def resolve_account_hash(self, require_unique: bool = False) -> Optional[str]:
        """Resolve account hash from linked accounts when not configured."""
        if self._account_hash:
            return self._account_hash

        account_hashes = self._fetch_account_hashes()
        if not account_hashes:
            if require_unique:
                raise RuntimeError(
                    "No linked Schwab accounts found for this token."
                )
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

    def _require_account_hash(self) -> str:
        account_hash = self.resolve_account_hash(require_unique=True)
        if not account_hash:
            raise RuntimeError("Account hash is required but could not be resolved.")
        return account_hash

    def _fetch_account_hashes(self) -> list[str]:
        """Fetch linked account hashes from Schwab."""
        resp = self.client.get_account_numbers()
        resp.raise_for_status()
        return _extract_account_hashes(resp.json())

    def get_account(self) -> dict:
        """Get account details including balances and positions."""
        account_hash = self._require_account_hash()
        resp = self.client.get_account(
            account_hash,
            fields=[schwab.client.Client.Account.Fields.POSITIONS],
        )
        resp.raise_for_status()
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
        return (
            account.get("securitiesAccount", {}).get("positions", [])
        )

    # ── Market Data ───────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> dict:
        """Get a real-time quote for a symbol."""
        resp = self.client.get_quote(symbol)
        resp.raise_for_status()
        data = resp.json()
        return data.get(symbol, data)

    def get_price(self, symbol: str) -> float:
        """Get the last price for a symbol."""
        quote = self.get_quote(symbol)
        ref = quote.get("quote", quote)
        return safe_float(ref.get("lastPrice", ref.get("mark", 0.0)))

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

        resp = self.client.get_option_chain(
            symbol,
            contract_type=ct_map.get(contract_type, ct_map["ALL"]),
            strike_count=strike_count,
            from_date=from_date,
            to_date=to_date,
            include_underlying_quote=True,
        )
        resp.raise_for_status()
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
            chain_data.get("underlyingPrice", chain_data.get("underlying", {}).get("mark", 0))
        )

        result = {
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
        resp = self.client.place_order(account_hash, order_spec)
        resp.raise_for_status()
        # Extract order ID from Location header
        order_id = None
        location = resp.headers.get("Location", "")
        if location:
            order_id = location.split("/")[-1]
        logger.info("Order placed. Order ID: %s", order_id)
        return {"order_id": order_id, "status": "PLACED"}

    def get_order(self, order_id: str) -> dict:
        """Get the status of an order."""
        account_hash = self._require_account_hash()
        resp = self.client.get_order(order_id, account_hash)
        resp.raise_for_status()
        return resp.json()

    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order."""
        account_hash = self._require_account_hash()
        resp = self.client.cancel_order(order_id, account_hash)
        resp.raise_for_status()
        logger.info("Order %s cancelled.", order_id)

    def get_orders(self, days_back: int = 7) -> list:
        """Get recent orders."""
        account_hash = self._require_account_hash()
        from_time = datetime.now() - timedelta(days=days_back)
        to_time = datetime.now()
        resp = self.client.get_orders_for_account(
            account_hash,
            from_entered_datetime=from_time,
            to_entered_datetime=to_time,
        )
        resp.raise_for_status()
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
    """Create a Schwab-compatible option symbol."""
    expiration_key = expiration.split("T", 1)[0].split(":", 1)[0]
    expiration_date = datetime.strptime(expiration_key, "%Y-%m-%d").date()
    strike_price = f"{float(strike):.3f}".rstrip("0").rstrip(".")
    return OptionSymbol(underlying, expiration_date, put_call, strike_price).build()


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
