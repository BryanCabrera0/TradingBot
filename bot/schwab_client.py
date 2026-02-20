"""Schwab API client wrapper for authentication, market data, and order execution."""

import logging
from datetime import datetime, timedelta
from typing import Optional

import schwab
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
from schwab.orders.common import Duration, Session

from bot.config import SchwabConfig

logger = logging.getLogger(__name__)


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

        try:
            self._client = schwab.auth.client_from_token_file(
                self.config.token_path,
                self.config.app_key,
                self.config.app_secret,
            )
            logger.info("Authenticated via existing token file.")
        except FileNotFoundError:
            logger.info(
                "No token file found. You need to run the initial auth flow."
            )
            logger.info(
                "Run: python -m bot.auth to complete browser-based authentication."
            )
            raise RuntimeError(
                "Token file not found. Run `python -m bot.auth` first to "
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
        return float(balance)

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
        return float(ref.get("lastPrice", ref.get("mark", 0.0)))

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
        underlying_price = float(
            chain_data.get("underlyingPrice", chain_data.get("underlying", {}).get("mark", 0))
        )

        result = {
            "underlying_price": underlying_price,
            "calls": {},
            "puts": {},
        }

        for exp_key, strikes in chain_data.get("callExpDateMap", {}).items():
            exp_date = exp_key.split(":")[0]
            options = []
            for strike_str, contracts in strikes.items():
                for c in contracts:
                    options.append(_normalize_contract(c, float(strike_str), exp_date))
            result["calls"][exp_date] = sorted(options, key=lambda o: o["strike"])

        for exp_key, strikes in chain_data.get("putExpDateMap", {}).items():
            exp_date = exp_key.split(":")[0]
            options = []
            for strike_str, contracts in strikes.items():
                for c in contracts:
                    options.append(_normalize_contract(c, float(strike_str), exp_date))
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
        """Build an iron condor order.

        Note: schwab-py does not currently expose a first-class helper for
        four-leg iron condors in this codebase's dependency range.
        """
        raise NotImplementedError(
            "Iron condor live order builder is not supported by installed schwab-py."
        )


# ── Helpers ──────────────────────────────────────────────────────────

def _normalize_contract(contract: dict, strike: float, expiration_date: str) -> dict:
    """Normalize a raw options contract from the API."""
    return {
        "symbol": contract.get("symbol", ""),
        "description": contract.get("description", ""),
        "strike": strike,
        "expiration": _extract_expiration_date(
            contract.get("expirationDate"), expiration_date
        ),
        "dte": int(contract.get("daysToExpiration", 0)),
        "bid": float(contract.get("bid", 0)),
        "ask": float(contract.get("ask", 0)),
        "mid": round(
            (float(contract.get("bid", 0)) + float(contract.get("ask", 0))) / 2, 2
        ),
        "last": float(contract.get("last", 0)),
        "volume": int(contract.get("totalVolume", 0)),
        "open_interest": int(contract.get("openInterest", 0)),
        "delta": float(contract.get("delta", 0)),
        "gamma": float(contract.get("gamma", 0)),
        "theta": float(contract.get("theta", 0)),
        "vega": float(contract.get("vega", 0)),
        "iv": float(contract.get("volatility", 0)),
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
