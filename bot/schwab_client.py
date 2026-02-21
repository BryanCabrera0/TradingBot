"""Schwab API client wrapper for authentication, market data, and order execution."""

import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import schwab
from schwab.orders.options import (
    bear_call_spread,
    bull_put_spread,
    iron_condor,
    option_symbol,
)
from schwab.orders.common import Duration, Session

from bot.config import SchwabConfig

logger = logging.getLogger(__name__)


class OrderAction(Enum):
    OPEN = "open"
    CLOSE = "close"


class SchwabClient:
    """Wrapper around the schwab-py library for trading operations."""

    def __init__(self, config: SchwabConfig):
        self.config = config
        self._client = None
        self._account_hash = config.account_hash

    def connect(self) -> None:
        """Authenticate and create the API client."""
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

    def get_account(self) -> dict:
        """Get account details including balances and positions."""
        resp = self.client.get_account(
            self._account_hash,
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
                    options.append(_normalize_contract(c, float(strike_str)))
            result["calls"][exp_date] = sorted(options, key=lambda o: o["strike"])

        for exp_key, strikes in chain_data.get("putExpDateMap", {}).items():
            exp_date = exp_key.split(":")[0]
            options = []
            for strike_str, contracts in strikes.items():
                for c in contracts:
                    options.append(_normalize_contract(c, float(strike_str)))
            result["puts"][exp_date] = sorted(options, key=lambda o: o["strike"])

        return result

    # ── Order Placement ───────────────────────────────────────────────

    def place_order(self, order_spec) -> dict:
        """Place an order and return the response."""
        resp = self.client.place_order(self._account_hash, order_spec)
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
        resp = self.client.get_order(order_id, self._account_hash)
        resp.raise_for_status()
        return resp.json()

    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order."""
        resp = self.client.cancel_order(order_id, self._account_hash)
        resp.raise_for_status()
        logger.info("Order %s cancelled.", order_id)

    def get_orders(self, days_back: int = 7) -> list:
        """Get recent orders."""
        from_time = datetime.now() - timedelta(days=days_back)
        to_time = datetime.now()
        resp = self.client.get_orders_for_account(
            self._account_hash,
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
        short_sym = _make_option_symbol(symbol, expiration, "P", short_strike)
        long_sym = _make_option_symbol(symbol, expiration, "P", long_strike)
        order = bull_put_spread(short_sym, long_sym, quantity, price)
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
        short_sym = _make_option_symbol(symbol, expiration, "C", short_strike)
        long_sym = _make_option_symbol(symbol, expiration, "C", long_strike)
        order = bear_call_spread(short_sym, long_sym, quantity, price)
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
        """Build an iron condor order."""
        put_long_sym = _make_option_symbol(symbol, expiration, "P", put_long_strike)
        put_short_sym = _make_option_symbol(symbol, expiration, "P", put_short_strike)
        call_short_sym = _make_option_symbol(symbol, expiration, "C", call_short_strike)
        call_long_sym = _make_option_symbol(symbol, expiration, "C", call_long_strike)
        order = iron_condor(
            put_long_sym, put_short_sym,
            call_short_sym, call_long_sym,
            quantity, price,
        )
        order.set_duration(Duration.DAY)
        order.set_session(Session.NORMAL)
        return order


# ── Helpers ──────────────────────────────────────────────────────────

def _normalize_contract(contract: dict, strike: float) -> dict:
    """Normalize a raw options contract from the API."""
    return {
        "symbol": contract.get("symbol", ""),
        "description": contract.get("description", ""),
        "strike": strike,
        "expiration": contract.get("expirationDate", ""),
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


def _make_option_symbol(
    underlying: str, expiration: str, put_call: str, strike: float
) -> str:
    """Create an OCC option symbol.

    Format: SYMBOL  YYMMDD[C/P]SSSSSSSS
    Example: SPY   250321C00500000
    """
    # Parse expiration date (YYYY-MM-DD)
    exp = datetime.strptime(expiration, "%Y-%m-%d")
    date_part = exp.strftime("%y%m%d")
    # Strike is multiplied by 1000 and zero-padded to 8 digits
    strike_part = f"{int(strike * 1000):08d}"
    # Underlying is left-padded with spaces to 6 chars
    sym_part = f"{underlying:<6}"
    return f"{sym_part}{date_part}{put_call}{strike_part}"
