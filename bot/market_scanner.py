"""Market scanner — dynamically finds the best stocks for options trading.

Instead of relying on a static watchlist, this module scans the entire market
to discover which tickers currently offer the best options trading opportunities.

Ranking criteria:
  1. Options volume & open interest (liquidity = tight fills)
  2. Implied volatility rank (elevated IV = richer premiums)
  3. Bid-ask spread tightness (narrow = less slippage)
  4. Underlying volume (ensures the stock itself is liquid)
  5. Price range suitability (too cheap = penny stock risk, too expensive = capital-heavy)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from bot.config import ScannerConfig

logger = logging.getLogger(__name__)


# ── Large universe of highly-traded options tickers ──────────────────
# Grouped by sector/type so the scanner always has a rich candidate pool.
# The scanner will score and rank these dynamically every cycle.

OPTIONS_UNIVERSE = {
    # Major ETFs (most liquid options in the world)
    "etfs": [
        "SPY", "QQQ", "IWM", "DIA", "EEM", "GLD", "SLV", "TLT",
        "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU", "XLB",
        "XLC", "XLRE", "XBI", "SMH", "ARKK", "HYG", "EFA", "VXX",
        "KWEB", "FXI", "USO", "IBIT", "MSOS",
    ],
    # Mega-cap tech
    "tech": [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA",
        "TSLA", "AMD", "INTC", "AVGO", "CRM", "ORCL", "ADBE",
        "NFLX", "UBER", "SHOP", "SQ", "SNOW", "PLTR", "COIN",
        "MSTR", "NET", "MU", "QCOM", "ARM", "SMCI", "DELL", "PANW",
    ],
    # Finance
    "finance": [
        "JPM", "BAC", "GS", "MS", "WFC", "C", "V", "MA",
        "AXP", "SCHW", "BLK", "COF", "PYPL", "SOFI",
    ],
    # Healthcare / Biotech
    "healthcare": [
        "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "BMY",
        "AMGN", "GILD", "MRNA", "BNTX",
    ],
    # Consumer / Retail
    "consumer": [
        "WMT", "COST", "HD", "NKE", "SBUX", "MCD", "DIS",
        "TGT", "LOW", "LULU", "CMG",
    ],
    # Energy / Industrial
    "energy_industrial": [
        "XOM", "CVX", "OXY", "SLB", "HAL", "BA", "CAT",
        "DE", "GE", "RTX", "LMT", "F", "GM", "RIVN", "LCID",
    ],
    # High-IV / Meme / Volatile (great for premium selling)
    "high_volatility": [
        "GME", "AMC", "RIOT", "MARA", "HOOD", "RBLX",
        "DKNG", "WYNN", "PENN", "SNAP", "PINS", "ROKU",
        "UPST", "AFRM", "PATH", "AI", "IONQ", "RGTI",
    ],
}


def get_full_universe() -> list[str]:
    """Return the complete de-duplicated ticker universe."""
    all_tickers = []
    for group in OPTIONS_UNIVERSE.values():
        all_tickers.extend(group)
    return list(dict.fromkeys(all_tickers))  # Preserve order, remove dupes


@dataclass
class TickerScore:
    """Scoring result for a single ticker's options tradability."""
    symbol: str
    # Raw metrics
    options_volume: int = 0
    options_open_interest: int = 0
    implied_volatility: float = 0.0  # Average IV
    iv_percentile: float = 0.0       # IV rank (0-100)
    bid_ask_quality: float = 0.0     # 0-1, higher = tighter spreads
    underlying_volume: int = 0
    underlying_price: float = 0.0
    avg_option_spread_pct: float = 0.0  # Average bid-ask spread as % of mid
    num_expirations: int = 0
    # Composite score
    score: float = 0.0


class MarketScanner:
    """Scans the market to find the best stocks for options trading."""

    def __init__(self, schwab_client, config: ScannerConfig):
        self.schwab = schwab_client
        self.config = config
        self._last_scan_results: list[TickerScore] = []
        self._last_scan_time: Optional[datetime] = None

    def scan(self) -> list[str]:
        """Run a full market scan and return the top tickers ranked by quality.

        This is the main entry point. Returns a list of ticker symbols
        ordered from best to worst for options trading right now.
        """
        logger.info("=" * 60)
        logger.info("MARKET SCANNER: Starting full market scan")
        logger.info("=" * 60)

        universe = self._build_universe()
        logger.info("Scanning %d tickers...", len(universe))

        scored: list[TickerScore] = []

        for i, symbol in enumerate(universe):
            try:
                ticker_score = self._score_ticker(symbol)
                if ticker_score and ticker_score.score > 0:
                    scored.append(ticker_score)

                if (i + 1) % 25 == 0:
                    logger.info("  Scanned %d/%d tickers...", i + 1, len(universe))

            except Exception as e:
                logger.debug("Error scoring %s: %s", symbol, e)
                continue

        # Sort by composite score descending
        scored.sort(key=lambda t: t.score, reverse=True)

        # Take the top N
        top_n = self.config.max_scan_results
        top_tickers = scored[:top_n]

        self._last_scan_results = scored
        self._last_scan_time = datetime.now()

        logger.info("=" * 60)
        logger.info("SCAN RESULTS: Top %d tickers for options trading", len(top_tickers))
        logger.info("-" * 60)
        for rank, ts in enumerate(top_tickers, 1):
            logger.info(
                "  #%2d  %-6s  Score: %5.1f | IV: %5.1f%% | OptVol: %8d | "
                "OI: %8d | Price: $%7.2f | B/A: %.2f%%",
                rank, ts.symbol, ts.score, ts.implied_volatility,
                ts.options_volume, ts.options_open_interest,
                ts.underlying_price, ts.avg_option_spread_pct * 100,
            )
        logger.info("=" * 60)

        return [ts.symbol for ts in top_tickers]

    def get_cached_results(self) -> list[str]:
        """Return the last scan results if still fresh, otherwise re-scan."""
        if (
            self._last_scan_results
            and self._last_scan_time
            and (datetime.now() - self._last_scan_time).total_seconds() < self.config.cache_seconds
        ):
            logger.info(
                "Using cached scan results (%d tickers, scanned %s ago)",
                len(self._last_scan_results),
                str(datetime.now() - self._last_scan_time).split(".")[0],
            )
            return [ts.symbol for ts in self._last_scan_results[:self.config.max_scan_results]]

        return self.scan()

    def _build_universe(self) -> list[str]:
        """Build the list of tickers to scan.

        Starts with our built-in universe, then optionally enriches
        with Schwab's movers API for trending stocks.
        """
        universe = get_full_universe()

        # Try to add today's market movers from Schwab
        if self.config.include_movers and self.schwab is not None:
            try:
                movers = self._fetch_movers()
                for ticker in movers:
                    if ticker not in universe:
                        universe.append(ticker)
                logger.info("Added %d movers to universe.", len(movers))
            except Exception as e:
                logger.debug("Could not fetch movers: %s", e)

        # Filter by price range if configured
        return universe

    def _fetch_movers(self) -> list[str]:
        """Fetch today's top movers from Schwab API."""
        movers = []
        try:
            # Get movers for major indices
            for index in ["$SPX", "$DJI", "$COMPX"]:
                try:
                    resp = self.schwab.client.get_movers(
                        index,
                        sort_type=self.schwab.client.Movers.SortType.VOLUME,
                        frequency=self.schwab.client.Movers.Frequency.ZERO,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for item in data.get("screeners", []):
                        sym = item.get("symbol", "")
                        if sym and sym.isalpha() and len(sym) <= 5:
                            movers.append(sym)
                except Exception:
                    continue
        except Exception as e:
            logger.debug("Movers fetch failed: %s", e)
        return list(dict.fromkeys(movers))  # Dedupe

    def _score_ticker(self, symbol: str) -> Optional[TickerScore]:
        """Score a single ticker for options trading quality.

        Uses the options chain data to evaluate:
        - Options liquidity (volume + open interest)
        - Implied volatility level
        - Bid-ask spread quality
        - Underlying price and volume
        """
        if self.schwab is None:
            return None

        ts = TickerScore(symbol=symbol)

        # Fetch a lightweight quote for underlying data
        try:
            quote = self.schwab.get_quote(symbol)
            q = quote.get("quote", quote)
            ts.underlying_price = _to_float(q.get("lastPrice", q.get("mark", 0)))
            ts.underlying_volume = _to_int(q.get("totalVolume", 0))
        except Exception:
            return None

        # Price range filter
        if ts.underlying_price < self.config.min_price:
            return None
        if self.config.max_price > 0 and ts.underlying_price > self.config.max_price:
            return None

        # Underlying volume filter
        if ts.underlying_volume < self.config.min_underlying_volume:
            return None

        # Fetch option chain (lightweight — just enough for scoring)
        try:
            from_date = datetime.now() + timedelta(days=14)
            to_date = datetime.now() + timedelta(days=50)
            chain = self.schwab.get_option_chain(
                symbol,
                strike_count=10,  # Fewer strikes = faster
                from_date=from_date,
                to_date=to_date,
            )
        except Exception:
            return None

        # Aggregate options metrics across the chain
        total_volume = 0
        total_oi = 0
        iv_values = []
        spread_ratios = []
        num_expirations = 0

        for date_map_key in ("callExpDateMap", "putExpDateMap"):
            date_map = chain.get(date_map_key, {})
            if not isinstance(date_map, dict):
                continue
            for exp_key, strikes in date_map.items():
                if not isinstance(strikes, dict):
                    continue
                if date_map_key == "callExpDateMap":
                    num_expirations += 1
                for strike_str, contracts in strikes.items():
                    for c in contracts or []:
                        if not isinstance(c, dict):
                            continue
                        vol = _to_int(c.get("totalVolume", 0))
                        oi = _to_int(c.get("openInterest", 0))
                        bid = _to_float(c.get("bid", 0))
                        ask = _to_float(c.get("ask", 0))
                        iv = _to_float(c.get("volatility", 0))

                        total_volume += vol
                        total_oi += oi

                        if iv > 0:
                            iv_values.append(iv)

                        mid = (bid + ask) / 2.0
                        if mid > 0.10 and bid > 0:
                            spread_ratio = (ask - bid) / mid
                            spread_ratios.append(spread_ratio)

        ts.options_volume = total_volume
        ts.options_open_interest = total_oi
        ts.implied_volatility = sum(iv_values) / len(iv_values) if iv_values else 0
        ts.avg_option_spread_pct = (
            sum(spread_ratios) / len(spread_ratios) if spread_ratios else 1.0
        )
        ts.num_expirations = num_expirations

        # ── Compute composite score ──────────────────────────────
        ts.score = self._compute_score(ts)

        return ts

    def _compute_score(self, ts: TickerScore) -> float:
        """Compute a 0-100 composite score for options trading quality."""
        score = 0.0

        # ── Options Volume (25% weight) ──────────────────────────
        # Higher volume = better fills, more liquidity
        vol_score = min(ts.options_volume / 100_000, 1.0)
        score += vol_score * 25

        # ── Open Interest (20% weight) ───────────────────────────
        # Deep open interest = tight markets, easy to enter/exit
        oi_score = min(ts.options_open_interest / 500_000, 1.0)
        score += oi_score * 20

        # ── Implied Volatility (20% weight) ──────────────────────
        # Elevated IV = richer premiums for selling strategies
        # Sweet spot: 25-60% IV (too low = no premium, too high = risky)
        if ts.implied_volatility <= 0:
            iv_score = 0
        elif ts.implied_volatility < 15:
            iv_score = ts.implied_volatility / 15 * 0.3  # Penalize very low IV
        elif ts.implied_volatility <= 60:
            iv_score = 0.3 + (min(ts.implied_volatility, 60) - 15) / 45 * 0.7
        else:
            # High IV: still tradeable but slightly penalized (more risk)
            iv_score = max(0.5, 1.0 - (ts.implied_volatility - 60) / 100)
        score += iv_score * 20

        # ── Bid-Ask Tightness (20% weight) ───────────────────────
        # Tighter spreads = less slippage = more profit kept
        if ts.avg_option_spread_pct <= 0:
            ba_score = 0
        elif ts.avg_option_spread_pct < 0.03:
            ba_score = 1.0  # Extremely tight (ETFs like SPY)
        elif ts.avg_option_spread_pct < 0.10:
            ba_score = 1.0 - (ts.avg_option_spread_pct - 0.03) / 0.07 * 0.4
        elif ts.avg_option_spread_pct < 0.25:
            ba_score = 0.6 - (ts.avg_option_spread_pct - 0.10) / 0.15 * 0.4
        else:
            ba_score = max(0, 0.2 - (ts.avg_option_spread_pct - 0.25) / 0.25 * 0.2)
        score += ba_score * 20

        # ── Underlying Volume (10% weight) ───────────────────────
        # Liquid underlying = reliable pricing
        uvol_score = min(ts.underlying_volume / 10_000_000, 1.0)
        score += uvol_score * 10

        # ── Expiration Availability (5% weight) ──────────────────
        # More expirations = more flexibility in DTE targeting
        exp_score = min(ts.num_expirations / 6, 1.0)
        score += exp_score * 5

        return round(score, 1)

    def get_scan_report(self) -> str:
        """Get a formatted report of the last scan results."""
        if not self._last_scan_results:
            return "No scan results available. Run a scan first."

        lines = [
            "=" * 70,
            "MARKET SCANNER REPORT",
            f"Scanned at: {self._last_scan_time}",
            f"Total scored: {len(self._last_scan_results)}",
            "=" * 70,
            f"{'Rank':>4}  {'Symbol':<6}  {'Score':>5}  {'IV%':>6}  "
            f"{'OptVol':>9}  {'OI':>9}  {'Price':>8}  {'B/A%':>6}",
            "-" * 70,
        ]

        for rank, ts in enumerate(self._last_scan_results[:50], 1):
            lines.append(
                f"{rank:>4}  {ts.symbol:<6}  {ts.score:>5.1f}  "
                f"{ts.implied_volatility:>5.1f}%  "
                f"{ts.options_volume:>9,}  {ts.options_open_interest:>9,}  "
                f"${ts.underlying_price:>7.2f}  "
                f"{ts.avg_option_spread_pct * 100:>5.2f}%"
            )

        lines.append("=" * 70)
        return "\n".join(lines)


def _to_float(value: object, default: float = 0.0) -> float:
    """Safely parse floats from API payloads that may contain nulls/strings."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    """Safely parse integers from API payloads that may contain nulls/strings."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default
