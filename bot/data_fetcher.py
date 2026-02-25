"""Historical option-chain downloader and cache manager."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from bot.data_store import ensure_data_dir
from bot.file_security import tighten_file_permissions

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    symbol: str
    trading_date: str
    rows: int
    path: str
    skipped: bool = False


class HistoricalDataFetcher:
    """Fetches and caches daily option-chain snapshots."""

    def __init__(
        self,
        schwab_client,
        data_dir: Path | str = "bot/data",
        max_attempts: int = 5,
        backoff_seconds: float = 1.0,
    ):
        self.schwab = schwab_client
        self.data_dir = ensure_data_dir(data_dir)
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_seconds = max(0.2, float(backoff_seconds))

    def fetch_range(
        self,
        *,
        start: str,
        end: str,
        symbols: list[str],
    ) -> list[FetchResult]:
        """Fetch and cache snapshots for each symbol and business day in range."""
        start_date = _parse_iso_date(start)
        end_date = _parse_iso_date(end)
        if end_date < start_date:
            raise ValueError("end date must be >= start date")
        if not symbols:
            raise ValueError("at least one symbol is required")

        results: list[FetchResult] = []
        for trading_day in _business_days(start_date, end_date):
            for symbol in symbols:
                result = self.fetch_day(symbol=symbol, trading_day=trading_day)
                results.append(result)

        return results

    def fetch_day(self, *, symbol: str, trading_day: date) -> FetchResult:
        symbol_key = symbol.upper().strip()
        output = self._snapshot_path(symbol_key, trading_day)
        if output.exists():
            return FetchResult(
                symbol=symbol_key,
                trading_date=trading_day.isoformat(),
                rows=0,
                path=str(output),
                skipped=True,
            )

        chain = self._fetch_chain_with_retry(symbol_key, trading_day)
        rows = _flatten_chain_rows(symbol_key, trading_day, chain)
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(
                [{"symbol": symbol_key, "snapshot_date": trading_day.isoformat()}]
            )

        self._write_frame(frame, output)
        logger.info(
            "Fetched %s %s snapshot with %d rows -> %s",
            symbol_key,
            trading_day.isoformat(),
            len(frame),
            output,
        )

        return FetchResult(
            symbol=symbol_key,
            trading_date=trading_day.isoformat(),
            rows=int(len(frame)),
            path=str(output),
            skipped=False,
        )

    def _fetch_chain_with_retry(self, symbol: str, trading_day: date) -> dict:
        start_dt = datetime.combine(trading_day, datetime.min.time())
        end_dt = start_dt + timedelta(days=60)

        attempt = 0
        while True:
            attempt += 1
            try:
                return self.schwab.get_option_chain(
                    symbol,
                    strike_count=20,
                    from_date=start_dt,
                    to_date=end_dt,
                )
            except Exception as exc:
                if attempt >= self.max_attempts:
                    raise
                delay = self.backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Option-chain fetch failed for %s (%s) attempt %d/%d: %s. Retrying in %.2fs",
                    symbol,
                    trading_day.isoformat(),
                    attempt,
                    self.max_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)

    def _snapshot_path(self, symbol: str, trading_day: date) -> Path:
        return self.data_dir / f"{symbol}_{trading_day.isoformat()}.parquet.gz"

    @staticmethod
    def _write_frame(frame: pd.DataFrame, path: Path) -> None:
        try:
            frame.to_parquet(path, index=False, compression="gzip")
            tighten_file_permissions(path, label=f"historical snapshot {path}")
            return
        except Exception as exc:
            logger.warning("Parquet write failed for %s (%s). Falling back to CSV.GZ.", path, exc)
            fallback = path.with_suffix(".csv.gz")
            frame.to_csv(fallback, index=False, compression="gzip")
            tighten_file_permissions(fallback, label=f"historical snapshot {fallback}")


def _business_days(start_date: date, end_date: date) -> list[date]:
    return [ts.date() for ts in pd.bdate_range(start=start_date, end=end_date).to_pydatetime()]


def _parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _flatten_chain_rows(symbol: str, trading_day: date, raw_chain: dict) -> list[dict]:
    rows: list[dict] = []
    snapshot_date = trading_day.isoformat()
    for side_key, contract_type in (("callExpDateMap", "CALL"), ("putExpDateMap", "PUT")):
        exp_map = raw_chain.get(side_key, {})
        if not isinstance(exp_map, dict):
            continue
        for exp_key, strikes in exp_map.items():
            if not isinstance(strikes, dict):
                continue
            expiration = str(exp_key).split(":")[0]
            for strike_text, contracts in strikes.items():
                try:
                    strike = float(strike_text)
                except (TypeError, ValueError):
                    strike = 0.0
                for contract in contracts or []:
                    if not isinstance(contract, dict):
                        continue
                    rows.append(
                        {
                            "symbol": symbol,
                            "snapshot_date": snapshot_date,
                            "expiration": expiration,
                            "side": contract_type,
                            "strike": strike,
                            "bid": float(contract.get("bid", 0.0) or 0.0),
                            "ask": float(contract.get("ask", 0.0) or 0.0),
                            "mid": round(
                                (
                                    float(contract.get("bid", 0.0) or 0.0)
                                    + float(contract.get("ask", 0.0) or 0.0)
                                )
                                / 2.0,
                                4,
                            ),
                            "delta": float(contract.get("delta", 0.0) or 0.0),
                            "gamma": float(contract.get("gamma", 0.0) or 0.0),
                            "theta": float(contract.get("theta", 0.0) or 0.0),
                            "vega": float(contract.get("vega", 0.0) or 0.0),
                            "iv": float(contract.get("volatility", 0.0) or 0.0),
                            "volume": int(contract.get("totalVolume", 0) or 0),
                            "open_interest": int(contract.get("openInterest", 0) or 0),
                            "dte": int(contract.get("daysToExpiration", 0) or 0),
                            "underlying_price": float(raw_chain.get("underlyingPrice", 0.0) or 0.0),
                        }
                    )
    return rows
