"""Historical options strategy backtesting engine."""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from bot.config import BotConfig, load_config
from bot.data_store import dump_json, ensure_data_dir
from bot.risk_manager import RiskManager
from bot.strategies.base import TradeSignal
from bot.strategies.covered_calls import CoveredCallStrategy
from bot.strategies.credit_spreads import CreditSpreadStrategy
from bot.strategies.iron_condors import IronCondorStrategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    report_path: str
    report: dict


class Backtester:
    """Replay historical option-chain snapshots through live strategy logic."""

    def __init__(
        self,
        config: Optional[BotConfig] = None,
        *,
        data_dir: Path | str = "bot/data",
        initial_balance: float = 100_000.0,
    ):
        self.config = config or load_config()
        self.data_dir = ensure_data_dir(data_dir)
        self.initial_balance = float(initial_balance)
        self.cash_balance = float(initial_balance)
        self.closed_trades: list[dict] = []
        self.positions: list[dict] = []
        self.equity_curve: list[dict] = []
        self._daily_realized: dict[str, float] = {}

        self.risk_manager = RiskManager(self.config.risk)
        self.strategies = self._build_strategies()

    def run(self, *, start: str, end: str) -> BacktestResult:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
        if end_date < start_date:
            raise ValueError("end date must be >= start date")

        logger.info("Backtest starting: %s -> %s", start_date, end_date)
        for day in _business_days(start_date, end_date):
            self._run_day(day)

        report = self._build_report(start_date, end_date)
        report_path = self._write_report(report)
        return BacktestResult(report_path=report_path, report=report)

    def _build_strategies(self) -> list:
        strategies = []
        if self.config.credit_spreads.enabled:
            strategies.append(CreditSpreadStrategy(vars(self.config.credit_spreads)))
        if self.config.iron_condors.enabled:
            strategies.append(IronCondorStrategy(vars(self.config.iron_condors)))
        if self.config.covered_calls.enabled:
            strategies.append(CoveredCallStrategy(vars(self.config.covered_calls)))
        return strategies

    def _run_day(self, trading_day: date) -> None:
        snapshots = self._load_snapshots_for_day(trading_day)
        if not snapshots:
            return

        self._mark_positions(trading_day, snapshots)
        self._process_exits(trading_day)

        daily_realized = self._daily_realized.get(trading_day.isoformat(), 0.0)
        self.risk_manager.update_portfolio(
            account_balance=self._current_equity(),
            open_positions=self.positions,
            daily_pnl=daily_realized,
        )
        self._process_entries(snapshots)
        self._record_equity_point(trading_day)

    def _process_entries(self, snapshots: dict[str, dict]) -> None:
        for symbol, chain_data in snapshots.items():
            underlying = float(chain_data.get("underlying_price", 0.0))
            if underlying <= 0:
                continue

            all_signals: list[TradeSignal] = []
            for strategy in self.strategies:
                all_signals.extend(strategy.scan_for_entries(symbol, chain_data, underlying))
            all_signals.sort(
                key=lambda signal: signal.analysis.score if signal.analysis else 0.0,
                reverse=True,
            )

            for signal in all_signals[:3]:
                if not self.risk_manager.can_open_more_positions():
                    return
                if signal.action != "open" or signal.analysis is None:
                    continue
                signal.quantity = self.risk_manager.calculate_position_size(signal)
                approved, _ = self.risk_manager.approve_trade(signal)
                if not approved:
                    continue
                self._open_position(signal)
                self.risk_manager.register_open_position(
                    symbol=signal.symbol,
                    max_loss_per_contract=self.risk_manager.effective_max_loss_per_contract(signal),
                    quantity=signal.quantity,
                    strategy=signal.strategy,
                )

    def _process_exits(self, trading_day: date) -> None:
        if not self.positions:
            return

        signals: list[TradeSignal] = []
        for strategy in self.strategies:
            signals.extend(strategy.check_exits(self.positions, market_client=None))

        for signal in signals:
            if signal.action != "close":
                continue
            self._close_position(signal.position_id, signal.reason, trading_day)

    def _open_position(self, signal: TradeSignal) -> None:
        analysis = signal.analysis
        if analysis is None:
            return
        pid = f"bt_{uuid.uuid4().hex[:10]}"
        entry_credit = max(0.0, float(analysis.credit))
        quantity = max(1, int(signal.quantity))
        self.cash_balance += entry_credit * quantity * 100.0

        position = {
            "position_id": pid,
            "strategy": signal.strategy,
            "symbol": signal.symbol,
            "entry_credit": round(entry_credit, 4),
            "current_value": round(entry_credit, 4),
            "quantity": quantity,
            "max_loss": float(analysis.max_loss),
            "open_date": datetime.utcnow().isoformat(),
            "expiration": analysis.expiration,
            "dte_remaining": int(analysis.dte),
            "details": {
                "expiration": analysis.expiration,
                "dte": analysis.dte,
                "short_strike": analysis.short_strike,
                "long_strike": analysis.long_strike,
                "put_short_strike": analysis.put_short_strike,
                "put_long_strike": analysis.put_long_strike,
                "call_short_strike": analysis.call_short_strike,
                "call_long_strike": analysis.call_long_strike,
                "score": analysis.score,
                "probability_of_profit": analysis.probability_of_profit,
            },
            "status": "open",
        }
        self.positions.append(position)

    def _close_position(self, position_id: Optional[str], reason: str, trading_day: date) -> None:
        if not position_id:
            return
        index = next(
            (i for i, position in enumerate(self.positions) if position.get("position_id") == position_id),
            None,
        )
        if index is None:
            return

        position = self.positions.pop(index)
        close_value = float(position.get("current_value", 0.0))
        quantity = max(1, int(position.get("quantity", 1)))
        self.cash_balance -= close_value * quantity * 100.0

        pnl = (float(position.get("entry_credit", 0.0)) - close_value) * quantity * 100.0
        opened_at = _parse_datetime(position.get("open_date"))
        days_in_trade = (
            max(0, (trading_day - opened_at.date()).days) if opened_at else max(0, int(position.get("dte_remaining", 0)))
        )
        closed = {
            **position,
            "close_date": trading_day.isoformat(),
            "close_value": round(close_value, 4),
            "pnl": round(pnl, 2),
            "reason": reason,
            "days_in_trade": days_in_trade,
            "status": "closed",
        }
        self.closed_trades.append(closed)
        self._daily_realized[trading_day.isoformat()] = (
            self._daily_realized.get(trading_day.isoformat(), 0.0) + pnl
        )

    def _mark_positions(self, trading_day: date, snapshots: dict[str, dict]) -> None:
        for position in self.positions:
            symbol = position.get("symbol", "")
            chain_data = snapshots.get(symbol)
            if not chain_data:
                continue
            estimate = _estimate_position_value(position, chain_data)
            if estimate is not None:
                position["current_value"] = round(estimate, 4)

            expiration = str(position.get("expiration", ""))
            try:
                exp_date = datetime.strptime(expiration.split("T")[0], "%Y-%m-%d").date()
                position["dte_remaining"] = (exp_date - trading_day).days
            except Exception:
                continue

    def _current_equity(self) -> float:
        mark_to_close = sum(
            float(position.get("current_value", 0.0)) * max(1, int(position.get("quantity", 1))) * 100.0
            for position in self.positions
        )
        return self.cash_balance - mark_to_close

    def _record_equity_point(self, trading_day: date) -> None:
        self.equity_curve.append(
            {
                "date": trading_day.isoformat(),
                "equity": round(self._current_equity(), 2),
                "cash": round(self.cash_balance, 2),
                "open_positions": len(self.positions),
            }
        )

    def _load_snapshots_for_day(self, trading_day: date) -> dict[str, dict]:
        iso = trading_day.isoformat()
        snapshots: dict[str, dict] = {}

        parquet_paths = sorted(self.data_dir.glob(f"*_{iso}.parquet.gz"))
        csv_paths = sorted(self.data_dir.glob(f"*_{iso}.parquet.csv.gz"))
        for path in parquet_paths + csv_paths:
            symbol = path.name.split("_", 1)[0].upper()
            try:
                if path.suffixes[-2:] == [".parquet", ".gz"]:
                    frame = pd.read_parquet(path)
                else:
                    frame = pd.read_csv(path)
            except Exception:
                continue
            parsed = _chain_from_snapshot(frame)
            if parsed:
                snapshots[symbol] = parsed
        return snapshots

    def _build_report(self, start_date: date, end_date: date) -> dict:
        equities = [point["equity"] for point in self.equity_curve]
        total_return = (
            ((equities[-1] - self.initial_balance) / self.initial_balance)
            if equities
            else 0.0
        )
        daily_returns = _daily_returns(equities)
        sharpe = _sharpe_ratio(daily_returns)
        sortino = _sortino_ratio(daily_returns)
        max_drawdown = _max_drawdown(equities)

        wins = [trade["pnl"] for trade in self.closed_trades if trade.get("pnl", 0.0) > 0]
        losses = [trade["pnl"] for trade in self.closed_trades if trade.get("pnl", 0.0) < 0]
        avg_days = (
            float(np.mean([trade.get("days_in_trade", 0) for trade in self.closed_trades]))
            if self.closed_trades
            else 0.0
        )

        monthly = _monthly_returns(self.equity_curve, self.initial_balance)
        report = {
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "initial_balance": round(self.initial_balance, 2),
            "ending_equity": round(equities[-1], 2) if equities else round(self.initial_balance, 2),
            "total_return": round(total_return, 6),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_drawdown, 6),
            "win_rate": round((len(wins) / max(1, len(self.closed_trades))), 4),
            "average_winner": round(float(np.mean(wins)) if wins else 0.0, 2),
            "average_loser": round(float(np.mean(losses)) if losses else 0.0, 2),
            "profit_factor": round((sum(wins) / abs(sum(losses))) if losses else float("inf"), 4),
            "average_days_in_trade": round(avg_days, 2),
            "monthly_returns": monthly,
            "closed_trades": len(self.closed_trades),
            "open_positions": len(self.positions),
            "equity_curve": self.equity_curve,
        }
        return report

    def _write_report(self, report: dict) -> str:
        logs_dir = ensure_data_dir("logs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = logs_dir / f"backtest_{timestamp}.json"
        dump_json(path, report)
        logger.info("Backtest report written: %s", path)
        return str(path)


def _chain_from_snapshot(frame: pd.DataFrame) -> dict:
    calls: dict[str, list[dict]] = {}
    puts: dict[str, list[dict]] = {}
    if frame.empty:
        return {}

    underlying_price = float(frame["underlying_price"].fillna(0.0).iloc[0]) if "underlying_price" in frame.columns else 0.0
    for _, row in frame.iterrows():
        expiration = str(row.get("expiration", ""))
        if not expiration:
            continue
        opt = {
            "symbol": str(row.get("symbol", "")),
            "expiration": expiration,
            "dte": int(row.get("dte", 0) or 0),
            "strike": float(row.get("strike", 0.0) or 0.0),
            "bid": float(row.get("bid", 0.0) or 0.0),
            "ask": float(row.get("ask", 0.0) or 0.0),
            "mid": float(row.get("mid", 0.0) or 0.0),
            "delta": float(row.get("delta", 0.0) or 0.0),
            "gamma": float(row.get("gamma", 0.0) or 0.0),
            "theta": float(row.get("theta", 0.0) or 0.0),
            "vega": float(row.get("vega", 0.0) or 0.0),
            "iv": float(row.get("iv", 0.0) or 0.0),
            "volume": int(row.get("volume", 0) or 0),
            "open_interest": int(row.get("open_interest", 0) or 0),
        }
        side = str(row.get("side", "")).upper()
        target = calls if side == "CALL" else puts
        target.setdefault(expiration, []).append(opt)

    for exp in calls:
        calls[exp].sort(key=lambda item: item.get("strike", 0.0))
    for exp in puts:
        puts[exp].sort(key=lambda item: item.get("strike", 0.0))

    return {"underlying_price": underlying_price, "calls": calls, "puts": puts}


def _estimate_position_value(position: dict, chain_data: dict) -> Optional[float]:
    details = position.get("details", {})
    strategy = str(position.get("strategy", ""))
    expiration = str(details.get("expiration", position.get("expiration", ""))).split("T")[0]
    if not expiration:
        return None

    calls = chain_data.get("calls", {}).get(expiration, [])
    puts = chain_data.get("puts", {}).get(expiration, [])

    def _mid(options: list[dict], strike: float) -> Optional[float]:
        for option in options:
            if abs(float(option.get("strike", 0.0)) - float(strike)) < 0.01:
                return float(option.get("mid", 0.0))
        return None

    if strategy == "bull_put_spread":
        short_mid = _mid(puts, float(details.get("short_strike", 0.0)))
        long_mid = _mid(puts, float(details.get("long_strike", 0.0)))
        if short_mid is None or long_mid is None:
            return None
        return max(short_mid - long_mid, 0.0)

    if strategy == "bear_call_spread":
        short_mid = _mid(calls, float(details.get("short_strike", 0.0)))
        long_mid = _mid(calls, float(details.get("long_strike", 0.0)))
        if short_mid is None or long_mid is None:
            return None
        return max(short_mid - long_mid, 0.0)

    if strategy == "iron_condor":
        put_short = _mid(puts, float(details.get("put_short_strike", 0.0)))
        put_long = _mid(puts, float(details.get("put_long_strike", 0.0)))
        call_short = _mid(calls, float(details.get("call_short_strike", 0.0)))
        call_long = _mid(calls, float(details.get("call_long_strike", 0.0)))
        if None in (put_short, put_long, call_short, call_long):
            return None
        return max((put_short - put_long) + (call_short - call_long), 0.0)

    if strategy == "covered_call":
        short_mid = _mid(calls, float(details.get("short_strike", 0.0)))
        if short_mid is None:
            return None
        return max(short_mid, 0.0)

    return None


def _business_days(start_date: date, end_date: date) -> list[date]:
    return [ts.date() for ts in pd.bdate_range(start=start_date, end=end_date).to_pydatetime()]


def _parse_datetime(value: object) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _daily_returns(equity: list[float]) -> list[float]:
    if len(equity) < 2:
        return []
    out: list[float] = []
    for prev, current in zip(equity[:-1], equity[1:]):
        if prev <= 0:
            continue
        out.append((current / prev) - 1.0)
    return out


def _sharpe_ratio(daily_returns: list[float]) -> float:
    if len(daily_returns) < 2:
        return 0.0
    mean = float(np.mean(daily_returns))
    stdev = float(np.std(daily_returns, ddof=1))
    if stdev <= 0:
        return 0.0
    return (mean / stdev) * math.sqrt(252)


def _sortino_ratio(daily_returns: list[float]) -> float:
    if len(daily_returns) < 2:
        return 0.0
    mean = float(np.mean(daily_returns))
    downside = [value for value in daily_returns if value < 0]
    if not downside:
        return 0.0
    downside_stdev = float(np.std(downside, ddof=1)) if len(downside) > 1 else abs(downside[0])
    if downside_stdev <= 0:
        return 0.0
    return (mean / downside_stdev) * math.sqrt(252)


def _max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            drawdown = min(drawdown, (value - peak) / peak)
    return abs(drawdown)


def _monthly_returns(equity_curve: list[dict], initial_balance: float) -> dict[str, float]:
    if not equity_curve:
        return {}
    by_month: dict[str, dict[str, float]] = {}
    for point in equity_curve:
        month = str(point.get("date", ""))[:7]
        eq = float(point.get("equity", initial_balance))
        month_data = by_month.setdefault(month, {"start": eq, "end": eq})
        month_data["end"] = eq

    out: dict[str, float] = {}
    for month, payload in sorted(by_month.items()):
        start = payload["start"]
        end = payload["end"]
        if start <= 0:
            out[month] = 0.0
        else:
            out[month] = round((end / start) - 1.0, 6)
    return out
