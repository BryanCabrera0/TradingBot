"""Historical options strategy backtesting engine."""

from __future__ import annotations

import logging
import math
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from bot.config import BotConfig, load_config
from bot.data_store import dump_json, ensure_data_dir
from bot.regime_detector import LOW_VOL_GRIND, MarketRegimeDetector
from bot.risk_manager import RiskManager
from bot.strategies.base import TradeSignal
from bot.strategies.calendar_spreads import CalendarSpreadStrategy
from bot.strategies.covered_calls import CoveredCallStrategy
from bot.strategies.credit_spreads import CreditSpreadStrategy
from bot.strategies.iron_condors import IronCondorStrategy
from bot.strategies.naked_puts import NakedPutStrategy
from bot.vol_surface import VolSurfaceAnalyzer

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
        commission_per_contract: float = 0.65,
        base_slippage_pct: float = 0.02,
    ):
        self.config = config or load_config()
        self.data_dir = ensure_data_dir(data_dir)
        self.initial_balance = float(initial_balance)
        self.cash_balance = float(initial_balance)
        self.closed_trades: list[dict] = []
        self.positions: list[dict] = []
        self.equity_curve: list[dict] = []
        self._daily_realized: dict[str, float] = {}
        self.commission_per_contract = max(0.0, float(commission_per_contract))
        self.base_slippage_pct = max(0.0, float(base_slippage_pct))
        self.total_fees = 0.0
        self.total_slippage = 0.0

        self.risk_manager = RiskManager(self.config.risk)
        self.strategies = self._build_strategies()
        self.vol_surface_analyzer = VolSurfaceAnalyzer() if self.config.vol_surface.enabled else None
        self.regime_detector = MarketRegimeDetector(config=vars(self.config.regime))
        self._spy_history: list[float] = []
        self._strategy_loss_streaks: dict[str, int] = defaultdict(int)
        self._strategy_pause_until: dict[str, date] = {}

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
        if self.config.naked_puts.enabled:
            strategies.append(NakedPutStrategy(vars(self.config.naked_puts)))
        if self.config.calendar_spreads.enabled:
            strategies.append(CalendarSpreadStrategy(vars(self.config.calendar_spreads)))
        return strategies

    def _run_day(self, trading_day: date) -> None:
        snapshots = self._load_snapshots_for_day(trading_day)
        if not snapshots:
            return
        regime_state = self._estimate_regime(trading_day, snapshots)

        self._mark_positions(trading_day, snapshots)
        self._process_exits(trading_day)

        daily_realized = self._daily_realized.get(trading_day.isoformat(), 0.0)
        self.risk_manager.update_portfolio(
            account_balance=self._current_equity(),
            open_positions=self.positions,
            daily_pnl=daily_realized,
        )
        self._process_entries(snapshots, regime_state=regime_state, trading_day=trading_day)
        self._record_equity_point(trading_day)

    def _process_entries(self, snapshots: dict[str, dict], *, regime_state, trading_day: date) -> None:
        for symbol, chain_data in snapshots.items():
            underlying = float(chain_data.get("underlying_price", 0.0))
            if underlying <= 0:
                continue

            all_signals: list[TradeSignal] = []
            for strategy in self.strategies:
                pause_until = self._strategy_pause_until.get(strategy.name)
                if pause_until and trading_day <= pause_until:
                    continue
                weight = float(regime_state.recommended_strategy_weights.get(strategy.name, 1.0))
                if weight <= 0:
                    continue
                market_context = {
                    "regime": regime_state.regime,
                    "regime_weights": regime_state.recommended_strategy_weights,
                    "position_size_scalar": regime_state.recommended_position_size_scalar,
                }
                if self.vol_surface_analyzer:
                    market_context["vol_surface"] = self.vol_surface_analyzer.analyze(
                        symbol=symbol,
                        chain_data=chain_data,
                        price_history=[],
                    ).to_dict()
                signals = strategy.scan_for_entries(symbol, chain_data, underlying, market_context=market_context)
                for signal in signals:
                    if signal.analysis is None:
                        continue
                    signal.analysis.score = max(
                        0.0,
                        min(100.0, float(signal.analysis.score or 0.0) * weight),
                    )
                    signal.size_multiplier *= float(regime_state.recommended_position_size_scalar or 1.0)
                    signal.metadata.setdefault("regime", regime_state.regime)
                all_signals.extend(signals)
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
        slippage_per_contract = entry_credit * self._slippage_factor(analysis.score)
        effective_credit = max(0.0, entry_credit - slippage_per_contract)
        commission = quantity * self.commission_per_contract
        self.cash_balance += effective_credit * quantity * 100.0
        self.cash_balance -= commission
        self.total_fees += commission
        self.total_slippage += slippage_per_contract * quantity * 100.0

        position = {
            "position_id": pid,
            "strategy": signal.strategy,
            "symbol": signal.symbol,
            "entry_credit": round(effective_credit, 4),
            "current_value": round(effective_credit, 4),
            "quantity": quantity,
            "max_loss": float(analysis.max_loss),
            "open_date": datetime.utcnow().isoformat(),
            "expiration": analysis.expiration,
            "dte_remaining": int(analysis.dte),
            "entry_slippage_per_contract": round(slippage_per_contract, 4),
            "commission_open": round(commission, 4),
            "regime": signal.metadata.get("regime", "unknown"),
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
        exit_slippage_per_contract = close_value * self._slippage_factor(float(position.get("details", {}).get("score", 50.0)))
        effective_close = max(0.0, close_value + exit_slippage_per_contract)
        commission = quantity * self.commission_per_contract
        self.cash_balance -= effective_close * quantity * 100.0
        self.cash_balance -= commission
        self.total_fees += commission
        self.total_slippage += exit_slippage_per_contract * quantity * 100.0

        pnl = (float(position.get("entry_credit", 0.0)) - effective_close) * quantity * 100.0
        pnl -= commission
        opened_at = _parse_datetime(position.get("open_date"))
        days_in_trade = (
            max(0, (trading_day - opened_at.date()).days) if opened_at else max(0, int(position.get("dte_remaining", 0)))
        )
        closed = {
            **position,
            "close_date": trading_day.isoformat(),
            "close_value": round(effective_close, 4),
            "pnl": round(pnl, 2),
            "reason": reason,
            "days_in_trade": days_in_trade,
            "status": "closed",
            "commission_close": round(commission, 4),
            "exit_slippage_per_contract": round(exit_slippage_per_contract, 4),
        }
        self.closed_trades.append(closed)
        self._daily_realized[trading_day.isoformat()] = (
            self._daily_realized.get(trading_day.isoformat(), 0.0) + pnl
        )
        strategy_key = str(position.get("strategy", ""))
        if pnl < 0:
            self._strategy_loss_streaks[strategy_key] += 1
            limit = max(2, int(getattr(self.config.circuit_breakers, "strategy_loss_streak_limit", 5)))
            if self._strategy_loss_streaks[strategy_key] >= limit:
                self._strategy_pause_until[strategy_key] = trading_day + timedelta(days=1)
        else:
            self._strategy_loss_streaks[strategy_key] = 0

    def _estimate_regime(self, trading_day: date, snapshots: dict[str, dict]):
        spy_snapshot = snapshots.get("SPY", {})
        spy_price = float(spy_snapshot.get("underlying_price", 0.0) or 0.0)
        if spy_price > 0:
            self._spy_history.append(spy_price)
        self._spy_history = self._spy_history[-260:]
        trend = 0.0
        if len(self._spy_history) >= 20:
            short = float(np.mean(self._spy_history[-20:]))
            long = float(np.mean(self._spy_history[-50:])) if len(self._spy_history) >= 50 else short
            if long > 0:
                trend = max(-1.0, min(1.0, (short - long) / long * 5.0))
        vix_proxy = 20.0
        inputs = {
            "vix_level": vix_proxy,
            "vix_term_ratio": 1.0,
            "spy_trend_score": trend,
            "breadth_above_50ma": 0.55 if trend >= 0 else 0.45,
            "put_call_ratio": 1.0,
            "vol_of_vol": 0.1,
            "realized_vs_implied_spread": 1.0,
        }
        state = self.regime_detector.detect_from_inputs(inputs)
        state.sub_signals["date"] = trading_day.isoformat()
        return state

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

    def _slippage_factor(self, score: float) -> float:
        """Simple liquidity-aware slippage proxy."""
        quality = max(0.0, min(100.0, float(score or 0.0))) / 100.0
        # Better scores imply tighter markets and lower slippage.
        return max(0.001, self.base_slippage_pct * (1.2 - quality))

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
        total_days = max(1, (end_date - start_date).days)
        annualized_return = ((1.0 + total_return) ** (365.0 / total_days) - 1.0) if total_return > -1.0 else -1.0
        calmar = (annualized_return / max_drawdown) if max_drawdown > 0 else 0.0

        wins = [trade["pnl"] for trade in self.closed_trades if trade.get("pnl", 0.0) > 0]
        losses = [trade["pnl"] for trade in self.closed_trades if trade.get("pnl", 0.0) < 0]
        avg_days = (
            float(np.mean([trade.get("days_in_trade", 0) for trade in self.closed_trades]))
            if self.closed_trades
            else 0.0
        )

        monthly = _monthly_returns(self.equity_curve, self.initial_balance)
        strategy_performance = _strategy_performance(self.closed_trades)
        regime_performance = _regime_performance(self.closed_trades)
        walk_forward = self._walk_forward_analysis(equities)
        monte_carlo = self._monte_carlo_simulation(self.closed_trades)
        report = {
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "initial_balance": round(self.initial_balance, 2),
            "ending_equity": round(equities[-1], 2) if equities else round(self.initial_balance, 2),
            "total_return": round(total_return, 6),
            "annualized_return": round(annualized_return, 6),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": round(calmar, 4),
            "max_drawdown": round(max_drawdown, 6),
            "win_rate": round((len(wins) / max(1, len(self.closed_trades))), 4),
            "average_winner": round(float(np.mean(wins)) if wins else 0.0, 2),
            "average_loser": round(float(np.mean(losses)) if losses else 0.0, 2),
            "profit_factor": round((sum(wins) / abs(sum(losses))) if losses else float("inf"), 4),
            "average_days_in_trade": round(avg_days, 2),
            "monthly_returns": monthly,
            "strategy_performance": strategy_performance,
            "regime_performance": regime_performance,
            "walk_forward": walk_forward,
            "monte_carlo": monte_carlo,
            "transaction_costs": {
                "total_fees": round(self.total_fees, 2),
                "total_slippage": round(self.total_slippage, 2),
                "commission_per_contract": round(self.commission_per_contract, 4),
                "base_slippage_pct": round(self.base_slippage_pct, 4),
            },
            "closed_trades": len(self.closed_trades),
            "open_positions": len(self.positions),
            "equity_curve": self.equity_curve,
        }
        return report

    def _write_report(self, report: dict) -> str:
        logs_dir = ensure_data_dir("logs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = logs_dir / f"backtest_{timestamp}.json"
        html_path = logs_dir / f"backtest_{timestamp}.html"
        report["chart_html_path"] = str(html_path)
        dump_json(path, report)
        self._write_html_report(report, html_path)
        logger.info("Backtest report written: %s", path)
        return str(path)

    def _walk_forward_analysis(self, equities: list[float]) -> dict:
        if len(equities) < 40:
            return {"windows": [], "avg_test_return": 0.0}
        window_train = 30
        window_test = 10
        cursor = 0
        windows = []
        while cursor + window_train + window_test <= len(equities):
            train = equities[cursor : cursor + window_train]
            test = equities[cursor + window_train : cursor + window_train + window_test]
            train_returns = _daily_returns(train)
            test_return = ((test[-1] / test[0]) - 1.0) if test and test[0] > 0 else 0.0
            windows.append(
                {
                    "train_mean_return": round(float(np.mean(train_returns)) if train_returns else 0.0, 6),
                    "test_return": round(test_return, 6),
                    "train_size": len(train),
                    "test_size": len(test),
                }
            )
            cursor += window_test
        avg_test_return = float(np.mean([w["test_return"] for w in windows])) if windows else 0.0
        return {"windows": windows, "avg_test_return": round(avg_test_return, 6)}

    def _monte_carlo_simulation(self, closed_trades: list[dict]) -> dict:
        pnls = [float(trade.get("pnl", 0.0) or 0.0) for trade in closed_trades if isinstance(trade, dict)]
        if len(pnls) < 2:
            return {
                "iterations": 0,
                "drawdown_p50": 0.0,
                "drawdown_p90": 0.0,
                "drawdown_p99": 0.0,
            }

        iterations = 10_000
        drawdowns: list[float] = []
        base = float(self.initial_balance)
        for _ in range(iterations):
            equity = base
            peak = base
            max_dd = 0.0
            for pnl in random.sample(pnls, len(pnls)):
                equity += pnl
                peak = max(peak, equity)
                if peak > 0:
                    max_dd = min(max_dd, (equity - peak) / peak)
            drawdowns.append(abs(max_dd))

        arr = np.array(drawdowns, dtype=float)
        return {
            "iterations": iterations,
            "drawdown_p50": round(float(np.percentile(arr, 50)), 6),
            "drawdown_p90": round(float(np.percentile(arr, 90)), 6),
            "drawdown_p99": round(float(np.percentile(arr, 99)), 6),
        }

    def _write_html_report(self, report: dict, output_path: Path) -> None:
        equity_curve = report.get("equity_curve", [])
        values = [float(point.get("equity", 0.0) or 0.0) for point in equity_curve]
        if not values:
            html = "<html><body><h1>Backtest</h1><p>No equity data.</p></body></html>"
            output_path.write_text(html, encoding="utf-8")
            return

        width = 900
        height = 320
        pad = 24
        min_value = min(values)
        max_value = max(values)
        span = max(1e-9, max_value - min_value)
        points = []
        for idx, value in enumerate(values):
            x = pad + (idx / max(1, len(values) - 1)) * (width - (pad * 2))
            y = height - pad - ((value - min_value) / span) * (height - (pad * 2))
            points.append(f"{x:.2f},{y:.2f}")
        polyline = " ".join(points)

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Backtest Report</title>
  <style>
    body {{ font-family: Helvetica, Arial, sans-serif; margin: 16px; color: #19222b; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(260px,1fr)); gap: 12px; }}
    .card {{ border:1px solid #dde6ec; border-radius:8px; padding:10px; }}
    svg {{ width:100%; height:auto; border:1px solid #e3eaef; border-radius:6px; }}
    table {{ border-collapse: collapse; width:100%; }}
    td, th {{ border-bottom:1px solid #eef3f7; padding:6px; text-align:left; }}
  </style>
</head>
<body>
  <h1>Backtest Report</h1>
  <div class="grid">
    <div class="card">
      <h3>Equity Curve</h3>
      <svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">
        <polyline points="{polyline}" fill="none" stroke="#0b6e8c" stroke-width="2"/>
      </svg>
    </div>
    <div class="card">
      <h3>Summary</h3>
      <table>
        <tr><th>Total Return</th><td>{report.get("total_return", 0):.2%}</td></tr>
        <tr><th>Sharpe</th><td>{report.get("sharpe_ratio", 0):.3f}</td></tr>
        <tr><th>Sortino</th><td>{report.get("sortino_ratio", 0):.3f}</td></tr>
        <tr><th>Calmar</th><td>{report.get("calmar_ratio", 0):.3f}</td></tr>
        <tr><th>Max DD</th><td>{report.get("max_drawdown", 0):.2%}</td></tr>
      </table>
    </div>
  </div>
</body>
</html>"""
        output_path.write_text(html, encoding="utf-8")


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


def _strategy_performance(closed_trades: list[dict]) -> dict:
    grouped: dict[str, list[float]] = {}
    for trade in closed_trades:
        if not isinstance(trade, dict):
            continue
        strategy = str(trade.get("strategy", "unknown"))
        grouped.setdefault(strategy, []).append(float(trade.get("pnl", 0.0) or 0.0))
    out = {}
    for strategy, pnls in grouped.items():
        wins = [value for value in pnls if value > 0]
        out[strategy] = {
            "trades": len(pnls),
            "win_rate": round(len(wins) / max(1, len(pnls)), 4),
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl": round(float(np.mean(pnls)) if pnls else 0.0, 2),
        }
    return out


def _regime_performance(closed_trades: list[dict]) -> dict:
    grouped: dict[str, list[float]] = {}
    for trade in closed_trades:
        if not isinstance(trade, dict):
            continue
        regime = str(trade.get("regime", "unknown"))
        grouped.setdefault(regime, []).append(float(trade.get("pnl", 0.0) or 0.0))
    out = {}
    for regime, pnls in grouped.items():
        wins = [value for value in pnls if value > 0]
        out[regime] = {
            "trades": len(pnls),
            "win_rate": round(len(wins) / max(1, len(pnls)), 4),
            "total_pnl": round(sum(pnls), 2),
        }
    return out


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
