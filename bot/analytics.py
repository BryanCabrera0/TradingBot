"""Performance analytics engine shared by live reporting and backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from typing import Optional

from bot.number_utils import safe_float, safe_int


@dataclass
class AnalyticsReport:
    core_metrics: dict = field(default_factory=dict)
    strategy_metrics: dict = field(default_factory=dict)
    regime_metrics: dict = field(default_factory=dict)
    monthly_metrics: dict = field(default_factory=dict)
    daily_pnl: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "core_metrics": dict(self.core_metrics),
            "strategy_metrics": dict(self.strategy_metrics),
            "regime_metrics": dict(self.regime_metrics),
            "monthly_metrics": dict(self.monthly_metrics),
            "daily_pnl": dict(self.daily_pnl),
        }


def compute(closed_trades: list[dict], *, initial_equity: float = 100_000.0) -> AnalyticsReport:
    """Compute portfolio analytics from closed trade history."""
    trades = [row for row in (closed_trades or []) if isinstance(row, dict)]
    if not trades:
        return AnalyticsReport(
            core_metrics={
                "sharpe": 0.0,
                "sortino": 0.0,
                "calmar": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_duration": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win_loss_ratio": 0.0,
                "expectancy_per_trade": 0.0,
                "current_consecutive_wins": 0,
                "current_consecutive_losses": 0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "risk_adjusted_return": 0.0,
                "total_pnl": 0.0,
                "trades": 0,
            }
        )

    ordered = sorted(trades, key=lambda row: str(row.get("close_date", "")))
    pnls = [safe_float(row.get("pnl", row.get("realized_pnl", 0.0)), 0.0) for row in ordered]
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]
    total_pnl = sum(pnls)
    total_trades = len(pnls)

    win_rate = (len(wins) / total_trades) if total_trades else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss_abs = (sum(losses) / len(losses)) if losses else 0.0
    profit_factor = (sum(wins) / sum(losses)) if losses else (float("inf") if wins else 0.0)
    avg_win_loss_ratio = (avg_win / avg_loss_abs) if avg_loss_abs > 0 else 0.0
    expectancy = (avg_win * win_rate) - (avg_loss_abs * (1.0 - win_rate))

    current_wins, current_losses, max_wins, max_losses = _streak_stats(pnls)

    daily_pnl = _daily_pnl_map(ordered)
    daily_returns = _daily_returns(daily_pnl, initial_equity=max(1.0, float(initial_equity)))
    sharpe = _sharpe(daily_returns)
    sortino = _sortino(daily_returns)

    equity_curve = _equity_curve(daily_pnl, initial_equity=max(1.0, float(initial_equity)))
    max_drawdown, max_dd_duration = _drawdown_stats(equity_curve)
    total_return = ((equity_curve[-1] / max(1.0, float(initial_equity))) - 1.0) if equity_curve else 0.0
    annual_factor = 252.0 / max(1, len(daily_returns)) if daily_returns else 0.0
    annualized_return = ((1.0 + total_return) ** annual_factor - 1.0) if daily_returns else 0.0
    calmar = (annualized_return / max_drawdown) if max_drawdown > 0 else 0.0

    total_risk_deployed = 0.0
    for row in ordered:
        max_loss = safe_float(row.get("max_loss"), 0.0)
        qty = max(1, safe_int(row.get("quantity"), 1))
        if max_loss > 0:
            total_risk_deployed += max_loss * qty * 100.0
    risk_adjusted_return = (total_pnl / total_risk_deployed) if total_risk_deployed > 0 else 0.0

    strategy_metrics = _bucket_metrics(ordered, key_name="strategy")
    regime_metrics = _bucket_metrics(ordered, key_name="regime")
    monthly_metrics = _bucket_metrics(ordered, key_name="month")

    core = {
        "sharpe": round(sharpe, 6),
        "sortino": round(sortino, 6),
        "calmar": round(calmar, 6),
        "max_drawdown": round(max_drawdown, 6),
        "max_drawdown_duration": int(max_dd_duration),
        "win_rate": round(win_rate, 6),
        "profit_factor": round(profit_factor, 6) if profit_factor != float("inf") else float("inf"),
        "avg_win_loss_ratio": round(avg_win_loss_ratio, 6),
        "expectancy_per_trade": round(expectancy, 6),
        "current_consecutive_wins": int(current_wins),
        "current_consecutive_losses": int(current_losses),
        "max_consecutive_wins": int(max_wins),
        "max_consecutive_losses": int(max_losses),
        "risk_adjusted_return": round(risk_adjusted_return, 6),
        "total_pnl": round(total_pnl, 4),
        "trades": int(total_trades),
    }

    return AnalyticsReport(
        core_metrics=core,
        strategy_metrics=strategy_metrics,
        regime_metrics=regime_metrics,
        monthly_metrics=monthly_metrics,
        daily_pnl=daily_pnl,
    )


def _daily_pnl_map(trades: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in trades:
        close_date = str(row.get("close_date", ""))
        day = close_date[:10] if len(close_date) >= 10 else "unknown"
        out[day] = out.get(day, 0.0) + safe_float(row.get("pnl", row.get("realized_pnl", 0.0)), 0.0)
    return dict(sorted(out.items(), key=lambda item: item[0]))


def _equity_curve(daily_pnl: dict[str, float], *, initial_equity: float) -> list[float]:
    equity = initial_equity
    out = [equity]
    for _, pnl in sorted(daily_pnl.items()):
        equity += safe_float(pnl, 0.0)
        out.append(equity)
    return out


def _daily_returns(daily_pnl: dict[str, float], *, initial_equity: float) -> list[float]:
    returns: list[float] = []
    equity = initial_equity
    for _, pnl in sorted(daily_pnl.items()):
        if equity <= 0:
            returns.append(0.0)
            equity += safe_float(pnl, 0.0)
            continue
        r = safe_float(pnl, 0.0) / equity
        returns.append(r)
        equity += safe_float(pnl, 0.0)
    return returns


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / max(1, len(returns) - 1)
    std = variance ** 0.5
    if std <= 0:
        return 0.0
    return (mean_r / std) * sqrt(252.0)


def _sortino(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    downside = [r for r in returns if r < 0]
    if not downside:
        return 0.0
    downside_dev = (sum(r * r for r in downside) / len(downside)) ** 0.5
    if downside_dev <= 0:
        return 0.0
    return (mean_r / downside_dev) * sqrt(252.0)


def _drawdown_stats(equity_curve: list[float]) -> tuple[float, int]:
    if len(equity_curve) < 2:
        return 0.0, 0
    peak = equity_curve[0]
    peak_idx = 0
    max_drawdown = 0.0
    max_duration = 0
    in_drawdown_start: Optional[int] = None

    for idx, value in enumerate(equity_curve):
        if value >= peak:
            peak = value
            peak_idx = idx
            if in_drawdown_start is not None:
                max_duration = max(max_duration, idx - in_drawdown_start)
                in_drawdown_start = None
            continue

        if peak > 0:
            dd = (peak - value) / peak
            max_drawdown = max(max_drawdown, dd)
        if in_drawdown_start is None:
            in_drawdown_start = peak_idx

    if in_drawdown_start is not None:
        max_duration = max(max_duration, len(equity_curve) - 1 - in_drawdown_start)
    return max_drawdown, max_duration


def _bucket_metrics(trades: list[dict], *, key_name: str) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {}
    for row in trades:
        if key_name == "month":
            close_date = str(row.get("close_date", ""))
            key = close_date[:7] if len(close_date) >= 7 else "unknown"
        else:
            key = str(row.get(key_name, "unknown") or "unknown")
        buckets.setdefault(key, []).append(row)

    out: dict[str, dict] = {}
    for key, rows in buckets.items():
        values = [safe_float(row.get("pnl", row.get("realized_pnl", 0.0)), 0.0) for row in rows]
        wins = [p for p in values if p > 0]
        losses = [abs(p) for p in values if p < 0]
        total = len(values)
        win_rate = (len(wins) / total) if total else 0.0
        avg_profit = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (-(sum(losses) / len(losses))) if losses else 0.0
        avg_loss_abs = (sum(losses) / len(losses)) if losses else 0.0
        expectancy = (avg_profit * win_rate) - (abs(avg_loss) * (1.0 - win_rate))
        profit_factor = (sum(wins) / sum(losses)) if losses else (float("inf") if wins else 0.0)
        avg_win_loss_ratio = (avg_profit / avg_loss_abs) if avg_loss_abs > 0 else 0.0
        current_wins, current_losses, max_wins, max_losses = _streak_stats(values)

        daily_pnl = _daily_pnl_map(rows)
        daily_returns = _daily_returns(daily_pnl, initial_equity=100_000.0)
        sharpe = _sharpe(daily_returns)
        sortino = _sortino(daily_returns)
        equity_curve = _equity_curve(daily_pnl, initial_equity=100_000.0)
        max_drawdown, max_dd_duration = _drawdown_stats(equity_curve)
        total_return = ((equity_curve[-1] / 100_000.0) - 1.0) if equity_curve else 0.0
        annual_factor = 252.0 / max(1, len(daily_returns)) if daily_returns else 0.0
        annualized_return = ((1.0 + total_return) ** annual_factor - 1.0) if daily_returns else 0.0
        calmar = (annualized_return / max_drawdown) if max_drawdown > 0 else 0.0
        out[key] = {
            "trades": total,
            "win_rate": round(win_rate * 100.0, 4),
            "avg_profit": round(avg_profit, 4),
            "avg_loss": round(avg_loss, 4),
            "avg_pnl": round(sum(values) / max(1, total), 4),
            "total_pnl": round(sum(values), 4),
            "profit_factor": round(profit_factor, 6) if profit_factor != float("inf") else float("inf"),
            "expectancy": round(expectancy, 6),
            "avg_win_loss_ratio": round(avg_win_loss_ratio, 6),
            "sharpe": round(sharpe, 6),
            "sortino": round(sortino, 6),
            "calmar": round(calmar, 6),
            "max_drawdown": round(max_drawdown, 6),
            "max_drawdown_duration": int(max_dd_duration),
            "current_consecutive_wins": int(current_wins),
            "current_consecutive_losses": int(current_losses),
            "max_consecutive_wins": int(max_wins),
            "max_consecutive_losses": int(max_losses),
        }
    return out


def _streak_stats(pnls: list[float]) -> tuple[int, int, int, int]:
    current_wins = 0
    current_losses = 0
    max_wins = 0
    max_losses = 0
    streak = 0
    for pnl in pnls:
        if pnl > 0:
            streak = streak + 1 if streak >= 0 else 1
        elif pnl < 0:
            streak = streak - 1 if streak <= 0 else -1
        else:
            streak = 0
        max_wins = max(max_wins, max(0, streak))
        max_losses = max(max_losses, max(0, -streak))
    if streak > 0:
        current_wins = streak
    elif streak < 0:
        current_losses = -streak
    return current_wins, current_losses, max_wins, max_losses
