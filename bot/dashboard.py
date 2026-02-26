"""HTML performance dashboard generation."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from bot.data_store import load_json

logger = logging.getLogger(__name__)

DASHBOARD_PATH = Path("logs/dashboard.html")
LLM_TRACK_RECORD_PATH = Path("bot/data/llm_track_record.json")
EXECUTION_QUALITY_PATH = Path("bot/data/execution_quality.json")


def generate_dashboard(payload: dict, output_path: Path | str = DASHBOARD_PATH) -> str:
    """Render dashboard HTML from a structured payload."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    html = _render_html(payload)
    output.write_text(html, encoding="utf-8")
    logger.info("Dashboard generated at %s", output)
    return str(output)


def enrich_dashboard_payload(payload: dict) -> dict:
    """Augment dashboard payload with LLM/execution analytics from persisted data."""
    out = dict(payload)

    llm_track = load_json(LLM_TRACK_RECORD_PATH, {"trades": []})
    trades = llm_track.get("trades", []) if isinstance(llm_track, dict) else []
    if isinstance(trades, list) and len(trades) >= 50:
        judged = [
            item
            for item in trades
            if isinstance(item, dict)
            and "verdict" in item
            and _coerce_outcome(item.get("outcome")) is not None
        ]
        hits = sum(1 for item in judged if _is_llm_hit(item))
        accuracy = hits / len(judged) if judged else 0.0
        approves = [item for item in judged if str(item.get("verdict", "")).lower() == "approve"]
        rejects = [item for item in judged if str(item.get("verdict", "")).lower() == "reject"]
        reduces = [item for item in judged if str(item.get("verdict", "")).lower() == "reduce_size"]
        approve_hits = sum(1 for item in approves if _is_llm_hit(item))
        reject_hits = sum(1 for item in rejects if _is_llm_hit(item))
        reduce_hits = sum(1 for item in reduces if _is_llm_hit(item))
        out["llm_accuracy"] = {
            "trades": len(judged),
            "hit_rate": round(accuracy, 4),
            "approve_accuracy": round((approve_hits / len(approves)) if approves else 0.0, 4),
            "reject_accuracy": round((reject_hits / len(rejects)) if rejects else 0.0, 4),
            "reduce_size_accuracy": round((reduce_hits / len(reduces)) if reduces else 0.0, 4),
        }

    execution = load_json(EXECUTION_QUALITY_PATH, {"fills": []})
    fills = execution.get("fills", []) if isinstance(execution, dict) else []
    slippages = [
        float(item.get("slippage", 0.0))
        for item in fills
        if isinstance(item, dict)
    ]
    if slippages:
        out["execution_quality"] = {
            "avg_slippage": round(float(np.mean(slippages)), 4),
            "samples": len(slippages),
        }

    return out


def _render_html(payload: dict) -> str:
    payload = enrich_dashboard_payload(payload)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    equity_curve = payload.get("equity_curve", [])
    equity_labels = [point.get("date", "") for point in equity_curve]
    equity_values = [float(point.get("equity", 0.0)) for point in equity_curve]
    monthly = payload.get("monthly_pnl", {})
    daily_calendar = payload.get("daily_pnl_calendar", {})
    strategy_stats = payload.get("strategy_breakdown", {})
    regime_stats = payload.get("regime_performance", {})
    winners = payload.get("top_winners", [])
    losers = payload.get("top_losers", [])
    risk = payload.get("risk_metrics", {})
    greeks = payload.get("portfolio_greeks", {})
    llm_accuracy = payload.get("llm_accuracy", {})
    execution = payload.get("execution_quality", {})
    sector_exposure = payload.get("sector_exposure", {})
    breakers = payload.get("circuit_breakers", {})
    regime_state = payload.get("regime_state", {})
    correlation_matrix = payload.get("correlation_matrix", {})
    var_metrics = payload.get("var_metrics", {})
    open_positions = payload.get("open_positions_table", [])
    journal_entries = payload.get("trade_journal", [])
    service_degradation = payload.get("service_degradation", {})
    hedge_costs = payload.get("hedge_costs", {})
    roll_metrics = payload.get("roll_metrics", {})

    monthly_rows = "".join(
        f"<tr><td>{month}</td><td>{value:.2f}</td></tr>"
        for month, value in sorted(monthly.items())
    ) or "<tr><td colspan='2' class='muted'>No data</td></tr>"
    strategy_rows = "".join(
        (
            "<tr>"
            f"<td>{strategy}</td>"
            f"<td>{stats.get('win_rate', 0):.1f}%</td>"
            f"<td>{stats.get('avg_profit', 0):.2f}</td>"
            f"<td>{stats.get('avg_loss', 0):.2f}</td>"
            f"<td>{stats.get('total_pnl', 0):.2f}</td>"
            "</tr>"
        )
        for strategy, stats in strategy_stats.items()
    ) or "<tr><td colspan='5' class='muted'>No strategy data</td></tr>"
    regime_rows = "".join(
        (
            "<tr>"
            f"<td>{regime}</td>"
            f"<td>{stats.get('trades', 0)}</td>"
            f"<td>{stats.get('win_rate', 0):.1f}%</td>"
            f"<td>{stats.get('total_pnl', 0):.2f}</td>"
            "</tr>"
        )
        for regime, stats in regime_stats.items()
    ) or "<tr><td colspan='4' class='muted'>No regime data</td></tr>"
    winner_rows = "".join(
        f"<li>{item.get('symbol', '')} {item.get('pnl', 0):.2f}</li>" for item in winners[:5]
    ) or "<li class='muted'>No winners yet</li>"
    loser_rows = "".join(
        f"<li>{item.get('symbol', '')} {item.get('pnl', 0):.2f}</li>" for item in losers[:5]
    ) or "<li class='muted'>No losers yet</li>"
    open_rows = "".join(
        (
            "<tr>"
            f"<td>{row.get('symbol', '')}</td>"
            f"<td>{row.get('strategy', '')}</td>"
            f"<td>{row.get('quantity', '')}</td>"
            f"<td>{row.get('dte_remaining', '')}</td>"
            f"<td>{row.get('pnl', 0):.2f}</td>"
            f"<td>{row.get('delta', 0):.2f}</td>"
            f"<td>{row.get('theta', 0):.2f}</td>"
            f"<td>{row.get('gamma', 0):.2f}</td>"
            f"<td>{row.get('vega', 0):.2f}</td>"
            "</tr>"
        )
        for row in open_positions[:40]
        if isinstance(row, dict)
    ) or "<tr><td colspan='9' class='muted'>No open positions</td></tr>"

    journal_rows = "".join(
        (
            "<tr>"
            f"<td>{item.get('symbol', '')}</td>"
            f"<td>{item.get('strategy', '')}</td>"
            f"<td>{item.get('verdict', '')}</td>"
            f"<td>{item.get('outcome', 0)}</td>"
            f"<td>{(item.get('analysis') or {}).get('adjustment', '') if isinstance(item.get('analysis'), dict) else ''}</td>"
            "</tr>"
        )
        for item in journal_entries
        if isinstance(item, dict)
    ) or "<tr><td colspan='5' class='muted'>No journal entries</td></tr>"

    correlation_html = _render_correlation_heatmap(correlation_matrix)
    equity_svg = _render_equity_svg(equity_values)
    strategy_bars = _render_strategy_bars(strategy_stats)
    sector_bars = _render_sector_bars(sector_exposure)
    greek_gauges = _render_greek_gauges(greeks)
    month_calendar_html = _render_monthly_calendar(daily_calendar)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>TradingBot Dashboard</title>
  <meta http-equiv="refresh" content="60">
  <style>
    :root {{
      --bg: #f4f7f8;
      --card: #ffffff;
      --text: #1a2128;
      --muted: #5f6b75;
      --accent: #0b6e8c;
      --good: #1f9d55;
      --bad: #cc3a3a;
      --warn: #c58a18;
      --grid: #e6edf1;
    }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }}
    header {{ padding: 16px 22px; background: linear-gradient(110deg, #d7e8ef, #f7efd8); border-bottom: 1px solid #d7e2e8; }}
    h1 {{ margin: 0; font-size: 1.4rem; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); padding: 14px; }}
    .card {{ background: var(--card); border-radius: 10px; border: 1px solid #dbe6ec; padding: 12px; }}
    h2 {{ margin: 0 0 8px 0; font-size: 1rem; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; border-bottom: 1px solid var(--grid); padding: 6px 4px; font-size: 0.88rem; }}
    .tiny {{ font-size: 0.78rem; }}
    ul {{ margin: 0; padding-left: 18px; }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #e7f2f6; color: #0f4e63; font-size: 0.78rem; }}
    .stack > div {{ margin-bottom: 6px; }}
    .bar {{ height: 10px; background: #edf2f5; border-radius: 8px; overflow: hidden; }}
    .bar > span {{ display: block; height: 100%; background: var(--accent); }}
    .heat td {{ text-align: center; font-size: 0.75rem; padding: 4px; }}
    .heat th {{ font-size: 0.75rem; }}
    .flex {{ display: flex; gap: 12px; align-items: center; }}
    .donut {{
      width: 96px;
      height: 96px;
      border-radius: 50%;
      background: conic-gradient(var(--good) {llm_accuracy.get("hit_rate", 0.0) * 360}deg, #e5ebf0 0deg);
      position: relative;
      flex: 0 0 auto;
    }}
    .donut:after {{
      content: "";
      position: absolute;
      inset: 16px;
      border-radius: 50%;
      background: #fff;
    }}
    .svg-wrap svg {{ width: 100%; height: 210px; display: block; }}
    @media (max-width: 640px) {{
      .grid {{ grid-template-columns: 1fr; }}
      th, td {{ font-size: 0.8rem; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>TradingBot Dashboard</h1>
    <div class="tiny muted">Generated {timestamp} • Auto-refresh 60s</div>
    <div class="tiny">
      <span class="pill">Regime: {regime_state.get("regime", breakers.get("regime", "normal"))}</span>
      <span class="pill">Confidence: {safe_pct(regime_state.get("confidence", 0.0))}</span>
      <span class="pill">VaR95: {var_metrics.get("var95", 0.0):.2f}</span>
      <span class="pill">VaR99: {var_metrics.get("var99", 0.0):.2f}</span>
    </div>
  </header>
  <div class="grid">
    <section class="card svg-wrap">
      <h2>Equity Curve</h2>
      {equity_svg}
    </section>
    <section class="card">
      <h2>Risk Metrics</h2>
      <div class="stack">
        <div>Sharpe: {risk.get("sharpe", 0):.3f}</div>
        <div>Sortino: {risk.get("sortino", 0):.3f}</div>
        <div>Max drawdown: {risk.get("max_drawdown", 0):.2%}</div>
        <div>Current drawdown: {risk.get("current_drawdown", 0):.2%}</div>
        <div>Execution avg slippage: {execution.get("avg_slippage", 0):.4f}</div>
        <div>Hedge cost (MTD): {float(hedge_costs.get("month_to_date", 0.0) or 0.0):.2f}</div>
        <div>Hedge cost (lifetime): {float(hedge_costs.get("lifetime", 0.0) or 0.0):.2f}</div>
        <div>Rolled positions: {int(roll_metrics.get("rolled_count", 0) or 0)}</div>
        <div>Avg roll credit captured: {float(roll_metrics.get("avg_roll_credit_captured", 0.0) or 0.0):.4f}</div>
      </div>
    </section>
    <section class="card">
      <h2>Portfolio Greeks</h2>
      {greek_gauges}
    </section>
    <section class="card">
      <h2>LLM Accuracy</h2>
      <div class="flex">
        <div class="donut"></div>
        <div class="stack">
          <div>Tracked trades: {llm_accuracy.get("trades", 0)}</div>
          <div>Hit rate: {llm_accuracy.get("hit_rate", 0):.2%}</div>
          <div>Approve accuracy: {llm_accuracy.get("approve_accuracy", 0):.2%}</div>
          <div>Reject accuracy: {llm_accuracy.get("reject_accuracy", 0):.2%}</div>
          <div>Reduce-size accuracy: {llm_accuracy.get("reduce_size_accuracy", 0):.2%}</div>
        </div>
      </div>
    </section>
    <section class="card">
      <h2>Monthly P&amp;L</h2>
      <table><thead><tr><th>Month</th><th>P&amp;L</th></tr></thead><tbody>{monthly_rows}</tbody></table>
    </section>
    <section class="card">
      <h2>Strategy Breakdown</h2>
      <table><thead><tr><th>Strategy</th><th>Win Rate</th><th>Avg Profit</th><th>Avg Loss</th><th>Total P&amp;L</th></tr></thead><tbody>{strategy_rows}</tbody></table>
      <div style="margin-top:8px;">{strategy_bars}</div>
    </section>
    <section class="card">
      <h2>Regime Performance</h2>
      <table><thead><tr><th>Regime</th><th>Trades</th><th>Win Rate</th><th>Total P&amp;L</th></tr></thead><tbody>{regime_rows}</tbody></table>
    </section>
    <section class="card">
      <h2>Top Winners / Losers</h2>
      <strong>Winners</strong>
      <ul>{winner_rows}</ul>
      <strong>Losers</strong>
      <ul>{loser_rows}</ul>
    </section>
    <section class="card">
      <h2>Sector Exposure</h2>
      {sector_bars}
    </section>
    <section class="card">
      <h2>Monthly P&amp;L Calendar</h2>
      {month_calendar_html}
    </section>
    <section class="card">
      <h2>Circuit Breakers</h2>
      <div>Halt entries: {breakers.get("halt_entries", False)}</div>
      <div>Consecutive-loss pause: {breakers.get("consecutive_loss_pause_until", "-")}</div>
      <div>Weekly-loss pause: {breakers.get("weekly_loss_pause_until", "-")}</div>
      <div>Portfolio halt: {payload.get("portfolio_halt_until") or "-"}</div>
      <div class="tiny muted">Degradation: {json.dumps(service_degradation, default=str)}</div>
    </section>
    <section class="card">
      <h2>Correlation Heat Map</h2>
      {correlation_html}
    </section>
    <section class="card" style="grid-column: 1 / -1;">
      <h2>Open Positions</h2>
      <table>
        <thead><tr><th>Symbol</th><th>Strategy</th><th>Qty</th><th>DTE</th><th>P&amp;L</th><th>Delta</th><th>Theta</th><th>Gamma</th><th>Vega</th></tr></thead>
        <tbody>{open_rows}</tbody>
      </table>
    </section>
    <section class="card" style="grid-column: 1 / -1;">
      <h2>Trade Journal</h2>
      <table>
        <thead><tr><th>Symbol</th><th>Strategy</th><th>Verdict</th><th>Outcome</th><th>Adjustment</th></tr></thead>
        <tbody>{journal_rows}</tbody>
      </table>
    </section>
  </div>
</body>
</html>"""


def _render_equity_svg(values: list[float]) -> str:
    if not values:
        return "<div class='muted tiny'>No equity data</div>"
    width = 640
    height = 210
    pad = 20
    min_value = min(values)
    max_value = max(values)
    span = max(1e-9, max_value - min_value)
    points = []
    for idx, value in enumerate(values):
        x = pad + (idx / max(1, len(values) - 1)) * (width - pad * 2)
        y = height - pad - ((value - min_value) / span) * (height - pad * 2)
        points.append(f"{x:.2f},{y:.2f}")
    polyline = " ".join(points)
    return (
        f"<svg viewBox='0 0 {width} {height}' preserveAspectRatio='none'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='#ffffff' />"
        f"<polyline points='{polyline}' fill='none' stroke='#0b6e8c' stroke-width='2'/>"
        f"</svg>"
    )


def _render_strategy_bars(strategy_stats: dict) -> str:
    if not isinstance(strategy_stats, dict) or not strategy_stats:
        return "<div class='muted tiny'>No strategy performance data</div>"
    totals = {name: float((stats or {}).get("total_pnl", 0.0) or 0.0) for name, stats in strategy_stats.items()}
    max_abs = max(abs(value) for value in totals.values()) if totals else 1.0
    rows = []
    for name, value in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        pct = (abs(value) / max(1e-9, max_abs)) * 100.0
        color = "#1f9d55" if value >= 0 else "#cc3a3a"
        rows.append(
            f"<div class='tiny'>{name}: {value:.2f}</div>"
            f"<div class='bar'><span style='width:{pct:.1f}%; background:{color};'></span></div>"
        )
    return "".join(rows)


def _render_sector_bars(sector_exposure: dict) -> str:
    if not isinstance(sector_exposure, dict) or not sector_exposure:
        return "<div class='muted tiny'>No sector exposure data</div>"
    rows = []
    for sector, pct in sorted(sector_exposure.items(), key=lambda item: item[1], reverse=True):
        value = float(pct or 0.0)
        rows.append(
            f"<div class='tiny'>{sector}: {value:.2f}%</div>"
            f"<div class='bar'><span style='width:{min(100.0, max(0.0, value)):.1f}%;'></span></div>"
        )
    return "".join(rows)


def _render_correlation_heatmap(matrix: dict) -> str:
    if not isinstance(matrix, dict) or not matrix:
        return "<div class='muted tiny'>No correlation matrix data</div>"
    symbols = sorted(matrix.keys())
    header = "".join(f"<th>{symbol}</th>" for symbol in symbols)
    rows = []
    for left in symbols:
        cols = []
        row = matrix.get(left, {}) if isinstance(matrix.get(left), dict) else {}
        for right in symbols:
            value = float(row.get(right, 0.0) or 0.0)
            color = _corr_color(value)
            cols.append(f"<td style='background:{color};'>{value:.2f}</td>")
        rows.append(f"<tr><th>{left}</th>{''.join(cols)}</tr>")
    return f"<table class='heat'><thead><tr><th></th>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _render_greek_gauges(greeks: dict) -> str:
    if not isinstance(greeks, dict):
        greeks = {}
    gauges = []
    for name in ("delta", "theta", "gamma", "vega"):
        value = float(greeks.get(name, 0.0) or 0.0)
        pct = min(100.0, abs(value))
        color = "#1f9d55" if value >= 0 else "#cc3a3a"
        gauges.append(
            f"<div class='tiny'>{name.title()}: {value:.2f}</div>"
            f"<div class='bar'><span style='width:{pct:.1f}%; background:{color};'></span></div>"
        )
    return "".join(gauges)


def _render_monthly_calendar(daily_pnl: dict) -> str:
    if not isinstance(daily_pnl, dict) or not daily_pnl:
        return "<div class='muted tiny'>No daily P&amp;L data</div>"
    month_key = datetime.now().strftime("%Y-%m")
    rows = []
    for day, pnl in sorted(daily_pnl.items()):
        if not str(day).startswith(month_key):
            continue
        value = float(pnl or 0.0)
        color = "#1f9d55" if value >= 0 else "#cc3a3a"
        rows.append(
            f"<tr><td>{day}</td><td style='color:{color};'>{value:.2f}</td></tr>"
        )
    if not rows:
        return "<div class='muted tiny'>No entries for current month</div>"
    return "<table><thead><tr><th>Date</th><th>P&amp;L</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _corr_color(value: float) -> str:
    v = max(-1.0, min(1.0, float(value)))
    if v >= 0:
        intensity = int(255 - (v * 120))
        return f"rgb(255,{intensity},{intensity})"
    intensity = int(255 - (abs(v) * 120))
    return f"rgb({intensity},255,{intensity})"


def safe_pct(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric <= 1.0:
        numeric *= 100.0
    return f"{numeric:.1f}%"


def _is_llm_hit(item: dict) -> bool:
    verdict = str(item.get("verdict", "")).lower()
    outcome = _coerce_outcome(item.get("outcome"))
    if outcome is None:
        return False
    if verdict == "approve":
        return outcome > 0
    if verdict == "reject":
        return outcome <= 0
    if verdict == "reduce_size":
        return True
    return False


def _coerce_outcome(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
