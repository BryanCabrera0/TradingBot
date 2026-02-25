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
            if isinstance(item, dict) and "verdict" in item and "outcome" in item
        ]
        hits = sum(1 for item in judged if _is_llm_hit(item))
        accuracy = hits / len(judged) if judged else 0.0
        out["llm_accuracy"] = {
            "trades": len(judged),
            "hit_rate": round(accuracy, 4),
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
    strategy_stats = payload.get("strategy_breakdown", {})
    winners = payload.get("top_winners", [])
    losers = payload.get("top_losers", [])
    risk = payload.get("risk_metrics", {})
    greeks = payload.get("portfolio_greeks", {})
    llm_accuracy = payload.get("llm_accuracy", {})
    execution = payload.get("execution_quality", {})
    sector_exposure = payload.get("sector_exposure", {})
    breakers = payload.get("circuit_breakers", {})

    monthly_rows = "".join(
        f"<tr><td>{month}</td><td>{value:.2f}</td></tr>"
        for month, value in sorted(monthly.items())
    )
    strategy_rows = "".join(
        (
            "<tr>"
            f"<td>{strategy}</td>"
            f"<td>{stats.get('win_rate', 0):.1f}%</td>"
            f"<td>{stats.get('avg_pnl', 0):.2f}</td>"
            f"<td>{stats.get('total_pnl', 0):.2f}</td>"
            "</tr>"
        )
        for strategy, stats in strategy_stats.items()
    )
    winner_rows = "".join(
        f"<li>{item.get('symbol', '')} {item.get('pnl', 0):.2f}</li>" for item in winners[:5]
    )
    loser_rows = "".join(
        f"<li>{item.get('symbol', '')} {item.get('pnl', 0):.2f}</li>" for item in losers[:5]
    )

    sector_labels = list(sector_exposure.keys())
    sector_values = [float(v) for v in sector_exposure.values()]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>TradingBot Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --bg: #f8f7f4;
      --text: #1e1f20;
      --card: #ffffff;
      --accent: #1b6ca8;
      --muted: #68707a;
    }}
    body {{ margin:0; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background: var(--bg); color: var(--text); }}
    header {{ padding: 18px 24px; background: linear-gradient(120deg, #dbe9f6, #f6efe2); }}
    h1 {{ margin: 0; font-size: 1.5rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; padding: 16px; }}
    .card {{ background: var(--card); border-radius: 10px; padding: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    h2 {{ margin: 0 0 8px 0; font-size: 1rem; }}
    p, li, td, th {{ font-size: 0.9rem; color: var(--text); }}
    .muted {{ color: var(--muted); }}
    table {{ width:100%; border-collapse: collapse; }}
    td, th {{ border-bottom: 1px solid #ece9e2; padding: 6px 4px; text-align:left; }}
    ul {{ margin: 0; padding-left: 18px; }}
    canvas {{ width:100%; max-height: 240px; }}
  </style>
</head>
<body>
  <header>
    <h1>TradingBot Dashboard</h1>
    <p class="muted">Generated {timestamp}</p>
  </header>
  <div class="grid">
    <section class="card">
      <h2>Equity Curve</h2>
      <canvas id="equityChart"></canvas>
    </section>
    <section class="card">
      <h2>Monthly P&amp;L</h2>
      <table><thead><tr><th>Month</th><th>P&amp;L</th></tr></thead><tbody>{monthly_rows}</tbody></table>
    </section>
    <section class="card">
      <h2>Strategy Breakdown</h2>
      <table>
        <thead><tr><th>Strategy</th><th>Win Rate</th><th>Avg P&amp;L</th><th>Total P&amp;L</th></tr></thead>
        <tbody>{strategy_rows}</tbody>
      </table>
    </section>
    <section class="card">
      <h2>Top Winners / Losers</h2>
      <strong>Winners</strong>
      <ul>{winner_rows}</ul>
      <strong>Losers</strong>
      <ul>{loser_rows}</ul>
    </section>
    <section class="card">
      <h2>Risk Metrics</h2>
      <p>Sharpe: {risk.get("sharpe", 0):.3f}</p>
      <p>Sortino: {risk.get("sortino", 0):.3f}</p>
      <p>Max drawdown: {risk.get("max_drawdown", 0):.2%}</p>
      <p>Current drawdown: {risk.get("current_drawdown", 0):.2%}</p>
    </section>
    <section class="card">
      <h2>Portfolio Greeks</h2>
      <p>Delta: {greeks.get("delta", 0):.2f}</p>
      <p>Theta: {greeks.get("theta", 0):.2f}</p>
      <p>Gamma: {greeks.get("gamma", 0):.2f}</p>
      <p>Vega: {greeks.get("vega", 0):.2f}</p>
    </section>
    <section class="card">
      <h2>LLM Accuracy</h2>
      <p>Tracked trades: {llm_accuracy.get("trades", 0)}</p>
      <p>Hit rate: {llm_accuracy.get("hit_rate", 0):.2%}</p>
    </section>
    <section class="card">
      <h2>Execution Quality</h2>
      <p>Average slippage: {execution.get("avg_slippage", 0):.4f}</p>
      <p>Samples: {execution.get("samples", 0)}</p>
    </section>
    <section class="card">
      <h2>Sector Exposure</h2>
      <canvas id="sectorChart"></canvas>
    </section>
    <section class="card">
      <h2>Circuit Breakers</h2>
      <p>Regime: {breakers.get("regime", "normal")}</p>
      <p>Halt entries: {breakers.get("halt_entries", False)}</p>
      <p>Consecutive-loss pause: {breakers.get("consecutive_loss_pause_until", "-")}</p>
      <p>Weekly-loss pause: {breakers.get("weekly_loss_pause_until", "-")}</p>
    </section>
  </div>
  <script>
    const equityLabels = {json.dumps(equity_labels)};
    const equityValues = {json.dumps(equity_values)};
    const sectorLabels = {json.dumps(sector_labels)};
    const sectorValues = {json.dumps(sector_values)};

    new Chart(document.getElementById('equityChart'), {{
      type: 'line',
      data: {{ labels: equityLabels, datasets: [{{ label: 'Equity', data: equityValues, borderColor: '#1b6ca8', tension: 0.2 }}] }},
      options: {{ plugins: {{ legend: {{ display: false }} }} }}
    }});

    new Chart(document.getElementById('sectorChart'), {{
      type: 'pie',
      data: {{ labels: sectorLabels, datasets: [{{ data: sectorValues }}] }},
      options: {{ plugins: {{ legend: {{ position: 'bottom' }} }} }}
    }});
  </script>
</body>
</html>
"""


def _is_llm_hit(item: dict) -> bool:
    verdict = str(item.get("verdict", "")).lower()
    outcome = float(item.get("outcome", 0.0))
    if verdict == "approve":
        return outcome > 0
    if verdict == "reject":
        return outcome <= 0
    if verdict == "reduce_size":
        return True
    return False
