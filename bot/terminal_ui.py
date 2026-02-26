"""Rich-based terminal UI dashboard for real-time bot monitoring."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import io
import logging
import threading
from typing import Optional
from zoneinfo import ZoneInfo

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)
EASTERN_TZ = ZoneInfo("America/New_York")


class _ActivityRichHandler(RichHandler):
    """Log bridge that feeds warning/error records into the terminal activity feed."""

    def __init__(self, ui: "TerminalUI") -> None:
        super().__init__(
            console=Console(file=io.StringIO(), force_terminal=False),
            show_time=False,
            show_path=False,
            markup=False,
            rich_tracebacks=False,
        )
        self.ui = ui

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno < logging.WARNING:
                return
            message = self.format(record) if self.formatter else record.getMessage()
            event_type = "circuit_breaker" if record.levelno >= logging.ERROR else "warning"
            self.ui.add_event(event_type, f"{record.name}: {message}")
        except Exception:
            # Logging handlers must never raise into application code.
            return


class TerminalUI:
    """Fire-and-forget state sink + rich Live renderer."""

    _EVENT_STYLES = {
        "opened": ("✅", "green"),
        "closed_profit": ("💰", "green"),
        "closed_loss": ("🔴", "red"),
        "closed": ("💰", "green"),
        "rejected": ("❌", "dim"),
        "rolled": ("🔄", "bright_blue"),
        "adjusted": ("🔧", "bright_blue"),
        "hedged": ("🛡️", "cyan"),
        "llm": ("🤖", "magenta"),
        "regime": ("📊", "yellow"),
        "warning": ("⚠️", "yellow"),
        "circuit_breaker": ("🚨", "bold red"),
        "paused": ("⏸️", "yellow"),
        "resumed": ("▶️", "green"),
    }

    def __init__(self, config):
        self.config = config
        ui_cfg = getattr(config, "terminal_ui", None)
        self.refresh_rate = float(getattr(ui_cfg, "refresh_rate", 0.5) or 0.5)
        self.max_activity_events = int(getattr(ui_cfg, "max_activity_events", 50) or 50)
        self.show_rejected_trades = bool(getattr(ui_cfg, "show_rejected_trades", True))
        self.compact_mode = bool(getattr(ui_cfg, "compact_mode", False))

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._live: Optional[Live] = None
        self._log_handler: Optional[logging.Handler] = None
        self._started_at = datetime.now(EASTERN_TZ)

        self._portfolio = {
            "balance": 0.0,
            "buying_power": None,
            "open_count": 0,
            "max_positions": 0,
            "daily_pnl": 0.0,
            "daily_risk_pct": 0.0,
            "max_daily_risk_pct": 0.0,
            "greeks": {},
        }
        self._metrics = {
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "wins": 0,
            "total": 0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "expectancy": 0.0,
            "today_pnl": 0.0,
            "today_pnl_pct": 0.0,
            "week_pnl": 0.0,
        }
        self._positions: list[dict] = []
        self._events: deque[dict] = deque(maxlen=self.max_activity_events)
        self._system_status: dict = {
            "scanner": "⏸️ Paused",
            "llm": "⏳ Initializing",
            "streaming": "⏳ Initializing",
            "api": "⏳ Initializing",
            "kill_switch": "🟢 Ready",
            "breakers": 0,
            "regime": "normal",
            "regime_confidence": None,
            "correlation": "normal",
            "econ": "N/A",
            "uptime": "0m",
            "last_scan": "N/A",
            "reconciliation": "⏳ Pending",
            "ml_scorer": "⏳ Pending",
            "theta_harvest": {"earned": 0.0, "target": 80.0},
            "next_scan": "N/A",
        }

    @classmethod
    def event_mapping(cls) -> dict[str, tuple[str, str]]:
        """Expose event type mapping for tests."""
        return dict(cls._EVENT_STYLES)

    def start(self) -> None:
        """Start the Live renderer in a daemon thread."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._install_logging_bridge()
            self._thread = threading.Thread(
                target=self._run_live_loop,
                name="terminal-ui",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Gracefully stop the renderer."""
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._remove_logging_bridge()

    def update_portfolio(
        self,
        balance,
        buying_power,
        open_count,
        max_positions,
        daily_pnl,
        daily_risk_pct,
        max_daily_risk_pct,
        greeks,
    ):
        """Update portfolio overview panel data."""
        with self._lock:
            self._portfolio = {
                "balance": float(balance or 0.0),
                "buying_power": buying_power,
                "open_count": int(open_count or 0),
                "max_positions": int(max_positions or 0),
                "daily_pnl": float(daily_pnl or 0.0),
                "daily_risk_pct": float(daily_risk_pct or 0.0),
                "max_daily_risk_pct": float(max_daily_risk_pct or 0.0),
                "greeks": dict(greeks or {}),
            }

    def update_metrics(
        self,
        sharpe,
        sortino,
        calmar,
        win_rate,
        wins,
        total,
        profit_factor,
        max_drawdown,
        expectancy,
        today_pnl,
        today_pnl_pct,
        week_pnl,
    ):
        """Update performance metrics panel data."""
        with self._lock:
            self._metrics = {
                "sharpe": float(sharpe or 0.0),
                "sortino": float(sortino or 0.0),
                "calmar": float(calmar or 0.0),
                "win_rate": float(win_rate or 0.0),
                "wins": int(wins or 0),
                "total": int(total or 0),
                "profit_factor": float(profit_factor or 0.0),
                "max_drawdown": float(max_drawdown or 0.0),
                "expectancy": float(expectancy or 0.0),
                "today_pnl": float(today_pnl or 0.0),
                "today_pnl_pct": float(today_pnl_pct or 0.0),
                "week_pnl": float(week_pnl or 0.0),
            }

    def update_positions(self, positions: list[dict]):
        """Update open positions table."""
        rows = [dict(item) for item in (positions or []) if isinstance(item, dict)]
        rows.sort(key=lambda item: int(item.get("dte", 9_999) or 9_999))
        with self._lock:
            self._positions = rows

    def add_trade_event(self, event_type: str, message: str):
        """Compatibility alias for add_event."""
        self.add_event(event_type, message)

    def add_event(self, event_type: str, message: str):
        """Add an event to the activity feed."""
        if event_type == "rejected" and not self.show_rejected_trades:
            return
        with self._lock:
            self._events.append(
                {
                    "time": datetime.now(EASTERN_TZ).strftime("%H:%M"),
                    "type": str(event_type or "warning"),
                    "message": str(message or ""),
                }
            )

    def update_system_status(self, **kwargs):
        """Update all system status indicators."""
        with self._lock:
            self._system_status.update(kwargs)

    def _build_layout(self) -> Layout:
        """Construct the full screen layout from current state."""
        with self._lock:
            portfolio = dict(self._portfolio)
            metrics = dict(self._metrics)
            positions = list(self._positions)
            events = list(self._events)
            status = dict(self._system_status)

        layout = Layout(name="root")
        if self.compact_mode:
            layout.split_column(
                Layout(self._build_header_panel(status), name="header", size=3),
                Layout(self._build_positions_panel(positions), name="positions", ratio=2),
                Layout(name="footer", ratio=1),
            )
            layout["footer"].split_row(
                Layout(self._build_activity_panel(events), name="activity"),
                Layout(self._build_system_status_panel(status), name="system"),
            )
            return layout

        layout.split_column(
            Layout(self._build_header_panel(status), name="header", size=3),
            Layout(name="top", size=12),
            Layout(self._build_positions_panel(positions), name="positions", ratio=2),
            Layout(name="bottom", size=14),
        )
        layout["top"].split_row(
            Layout(self._build_portfolio_panel(portfolio), name="portfolio"),
            Layout(self._build_metrics_panel(metrics, portfolio), name="metrics"),
        )
        layout["bottom"].split_row(
            Layout(self._build_activity_panel(events), name="activity"),
            Layout(self._build_system_status_panel(status), name="system"),
        )
        return layout

    def _run_live_loop(self) -> None:
        refresh_per_second = max(0.1, self.refresh_rate)
        wait_seconds = 1.0 / refresh_per_second
        try:
            with Live(
                self._build_layout(),
                refresh_per_second=refresh_per_second,
                screen=True,
                auto_refresh=False,
                transient=False,
            ) as live:
                self._live = live
                while not self._stop_event.is_set():
                    try:
                        live.update(self._build_layout(), refresh=True)
                    except Exception as exc:
                        logger.debug("Terminal UI render error: %s", exc)
                    self._stop_event.wait(timeout=wait_seconds)
        except Exception as exc:
            logger.warning("Terminal UI disabled due to renderer error: %s", exc)
        finally:
            self._live = None

    def _install_logging_bridge(self) -> None:
        """Keep file handlers; replace stream handlers with a warning/error feed bridge."""
        root = logging.getLogger()
        for handler in list(root.handlers):
            if isinstance(handler, logging.FileHandler):
                continue
            root.removeHandler(handler)

        if any(isinstance(handler, _ActivityRichHandler) for handler in root.handlers):
            return

        bridge = _ActivityRichHandler(self)
        bridge.setLevel(logging.WARNING)
        bridge.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(bridge)
        self._log_handler = bridge

    def _remove_logging_bridge(self) -> None:
        root = logging.getLogger()
        handler = self._log_handler
        if not handler:
            return
        try:
            root.removeHandler(handler)
        except Exception:
            pass
        self._log_handler = None

    def _build_header_panel(self, status: dict) -> Panel:
        now_et = datetime.now(EASTERN_TZ).strftime("%H:%M:%S ET")
        mode = str(getattr(self.config, "trading_mode", "paper")).upper()
        mode_style = "green" if mode == "PAPER" else "red"
        regime = str(status.get("regime", "normal")).upper()
        regime_conf = status.get("regime_confidence")
        conf_text = (
            f" ({float(regime_conf) * 100.0:.0f}%)"
            if isinstance(regime_conf, (int, float)) and float(regime_conf) <= 1.0
            else (f" ({float(regime_conf):.0f}%)" if isinstance(regime_conf, (int, float)) else "")
        )
        correlation = str(status.get("correlation", "normal"))

        title = Text("⚡ OPTIONS TRADING BOT ⚡", style="bold white", justify="center")
        subtitle = Text.assemble(
            "Mode: ",
            (mode, mode_style),
            " | Regime: ",
            (regime + conf_text, self._regime_color(regime)),
            " | Correlation: ",
            (correlation, self._correlation_color(correlation)),
            " | ⏱ ",
            (now_et, "cyan"),
        )
        subtitle.justify = "center"
        return Panel(Group(title, subtitle), border_style="bright_black")

    def _build_portfolio_panel(self, portfolio: dict) -> Panel:
        table = Table(show_header=False, box=None, expand=True, pad_edge=False)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(justify="right")

        balance = float(portfolio.get("balance", 0.0) or 0.0)
        buying_power = portfolio.get("buying_power")
        open_count = int(portfolio.get("open_count", 0) or 0)
        max_positions = max(1, int(portfolio.get("max_positions", 0) or 1))
        open_ratio = open_count / max_positions

        daily_risk_used = float(portfolio.get("daily_risk_pct", 0.0) or 0.0)
        max_daily_risk = float(portfolio.get("max_daily_risk_pct", 0.0) or 0.0)
        risk_ratio = (daily_risk_used / max_daily_risk) if max_daily_risk > 0 else 0.0

        greeks = portfolio.get("greeks", {}) if isinstance(portfolio.get("greeks"), dict) else {}
        delta = float(greeks.get("delta", 0.0) or 0.0)
        theta = float(greeks.get("theta", 0.0) or 0.0)
        vega = float(greeks.get("vega", 0.0) or 0.0)
        gamma = float(greeks.get("gamma", 0.0) or 0.0)

        table.add_row("Balance", f"${balance:,.2f}")
        table.add_row(
            "Buying Power",
            "N/A" if buying_power is None else f"${float(buying_power):,.2f}",
        )
        table.add_row(
            "Open Positions",
            f"[{self._utilization_color(open_ratio)}]{open_count}/{max_positions}[/]",
        )
        table.add_row(
            "Daily Risk Used",
            f"[{self._utilization_color(risk_ratio)}]{daily_risk_used:.2f}% / {max_daily_risk:.2f}%[/]",
        )
        table.add_row(
            "Portfolio Delta",
            f"[{'yellow' if abs(delta) > 40 else 'white'}]{delta:+.2f}[/]",
        )
        table.add_row(
            "Portfolio Theta",
            f"[{'green' if theta >= 0 else 'red'}]${theta:+.2f}/day[/]",
        )
        table.add_row("Portfolio Vega", f"{vega:+.2f}")
        table.add_row("Portfolio Gamma", f"{gamma:+.2f}")
        return Panel(table, title="PORTFOLIO OVERVIEW", border_style="bright_black")

    def _build_metrics_panel(self, metrics: dict, portfolio: dict) -> Panel:
        table = Table(show_header=False, box=None, expand=True, pad_edge=False)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(justify="right")

        balance = max(1.0, float(portfolio.get("balance", 0.0) or 0.0))
        today_pnl = float(metrics.get("today_pnl", 0.0) or 0.0)
        today_pct = float(metrics.get("today_pnl_pct", 0.0) or (today_pnl / balance * 100.0))
        week_pnl = float(metrics.get("week_pnl", 0.0) or 0.0)

        sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
        sortino = float(metrics.get("sortino", 0.0) or 0.0)
        calmar = float(metrics.get("calmar", 0.0) or 0.0)
        win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
        wins = int(metrics.get("wins", 0) or 0)
        total = int(metrics.get("total", 0) or 0)
        profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
        max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
        expectancy = float(metrics.get("expectancy", 0.0) or 0.0)

        table.add_row(
            "Today P/L",
            f"[{self._pnl_color(today_pnl)}]{today_pnl:+,.2f} ({today_pct:+.2f}%)[/]",
        )
        table.add_row("Week P/L", f"[{self._pnl_color(week_pnl)}]{week_pnl:+,.2f}[/]")
        table.add_row("Sharpe", f"[{self._ratio_color(sharpe)}]{sharpe:.2f}[/]")
        table.add_row("Sortino", f"[{self._ratio_color(sortino)}]{sortino:.2f}[/]")
        table.add_row("Calmar", f"[{self._ratio_color(calmar)}]{calmar:.2f}[/]")
        table.add_row("Win Rate", f"{win_rate * 100.0:.1f}% ({wins}/{total})")
        table.add_row(
            "Profit Factor",
            f"[{self._profit_factor_color(profit_factor)}]{profit_factor:.2f}[/]",
        )
        table.add_row(
            "Max Drawdown",
            f"[{self._drawdown_color(max_drawdown)}]-{max_drawdown * 100.0:.2f}%[/]",
        )
        table.add_row("Expectancy", f"${expectancy:+,.2f}/trade")
        return Panel(table, title="PERFORMANCE METRICS", border_style="bright_black")

    def _build_positions_panel(self, positions: list[dict]) -> Panel:
        table = Table(expand=True)
        table.add_column("Symbol", no_wrap=True)
        table.add_column("Strategy", overflow="fold")
        table.add_column("Qty", justify="right")
        table.add_column("DTE", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P/L", justify="right")
        table.add_column("% Max", justify="right")
        table.add_column("Δ", justify="right")

        if not positions:
            return Panel(Text("No open positions", style="dim", justify="center"), title="OPEN POSITIONS")

        for row in positions:
            symbol = str(row.get("symbol", "")).upper()
            strategy = self._strategy_label(str(row.get("strategy", "")))
            qty = int(row.get("qty", row.get("quantity", 0)) or 0)
            dte = int(row.get("dte", row.get("dte_remaining", 0)) or 0)
            entry = float(row.get("entry_price", row.get("entry_credit", 0.0)) or 0.0)
            current = float(row.get("current_price", row.get("current_value", 0.0)) or 0.0)

            pnl = row.get("pnl")
            if pnl is None:
                pnl = (entry - current) * qty * 100.0
            pnl = float(pnl or 0.0)

            pct_max = row.get("pct_of_max_profit", row.get("pct_max", None))
            if pct_max is None:
                max_profit = max(1e-6, entry * qty * 100.0)
                pct_max = (pnl / max_profit) * 100.0
            pct_max = float(pct_max or 0.0)

            delta = float(row.get("delta", 0.0) or 0.0)
            dte_style = self._dte_color(dte)
            pct_style = self._pct_max_color(pct_max)
            table.add_row(
                symbol,
                strategy,
                str(qty),
                f"[{dte_style}]{dte}[/]",
                f"${entry:.2f}",
                f"${current:.2f}",
                f"[{self._pnl_color(pnl)}]{pnl:+,.0f}[/]",
                f"[{pct_style}]{pct_max:.0f}%[/]",
                f"{delta:+.2f}",
            )

        return Panel(table, title="OPEN POSITIONS", border_style="bright_black")

    def _build_activity_panel(self, events: list[dict]) -> Panel:
        lines: list[Text] = []
        for item in reversed(events[-12:]):
            event_type = str(item.get("type", "warning"))
            emoji, style = self._EVENT_STYLES.get(event_type, ("•", "white"))
            stamp = str(item.get("time", "--:--"))
            message = str(item.get("message", ""))
            line = Text.assemble(
                (stamp, "cyan"),
                " ",
                (emoji, style),
                " ",
                (message, style),
            )
            lines.append(line)
        if not lines:
            lines = [Text("No recent activity", style="dim")]
        return Panel(Group(*lines), title="RECENT ACTIVITY", border_style="bright_black")

    def _build_system_status_panel(self, status: dict) -> Panel:
        table = Table(show_header=False, box=None, expand=True, pad_edge=False)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(justify="left")

        table.add_row("Scanner", str(status.get("scanner", "N/A")))
        table.add_row("LLM", str(status.get("llm", "N/A")))
        table.add_row("Streaming", str(status.get("streaming", "N/A")))
        table.add_row("Schwab API", str(status.get("api", "N/A")))
        table.add_row("Kill Switch", str(status.get("kill_switch", "N/A")))

        breakers = int(status.get("breakers", 0) or 0)
        breaker_color = "green" if breakers == 0 else ("yellow" if breakers <= 2 else "red")
        table.add_row("Circuit Breakers", f"[{breaker_color}]{breakers} tripped[/]")

        regime = str(status.get("regime", "normal")).upper()
        regime_dur = str(status.get("regime_duration", ""))
        table.add_row("Regime", f"[{self._regime_color(regime)}]{regime}[/] {regime_dur}")
        table.add_row("Next Econ Event", str(status.get("econ", "N/A")))

        uptime = status.get("uptime") or self._format_uptime()
        table.add_row("Uptime", str(uptime))
        table.add_row("Last Scan", str(status.get("last_scan", "N/A")))
        table.add_row("Reconciliation", str(status.get("reconciliation", "N/A")))
        table.add_row("ML Scorer", str(status.get("ml_scorer", "N/A")))

        theta = status.get("theta_harvest", {}) if isinstance(status.get("theta_harvest"), dict) else {}
        earned = float(theta.get("earned", 0.0) or 0.0)
        target = max(1.0, float(theta.get("target", 80.0) or 80.0))
        progress = ProgressBar(total=100, completed=max(0.0, min(100.0, earned)))
        theta_renderable = Group(progress, Text(f"{earned:.0f}% / {target:.0f}% target", style="white"))
        table.add_row("Theta Harvest", theta_renderable)

        return Panel(table, title="SYSTEM STATUS", border_style="bright_black")

    def _format_uptime(self) -> str:
        elapsed = datetime.now(EASTERN_TZ) - self._started_at
        seconds = max(0, int(elapsed.total_seconds()))
        hours, remainder = divmod(seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    @staticmethod
    def _utilization_color(value: float) -> str:
        if value > 0.9:
            return "red"
        if value >= 0.7:
            return "yellow"
        return "green"

    @staticmethod
    def _pnl_color(value: float) -> str:
        return "green" if value >= 0 else "red"

    @staticmethod
    def _dte_color(dte: int) -> str:
        if int(dte) <= 7:
            return "red"
        if int(dte) <= 14:
            return "yellow"
        return "white"

    @staticmethod
    def _pct_max_color(pct_max: float) -> str:
        if float(pct_max) > 50:
            return "green"
        if float(pct_max) < 0:
            return "red"
        return "white"

    @staticmethod
    def _ratio_color(value: float) -> str:
        if value > 1.0:
            return "green"
        if value >= 0.5:
            return "yellow"
        return "red"

    @staticmethod
    def _profit_factor_color(value: float) -> str:
        if value > 1.5:
            return "green"
        if value >= 1.0:
            return "yellow"
        return "red"

    @staticmethod
    def _drawdown_color(value: float) -> str:
        pct = abs(float(value or 0.0)) * 100.0
        if pct < 2.0:
            return "green"
        if pct <= 4.0:
            return "yellow"
        return "red"

    @staticmethod
    def _regime_color(regime: str) -> str:
        key = str(regime or "").upper()
        if "BULL" in key:
            return "green"
        if "BEAR" in key or "CRASH" in key:
            return "red"
        if "CHOP" in key:
            return "yellow"
        if "LOW_VOL" in key:
            return "bright_blue"
        if "MEAN_REVERSION" in key:
            return "magenta"
        return "white"

    @staticmethod
    def _correlation_color(correlation: str) -> str:
        key = str(correlation or "").lower()
        if key == "normal":
            return "green"
        if key == "stressed":
            return "yellow"
        if key == "crisis":
            return "red"
        return "white"

    @staticmethod
    def _strategy_label(strategy_name: str) -> str:
        mapping = {
            "bull_put_spread": "Credit Spread",
            "bear_call_spread": "Credit Spread",
            "credit_spreads": "Credit Spread",
            "iron_condor": "Iron Condor",
            "calendar_spread": "Calendar",
            "calendar_spreads": "Calendar",
            "naked_put": "Naked Put",
            "naked_puts": "Naked Put",
            "covered_call": "Covered Call",
            "covered_calls": "Covered Call",
            "strangle": "Strangle",
            "strangles": "Strangle",
            "broken_wing_butterfly": "BWB",
            "earnings_vol_crush": "EarningsVC",
        }
        key = str(strategy_name or "").strip().lower()
        label = mapping.get(key, strategy_name or "Unknown")
        return label if len(label) <= 14 else f"{label[:13]}…"
