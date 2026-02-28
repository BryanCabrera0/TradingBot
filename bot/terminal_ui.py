"""Minimal terminal dashboard for real-time bot monitoring.

Apple-inspired design: no borders, clean typography, intentional colour.
"""

from __future__ import annotations

import io
import logging
import select
import sys
import threading
from collections import deque
from datetime import datetime
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)
EASTERN_TZ = ZoneInfo("America/New_York")

# Semantic colours — used only where meaning requires it
_POS = "green"
_NEG = "red"
_WARN = "yellow"

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


# ── Logging bridge ─────────────────────────────────────────────────────────

class _ActivityRichHandler(RichHandler):
    """Feeds warning/error log records into the activity feed."""

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
            return


# ── Main TUI class ─────────────────────────────────────────────────────────

class TerminalUI:
    """Fire-and-forget state sink with Rich Live renderer."""

    # (dot_indicator, colour) — kept for test-API compatibility
    _EVENT_STYLES = {
        "opened":         ("·", _POS),
        "closed_profit":  ("·", _POS),
        "closed_loss":    ("·", _NEG),
        "closed":         ("·", _POS),
        "rejected":       ("·", "dim"),
        "rolled":         ("·", "white"),
        "adjusted":       ("·", "white"),
        "hedged":         ("·", "white"),
        "llm":            ("·", "white"),
        "regime":         ("·", "white"),
        "warning":        ("·", _WARN),
        "circuit_breaker":("·", _NEG),
        "paused":         ("·", _WARN),
        "resumed":        ("·", _POS),
    }

    _EVENT_LABELS = {
        "opened":         "opened",
        "closed_profit":  "profit",
        "closed_loss":    "loss",
        "closed":         "closed",
        "rejected":       "skip",
        "rolled":         "rolled",
        "adjusted":       "adjusted",
        "hedged":         "hedged",
        "llm":            "llm",
        "regime":         "regime",
        "warning":        "warn",
        "circuit_breaker":"alert",
        "paused":         "paused",
        "resumed":        "resumed",
    }

    _EVENT_COLORS = {
        "opened":         _POS,
        "closed_profit":  _POS,
        "closed":         _POS,
        "closed_loss":    _NEG,
        "rejected":       "dim",
        "warning":        _WARN,
        "circuit_breaker":_NEG,
        "paused":         _WARN,
        "resumed":        _POS,
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

        self._spinner_idx = 0
        self._scan_active = False

        self._command_listener: Optional[threading.Thread] = None
        self._on_command: Optional[Callable[[str], None]] = None
        self._last_command: Optional[str] = None

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
            "scanner": "Initializing",
            "llm": "Initializing",
            "streaming": "Initializing",
            "api": "Initializing",
            "kill_switch": "Ready",
            "breakers": 0,
            "regime": "normal",
            "regime_confidence": None,
            "regime_duration": "",
            "correlation": "normal",
            "econ": "N/A",
            "uptime": "0m",
            "last_scan": "N/A",
            "reconciliation": "Pending",
            "ml_scorer": "Pending",
            "theta_harvest": {"earned": 0.0, "target": 80.0},
            "next_scan": "N/A",
        }

    @classmethod
    def event_mapping(cls) -> dict[str, tuple[str, str]]:
        """Expose event type mapping for tests."""
        return dict(cls._EVENT_STYLES)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
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
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._remove_logging_bridge()

    # ── Public state setters ───────────────────────────────────────────────

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
        rows = [dict(item) for item in (positions or []) if isinstance(item, dict)]
        rows.sort(key=lambda item: int(item.get("dte", 9_999) or 9_999))
        with self._lock:
            self._positions = rows

    def add_trade_event(self, event_type: str, message: str):
        self.add_event(event_type, message)

    def add_event(self, event_type: str, message: str):
        if event_type == "rejected" and not self.show_rejected_trades:
            return
        with self._lock:
            self._events.append({
                "time": datetime.now(EASTERN_TZ).strftime("%H:%M"),
                "type": str(event_type or "warning"),
                "message": str(message or ""),
            })

    def update_system_status(self, **kwargs):
        with self._lock:
            self._system_status.update(kwargs)

    def set_scan_active(self, active: bool = True):
        with self._lock:
            self._scan_active = bool(active)

    def start_command_listener(self, on_command: Callable[[str], None]) -> None:
        self._on_command = on_command
        if self._command_listener and self._command_listener.is_alive():
            return
        self._command_listener = threading.Thread(
            target=self._stdin_loop,
            name="tui-command-listener",
            daemon=True,
        )
        self._command_listener.start()

    @property
    def last_command(self) -> Optional[str]:
        return self._last_command

    def _stdin_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.25)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            raw = sys.stdin.readline()
            if raw == "":
                return
            choice = raw.strip().lower()
            if choice in {"1", "menu", "return", "m"}:
                self._last_command = "menu"
                self.add_event("regime", "Returning to menu…")
                if self._on_command:
                    self._on_command("menu")
                return
            if choice in {"2", "exit", "quit", "q"}:
                self._last_command = "exit"
                self.add_event("regime", "Shutting down…")
                if self._on_command:
                    self._on_command("exit")
                return

    # ── Layout ────────────────────────────────────────────────────────────

    def _build_layout(self) -> Layout:
        with self._lock:
            portfolio = dict(self._portfolio)
            metrics = dict(self._metrics)
            positions = list(self._positions)
            events = list(self._events)
            status = dict(self._system_status)
            self._spinner_idx = (self._spinner_idx + 1) % len(SPINNER_FRAMES)
            spinner = SPINNER_FRAMES[self._spinner_idx] if self._scan_active else ""

        layout = Layout(name="root")

        if self.compact_mode:
            layout.split_column(
                Layout(self._render_header(status, spinner), name="header", size=1),
                Layout(self._render_positions(positions), name="positions", ratio=2),
                Layout(name="bottom", ratio=1),
                Layout(self._render_footer(), name="footer", size=2),
            )
            layout["bottom"].split_row(
                Layout(self._render_activity(events), name="activity"),
                Layout(self._render_system(status), name="system"),
            )
            return layout

        layout.split_column(
            Layout(self._render_header(status, spinner), name="header", size=1),
            Layout(name="body", ratio=1),
            Layout(self._render_footer(), name="footer", size=2),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=5),
            Layout(name="right", ratio=4),
        )
        layout["left"].split_column(
            Layout(self._render_portfolio(portfolio, metrics, status), name="portfolio", size=8),
            Layout(self._render_positions(positions), name="positions", ratio=1),
        )
        layout["right"].split_column(
            Layout(self._render_activity(events), name="activity", ratio=1),
            Layout(self._render_system(status), name="system", size=8),
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
            logger.warning("Terminal UI disabled: %s", exc)
        finally:
            self._live = None

    # ── Logging bridge ─────────────────────────────────────────────────────

    def _install_logging_bridge(self) -> None:
        root = logging.getLogger()
        for handler in list(root.handlers):
            if isinstance(handler, logging.FileHandler):
                continue
            root.removeHandler(handler)
        if any(isinstance(h, _ActivityRichHandler) for h in root.handlers):
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

    # ── Renderers ─────────────────────────────────────────────────────────

    def _render_header(self, status: dict, spinner: str) -> Table:
        """Single-line header: name · mode · time · spinner."""
        mode = str(getattr(self.config, "trading_mode", "paper")).lower()
        mode_style = "bold red" if mode == "live" else "dim"
        time_str = datetime.now(EASTERN_TZ).strftime("%H:%M:%S ET")
        spin = f"  {spinner}" if spinner else ""

        t = Table(box=None, show_header=False, expand=True, padding=(0, 1, 0, 1))
        t.add_column(justify="left")
        t.add_column(justify="right")
        t.add_row(
            Text.assemble(("TradingBot", "bold"), ("  ·  ", "dim"), (mode, mode_style)),
            Text.assemble((time_str, "dim"), (spin, "cyan")),
        )
        return t

    @staticmethod
    def _section(title: str, content) -> Group:
        """Wrap content under a titled horizontal rule."""
        return Group(
            Rule(title=f"[dim]{title}[/dim]", align="left", style="dim"),
            Padding(content, (0, 2, 0, 2)),
        )

    def _render_portfolio(self, portfolio: dict, metrics: dict, status: dict) -> Group:
        """Portfolio + performance in a compact 6-column grid."""
        balance = float(portfolio.get("balance", 0.0) or 0.0)
        daily_pnl = float(portfolio.get("daily_pnl", 0.0) or 0.0)
        open_count = int(portfolio.get("open_count", 0) or 0)
        max_pos = max(1, int(portfolio.get("max_positions", 1) or 1))
        risk_pct = float(portfolio.get("daily_risk_pct", 0.0) or 0.0)
        max_risk = float(portfolio.get("max_daily_risk_pct", 0.0) or 0.0)
        greeks = portfolio.get("greeks") if isinstance(portfolio.get("greeks"), dict) else {}
        delta = float(greeks.get("delta", 0.0) or 0.0)
        theta = float(greeks.get("theta", 0.0) or 0.0)
        vega = float(greeks.get("vega", 0.0) or 0.0)

        sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
        win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
        max_dd = float(metrics.get("max_drawdown", 0.0) or 0.0)
        expectancy = float(metrics.get("expectancy", 0.0) or 0.0)

        regime = str(status.get("regime", "—")).upper()
        regime_conf = status.get("regime_confidence")
        conf_str = ""
        if isinstance(regime_conf, (int, float)):
            v = float(regime_conf)
            conf_str = f" {v * 100:.0f}%" if v <= 1.0 else f" {v:.0f}%"

        risk_ratio = risk_pct / max_risk if max_risk > 0 else 0.0
        risk_color = _POS if risk_ratio < 0.7 else (_WARN if risk_ratio < 0.9 else _NEG)
        pos_color = _POS if open_count < max_pos else _WARN

        # 6-column grid: [label, value] × 3
        t = Table(box=None, show_header=False, padding=(0, 3, 0, 0), expand=True)
        for _ in range(6):
            t.add_column()

        t.add_row(
            "[dim]Balance[/]",    f"[white]${balance:,.0f}[/]",
            "[dim]Regime[/]",     f"[{self._regime_color(regime)}]{regime}{conf_str}[/]",
            "[dim]Sharpe[/]",     f"[{self._ratio_color(sharpe)}]{sharpe:.2f}[/]",
        )
        t.add_row(
            "[dim]Daily P/L[/]",  f"[{self._pnl_color(daily_pnl)}]{daily_pnl:+,.0f}[/]",
            "[dim]Δ Delta[/]",    f"[{'yellow' if abs(delta) > 40 else 'white'}]{delta:+.1f}[/]",
            "[dim]Win Rate[/]",   f"[white]{win_rate * 100:.0f}%[/]",
        )
        t.add_row(
            "[dim]Positions[/]",  f"[{pos_color}]{open_count}/{max_pos}[/]",
            "[dim]Θ Theta[/]",    f"[{self._pnl_color(theta)}]${theta:+.0f}/d[/]",
            "[dim]Drawdown[/]",   f"[{self._drawdown_color(max_dd)}]{max_dd * 100:.1f}%[/]",
        )
        t.add_row(
            "[dim]Risk[/]",       f"[{risk_color}]{risk_pct:.1f}%[/]",
            "[dim]ν Vega[/]",     f"[white]{vega:+.1f}[/]",
            "[dim]Expectancy[/]", f"[{self._pnl_color(expectancy)}]${expectancy:+.0f}[/]",
        )
        return self._section("Portfolio", t)

    def _render_positions(self, positions: list[dict]) -> Group:
        if not positions:
            return self._section("Positions", Text("No open positions", style="dim"))

        t = Table(
            box=None,
            show_header=True,
            expand=True,
            header_style="dim",
            padding=(0, 2, 0, 0),
        )
        t.add_column("Symbol", no_wrap=True)
        t.add_column("Strategy", style="dim")
        t.add_column("Qty", justify="right")
        t.add_column("DTE", justify="right")
        t.add_column("Entry", justify="right", style="dim")
        t.add_column("Current", justify="right")
        t.add_column("P/L", justify="right")
        t.add_column("Max%", justify="right")
        t.add_column("Δ", justify="right", style="dim")

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
            dte_color = _NEG if dte <= 7 else (_WARN if dte <= 14 else "white")

            t.add_row(
                f"[white]{symbol}[/]",
                strategy,
                f"[white]{qty}[/]",
                f"[{dte_color}]{dte}[/]",
                f"${entry:.2f}",
                f"[white]${current:.2f}[/]",
                f"[{self._pnl_color(pnl)}]{pnl:+,.0f}[/]",
                f"[{self._pct_max_color(pct_max)}]{pct_max:.0f}%[/]",
                f"{delta:+.2f}",
            )
        return self._section("Positions", t)

    def _render_activity(self, events: list[dict]) -> Group:
        lines: list[Text] = []
        for item in reversed(list(events)[-14:]):
            event_type = str(item.get("type", "warning"))
            label = self._EVENT_LABELS.get(event_type, event_type)
            color = self._EVENT_COLORS.get(event_type, "white")
            stamp = str(item.get("time", "--:--"))
            message = str(item.get("message", ""))
            line = Text.assemble(
                (stamp, "dim"),
                ("  ", ""),
                (f"{label:<10}", color),
                (message, "white"),
            )
            lines.append(line)
        if not lines:
            lines = [Text("Waiting for activity…", style="dim")]
        return self._section("Activity", Group(*lines))

    def _render_system(self, status: dict) -> Group:
        """Three-column system status grid."""
        scanner = self._status_text(status.get("scanner", "—"))
        streaming = self._status_text(status.get("streaming", "—"))
        llm = self._status_text(status.get("llm", "—"))
        api = self._status_text(status.get("api", "—"))
        ks = self._status_text(status.get("kill_switch", "—"))
        breakers = int(status.get("breakers", 0) or 0)
        bc_color = _POS if breakers == 0 else (_WARN if breakers <= 2 else _NEG)

        uptime = status.get("uptime") or self._format_uptime()
        last_scan = str(status.get("last_scan", "—"))
        regime = str(status.get("regime", "—")).upper()

        t = Table(box=None, show_header=False, padding=(0, 4, 0, 0), expand=True)
        for _ in range(6):
            t.add_column()

        t.add_row(
            "[dim]Scanner[/]",    scanner,
            "[dim]Streaming[/]",  streaming,
            "[dim]Kill Switch[/]",ks,
        )
        t.add_row(
            "[dim]LLM[/]",        llm,
            "[dim]API[/]",        api,
            "[dim]Breakers[/]",   Text(f"{breakers} tripped", style=bc_color),
        )
        t.add_row(
            "[dim]Uptime[/]",     Text(uptime, style="white"),
            "[dim]Last Scan[/]",  Text(last_scan, style="white"),
            "[dim]Regime[/]",     Text(regime, style=self._regime_color(regime)),
        )
        return self._section("System", t)

    @staticmethod
    def _render_footer() -> Group:
        return Group(
            Rule(style="dim"),
            Text("  m  menu    q  quit", style="dim"),
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _status_text(raw: str) -> Text:
        """Normalise status strings to clean coloured text."""
        text = str(raw or "—")
        for prefix in ("✅ ", "❌ ", "⚠️ ", "⏸️ ", "⏳ ", "🟢 ", "🟡 ", "● "):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        lower = text.lower()
        if any(k in lower for k in ("active", "healthy", "ok", "trained", "ready",
                                    "synced", "connected", "running")):
            return Text(text, style=_POS)
        if any(k in lower for k in ("fail", "down", "disabled", "error")):
            return Text(text, style=_NEG)
        if any(k in lower for k in ("degrad", "fallback", "stale", "warn",
                                    "paused", "pending")):
            return Text(text, style=_WARN)
        if "initializing" in lower:
            return Text(text, style="dim")
        return Text(text, style="white")

    # Backwards-compatible alias used by system panel previously
    _dot_status = _status_text

    def _format_uptime(self) -> str:
        elapsed = datetime.now(EASTERN_TZ) - self._started_at
        seconds = max(0, int(elapsed.total_seconds()))
        hours, remainder = divmod(seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

    @staticmethod
    def _pnl_color(value: float) -> str:
        return _POS if float(value) >= 0 else _NEG

    @staticmethod
    def _ratio_color(value: float) -> str:
        v = float(value)
        return _POS if v > 1.0 else (_WARN if v >= 0.5 else _NEG)

    @staticmethod
    def _pct_max_color(pct_max: float) -> str:
        v = float(pct_max)
        return _POS if v > 50 else (_NEG if v < 0 else "white")

    @staticmethod
    def _drawdown_color(value: float) -> str:
        pct = abs(float(value or 0.0)) * 100.0
        return _POS if pct < 2.0 else (_WARN if pct <= 4.0 else _NEG)

    @staticmethod
    def _regime_color(regime: str) -> str:
        key = str(regime or "").upper()
        if "BULL" in key:
            return _POS
        if "BEAR" in key or "CRASH" in key:
            return _NEG
        if "CHOP" in key:
            return _WARN
        return "white"

    @staticmethod
    def _utilization_color(value: float) -> str:
        return _POS if value < 0.7 else (_WARN if value < 0.9 else _NEG)

    @staticmethod
    def _profit_factor_color(value: float) -> str:
        return _POS if value > 1.5 else (_WARN if value >= 1.0 else _NEG)

    @staticmethod
    def _dte_color(dte: int) -> str:
        return _NEG if dte <= 7 else (_WARN if dte <= 14 else "white")

    @staticmethod
    def _correlation_color(correlation: str) -> str:
        key = str(correlation or "").lower()
        return _POS if key == "normal" else (_WARN if key == "stressed" else _NEG)

    @staticmethod
    def _strategy_label(strategy_name: str) -> str:
        mapping = {
            "bull_put_spread":      "credit spread",
            "bear_call_spread":     "credit spread",
            "credit_spreads":       "credit spread",
            "iron_condor":          "iron condor",
            "calendar_spread":      "calendar",
            "calendar_spreads":     "calendar",
            "naked_put":            "naked put",
            "naked_puts":           "naked put",
            "covered_call":         "covered call",
            "covered_calls":        "covered call",
            "strangle":             "strangle",
            "strangles":            "strangle",
            "broken_wing_butterfly":"butterfly",
            "earnings_vol_crush":   "earnings vc",
        }
        key = str(strategy_name or "").strip().lower()
        label = mapping.get(key, strategy_name or "unknown")
        return label if len(label) <= 14 else f"{label[:13]}…"
