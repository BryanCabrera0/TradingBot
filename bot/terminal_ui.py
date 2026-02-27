"""Claude CLI-inspired terminal UI dashboard for real-time bot monitoring.

Uses Rich with rounded borders, muted lavender/cyan palette, braille spinners,
and clean dot-prefixed activity indicators.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
import io
import logging
import select
import sys
import threading
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from rich import box

logger = logging.getLogger(__name__)
EASTERN_TZ = ZoneInfo("America/New_York")

# ── Claude-style palette ──────────────────────────────────────────────
BORDER = "dim"
ACCENT = "#a78bfa"          # lavender
ACCENT_DIM = "#818cf8"      # indigo
LABEL = "dim cyan"
POS_COLOR = "bright_green"
NEG_COLOR = "#ff6b6b"
MUTED = "dim white"
HEADER_BG = "bold white"
BADGE_PAPER = "bold black on bright_green"
BADGE_LIVE = "bold white on red"

SPINNER_FRAMES = list("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")


# ── Logging bridge ────────────────────────────────────────────────────
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
            return


# ── Main TUI class ────────────────────────────────────────────────────
class TerminalUI:
    """Fire-and-forget state sink + Rich Live renderer — Claude CLI aesthetic."""

    # Dot-based event indicators (replaces emoji clutter)
    _EVENT_STYLES = {
        "opened":          ("●", POS_COLOR),
        "closed_profit":   ("●", POS_COLOR),
        "closed_loss":     ("●", NEG_COLOR),
        "closed":          ("●", POS_COLOR),
        "rejected":        ("○", "dim"),
        "rolled":          ("●", ACCENT),
        "adjusted":        ("●", ACCENT_DIM),
        "hedged":          ("●", "cyan"),
        "llm":             ("●", ACCENT),
        "regime":          ("●", "yellow"),
        "warning":         ("▲", "yellow"),
        "circuit_breaker": ("▲", "bold red"),
        "paused":          ("‖", "yellow"),
        "resumed":         ("▶", POS_COLOR),
    }

    _EVENT_LABELS = {
        "opened":          "OPENED",
        "closed_profit":   "PROFIT",
        "closed_loss":     "LOSS",
        "closed":          "CLOSED",
        "rejected":        "SKIP",
        "rolled":          "ROLLED",
        "adjusted":        "ADJUST",
        "hedged":          "HEDGE",
        "llm":             "LLM",
        "regime":          "REGIME",
        "warning":         "WARN",
        "circuit_breaker": "ALERT",
        "paused":          "PAUSE",
        "resumed":         "RESUME",
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

        # Spinner state
        self._spinner_idx = 0
        self._scan_active = False

        # Command listener state
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

    # ── Lifecycle ─────────────────────────────────────────────────────

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

    # ── Public state setters (API contract) ───────────────────────────

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

    def set_scan_active(self, active: bool = True):
        """Toggle the header spinner for active scanning."""
        with self._lock:
            self._scan_active = bool(active)

    def start_command_listener(self, on_command: Callable[[str], None]) -> None:
        """Start a stdin listener that displays keybindings in the footer.

        ``on_command`` is called with ``"menu"`` or ``"exit"`` when the
        user presses the corresponding key.
        """
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
        """Return the last command the user entered, or None."""
        return self._last_command

    def _stdin_loop(self) -> None:
        """Read stdin in a non-blocking loop; dispatch recognised commands."""
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
                self.add_event("regime", "Returning to mode menu…")
                if self._on_command:
                    self._on_command("menu")
                return
            if choice in {"2", "exit", "quit", "q"}:
                self._last_command = "exit"
                self.add_event("regime", "Shutting down…")
                if self._on_command:
                    self._on_command("exit")
                return

    # ── Layout builder ────────────────────────────────────────────────

    def _build_layout(self) -> Layout:
        """Construct the full screen layout from current state."""
        with self._lock:
            portfolio = dict(self._portfolio)
            metrics = dict(self._metrics)
            positions = list(self._positions)
            events = list(self._events)
            status = dict(self._system_status)
            self._spinner_idx = (self._spinner_idx + 1) % len(SPINNER_FRAMES)
            spinner = SPINNER_FRAMES[self._spinner_idx] if self._scan_active else ""

        layout = Layout(name="root")
        keyhint = self._build_keyhint_bar()

        if self.compact_mode:
            layout.split_column(
                Layout(self._build_title_bar(status, spinner), name="header", size=3),
                Layout(self._build_positions_panel(positions), name="positions", ratio=2),
                Layout(name="panels", ratio=1),
                Layout(keyhint, name="keyhint", size=1),
            )
            layout["panels"].split_row(
                Layout(self._build_activity_panel(events), name="activity"),
                Layout(self._build_system_panel(status), name="system"),
            )
            return layout

        layout.split_column(
            Layout(self._build_title_bar(status, spinner), name="header", size=3),
            Layout(name="top", size=12),
            Layout(self._build_positions_panel(positions), name="positions", ratio=2),
            Layout(name="bottom", size=14),
            Layout(keyhint, name="keyhint", size=1),
        )
        layout["top"].split_row(
            Layout(self._build_portfolio_panel(portfolio), name="portfolio"),
            Layout(self._build_metrics_panel(metrics, portfolio), name="metrics"),
        )
        layout["bottom"].split_row(
            Layout(self._build_activity_panel(events), name="activity"),
            Layout(self._build_system_panel(status), name="system"),
        )
        return layout

    @staticmethod
    def _build_keyhint_bar() -> Text:
        """Render the bottom keybinding hint bar."""
        bar = Text.assemble(
            ("  [", MUTED),
            ("m", ACCENT),
            ("] Menu", MUTED),
            ("  ·  ", BORDER),
            ("[", MUTED),
            ("q", ACCENT),
            ("] Quit", MUTED),
            ("  ", ""),
        )
        bar.justify = "center"
        return bar

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

    # ── Logging bridge ────────────────────────────────────────────────

    def _install_logging_bridge(self) -> None:
        """Keep file handlers; replace stream handlers with a warning/error bridge."""
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

    # ── Panel builders ────────────────────────────────────────────────

    def _build_title_bar(self, status: dict, spinner: str) -> Panel:
        """Claude-style minimal title bar with mode/regime badges."""
        now_et = datetime.now(EASTERN_TZ).strftime("%H:%M:%S ET")
        mode = str(getattr(self.config, "trading_mode", "paper")).upper()
        mode_style = BADGE_PAPER if mode == "PAPER" else BADGE_LIVE
        regime = str(status.get("regime", "normal")).upper()
        regime_conf = status.get("regime_confidence")
        conf_text = ""
        if isinstance(regime_conf, (int, float)):
            val = float(regime_conf)
            conf_text = f" {val * 100.0:.0f}%" if val <= 1.0 else f" {val:.0f}%"
        correlation = str(status.get("correlation", "normal")).upper()

        spinner_text = f"[{ACCENT_DIM}]{spinner}[/] " if spinner else "  "

        title = Text.assemble(
            (spinner_text, ""),
            ("TradingBot", HEADER_BG),
            ("  ", ""),
            (f" {mode} ", mode_style),
            ("  ", ""),
        )
        subtitle = Text.assemble(
            ("Regime ", LABEL),
            (regime + conf_text, self._regime_color(regime)),
            ("  │  ", BORDER),
            ("Correlation ", LABEL),
            (correlation, self._correlation_color(correlation)),
            ("  │  ", BORDER),
            (now_et, MUTED),
        )
        subtitle.justify = "center"
        title.justify = "center"
        return Panel(Group(title, subtitle), box=box.ROUNDED, border_style=BORDER)

    def _build_portfolio_panel(self, portfolio: dict) -> Panel:
        table = Table(show_header=False, box=None, expand=True, pad_edge=False)
        table.add_column(style=LABEL, no_wrap=True)
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

        table.add_row("Balance", f"[white]${balance:,.2f}[/]")
        table.add_row(
            "Buying Power",
            "[white]N/A[/]" if buying_power is None else f"[white]${float(buying_power):,.2f}[/]",
        )
        table.add_row(
            "Positions",
            f"[{self._utilization_color(open_ratio)}]{open_count}/{max_positions}[/]",
        )
        table.add_row(
            "Daily Risk",
            f"[{self._utilization_color(risk_ratio)}]{daily_risk_used:.2f}% / {max_daily_risk:.2f}%[/]",
        )
        table.add_row(
            "Δ Delta",
            f"[{'yellow' if abs(delta) > 40 else 'white'}]{delta:+.2f}[/]",
        )
        table.add_row(
            "Θ Theta",
            f"[{POS_COLOR if theta >= 0 else NEG_COLOR}]${theta:+.2f}/day[/]",
        )
        table.add_row("ν Vega", f"[white]{vega:+.2f}[/]")
        table.add_row("Γ Gamma", f"[white]{gamma:+.2f}[/]")
        return Panel(table, title="[dim]Portfolio[/]", box=box.ROUNDED, border_style=BORDER)

    def _build_metrics_panel(self, metrics: dict, portfolio: dict) -> Panel:
        table = Table(show_header=False, box=None, expand=True, pad_edge=False)
        table.add_column(style=LABEL, no_wrap=True)
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
        table.add_row("Win Rate", f"[white]{win_rate * 100.0:.1f}%[/] [{MUTED}]({wins}/{total})[/]")
        table.add_row(
            "Profit Factor",
            f"[{self._profit_factor_color(profit_factor)}]{profit_factor:.2f}[/]",
        )
        table.add_row(
            "Max Drawdown",
            f"[{self._drawdown_color(max_drawdown)}]-{max_drawdown * 100.0:.2f}%[/]",
        )
        table.add_row("Expectancy", f"[white]${expectancy:+,.2f}/trade[/]")
        return Panel(table, title="[dim]Performance[/]", box=box.ROUNDED, border_style=BORDER)

    def _build_positions_panel(self, positions: list[dict]) -> Panel:
        table = Table(expand=True, box=box.SIMPLE_HEAVY, header_style="dim bold")
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
            return Panel(
                Text("No open positions", style=MUTED, justify="center"),
                title="[dim]Positions[/]",
                box=box.ROUNDED,
                border_style=BORDER,
            )

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
            table.add_row(
                f"[white]{symbol}[/]",
                f"[{MUTED}]{strategy}[/]",
                f"[white]{qty}[/]",
                f"[{self._dte_color(dte)}]{dte}[/]",
                f"[{MUTED}]${entry:.2f}[/]",
                f"[white]${current:.2f}[/]",
                f"[{self._pnl_color(pnl)}]{pnl:+,.0f}[/]",
                f"[{self._pct_max_color(pct_max)}]{pct_max:.0f}%[/]",
                f"[white]{delta:+.2f}[/]",
            )

        return Panel(table, title="[dim]Positions[/]", box=box.ROUNDED, border_style=BORDER)

    def _build_activity_panel(self, events: list[dict]) -> Panel:
        """Clean dot-prefixed activity feed."""
        lines: list[Text] = []
        for item in reversed(events[-12:]):
            event_type = str(item.get("type", "warning"))
            dot, style = self._EVENT_STYLES.get(event_type, ("●", "white"))
            label = self._EVENT_LABELS.get(event_type, "EVENT")
            stamp = str(item.get("time", "--:--"))
            message = str(item.get("message", ""))
            line = Text.assemble(
                (stamp, MUTED),
                ("  ", ""),
                (dot, style),
                (" ", ""),
                (f"{label:<7}", style),
                ("  ", ""),
                (message, "white"),
            )
            lines.append(line)
        if not lines:
            lines = [Text("Waiting for activity…", style=MUTED)]
        return Panel(Group(*lines), title="[dim]Activity[/]", box=box.ROUNDED, border_style=BORDER)

    def _build_system_panel(self, status: dict) -> Panel:
        """Two-column system status with dot indicators."""
        table = Table(show_header=False, box=None, expand=True, pad_edge=False)
        table.add_column(style=LABEL, no_wrap=True)
        table.add_column(justify="left")

        # Status helpers
        table.add_row("Scanner", self._dot_status(status.get("scanner", "N/A")))
        table.add_row("LLM", self._dot_status(status.get("llm", "N/A")))
        table.add_row("Streaming", self._dot_status(status.get("streaming", "N/A")))
        table.add_row("Schwab API", self._dot_status(status.get("api", "N/A")))
        table.add_row("Kill Switch", self._dot_status(status.get("kill_switch", "N/A")))

        breakers = int(status.get("breakers", 0) or 0)
        bc_color = POS_COLOR if breakers == 0 else ("yellow" if breakers <= 2 else NEG_COLOR)
        table.add_row("Breakers", f"[{bc_color}]{breakers} tripped[/]")

        regime = str(status.get("regime", "normal")).upper()
        regime_dur = str(status.get("regime_duration", ""))
        table.add_row("Regime", f"[{self._regime_color(regime)}]{regime}[/] [{MUTED}]{regime_dur}[/]")
        table.add_row("Econ Event", f"[white]{str(status.get('econ', 'N/A'))}[/]")

        uptime = status.get("uptime") or self._format_uptime()
        table.add_row("Uptime", f"[white]{uptime}[/]")
        table.add_row("Last Scan", f"[white]{str(status.get('last_scan', 'N/A'))}[/]")
        table.add_row("Recon", self._dot_status(status.get("reconciliation", "N/A")))
        table.add_row("ML Scorer", self._dot_status(status.get("ml_scorer", "N/A")))

        theta = status.get("theta_harvest", {}) if isinstance(status.get("theta_harvest"), dict) else {}
        earned = float(theta.get("earned", 0.0) or 0.0)
        target = max(1.0, float(theta.get("target", 80.0) or 80.0))
        progress = ProgressBar(total=100, completed=max(0.0, min(100.0, earned)))
        theta_renderable = Group(progress, Text(f"{earned:.0f}% / {target:.0f}%", style=MUTED))
        table.add_row("Θ Harvest", theta_renderable)

        return Panel(table, title="[dim]System[/]", box=box.ROUNDED, border_style=BORDER)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _dot_status(raw: str) -> Text:
        """Convert status strings with emojis to clean dot indicators."""
        text = str(raw or "N/A")
        # Strip existing emojis/indicators for re-rendering
        cleaned = text
        for prefix in ("✅ ", "❌ ", "⚠️ ", "⏸️ ", "⏳ ", "🟢 ", "🟡 ", "● "):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break

        # Determine color
        lower = text.lower()
        if any(k in lower for k in ("active", "healthy", "ok", "trained", "ready", "synced", "connected")):
            return Text.assemble(("● ", POS_COLOR), (cleaned, "white"))
        if any(k in lower for k in ("fail", "down", "disabled", "error")):
            return Text.assemble(("● ", NEG_COLOR), (cleaned, "white"))
        if any(k in lower for k in ("degrad", "fallback", "stale", "warn", "paused", "pending", "not enough")):
            return Text.assemble(("● ", "yellow"), (cleaned, "white"))
        if "initializing" in lower:
            return Text.assemble(("○ ", MUTED), (cleaned, MUTED))
        return Text.assemble(("● ", "white"), (cleaned, "white"))

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
            return NEG_COLOR
        if value >= 0.7:
            return "yellow"
        return POS_COLOR

    @staticmethod
    def _pnl_color(value: float) -> str:
        return POS_COLOR if value >= 0 else NEG_COLOR

    @staticmethod
    def _dte_color(dte: int) -> str:
        if int(dte) <= 7:
            return NEG_COLOR
        if int(dte) <= 14:
            return "yellow"
        return "white"

    @staticmethod
    def _pct_max_color(pct_max: float) -> str:
        if float(pct_max) > 50:
            return POS_COLOR
        if float(pct_max) < 0:
            return NEG_COLOR
        return "white"

    @staticmethod
    def _ratio_color(value: float) -> str:
        if value > 1.0:
            return POS_COLOR
        if value >= 0.5:
            return "yellow"
        return NEG_COLOR

    @staticmethod
    def _profit_factor_color(value: float) -> str:
        if value > 1.5:
            return POS_COLOR
        if value >= 1.0:
            return "yellow"
        return NEG_COLOR

    @staticmethod
    def _drawdown_color(value: float) -> str:
        pct = abs(float(value or 0.0)) * 100.0
        if pct < 2.0:
            return POS_COLOR
        if pct <= 4.0:
            return "yellow"
        return NEG_COLOR

    @staticmethod
    def _regime_color(regime: str) -> str:
        key = str(regime or "").upper()
        if "BULL" in key:
            return POS_COLOR
        if "BEAR" in key or "CRASH" in key:
            return NEG_COLOR
        if "CHOP" in key:
            return "yellow"
        if "LOW_VOL" in key:
            return ACCENT
        if "MEAN_REVERSION" in key:
            return ACCENT_DIM
        return "white"

    @staticmethod
    def _correlation_color(correlation: str) -> str:
        key = str(correlation or "").lower()
        if key == "normal":
            return POS_COLOR
        if key == "stressed":
            return "yellow"
        if key == "crisis":
            return NEG_COLOR
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
