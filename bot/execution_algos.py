"""Institutional-style execution algorithms for options order routing."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
import inspect
import time

from bot.config import ExecutionAlgoConfig
from bot.data_store import dump_json, load_json
from bot.number_utils import safe_float, safe_int


DEFAULT_SLIPPAGE_PATH = Path("bot/data/slippage_history.json")


class ExecutionAgent:
    """Spread-aware pace controller used by the adaptive execution mode."""

    def __init__(
        self,
        *,
        pause_threshold: float = 1.5,
        accelerate_threshold: float = 0.8,
        history_size: int = 20,
    ):
        self.pause_threshold = max(1.0, float(pause_threshold))
        self.accelerate_threshold = max(0.1, min(1.0, float(accelerate_threshold)))
        self._history: deque[float] = deque(maxlen=max(3, int(history_size)))

    def observe(self, spread_width: float) -> dict:
        spread = max(0.0001, float(spread_width))
        average = spread
        if self._history:
            average = sum(self._history) / len(self._history)
        ratio = spread / max(average, 0.0001)

        action = "normal"
        if len(self._history) >= 3:
            if ratio >= self.pause_threshold:
                action = "pause"
            elif ratio <= self.accelerate_threshold:
                action = "accelerate"

        self._history.append(spread)
        return {
            "action": action,
            "ratio": round(ratio, 4),
            "average_spread": round(average, 6),
            "latest_spread": round(spread, 6),
        }


class ExecutionAlgoEngine:
    """Route orders through smart-ladder, TWAP, iceberg, or adaptive algorithms."""

    def __init__(
        self,
        config: ExecutionAlgoConfig,
        schwab_client,
        *,
        slippage_path: Path | str = DEFAULT_SLIPPAGE_PATH,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ):
        self.config = config
        self.schwab = schwab_client
        self.slippage_path = Path(slippage_path)
        self.sleep_fn = sleep_fn or time.sleep
        self.execution_agent = ExecutionAgent(
            pause_threshold=float(config.adaptive_spread_pause_threshold),
            accelerate_threshold=float(config.adaptive_spread_accelerate_threshold),
        )

    @staticmethod
    def schedule_twap_slices(quantity: int, slices: int) -> list[int]:
        """Split total quantity into near-equal TWAP child slices."""
        qty = max(1, int(quantity))
        count = max(1, min(int(slices), qty))
        base = qty // count
        remainder = qty % count
        out: list[int] = []
        for idx in range(count):
            out.append(base + (1 if idx < remainder else 0))
        return out

    @staticmethod
    def generate_iceberg_children(quantity: int, visible_qty: int) -> list[int]:
        """Generate sequential child order quantities for iceberg execution."""
        qty = max(1, int(quantity))
        visible = max(1, int(visible_qty))
        out: list[int] = []
        remaining = qty
        while remaining > 0:
            child = min(visible, remaining)
            out.append(child)
            remaining -= child
        return out

    def execute(
        self,
        *,
        order_factory,
        midpoint_price: float,
        spread_width: float,
        side: str,
        quantity: int = 1,
        step_timeout_seconds: int = 45,
        step_timeouts: Optional[list[int]] = None,
        max_attempts: int = 4,
        shifts: Optional[list[float]] = None,
        total_timeout_seconds: Optional[int] = None,
        symbol: str = "",
        strategy: str = "",
        quote_spread_provider: Optional[Callable[[], float]] = None,
    ) -> dict:
        """Execute an order using the configured algorithm."""
        algo_type = str(self.config.algo_type or "smart_ladder").strip().lower()
        if not bool(getattr(self.config, "enabled", False)):
            algo_type = "smart_ladder"

        if algo_type == "twap":
            result = self._execute_twap(
                order_factory=order_factory,
                midpoint_price=midpoint_price,
                spread_width=spread_width,
                side=side,
                quantity=quantity,
                step_timeout_seconds=step_timeout_seconds,
                step_timeouts=step_timeouts,
                max_attempts=max_attempts,
                shifts=shifts,
            )
        elif algo_type == "iceberg":
            result = self._execute_iceberg(
                order_factory=order_factory,
                midpoint_price=midpoint_price,
                spread_width=spread_width,
                side=side,
                quantity=quantity,
                step_timeout_seconds=step_timeout_seconds,
                step_timeouts=step_timeouts,
                max_attempts=max_attempts,
                shifts=shifts,
                total_timeout_seconds=total_timeout_seconds,
            )
        elif algo_type == "adaptive":
            result = self._execute_adaptive(
                order_factory=order_factory,
                midpoint_price=midpoint_price,
                spread_width=spread_width,
                side=side,
                quantity=quantity,
                step_timeout_seconds=step_timeout_seconds,
                step_timeouts=step_timeouts,
                max_attempts=max_attempts,
                shifts=shifts,
                total_timeout_seconds=total_timeout_seconds,
                quote_spread_provider=quote_spread_provider,
            )
        else:
            result = self._execute_smart_ladder(
                order_factory=order_factory,
                midpoint_price=midpoint_price,
                spread_width=spread_width,
                side=side,
                quantity=quantity,
                step_timeout_seconds=step_timeout_seconds,
                step_timeouts=step_timeouts,
                max_attempts=max_attempts,
                shifts=shifts,
                total_timeout_seconds=total_timeout_seconds,
            )

        if not isinstance(result, dict):
            result = {"status": "REJECTED"}
        expected = safe_float(result.get("expected_fill_price"), midpoint_price)
        realized = safe_float(result.get("fill_price", result.get("requested_price")), expected)
        self._record_slippage(
            symbol=symbol,
            strategy=strategy,
            side=side,
            algo_type=algo_type,
            quantity=quantity,
            expected_fill_price=expected,
            realized_fill_price=realized,
            status=str(result.get("status", "")).upper(),
        )
        result["algo_type"] = algo_type
        return result

    def _execute_smart_ladder(
        self,
        *,
        order_factory,
        midpoint_price: float,
        spread_width: float,
        side: str,
        quantity: int,
        step_timeout_seconds: int,
        step_timeouts: Optional[list[int]],
        max_attempts: int,
        shifts: Optional[list[float]],
        total_timeout_seconds: Optional[int],
    ) -> dict:
        return self._execute_child(
            order_factory=order_factory,
            midpoint_price=midpoint_price,
            spread_width=spread_width,
            side=side,
            child_quantity=max(1, int(quantity)),
            step_timeout_seconds=step_timeout_seconds,
            step_timeouts=step_timeouts,
            max_attempts=max_attempts,
            shifts=shifts,
            total_timeout_seconds=total_timeout_seconds,
        )

    def _execute_twap(
        self,
        *,
        order_factory,
        midpoint_price: float,
        spread_width: float,
        side: str,
        quantity: int,
        step_timeout_seconds: int,
        step_timeouts: Optional[list[int]],
        max_attempts: int,
        shifts: Optional[list[float]],
    ) -> dict:
        schedule = self.schedule_twap_slices(quantity, int(self.config.twap_slices))
        window_seconds = max(10, int(self.config.twap_window_seconds))
        spacing = max(0.0, float(window_seconds) / max(1, len(schedule)))

        concession_ticks = 0
        child_results: list[dict] = []
        weighted_fill = 0.0
        filled_qty = 0
        filled_children = 0
        last_order_id = ""

        for idx, child_qty in enumerate(schedule):
            concession = concession_ticks * 0.01
            expected_price = self._price_with_concession(midpoint_price, side, concession)
            child = self._execute_child(
                order_factory=order_factory,
                midpoint_price=expected_price,
                spread_width=spread_width,
                side=side,
                child_quantity=child_qty,
                step_timeout_seconds=max(5, int(spacing) or int(step_timeout_seconds)),
                step_timeouts=step_timeouts,
                max_attempts=1,
                shifts=[0.0],
                total_timeout_seconds=max(5, int(spacing) or int(step_timeout_seconds)),
            )
            child["child_quantity"] = child_qty
            child["expected_fill_price"] = round(expected_price, 4)
            child_results.append(child)
            if str(child.get("order_id", "")).strip():
                last_order_id = str(child.get("order_id", "")).strip()

            status = str(child.get("status", "")).upper()
            if status == "FILLED":
                fill_price = safe_float(child.get("fill_price", child.get("requested_price")), expected_price)
                weighted_fill += fill_price * child_qty
                filled_qty += child_qty
                filled_children += 1
            else:
                concession_ticks += 1

            if idx < len(schedule) - 1 and spacing > 0:
                self.sleep_fn(min(spacing, 5.0))

        if filled_qty > 0:
            avg_fill = weighted_fill / filled_qty
            status = "FILLED" if filled_qty == sum(schedule) else "PARTIALLY_FILLED"
        else:
            avg_fill = midpoint_price
            status = str(child_results[-1].get("status", "REJECTED")).upper() if child_results else "REJECTED"

        return {
            "status": status,
            "filled_quantity": filled_qty,
            "requested_quantity": int(sum(schedule)),
            "filled_slices": filled_children,
            "slice_plan": list(schedule),
            "fill_price": round(avg_fill, 4),
            "expected_fill_price": round(midpoint_price, 4),
            "children": child_results,
            "order_id": last_order_id,
        }

    def _execute_iceberg(
        self,
        *,
        order_factory,
        midpoint_price: float,
        spread_width: float,
        side: str,
        quantity: int,
        step_timeout_seconds: int,
        step_timeouts: Optional[list[int]],
        max_attempts: int,
        shifts: Optional[list[float]],
        total_timeout_seconds: Optional[int],
    ) -> dict:
        children = self.generate_iceberg_children(quantity, int(self.config.iceberg_visible_qty))
        results: list[dict] = []
        weighted_fill = 0.0
        filled_qty = 0
        last_order_id = ""

        for child_qty in children:
            child = self._execute_child(
                order_factory=order_factory,
                midpoint_price=midpoint_price,
                spread_width=spread_width,
                side=side,
                child_quantity=child_qty,
                step_timeout_seconds=step_timeout_seconds,
                step_timeouts=step_timeouts,
                max_attempts=max_attempts,
                shifts=shifts,
                total_timeout_seconds=total_timeout_seconds,
            )
            child["child_quantity"] = child_qty
            results.append(child)
            if str(child.get("order_id", "")).strip():
                last_order_id = str(child.get("order_id", "")).strip()
            status = str(child.get("status", "")).upper()
            if status != "FILLED":
                break
            fill = safe_float(child.get("fill_price", child.get("requested_price")), midpoint_price)
            weighted_fill += fill * child_qty
            filled_qty += child_qty

        if filled_qty > 0:
            fill_price = weighted_fill / filled_qty
            status = "FILLED" if filled_qty == sum(children) else "PARTIALLY_FILLED"
        else:
            fill_price = midpoint_price
            status = str(results[-1].get("status", "REJECTED")).upper() if results else "REJECTED"

        return {
            "status": status,
            "fill_price": round(fill_price, 4),
            "expected_fill_price": round(midpoint_price, 4),
            "filled_quantity": filled_qty,
            "requested_quantity": int(sum(children)),
            "child_orders": list(children),
            "children": results,
            "order_id": last_order_id,
        }

    def _execute_adaptive(
        self,
        *,
        order_factory,
        midpoint_price: float,
        spread_width: float,
        side: str,
        quantity: int,
        step_timeout_seconds: int,
        step_timeouts: Optional[list[int]],
        max_attempts: int,
        shifts: Optional[list[float]],
        total_timeout_seconds: Optional[int],
        quote_spread_provider: Optional[Callable[[], float]],
    ) -> dict:
        current_spread = max(0.01, float(spread_width))
        if quote_spread_provider is not None:
            try:
                current_spread = max(0.01, float(quote_spread_provider()))
            except Exception:
                current_spread = max(0.01, float(spread_width))

        state = self.execution_agent.observe(current_spread)
        pause_checks = 0
        while state.get("action") == "pause" and pause_checks < 5:
            self.sleep_fn(0.1)
            pause_checks += 1
            if quote_spread_provider is None:
                break
            try:
                current_spread = max(0.01, float(quote_spread_provider()))
            except Exception:
                break
            state = self.execution_agent.observe(current_spread)

        local_shifts = list(shifts or [0.0, 0.10, 0.25, 0.40])
        local_timeout = int(step_timeout_seconds)
        if state.get("action") == "accelerate":
            local_shifts = [0.20, 0.35, 0.50]
            local_timeout = max(5, int(step_timeout_seconds * 0.6))

        result = self._execute_child(
            order_factory=order_factory,
            midpoint_price=midpoint_price,
            spread_width=current_spread,
            side=side,
            child_quantity=max(1, int(quantity)),
            step_timeout_seconds=local_timeout,
            step_timeouts=step_timeouts,
            max_attempts=max_attempts,
            shifts=local_shifts,
            total_timeout_seconds=total_timeout_seconds,
        )
        result["adaptive_state"] = state
        return result

    def _execute_child(
        self,
        *,
        order_factory,
        midpoint_price: float,
        spread_width: float,
        side: str,
        child_quantity: int,
        step_timeout_seconds: int,
        step_timeouts: Optional[list[int]],
        max_attempts: int,
        shifts: Optional[list[float]],
        total_timeout_seconds: Optional[int],
    ) -> dict:
        quantity = max(1, int(child_quantity))

        def _factory(price: float):
            return _call_order_factory(order_factory, price, quantity)

        result = self.schwab.place_order_with_ladder(
            order_factory=_factory,
            midpoint_price=float(midpoint_price),
            spread_width=max(0.01, float(spread_width)),
            side=str(side),
            step_timeout_seconds=max(5, int(step_timeout_seconds)),
            step_timeouts=step_timeouts,
            max_attempts=max(1, int(max_attempts)),
            shifts=shifts,
            total_timeout_seconds=total_timeout_seconds,
        )
        if not isinstance(result, dict):
            result = {"status": "REJECTED"}
        result.setdefault("expected_fill_price", round(float(midpoint_price), 4))
        return result

    @staticmethod
    def _price_with_concession(midpoint_price: float, side: str, concession: float) -> float:
        mid = max(0.01, float(midpoint_price))
        slip = max(0.0, float(concession))
        if str(side).lower() == "credit":
            return max(0.01, mid - slip)
        return max(0.01, mid + slip)

    def _record_slippage(
        self,
        *,
        symbol: str,
        strategy: str,
        side: str,
        algo_type: str,
        quantity: int,
        expected_fill_price: float,
        realized_fill_price: float,
        status: str,
    ) -> None:
        payload = load_json(
            self.slippage_path,
            {
                "fills": [],
                "by_strategy": {},
                "by_symbol": {},
                "by_dte_bucket": {},
                "algo_history": [],
            },
        )
        if not isinstance(payload, dict):
            payload = {
                "fills": [],
                "by_strategy": {},
                "by_symbol": {},
                "by_dte_bucket": {},
                "algo_history": [],
            }
        history = payload.get("algo_history")
        if not isinstance(history, list):
            history = []
            payload["algo_history"] = history

        slippage = realized_fill_price - expected_fill_price
        row = {
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "symbol": str(symbol).upper(),
            "strategy": str(strategy),
            "side": str(side),
            "algo_type": str(algo_type),
            "quantity": max(1, safe_int(quantity, 1)),
            "expected_fill_price": round(float(expected_fill_price), 4),
            "realized_fill_price": round(float(realized_fill_price), 4),
            "slippage": round(float(slippage), 4),
            "status": str(status).upper(),
        }
        history.append(row)
        payload["algo_history"] = history[-8000:]
        dump_json(self.slippage_path, payload)


def _call_order_factory(order_factory, price: float, quantity: int):
    """Support order_factory(price[, quantity]) for child-order algorithms."""
    try:
        sig = inspect.signature(order_factory)
        if len(sig.parameters) >= 2:
            return order_factory(price, quantity)
    except Exception:
        # Fall back to best effort call styles.
        pass

    try:
        return order_factory(price, quantity)
    except TypeError:
        return order_factory(price)
