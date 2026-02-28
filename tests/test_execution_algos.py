import tempfile
import unittest
from pathlib import Path

from bot.config import ExecutionAlgoConfig
from bot.data_store import load_json
from bot.execution_algos import ExecutionAgent, ExecutionAlgoEngine


class StubSchwab:
    def __init__(self):
        self.calls: list[dict] = []

    def place_order_with_ladder(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        requested = float(kwargs.get("midpoint_price", 1.0))
        order_spec = kwargs["order_factory"](requested)
        qty = int(order_spec.get("qty", 1))
        return {
            "status": "FILLED",
            "fill_price": requested,
            "requested_price": requested,
            "filled_quantity": qty,
        }


def order_factory(price: float, qty: int = 1) -> dict:
    return {"price": price, "qty": qty}


class ExecutionAlgoTests(unittest.TestCase):
    def test_twap_slice_scheduling(self) -> None:
        schwab = StubSchwab()
        cfg = ExecutionAlgoConfig(
            enabled=True, algo_type="twap", twap_slices=4, twap_window_seconds=60
        )
        engine = ExecutionAlgoEngine(cfg, schwab, sleep_fn=lambda _: None)

        result = engine.execute(
            order_factory=order_factory,
            midpoint_price=1.25,
            spread_width=0.4,
            side="credit",
            quantity=7,
            step_timeout_seconds=15,
            symbol="SPY",
            strategy="bull_put_spread",
        )

        self.assertEqual(result["slice_plan"], [2, 2, 2, 1])
        self.assertEqual(len(schwab.calls), 4)

    def test_iceberg_child_order_generation(self) -> None:
        schwab = StubSchwab()
        cfg = ExecutionAlgoConfig(
            enabled=True, algo_type="iceberg", iceberg_visible_qty=1
        )
        engine = ExecutionAlgoEngine(cfg, schwab, sleep_fn=lambda _: None)

        result = engine.execute(
            order_factory=order_factory,
            midpoint_price=1.10,
            spread_width=0.3,
            side="credit",
            quantity=4,
            step_timeout_seconds=10,
            symbol="QQQ",
            strategy="iron_condor",
        )

        self.assertEqual(result["child_orders"], [1, 1, 1, 1])
        self.assertEqual(len(schwab.calls), 4)

    def test_adaptive_spread_pause_resume_logic(self) -> None:
        agent = ExecutionAgent(pause_threshold=1.5, accelerate_threshold=0.8)
        agent.observe(0.20)
        agent.observe(0.22)
        agent.observe(0.21)

        paused = agent.observe(0.40)
        resumed = agent.observe(0.12)

        self.assertEqual(paused["action"], "pause")
        self.assertEqual(resumed["action"], "accelerate")

    def test_slippage_tracking_calculation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            schwab = StubSchwab()
            path = Path(tmp_dir) / "slippage_history.json"
            cfg = ExecutionAlgoConfig(enabled=True, algo_type="smart_ladder")
            engine = ExecutionAlgoEngine(
                cfg, schwab, slippage_path=path, sleep_fn=lambda _: None
            )

            engine.execute(
                order_factory=order_factory,
                midpoint_price=1.00,
                spread_width=0.20,
                side="credit",
                quantity=1,
                step_timeout_seconds=10,
                symbol="AAPL",
                strategy="bull_put_spread",
            )

            payload = load_json(path, {})
            history = payload.get("algo_history", [])
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["expected_fill_price"], 1.0)
            self.assertEqual(history[0]["realized_fill_price"], 1.0)
            self.assertEqual(history[0]["slippage"], 0.0)


if __name__ == "__main__":
    unittest.main()
