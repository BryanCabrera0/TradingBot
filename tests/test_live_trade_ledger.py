import tempfile
import unittest
from pathlib import Path

from bot.live_trade_ledger import LiveTradeLedger


class LiveTradeLedgerTests(unittest.TestCase):
    def test_register_and_reconcile_entry_and_exit_orders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger_path = Path(tmp_dir) / "live_ledger.json"
            ledger = LiveTradeLedger(state_file=str(ledger_path))

            position_id = ledger.register_entry_order(
                strategy="bull_put_spread",
                symbol="SPY",
                quantity=1,
                max_loss=3.5,
                entry_credit=1.2,
                details={
                    "expiration": "2026-03-20",
                    "short_strike": 100,
                    "long_strike": 95,
                },
                entry_order_id="entry-1",
            )

            self.assertIn("entry-1", ledger.pending_entry_order_ids())

            ledger.reconcile_entry_order(
                "entry-1",
                status="FILLED",
                filled_at="2026-02-23T10:00:00-05:00",
                entry_credit=1.1,
                filled_quantity=1,
            )
            position = ledger.get_position(position_id)
            self.assertEqual(position["status"], "open")
            self.assertEqual(position["entry_credit"], 1.1)

            ok = ledger.register_exit_order(
                position_id=position_id,
                exit_order_id="exit-1",
                reason="profit target",
            )
            self.assertTrue(ok)
            self.assertIn("exit-1", ledger.pending_exit_order_ids())

            ledger.reconcile_exit_order(
                "exit-1",
                status="FILLED",
                filled_at="2026-02-23T11:00:00-05:00",
                close_value=0.4,
            )
            closed = ledger.get_position(position_id)
            self.assertEqual(closed["status"], "closed")
            self.assertEqual(closed["realized_pnl"], 70.0)

    def test_partial_fill_helpers_update_pending_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger = LiveTradeLedger(state_file=str(Path(tmp_dir) / "live_ledger.json"))
            position_id = ledger.register_entry_order(
                strategy="bull_put_spread",
                symbol="SPY",
                quantity=2,
                max_loss=4.0,
                entry_credit=1.4,
                details={
                    "expiration": "2026-03-20",
                    "short_strike": 100,
                    "long_strike": 95,
                },
                entry_order_id="entry-partial",
            )

            changed = ledger.apply_partial_entry_fill(
                "entry-partial",
                filled_quantity=1.0,
                entry_credit=1.25,
            )
            self.assertTrue(changed)
            position = ledger.get_position(position_id)
            self.assertEqual(position["entry_filled_quantity"], 1.0)
            self.assertEqual(position["entry_credit"], 1.25)

    def test_close_missing_from_broker_marks_external_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger_path = Path(tmp_dir) / "live_ledger.json"
            ledger = LiveTradeLedger(state_file=str(ledger_path))

            position_id = ledger.register_entry_order(
                strategy="covered_call",
                symbol="AAPL",
                quantity=1,
                max_loss=0.0,
                entry_credit=1.0,
                details={"expiration": "2026-03-20", "short_strike": 220},
                entry_order_id="",
                entry_order_status="FILLED",
                opened_at="2026-02-23T09:40:00-05:00",
            )

            changed = ledger.close_missing_from_broker(
                open_strategy_symbols=set(),
                position_symbol_resolver=lambda _position: {"AAPL_032026C220"},
                close_metadata_resolver=lambda _position, _symbols: {
                    "close_value": 0.0,
                    "realized_pnl": 100.0,
                    "exit_reason": "exercise_or_assignment",
                },
            )
            self.assertEqual(changed, 1)
            closed = ledger.get_position(position_id)
            self.assertEqual(closed["status"], "closed_external")
            self.assertEqual(closed["realized_pnl"], 100.0)

    def test_partial_exit_fill_keeps_position_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger = LiveTradeLedger(state_file=str(Path(tmp_dir) / "live_ledger.json"))
            position_id = ledger.register_entry_order(
                strategy="bull_put_spread",
                symbol="SPY",
                quantity=3,
                max_loss=4.0,
                entry_credit=1.2,
                details={
                    "expiration": "2026-03-20",
                    "short_strike": 100,
                    "long_strike": 95,
                },
                entry_order_id="",
                entry_order_status="FILLED",
                opened_at="2026-02-23T09:45:00-05:00",
            )
            ledger.register_exit_order(
                position_id=position_id,
                exit_order_id="exit-partial",
                reason="scale out",
                quantity=1,
            )

            ledger.reconcile_exit_order(
                "exit-partial",
                status="FILLED",
                filled_at="2026-02-23T10:30:00-05:00",
                close_value=0.6,
            )
            updated = ledger.get_position(position_id)
            self.assertEqual(updated["status"], "open")
            self.assertEqual(updated["quantity"], 2)
            self.assertTrue(updated["partial_closed"])

    def test_reconcile_exit_order_handles_debit_entry_pnl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger = LiveTradeLedger(state_file=str(Path(tmp_dir) / "live_ledger.json"))
            position_id = ledger.register_entry_order(
                strategy="hedge",
                symbol="SPY",
                quantity=1,
                max_loss=2.0,
                entry_credit=-2.0,  # debit paid to open
                details={"expiration": "2026-03-20", "short_strike": 500},
                entry_order_id="",
                entry_order_status="FILLED",
                opened_at="2026-02-23T09:45:00-05:00",
            )
            ledger.register_exit_order(
                position_id=position_id,
                exit_order_id="exit-debit",
                reason="hedge unwind",
            )

            ledger.reconcile_exit_order(
                "exit-debit",
                status="FILLED",
                filled_at="2026-02-23T11:30:00-05:00",
                close_value=3.5,  # credit received to close
            )
            closed = ledger.get_position(position_id)
            self.assertEqual(closed["status"], "closed")
            self.assertEqual(closed["realized_pnl"], 150.0)


if __name__ == "__main__":
    unittest.main()
