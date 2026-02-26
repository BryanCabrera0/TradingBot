import json
import unittest
from datetime import datetime
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock

from bot.analysis import SpreadAnalysis
from bot.config import BotConfig
from bot.orchestrator import TradingBot
from bot.strategies.base import TradeSignal


def make_live_config() -> BotConfig:
    cfg = BotConfig()
    cfg.trading_mode = "live"
    cfg.scanner.enabled = False
    cfg.news.enabled = False
    cfg.llm.enabled = False
    cfg.credit_spreads.enabled = True
    cfg.covered_calls.enabled = False
    cfg.iron_condors.enabled = False
    return cfg


def make_paper_config() -> BotConfig:
    cfg = make_live_config()
    cfg.trading_mode = "paper"
    return cfg


class OrchestratorLiveExecutionTests(unittest.TestCase):
    def test_live_entry_registers_pending_ledger_position(self) -> None:
        bot = TradingBot(make_live_config())
        bot.live_ledger = mock.Mock()
        bot.schwab.build_bull_put_spread = mock.Mock(return_value={"order": "spec"})
        bot.schwab.place_order_with_ladder = mock.Mock(
            return_value={
                "order_id": "abc123",
                "status": "FILLED",
                "midpoint_price": 1.4,
                "fill_price": 1.35,
                "requested_price": 1.35,
            }
        )

        signal = TradeSignal(
            action="open",
            strategy="bull_put_spread",
            symbol="SPY",
            quantity=2,
            analysis=SpreadAnalysis(
                symbol="SPY",
                strategy="bull_put_spread",
                expiration="2026-03-20",
                dte=25,
                short_strike=100,
                long_strike=95,
                credit=1.4,
                max_loss=3.6,
                probability_of_profit=0.63,
                score=58,
            ),
        )

        opened = bot._execute_live_entry(
            signal,
            details={
                "expiration": "2026-03-20",
                "short_strike": 100,
                "long_strike": 95,
            },
        )

        self.assertTrue(opened)
        bot.live_ledger.register_entry_order.assert_called_once()

    def test_live_entry_uses_smart_ladder_width_fractions(self) -> None:
        bot = TradingBot(make_live_config())
        bot.live_ledger = mock.Mock()
        bot.config.execution.smart_ladder_enabled = True
        bot.config.execution.ladder_width_fractions = [0.0, 0.10, 0.25, 0.40]
        bot.config.execution.ladder_step_timeouts_seconds = [45, 45, 45, 30]
        bot.schwab.build_bull_put_spread = mock.Mock(return_value={"order": "spec"})
        bot.schwab.place_order_with_ladder = mock.Mock(
            return_value={
                "order_id": "abc123",
                "status": "FILLED",
                "midpoint_price": 1.4,
                "fill_price": 1.32,
                "requested_price": 1.32,
            }
        )

        signal = TradeSignal(
            action="open",
            strategy="bull_put_spread",
            symbol="SPY",
            quantity=1,
            analysis=SpreadAnalysis(
                symbol="SPY",
                strategy="bull_put_spread",
                expiration="2026-03-20",
                dte=25,
                short_strike=100,
                long_strike=95,
                credit=1.4,
                max_loss=3.6,
                probability_of_profit=0.63,
                score=58,
            ),
        )

        opened = bot._execute_live_entry(signal, details={"expiration": "2026-03-20", "short_strike": 100, "long_strike": 95})

        self.assertTrue(opened)
        kwargs = bot.schwab.place_order_with_ladder.call_args.kwargs
        self.assertEqual(kwargs["shifts"], [0.0, 0.10, 0.25, 0.40])
        self.assertEqual(kwargs["step_timeouts"], [45, 45, 45, 30])

    def test_live_entry_partial_fill_tracks_partial_position(self) -> None:
        bot = TradingBot(make_live_config())
        bot.live_ledger = mock.Mock()
        bot.schwab.get_quote = mock.Mock(return_value={"quote": {"lastPrice": 500.0}})
        bot.schwab.build_bull_put_spread = mock.Mock(return_value={"order": "spec"})
        bot.schwab.place_order_with_ladder = mock.Mock(
            return_value={
                "order_id": "abc123",
                "status": "PARTIALLY_FILLED",
                "filled_quantity": 1,
                "midpoint_price": 1.4,
                "fill_price": 1.35,
                "requested_price": 1.35,
            }
        )

        signal = TradeSignal(
            action="open",
            strategy="bull_put_spread",
            symbol="SPY",
            quantity=2,
            analysis=SpreadAnalysis(
                symbol="SPY",
                strategy="bull_put_spread",
                expiration="2026-03-20",
                dte=25,
                short_strike=100,
                long_strike=95,
                credit=1.4,
                max_loss=3.6,
                probability_of_profit=0.63,
                score=58,
            ),
        )

        opened = bot._execute_live_entry(
            signal,
            details={"expiration": "2026-03-20", "short_strike": 100, "long_strike": 95},
        )

        self.assertTrue(opened)
        kwargs = bot.live_ledger.register_entry_order.call_args.kwargs
        self.assertEqual(kwargs["quantity"], 1)
        self.assertEqual(kwargs["entry_order_id"], "")
        self.assertEqual(kwargs["entry_order_status"], "FILLED")
        self.assertEqual(signal.quantity, 1)

    def test_live_exit_places_close_order_and_marks_closing(self) -> None:
        bot = TradingBot(make_live_config())
        bot.live_ledger = mock.Mock()
        bot.live_ledger.get_position.return_value = {
            "position_id": "live_1",
            "strategy": "bull_put_spread",
            "symbol": "SPY",
            "status": "open",
            "quantity": 1,
            "entry_credit": 1.25,
            "current_value": 0.55,
            "details": {
                "expiration": "2026-03-20",
                "short_strike": 100,
                "long_strike": 95,
            },
        }
        bot.schwab.build_bull_put_spread_close = mock.Mock(return_value={"order": "close"})
        bot.schwab.place_order = mock.Mock(
            return_value={"order_id": "close123", "status": "PLACED"}
        )

        signal = TradeSignal(
            action="close",
            strategy="bull_put_spread",
            symbol="SPY",
            position_id="live_1",
            reason="profit target",
        )

        closed = bot._execute_live_exit(signal)

        self.assertTrue(closed)
        bot.schwab.build_bull_put_spread_close.assert_called_once()
        bot.live_ledger.register_exit_order.assert_called_once_with(
            position_id="live_1",
            exit_order_id="close123",
            reason="profit target",
        )

    def test_live_iron_condor_entry_uses_condor_builder(self) -> None:
        bot = TradingBot(make_live_config())
        bot.live_ledger = mock.Mock()
        bot.schwab.build_iron_condor = mock.Mock(return_value={"order": "condor"})
        bot.schwab.place_order = mock.Mock(
            return_value={"order_id": "condor1", "status": "PLACED"}
        )

        signal = TradeSignal(
            action="open",
            strategy="iron_condor",
            symbol="SPY",
            quantity=1,
            analysis=SpreadAnalysis(
                symbol="SPY",
                strategy="iron_condor",
                expiration="2026-03-20",
                dte=30,
                short_strike=0,
                long_strike=0,
                put_short_strike=95,
                put_long_strike=90,
                call_short_strike=110,
                call_long_strike=115,
                credit=1.1,
                max_loss=3.9,
                probability_of_profit=0.61,
                score=57,
            ),
        )

        opened = bot._execute_live_entry(signal, details={"expiration": "2026-03-20"})

        self.assertTrue(opened)
        bot.schwab.build_iron_condor.assert_called_once()

    def test_reconcile_cancels_stale_entry_orders(self) -> None:
        bot = TradingBot(make_live_config())
        bot.live_ledger = mock.Mock()
        bot.live_ledger.pending_entry_order_ids.return_value = ["stale-order"]
        bot.live_ledger.pending_exit_order_ids.return_value = []
        bot.live_ledger.get_position_by_order_id.return_value = {
            "entry_order_time": "2026-02-23T09:00:00-05:00"
        }
        bot.schwab.get_order = mock.Mock(
            return_value={"status": "WORKING", "enteredTime": "2026-02-23T09:00:00-05:00"}
        )
        bot.schwab.cancel_order = mock.Mock()
        bot._now_eastern = mock.Mock(
            return_value=datetime.fromisoformat("2026-02-23T09:30:00-05:00")
        )
        bot.config.execution.stale_order_minutes = 5
        bot.config.execution.cancel_stale_orders = True

        bot._reconcile_live_orders()

        bot.schwab.cancel_order.assert_called_once_with("stale-order")
        bot.live_ledger.reconcile_entry_order.assert_called_with(
            "stale-order",
            status="CANCELED",
        )

    def test_reconcile_cancels_working_entry_on_market_move(self) -> None:
        bot = TradingBot(make_live_config())
        bot.live_ledger = mock.Mock()
        bot.live_ledger.pending_entry_order_ids.return_value = ["work-order"]
        bot.live_ledger.pending_exit_order_ids.return_value = []
        bot.live_ledger.get_position_by_order_id.return_value = {
            "position_id": "live_1",
            "status": "working",
            "symbol": "SPY",
            "strategy": "bull_put_spread",
            "quantity": 1,
            "max_loss": 3.6,
            "entry_credit": 1.3,
            "details": {"entry_underlying_price": 100.0},
        }
        bot.schwab.get_order = mock.Mock(return_value={"status": "WORKING", "enteredTime": "2026-02-23T09:00:00-05:00"})
        bot.schwab.get_quote = mock.Mock(return_value={"quote": {"lastPrice": 102.0}})
        bot._cancel_stale_live_order = mock.Mock()
        bot._reenter_canceled_working_entry = mock.Mock()
        bot.config.execution.market_move_cancel_pct = 1.0

        bot._reconcile_live_orders()

        bot._cancel_stale_live_order.assert_called_once_with("work-order", side="entry")
        bot._reenter_canceled_working_entry.assert_called_once()

    def test_bootstrap_imports_existing_broker_spread_position(self) -> None:
        bot = TradingBot(make_live_config())
        bot.live_ledger = mock.Mock()
        bot.live_ledger.list_positions.return_value = []
        bot.schwab.get_positions = mock.Mock(
            return_value=[
                {
                    "instrument": {"assetType": "OPTION", "symbol": "SPY  260320P00100000"},
                    "longQuantity": 0.0,
                    "shortQuantity": 1.0,
                },
                {
                    "instrument": {"assetType": "OPTION", "symbol": "SPY  260320P00095000"},
                    "longQuantity": 1.0,
                    "shortQuantity": 0.0,
                },
            ]
        )

        imported = bot._bootstrap_live_ledger_from_broker()

        self.assertEqual(imported, 1)
        bot.live_ledger.register_entry_order.assert_called_once()

    def test_execute_roll_does_not_close_without_replacement(self) -> None:
        bot = TradingBot(make_live_config())
        bot._get_tracked_positions = mock.Mock(
            return_value=[
                {
                    "position_id": "pos1",
                    "status": "open",
                    "symbol": "SPY",
                    "strategy": "bull_put_spread",
                    "dte_remaining": 7,
                    "details": {"roll_count": 0},
                }
            ]
        )
        bot.roll_manager.evaluate = mock.Mock(
            return_value=mock.Mock(should_roll=True, reason="ok", min_credit_required=0.1)
        )
        bot._get_chain_data = mock.Mock(return_value=({"calls": {}, "puts": {}}, 0.0))
        bot._execute_exit = mock.Mock()
        bot._try_execute_entry = mock.Mock()

        bot._execute_roll(
            TradeSignal(
                action="roll",
                strategy="bull_put_spread",
                symbol="SPY",
                position_id="pos1",
                quantity=1,
                metadata={},
            )
        )

        bot._execute_exit.assert_not_called()
        bot._try_execute_entry.assert_not_called()

    def test_execute_roll_closes_and_opens_when_replacement_exists(self) -> None:
        bot = TradingBot(make_live_config())
        bot._get_tracked_positions = mock.Mock(
            return_value=[
                {
                    "position_id": "pos1",
                    "status": "open",
                    "symbol": "SPY",
                    "strategy": "bull_put_spread",
                    "dte_remaining": 7,
                    "details": {"roll_count": 0},
                }
            ]
        )
        bot.roll_manager.evaluate = mock.Mock(
            return_value=mock.Mock(should_roll=True, reason="ok", min_credit_required=0.1)
        )
        bot._get_chain_data = mock.Mock(return_value=({"calls": {"2026-04-17": [{"dte": 40}]}, "puts": {"2026-04-17": [{"dte": 40}]}}, 500.0))
        bot.technicals.get_context = mock.Mock(return_value=None)
        candidate = TradeSignal(
            action="open",
            strategy="bull_put_spread",
            symbol="SPY",
            analysis=SpreadAnalysis(
                symbol="SPY",
                strategy="bull_put_spread",
                expiration="2026-04-17",
                dte=40,
                short_strike=490,
                long_strike=480,
                credit=1.2,
                max_loss=8.8,
                probability_of_profit=0.65,
                score=70,
            ),
            metadata={},
        )
        strategy = mock.Mock()
        strategy.scan_for_entries = mock.Mock(return_value=[candidate])
        bot.strategies = [strategy]
        bot._execute_exit = mock.Mock()
        bot._try_execute_entry = mock.Mock()

        bot._execute_roll(
            TradeSignal(
                action="roll",
                strategy="bull_put_spread",
                symbol="SPY",
                position_id="pos1",
                quantity=1,
                metadata={},
            )
        )

        bot._execute_exit.assert_called_once()
        bot._try_execute_entry.assert_called_once()

    def test_orchestrator_roll_execution_links_metadata(self) -> None:
        bot = TradingBot(make_paper_config())
        source_position = {
            "position_id": "paper_src",
            "status": "open",
            "symbol": "SPY",
            "strategy": "bull_put_spread",
            "dte_remaining": 5,
            "quantity": 1,
            "entry_credit": 1.0,
            "current_value": 0.45,
            "details": {"roll_count": 0},
        }
        bot._get_tracked_positions = mock.Mock(return_value=[source_position])
        bot.roll_manager.evaluate = mock.Mock(
            return_value=SimpleNamespace(
                should_roll=True,
                reason="profit roll",
                min_credit_required=0.1,
                roll_type="profit",
            )
        )
        bot.paper_trader.execute_close = mock.Mock(return_value={"status": "FILLED"})
        bot.paper_trader.closed_trades = [
            {"position_id": "paper_src", "status": "closed", "details": {}}
        ]
        bot.paper_trader._save_state = mock.Mock()

        replacement = TradeSignal(
            action="open",
            strategy="bull_put_spread",
            symbol="SPY",
            analysis=SpreadAnalysis(
                symbol="SPY",
                strategy="bull_put_spread",
                expiration="2026-04-17",
                dte=35,
                short_strike=98,
                long_strike=93,
                credit=1.3,
                max_loss=3.7,
                probability_of_profit=0.66,
                score=72,
            ),
            metadata={},
        )
        bot._find_roll_replacement_signal = mock.Mock(return_value=replacement)
        bot._try_execute_entry = mock.Mock(
            side_effect=lambda signal: signal.metadata.__setitem__("paper_position_id", "paper_new") or True
        )

        bot._execute_roll(
            TradeSignal(
                action="roll",
                strategy="bull_put_spread",
                symbol="SPY",
                position_id="paper_src",
                quantity=1,
                reason="roll test",
                metadata={},
            )
        )

        bot.paper_trader.execute_close.assert_called_once()
        bot._try_execute_entry.assert_called_once()
        rolled = bot.paper_trader.closed_trades[0]
        self.assertEqual(rolled["status"], "rolled")
        self.assertEqual(rolled["rolled_to_position_id"], "paper_new")
        self.assertEqual(rolled["details"].get("roll_status"), "rolled")

    def test_orchestrator_adjustment_execution(self) -> None:
        bot = TradingBot(make_paper_config())
        bot._review_adjustment_with_llm = mock.Mock(return_value=True)
        bot.paper_trader._save_state = mock.Mock()
        bot.paper_trader.positions = [
            {
                "position_id": "pos_adj",
                "strategy": "bull_put_spread",
                "symbol": "SPY",
                "status": "open",
                "details": {
                    "expiration": "2026-03-20",
                    "short_strike": 100.0,
                    "long_strike": 95.0,
                    "adjustment_count": 0,
                    "adjustment_cost": 0.0,
                },
            }
        ]
        position = dict(bot.paper_trader.positions[0])
        position["underlying_price"] = 99.0
        bot._get_chain_data = mock.Mock(
            return_value=(
                {
                    "puts": {
                        "2026-03-20": [
                            {"strike": 90.0, "delta": -0.12, "mid": 0.22},
                            {"strike": 88.0, "delta": -0.08, "mid": 0.14},
                        ]
                    },
                    "calls": {},
                },
                99.0,
            )
        )

        plan = SimpleNamespace(action="add_wing", reason="short strike tested")
        bot._execute_adjustment_plan(position, plan)

        self.assertTrue(any(order.get("type") == "adjustment" for order in bot.paper_trader.orders))
        details = bot.paper_trader.positions[0]["details"]
        self.assertEqual(details.get("adjustment_count"), 1)
        self.assertGreater(float(details.get("adjustment_cost", 0.0)), 0.0)

    def test_startup_reconciliation(self) -> None:
        bot = TradingBot(make_live_config())
        bot.live_ledger = mock.Mock()
        bot.live_ledger.close_missing_from_broker.return_value = 1
        bot.live_ledger.list_positions.return_value = [
            {
                "position_id": "p1",
                "strategy": "bull_put_spread",
                "symbol": "SPY",
                "details": {
                    "expiration": "2026-03-20",
                    "short_strike": 100.0,
                    "long_strike": 95.0,
                },
            }
        ]
        bot._get_broker_positions = mock.Mock(
            return_value=[
                {
                    "instrument": {"assetType": "OPTION", "symbol": "QQQ   260320P00100000"},
                    "shortQuantity": 1.0,
                    "longQuantity": 0.0,
                }
            ]
        )

        reconciled = bot._startup_reconcile_positions()

        self.assertEqual(reconciled, 2)
        bot.live_ledger.register_entry_order.assert_called_once()
        kwargs = bot.live_ledger.register_entry_order.call_args.kwargs
        self.assertEqual(kwargs["strategy"], "unknown_external")

    def test_periodic_reconciliation_watchdog_imports_orphans(self) -> None:
        bot = TradingBot(make_live_config())
        bot.live_ledger = mock.Mock()
        bot.live_ledger.close_missing_from_broker.return_value = 1
        bot.live_ledger.list_positions.return_value = []
        bot._get_broker_positions = mock.Mock(
            return_value=[
                {
                    "instrument": {"assetType": "OPTION", "symbol": "QQQ   260320P00100000"},
                    "shortQuantity": 1.0,
                    "longQuantity": 0.0,
                }
            ]
        )
        bot._alert = mock.Mock()

        actions = bot._maybe_run_reconciliation_watchdog(force=True)

        self.assertEqual(actions, 2)
        bot.live_ledger.register_entry_order.assert_called_once()
        bot._alert.assert_called_once()

    def test_periodic_reconciliation_respects_auto_import_toggle(self) -> None:
        cfg = make_live_config()
        cfg.reconciliation.auto_import = False
        bot = TradingBot(cfg)
        bot.live_ledger = mock.Mock()
        bot.live_ledger.close_missing_from_broker.return_value = 0
        bot.live_ledger.list_positions.return_value = []
        bot._get_broker_positions = mock.Mock(
            return_value=[
                {
                    "instrument": {"assetType": "OPTION", "symbol": "QQQ   260320P00100000"},
                    "shortQuantity": 1.0,
                    "longQuantity": 0.0,
                }
            ]
        )

        actions = bot._maybe_run_reconciliation_watchdog(force=True)

        self.assertEqual(actions, 0)
        bot.live_ledger.register_entry_order.assert_not_called()

    def test_stream_quote_triggers_immediate_exit_check(self) -> None:
        bot = TradingBot(make_paper_config())
        bot._service_degradation["stream_down"] = False
        bot.paper_trader.positions = [
            {
                "position_id": "stream_pos",
                "strategy": "bull_put_spread",
                "symbol": "SPY",
                "status": "open",
                "entry_credit": 1.0,
                "current_value": 0.8,
                "dte_remaining": 20,
                "details": {"short_strike": 95.0},
            }
        ]
        strategy = mock.Mock()
        strategy.check_exits.return_value = [
            TradeSignal(
                action="close",
                strategy="bull_put_spread",
                symbol="SPY",
                position_id="stream_pos",
                reason="stream trigger",
                quantity=1,
            )
        ]
        bot.strategies = [strategy]
        bot._get_chain_data = mock.Mock(return_value=({}, 0.0))
        bot._execute_exit = mock.Mock()

        bot._process_stream_exit_check({"symbol": "SPY", "lastPrice": 99.5})

        bot._execute_exit.assert_called_once()

    def test_daily_report_triggers_daily_and_weekly_alerts(self) -> None:
        bot = TradingBot(make_paper_config())
        bot.paper_trader.get_performance_summary = mock.Mock(
            return_value={
                "balance": 101000.0,
                "total_trades": 12,
                "win_rate": 58.0,
                "total_pnl": 1000.0,
                "return_pct": 1.0,
                "open_positions": 2,
            }
        )
        bot._now_eastern = mock.Mock(return_value=datetime.fromisoformat("2026-02-20T16:10:00-05:00"))
        bot.alerts = mock.Mock()
        bot._auto_generate_dashboard_if_due = mock.Mock()

        bot._daily_report()

        bot.alerts.daily_summary.assert_called_once()
        bot.alerts.weekly_summary.assert_called_once()
        daily_ctx = bot.alerts.daily_summary.call_args.kwargs.get("context", {})
        weekly_ctx = bot.alerts.weekly_summary.call_args.kwargs.get("context", {})
        self.assertIn("calmar", daily_ctx)
        self.assertIn("profit_factor", daily_ctx)
        self.assertIn("expectancy_per_trade", daily_ctx)
        self.assertIn("calmar", weekly_ctx)

    def test_persist_runtime_exit_state_writes_trailing_fields_to_live_ledger(self) -> None:
        bot = TradingBot(make_live_config())
        bot.live_ledger = mock.Mock()
        positions = [
            {
                "position_id": "live_1",
                "strategy": "short_strangle",
                "status": "open",
                "trailing_stop_high": 0.42,
                "max_profit_pct_seen": 0.51,
                "details": {"trailing_stop_high": 0.42},
            }
        ]

        bot._persist_runtime_exit_state(positions)

        bot.live_ledger.update_position_metadata.assert_called_once()

    def test_record_daily_pnl_attribution_calls_engine(self) -> None:
        bot = TradingBot(make_paper_config())
        bot.risk_manager.portfolio.open_positions = [
            {
                "position_id": "p1",
                "symbol": "SPY",
                "strategy": "bull_put_spread",
                "quantity": 1,
                "entry_credit": 1.2,
                "current_value": 1.0,
                "details": {"net_delta": 2.0, "net_gamma": 0.1, "net_theta": 0.1, "net_vega": 0.5},
            }
        ]
        bot.schwab.get_price_history = mock.Mock(return_value=[{"close": 500.0}, {"close": 503.0}])
        bot.pnl_attribution = mock.Mock()
        bot.pnl_attribution.compute_attribution = mock.Mock(return_value={"portfolio": {"delta_pnl": 5.0}})
        bot.pnl_attribution.record_daily_snapshot = mock.Mock()

        summary = bot._record_daily_pnl_attribution()

        self.assertIn("delta_pnl", summary)
        bot.pnl_attribution.compute_attribution.assert_called_once()
        bot.pnl_attribution.record_daily_snapshot.assert_called_once()

    def test_entry_timing_block_queues_signal(self) -> None:
        bot = TradingBot(make_paper_config())
        bot.paper_trader.execute_open = mock.Mock(return_value={"status": "FILLED", "position_id": "p1"})
        bot._now_eastern = mock.Mock(return_value=datetime.fromisoformat("2026-02-23T09:35:00-05:00"))

        signal = TradeSignal(
            action="open",
            strategy="bull_put_spread",
            symbol="SPY",
            analysis=SpreadAnalysis(
                symbol="SPY",
                strategy="bull_put_spread",
                expiration="2026-03-20",
                dte=25,
                short_strike=100,
                long_strike=95,
                credit=1.2,
                max_loss=3.8,
                probability_of_profit=0.62,
                score=55,
            ),
            metadata={"apply_timing_blocks": True},
        )

        executed = bot._try_execute_entry(signal)

        self.assertFalse(executed)
        self.assertEqual(len(bot.signal_queue), 1)
        bot.paper_trader.execute_open.assert_not_called()

    def test_queued_signal_executes_in_optimal_window(self) -> None:
        bot = TradingBot(make_paper_config())
        bot._entries_allowed = mock.Mock(return_value=True)
        bot.risk_manager.can_open_more_positions = mock.Mock(return_value=True)
        bot.risk_manager.calculate_position_size = mock.Mock(return_value=1)
        bot.risk_manager.approve_trade = mock.Mock(return_value=(True, "ok"))
        bot.paper_trader.execute_open = mock.Mock(return_value={"status": "FILLED", "position_id": "pq1"})
        bot.risk_manager.register_open_position = mock.Mock()
        bot._now_eastern = mock.Mock(return_value=datetime.fromisoformat("2026-02-23T11:00:00-05:00"))

        signal = TradeSignal(
            action="open",
            strategy="bull_put_spread",
            symbol="SPY",
            analysis=SpreadAnalysis(
                symbol="SPY",
                strategy="bull_put_spread",
                expiration="2026-03-20",
                dte=25,
                short_strike=100,
                long_strike=95,
                credit=1.2,
                max_loss=3.8,
                probability_of_profit=0.62,
                score=55,
            ),
            metadata={"apply_timing_blocks": True},
        )
        bot._queue_entry_signal(signal, "outside_optimal_window")

        bot._process_signal_queue()

        bot.paper_trader.execute_open.assert_called_once()
        self.assertEqual(bot.signal_queue, [])

    def test_timing_bypass_allows_priority_entry_outside_window(self) -> None:
        bot = TradingBot(make_paper_config())
        bot._now_eastern = mock.Mock(return_value=datetime.fromisoformat("2026-02-23T09:35:00-05:00"))
        bot.risk_manager.calculate_position_size = mock.Mock(return_value=1)
        bot.risk_manager.approve_trade = mock.Mock(return_value=(True, "ok"))
        bot.paper_trader.execute_open = mock.Mock(return_value={"status": "FILLED", "position_id": "p2"})
        bot.risk_manager.register_open_position = mock.Mock()

        signal = TradeSignal(
            action="open",
            strategy="bull_put_spread",
            symbol="SPY",
            analysis=SpreadAnalysis(
                symbol="SPY",
                strategy="bull_put_spread",
                expiration="2026-03-20",
                dte=25,
                short_strike=100,
                long_strike=95,
                credit=1.2,
                max_loss=3.8,
                probability_of_profit=0.62,
                score=55,
            ),
            metadata={"apply_timing_blocks": True, "bypass_timing_blocks": True},
        )

        executed = bot._try_execute_entry(signal)

        self.assertTrue(executed)
        bot.paper_trader.execute_open.assert_called_once()

    def test_adaptive_execution_timing_queues_non_preferred_bucket(self) -> None:
        bot = TradingBot(make_paper_config())
        bot._now_eastern = mock.Mock(return_value=datetime.fromisoformat("2026-02-23T11:20:00-05:00"))
        bot._entry_timing_state = mock.Mock(return_value={"allowed": True, "optimal": True, "reason": "optimal_window"})
        bot._refresh_execution_timing_analysis = mock.Mock()
        bot._preferred_execution_bucket = mock.Mock(return_value={"active": True, "preferred_bucket": "10:00-11:00"})
        bot._execution_time_bucket = mock.Mock(return_value="11:00-13:00")
        bot.paper_trader.execute_open = mock.Mock(return_value={"status": "FILLED", "position_id": "p2"})

        signal = TradeSignal(
            action="open",
            strategy="bull_put_spread",
            symbol="SPY",
            analysis=SpreadAnalysis(
                symbol="SPY",
                strategy="bull_put_spread",
                expiration="2026-03-20",
                dte=25,
                short_strike=100,
                long_strike=95,
                credit=1.2,
                max_loss=3.8,
                probability_of_profit=0.62,
                score=55,
            ),
            metadata={"apply_timing_blocks": True},
        )

        executed = bot._try_execute_entry(signal)

        self.assertFalse(executed)
        self.assertEqual(len(bot.signal_queue), 1)
        bot.paper_trader.execute_open.assert_not_called()

    def test_record_slippage_history_applies_symbol_penalty_after_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            slippage_path = Path(tmp_dir) / "slippage_history.json"
            with mock.patch("bot.orchestrator.SLIPPAGE_HISTORY_PATH", slippage_path):
                bot = TradingBot(make_paper_config())
                row = {
                    "symbol": "ILLQ",
                    "strategy": "bull_put_spread",
                    "dte": 25,
                    "slippage": 0.12,
                    "adverse_slippage": 0.12,
                }
                for _ in range(4):
                    bot._record_slippage_history(dict(row))
                self.assertEqual(bot._symbol_slippage_penalty("ILLQ"), 0.0)

                bot._record_slippage_history(dict(row))
                self.assertAlmostEqual(bot._symbol_slippage_penalty("ILLQ"), 0.12, places=4)

    def test_slippage_penalty_rejects_low_credit_signal(self) -> None:
        bot = TradingBot(make_paper_config())
        bot._symbol_slippage_penalty = mock.Mock(return_value=0.20)
        bot.risk_manager.calculate_position_size = mock.Mock(return_value=1)
        bot.risk_manager.approve_trade = mock.Mock(return_value=(True, "ok"))
        bot.paper_trader.execute_open = mock.Mock()

        signal = TradeSignal(
            action="open",
            strategy="bull_put_spread",
            symbol="SPY",
            analysis=SpreadAnalysis(
                symbol="SPY",
                strategy="bull_put_spread",
                expiration="2026-03-20",
                dte=25,
                short_strike=100,
                long_strike=95,
                credit=1.40,
                max_loss=3.6,
                probability_of_profit=0.62,
                score=58,
            ),
        )

        executed = bot._try_execute_entry(signal)

        self.assertFalse(executed)
        bot.risk_manager.approve_trade.assert_not_called()
        bot.paper_trader.execute_open.assert_not_called()

    def test_refresh_strategy_stats_persists_regime_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stats_path = Path(tmp_dir) / "strategy_stats.json"
            with mock.patch("bot.orchestrator.STRATEGY_STATS_PATH", stats_path):
                bot = TradingBot(make_paper_config())
                bot._refresh_strategy_stats(
                    [
                        {"strategy": "bull_put_spread", "regime": "BULL_TREND", "pnl": 120.0},
                        {"strategy": "bear_call_spread", "regime": "BULL_TREND", "pnl": -80.0},
                    ]
                )

                payload = json.loads(stats_path.read_text(encoding="utf-8"))
                self.assertIn("credit_spreads", payload)
                row = payload["credit_spreads"]["BULL_TREND"]
                self.assertEqual(row["wins"], 1)
                self.assertEqual(row["losses"], 1)
                self.assertAlmostEqual(row["win_rate"], 50.0, places=4)


if __name__ == "__main__":
    unittest.main()
