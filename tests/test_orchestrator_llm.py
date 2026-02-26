import unittest
from types import SimpleNamespace
from unittest import mock

import schedule

from bot.analysis import SpreadAnalysis
from bot.config import BotConfig
from bot.llm_advisor import LLMDecision
from bot.orchestrator import TradingBot
from bot.strategies.base import TradeSignal


def make_signal(symbol: str = "SPY") -> TradeSignal:
    return TradeSignal(
        action="open",
        strategy="bull_put_spread",
        symbol=symbol,
        analysis=SpreadAnalysis(
            symbol=symbol,
            strategy="bull_put_spread",
            expiration="2026-03-20",
            dte=30,
            short_strike=95,
            long_strike=90,
            credit=1.25,
            max_loss=3.75,
            probability_of_profit=0.68,
            score=64,
        ),
    )


def make_config(mode: str) -> BotConfig:
    cfg = BotConfig()
    cfg.trading_mode = "paper"
    cfg.scanner.enabled = False
    cfg.llm.enabled = True
    cfg.news.enabled = False
    cfg.llm.mode = mode
    cfg.credit_spreads.enabled = True
    cfg.iron_condors.enabled = False
    cfg.covered_calls.enabled = False
    return cfg


class OrchestratorLLMTests(unittest.TestCase):
    def test_setup_schedule_registers_ml_retrain_job(self) -> None:
        schedule.clear()
        bot = TradingBot(make_config("advisory"))
        bot.config.ml_scorer.enabled = True
        bot.config.ml_scorer.retrain_day = "sunday"
        bot.config.ml_scorer.retrain_time = "18:00"

        bot.setup_schedule()
        funcs = [repr(job.job_func) for job in schedule.get_jobs()]
        self.assertTrue(any("_scheduled_ml_retrain" in fn for fn in funcs))
        schedule.clear()

    def test_blocking_mode_rejects_trade(self) -> None:
        bot = TradingBot(make_config("blocking"))
        bot.risk_manager.calculate_position_size = mock.Mock(return_value=2)
        bot.risk_manager.approve_trade = mock.Mock(return_value=(True, "Approved"))
        bot.llm_advisor = mock.Mock()
        bot.llm_advisor.review_trade.return_value = LLMDecision(
            approve=False,
            confidence=0.9,
            risk_adjustment=1.0,
            reason="reject",
        )
        bot.paper_trader.execute_open = mock.Mock()

        executed = bot._try_execute_entry(make_signal())

        self.assertFalse(executed)
        bot.paper_trader.execute_open.assert_not_called()

    def test_advisory_mode_applies_size_adjustment(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.risk_manager.calculate_position_size = mock.Mock(return_value=4)
        bot.risk_manager.approve_trade = mock.Mock(return_value=(True, "Approved"))
        bot.llm_advisor = mock.Mock()
        bot.llm_advisor.review_trade.return_value = LLMDecision(
            approve=True,
            confidence=0.91,
            risk_adjustment=0.5,
            reason="reduce size",
        )
        bot.paper_trader.execute_open = mock.Mock(return_value={"status": "FILLED"})

        executed = bot._try_execute_entry(make_signal())

        self.assertTrue(executed)
        _, kwargs = bot.paper_trader.execute_open.call_args
        self.assertEqual(kwargs["quantity"], 2)

    def test_successful_open_updates_intra_cycle_risk_state(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.risk_manager.calculate_position_size = mock.Mock(return_value=1)
        bot.risk_manager.approve_trade = mock.Mock(return_value=(True, "Approved"))
        bot.llm_advisor = mock.Mock()
        bot.llm_advisor.review_trade.return_value = LLMDecision(
            approve=True,
            confidence=0.9,
            risk_adjustment=1.0,
            reason="ok",
        )
        bot.paper_trader.execute_open = mock.Mock(return_value={"status": "FILLED"})

        signal = make_signal()
        executed = bot._try_execute_entry(signal)

        self.assertTrue(executed)
        self.assertEqual(len(bot.risk_manager.portfolio.open_positions), 1)
        self.assertEqual(
            bot.risk_manager.portfolio.total_risk_deployed,
            signal.analysis.max_loss * 100,
        )

    def test_llm_context_includes_sector_relative_strength(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.risk_manager.earnings_calendar = mock.Mock(
            earnings_within_window=mock.Mock(return_value=(False, None))
        )
        bot.llm_advisor = mock.Mock()
        bot.llm_advisor.review_trade.return_value = LLMDecision(
            approve=True,
            confidence=0.9,
            risk_adjustment=1.0,
            reason="ok",
        )

        prices = {
            "AAPL": [{"close": 100.0}, {"close": 110.0}],
            "XLK": [{"close": 50.0}, {"close": 52.0}],
            "SPY": [{"close": 400.0}, {"close": 404.0}],
        }
        bot.schwab.get_price_history = mock.Mock(
            side_effect=lambda symbol, days=40: prices.get(symbol, [])
        )

        allowed = bot._review_entry_with_llm(make_signal("AAPL"))

        self.assertTrue(allowed)
        _, context = bot.llm_advisor.review_trade.call_args.args
        sector = context["sector_performance"]
        self.assertEqual(sector["sector_etf"], "XLK")
        self.assertGreater(sector["sector_vs_spy"], 0.0)
        self.assertGreater(sector["symbol_vs_sector"], 0.0)

    def test_portfolio_strategist_skip_sector_directive_is_enforced(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.llm_strategist = mock.Mock()
        bot.llm_strategist.review_portfolio.return_value = [
            SimpleNamespace(
                action="skip_sector",
                reason="Too concentrated",
                payload={"sector": "Information Technology"},
            )
        ]
        bot.risk_manager.sector_map["AAPL"] = "Information Technology"
        bot.risk_manager.sector_map["XOM"] = "Energy"

        bot._apply_portfolio_strategist()

        self.assertTrue(bot._is_symbol_sector_skipped("AAPL"))
        self.assertFalse(bot._is_symbol_sector_skipped("XOM"))

    def test_portfolio_strategist_reduce_delta_closes_heaviest_position(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.llm_strategist = mock.Mock()
        bot.llm_strategist.review_portfolio.return_value = [
            SimpleNamespace(
                action="reduce_delta",
                reason="Net delta too high",
                payload={"count": 1, "direction": "positive"},
            )
        ]
        bot.risk_manager.portfolio.net_delta = 42.0
        bot._get_tracked_positions = mock.Mock(
            return_value=[
                {
                    "position_id": "p_hi",
                    "status": "open",
                    "symbol": "AAPL",
                    "strategy": "bull_put_spread",
                    "quantity": 1,
                    "details": {"net_delta": 18.0},
                },
                {
                    "position_id": "p_lo",
                    "status": "open",
                    "symbol": "MSFT",
                    "strategy": "bull_put_spread",
                    "quantity": 1,
                    "details": {"net_delta": 7.0},
                },
                {
                    "position_id": "p_neg",
                    "status": "open",
                    "symbol": "QQQ",
                    "strategy": "bear_call_spread",
                    "quantity": 1,
                    "details": {"net_delta": -10.0},
                },
            ]
        )
        bot._execute_exit = mock.Mock()

        bot._apply_portfolio_strategist()

        bot._execute_exit.assert_called_once()
        signal = bot._execute_exit.call_args.args[0]
        self.assertEqual(signal.position_id, "p_hi")
        self.assertEqual(signal.action, "close")

    def test_option_stream_subscription_uses_atm_contracts(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.schwab.streaming_connected = mock.Mock(return_value=True)
        bot.schwab.stream_option_level_one = mock.Mock(return_value=True)
        chain = {
            "calls": {
                "2026-04-17": [
                    {"dte": 30, "strike": 95.0, "symbol": "SPY_041726C95"},
                    {"dte": 30, "strike": 100.0, "symbol": "SPY_041726C100"},
                    {"dte": 30, "strike": 105.0, "symbol": "SPY_041726C105"},
                ]
            },
            "puts": {
                "2026-04-17": [
                    {"dte": 30, "strike": 95.0, "symbol": "SPY_041726P95"},
                    {"dte": 30, "strike": 100.0, "symbol": "SPY_041726P100"},
                    {"dte": 30, "strike": 105.0, "symbol": "SPY_041726P105"},
                ]
            },
        }

        bot._subscribe_option_stream_for_symbol("SPY", chain, 100.0)
        bot._subscribe_option_stream_for_symbol("SPY", chain, 100.0)

        bot.schwab.stream_option_level_one.assert_called_once()
        symbols = bot.schwab.stream_option_level_one.call_args.args[0]
        self.assertEqual(set(symbols), {"SPY_041726C100", "SPY_041726P100"})

    def test_portfolio_strategist_close_long_dte_directive_executes_closes(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.llm_strategist = mock.Mock()
        bot.llm_strategist.review_portfolio.return_value = [
            SimpleNamespace(
                action="close_long_dte",
                reason="trim duration",
                payload={"max_dte": 30},
            )
        ]
        bot._get_tracked_positions = mock.Mock(
            return_value=[
                {
                    "position_id": "long_dte",
                    "status": "open",
                    "symbol": "SPY",
                    "strategy": "bull_put_spread",
                    "dte_remaining": 45,
                    "quantity": 1,
                },
                {
                    "position_id": "short_dte",
                    "status": "open",
                    "symbol": "QQQ",
                    "strategy": "bull_put_spread",
                    "dte_remaining": 20,
                    "quantity": 1,
                },
            ]
        )
        bot._execute_exit = mock.Mock()

        bot._apply_portfolio_strategist()

        bot._execute_exit.assert_called_once()
        signal = bot._execute_exit.call_args.args[0]
        self.assertEqual(signal.position_id, "long_dte")

    def test_portfolio_strategist_scale_size_uses_factor_payload(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.llm_strategist = mock.Mock()
        bot.llm_strategist.review_portfolio.return_value = [
            SimpleNamespace(
                action="scale_size",
                reason="risk-off",
                payload={"factor": 0.7},
            )
        ]

        bot._apply_portfolio_strategist()

        self.assertAlmostEqual(bot._cycle_size_scalar, 0.7, places=4)

    def test_scan_applies_regime_weight_and_size_scalar(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.config.watchlist = ["SPY"]
        bot.risk_manager.can_open_more_positions = mock.Mock(return_value=True)
        bot._get_chain_data = mock.Mock(
            return_value=(
                {"calls": {"2026-03-20": [{}]}, "puts": {"2026-03-20": [{}]}},
                500.0,
            )
        )
        bot._subscribe_option_stream_for_symbol = mock.Mock()
        bot.technicals.get_context = mock.Mock(return_value=None)
        bot._build_market_context = mock.Mock(
            return_value={
                "regime": "HIGH_VOL_CHOP",
                "regime_weights": {"credit_spreads": 1.5, "iron_condors": 0.0},
                "position_size_scalar": 0.8,
            }
        )
        bot._filter_signals_by_context = mock.Mock(side_effect=lambda signals, _ctx: signals)
        bot._try_execute_entry = mock.Mock(return_value=False)

        sig = make_signal("SPY")
        sig.analysis.score = 20.0
        sig.size_multiplier = 1.0

        strategy_a = SimpleNamespace(
            name="credit_spreads",
            scan_for_entries=mock.Mock(return_value=[sig]),
        )
        strategy_b = SimpleNamespace(
            name="iron_condors",
            scan_for_entries=mock.Mock(return_value=[make_signal("SPY")]),
        )
        bot.strategies = [strategy_a, strategy_b]

        bot._scan_for_entries()

        strategy_a.scan_for_entries.assert_called_once()
        strategy_b.scan_for_entries.assert_not_called()
        executed_signal = bot._try_execute_entry.call_args.args[0]
        self.assertAlmostEqual(executed_signal.analysis.score, 30.0, places=4)
        self.assertAlmostEqual(executed_signal.size_multiplier, 0.8, places=4)

    def test_scan_ranks_signals_by_composite_score_across_symbols(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.config.watchlist = ["AAA", "BBB"]
        bot.config.signal_ranking.enabled = True
        bot.risk_manager.can_open_more_positions = mock.Mock(side_effect=[True, True, True, False])
        bot._get_chain_data = mock.Mock(
            return_value=(
                {"calls": {"2026-03-20": [{}]}, "puts": {"2026-03-20": [{}]}},
                100.0,
            )
        )
        bot._subscribe_option_stream_for_symbol = mock.Mock()
        bot.technicals.get_context = mock.Mock(return_value=None)
        bot._filter_signals_by_context = mock.Mock(side_effect=lambda signals, _ctx: signals)

        def _context_for_symbol(symbol: str, _chain: dict) -> dict:
            spread = 2.0 if symbol == "AAA" else 3.0
            return {
                "regime": "HIGH_VOL_CHOP",
                "regime_weights": {"credit_spreads": 1.0},
                "position_size_scalar": 1.0,
                "vol_surface": {"realized_implied_spread": spread},
                "max_signals_per_symbol_per_strategy": 2,
            }

        bot._build_market_context = mock.Mock(side_effect=_context_for_symbol)
        bot._try_execute_entry = mock.Mock(return_value=True)

        sig_a = make_signal("AAA")
        sig_a.analysis.score = 50.0
        sig_a.analysis.probability_of_profit = 0.60
        sig_a.analysis.credit_pct_of_width = 0.30

        sig_b = make_signal("BBB")
        sig_b.analysis.score = 45.0
        sig_b.analysis.probability_of_profit = 0.80
        sig_b.analysis.credit_pct_of_width = 0.50

        strategy = SimpleNamespace(
            name="credit_spreads",
            scan_for_entries=mock.Mock(side_effect=lambda symbol, *_args, **_kwargs: [sig_a] if symbol == "AAA" else [sig_b]),
        )
        bot.strategies = [strategy]

        bot._scan_for_entries()

        first = bot._try_execute_entry.call_args_list[0].args[0]
        second = bot._try_execute_entry.call_args_list[1].args[0]
        self.assertEqual(first.symbol, "BBB")
        self.assertEqual(second.symbol, "AAA")
        self.assertTrue(bot._last_ranked_signals)
        self.assertEqual(bot._last_ranked_signals[0]["symbol"], "BBB")

    def test_composite_score_includes_ml_score_and_low_confidence_flag(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.ml_scorer = mock.Mock()
        bot.ml_scorer.predict_score = mock.Mock(return_value=0.20)
        signal = make_signal("SPY")
        signal.metadata["regime_score_weight"] = 1.0
        signal.metadata["vol_surface"] = {"realized_implied_spread": 2.0}

        composite = bot._composite_signal_score(signal)

        self.assertGreater(composite, 0.0)
        self.assertEqual(signal.metadata.get("ml_flag"), "low_confidence")
        self.assertAlmostEqual(signal.metadata.get("ml_score"), 0.2, places=4)

    def test_llm_context_includes_ml_flags(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.risk_manager.earnings_calendar = mock.Mock(
            earnings_within_window=mock.Mock(return_value=(False, None))
        )
        bot.llm_advisor = mock.Mock()
        bot.llm_advisor.review_trade.return_value = LLMDecision(
            approve=True,
            confidence=0.9,
            risk_adjustment=1.0,
            reason="ok",
        )
        signal = make_signal("AAPL")
        signal.metadata["ml_score"] = 0.22
        signal.metadata["ml_flag"] = "low_confidence"

        approved = bot._review_entry_with_llm(signal)

        self.assertTrue(approved)
        _, context = bot.llm_advisor.review_trade.call_args.args
        self.assertEqual(context.get("ml_flag"), "low_confidence")
        self.assertAlmostEqual(float(context.get("ml_score")), 0.22, places=4)

    def test_strategy_regime_gate_rejects_when_historical_win_rate_is_weak(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot._strategy_stats = {
            "credit_spreads": {
                "HIGH_VOL_CHOP": {"wins": 8, "losses": 14, "win_rate": 36.36}
            }
        }
        bot.risk_manager.calculate_position_size = mock.Mock(return_value=1)
        bot.risk_manager.approve_trade = mock.Mock(return_value=(True, "ok"))

        signal = make_signal("SPY")
        signal.analysis.score = 55.0
        signal.metadata["regime"] = "HIGH_VOL_CHOP"

        executed = bot._try_execute_entry(signal)

        self.assertFalse(executed)
        bot.risk_manager.approve_trade.assert_not_called()

    def test_strategy_regime_gate_relaxes_when_historical_win_rate_is_strong(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.llm_advisor = None
        bot._strategy_stats = {
            "credit_spreads": {
                "HIGH_VOL_CHOP": {"wins": 15, "losses": 7, "win_rate": 68.18}
            }
        }
        bot.risk_manager.calculate_position_size = mock.Mock(return_value=1)
        bot.risk_manager.approve_trade = mock.Mock(return_value=(True, "ok"))
        bot.paper_trader.execute_open = mock.Mock(return_value={"status": "FILLED", "position_id": "p1"})
        bot.risk_manager.register_open_position = mock.Mock()

        signal = make_signal("SPY")
        signal.analysis.score = 47.0
        signal.metadata["regime"] = "HIGH_VOL_CHOP"

        executed = bot._try_execute_entry(signal)

        self.assertTrue(executed)
        bot.risk_manager.approve_trade.assert_called_once()

    def test_theta_harvest_blocks_short_premium_entries_after_cutoff(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.llm_advisor = None
        bot.risk_manager.calculate_position_size = mock.Mock(return_value=1)
        bot.risk_manager.approve_trade = mock.Mock(return_value=(True, "ok"))
        bot.paper_trader.execute_open = mock.Mock(return_value={"status": "FILLED", "position_id": "p1"})
        bot._refresh_theta_harvest_state = mock.Mock()
        bot._theta_harvest_state = {
            "date": bot._now_eastern().date().isoformat(),
            "expected_theta": 100.0,
            "theta_earned": 85.0,
            "ratio": 0.85,
        }
        bot._now_eastern = mock.Mock(return_value=bot._now_eastern().replace(hour=14, minute=5))

        executed = bot._try_execute_entry(make_signal("SPY"))

        self.assertFalse(executed)
        bot.paper_trader.execute_open.assert_not_called()

    def test_theta_harvest_underperforming_relaxes_score_floor(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.llm_advisor = None
        bot._strategy_stats = {
            "credit_spreads": {
                "HIGH_VOL_CHOP": {"wins": 15, "losses": 7, "win_rate": 68.18}
            }
        }
        bot.risk_manager.calculate_position_size = mock.Mock(return_value=1)
        bot.risk_manager.approve_trade = mock.Mock(return_value=(True, "ok"))
        bot.paper_trader.execute_open = mock.Mock(return_value={"status": "FILLED", "position_id": "p1"})
        bot.risk_manager.register_open_position = mock.Mock()
        now_et = bot._now_eastern().replace(hour=14, minute=10)
        bot._now_eastern = mock.Mock(return_value=now_et)
        bot._refresh_theta_harvest_state = mock.Mock()
        bot._theta_harvest_state = {
            "date": now_et.date().isoformat(),
            "expected_theta": 100.0,
            "theta_earned": 20.0,
            "ratio": 0.20,
        }

        signal = make_signal("SPY")
        signal.analysis.score = 46.0
        signal.metadata["regime"] = "HIGH_VOL_CHOP"
        executed = bot._try_execute_entry(signal)

        self.assertTrue(executed)
        bot.risk_manager.approve_trade.assert_called_once()

    def test_monte_carlo_high_risk_reduces_entry_quantity(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.llm_advisor = None
        bot._monte_carlo_state = {"high_risk": True, "var99_pct_account": 4.2}
        bot.risk_manager.calculate_position_size = mock.Mock(return_value=4)
        bot.risk_manager.approve_trade = mock.Mock(return_value=(True, "ok"))
        bot.paper_trader.execute_open = mock.Mock(return_value={"status": "FILLED", "position_id": "p1"})
        bot.risk_manager.register_open_position = mock.Mock()
        bot._refresh_monte_carlo_risk = mock.Mock()

        signal = make_signal("SPY")
        executed = bot._try_execute_entry(signal)

        self.assertTrue(executed)
        _, kwargs = bot.paper_trader.execute_open.call_args
        self.assertEqual(kwargs["quantity"], 2)

    def test_portfolio_strategist_context_includes_regime_streak_and_risk_contributors(self) -> None:
        bot = TradingBot(make_config("advisory"))
        bot.llm_strategist = mock.Mock()
        bot.llm_strategist.review_portfolio.return_value = []
        bot._recent_regime_states = mock.Mock(return_value=[{"regime": "BULL_TREND"}])
        bot._strategy_streak_summary = mock.Mock(return_value={"credit_spreads": {"streak": 2, "type": "wins"}})
        bot._top_risk_contributors = mock.Mock(return_value=[{"symbol": "SPY", "risk_dollars": 1200.0}])
        bot.current_regime_state.sub_signals = {"term_structure_steepness": 0.12, "term_structure_momentum": -0.01}

        bot._apply_portfolio_strategist()

        context = bot.llm_strategist.review_portfolio.call_args.args[0]
        self.assertIn("recent_regime_states", context)
        self.assertIn("strategy_streaks", context)
        self.assertIn("top_risk_contributors", context)
        self.assertIn("regime_sub_signals", context)
        self.assertIn("term_structure_steepness", context["regime_sub_signals"])


if __name__ == "__main__":
    unittest.main()
