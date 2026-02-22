import unittest
from unittest import mock

from bot.analysis import SpreadAnalysis
from bot.config import BotConfig
from bot.llm_advisor import LLMDecision
from bot.orchestrator import TradingBot
from bot.strategies.base import TradeSignal


def make_signal() -> TradeSignal:
    return TradeSignal(
        action="open",
        strategy="bull_put_spread",
        symbol="SPY",
        analysis=SpreadAnalysis(
            symbol="SPY",
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
    cfg.llm.mode = mode
    cfg.credit_spreads.enabled = True
    cfg.iron_condors.enabled = False
    cfg.covered_calls.enabled = False
    return cfg


class OrchestratorLLMTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
