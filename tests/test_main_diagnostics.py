import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import main


class _DummySchwabClient:
    def __init__(self, _config) -> None:
        pass

    def connect(self) -> None:
        return None

    def stop_streaming(self) -> None:
        return None

    def get_option_chain(self, _symbol: str) -> dict:
        return {"ok": True}

    def get_quote(self, _symbol: str) -> dict:
        return {"quote": {"lastPrice": 500.0}}

    @staticmethod
    def parse_option_chain(_raw: dict) -> dict:
        return {
            "underlying_price": 500.0,
            "calls": {
                "2026-03-20": [
                    {
                        "symbol": "SPY_CALL",
                        "expiration": "2026-03-20",
                        "strike": 505.0,
                        "dte": 20,
                        "bid": 1.2,
                    }
                ]
            },
            "puts": {
                "2026-03-20": [
                    {
                        "symbol": "SPY_PUT",
                        "expiration": "2026-03-20",
                        "strike": 495.0,
                        "dte": 20,
                        "bid": 1.1,
                    }
                ]
            },
        }


class _DummyStrategy:
    name = "dummy_strategy"

    def scan_for_entries(
        self,
        _symbol: str,
        _chain_data: dict,
        _underlying: float,
        technical_context=None,
        market_context=None,
    ) -> list[SimpleNamespace]:
        del technical_context, market_context
        return [
            SimpleNamespace(
                strategy="dummy_strategy",
                symbol="SPY",
                quantity=1,
                metadata={},
                analysis=SimpleNamespace(
                    score=55.0,
                    probability_of_profit=0.62,
                    credit=1.25,
                ),
            )
        ]


class _DummyRiskManager:
    def __init__(self) -> None:
        self.portfolio = SimpleNamespace(open_positions=[])

    def approve_trade(self, _signal) -> tuple[bool, str]:
        return True, "ok"

    def evaluate_greeks_budget(
        self,
        _signal,
        regime: str,
        quantity: int,
        allow_resize: bool = True,
    ) -> tuple[bool, int, str]:
        del regime, quantity, allow_resize
        return True, 1, "ok"


class _DummyBot:
    def __init__(self, _config, **_kwargs) -> None:
        self.schwab = SimpleNamespace(stop_streaming=lambda: None)
        self.risk_manager = _DummyRiskManager()
        self.circuit_state = {}
        self.strategies = [_DummyStrategy()]
        self.technicals = SimpleNamespace(get_context=lambda *_args, **_kwargs: {})
        self.config = SimpleNamespace(
            risk=SimpleNamespace(max_open_positions=5),
            multi_timeframe=SimpleNamespace(min_agreement=2),
        )

    def connect(self) -> None:
        return None

    def _update_portfolio_state(self) -> None:
        return None

    def _entries_allowed(self) -> bool:
        return True

    def _entry_timing_state(self) -> dict:
        return {"allowed": True, "optimal": True, "reason": "ok"}

    def _get_chain_data(self, _symbol: str) -> tuple[dict, float]:
        return {"chain": "ok"}, 500.0

    def _build_market_context(self, _symbol: str, _chain_data: dict) -> dict:
        return {"market": "ok"}

    def _filter_signals_by_context(
        self, signals: list[SimpleNamespace], _market_context: dict
    ) -> list[SimpleNamespace]:
        return signals

    def _passes_multi_timeframe_confirmation(self, _signal) -> tuple[bool, int, dict]:
        return True, 2, {}

    def _strategy_regime_min_score(self, _signal) -> float:
        return 10.0

    def _symbol_slippage_penalty(self, _symbol: str) -> float:
        return 0.0

    def _signal_width(self, _signal) -> float:
        return 5.0

    def _strategy_min_credit_pct(self, _strategy: str) -> float:
        return 0.2


class MainDiagnosticsTests(unittest.TestCase):
    def test_run_integrated_diagnostics_returns_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            token_path = Path(tmp_dir) / "token.json"
            token_path.write_text("{}", encoding="utf-8")
            os.chmod(token_path, 0o600)

            cfg = SimpleNamespace(
                schwab=SimpleNamespace(token_path=str(token_path)),
                terminal_ui=SimpleNamespace(enabled=True),
            )

            with (
                mock.patch("bot.schwab_client.SchwabClient", _DummySchwabClient),
                mock.patch("bot.orchestrator.TradingBot", _DummyBot),
            ):
                code = main.run_integrated_diagnostics(
                    config=cfg,
                    mode_hint="paper",
                    symbol="SPY",
                )

        self.assertEqual(code, 0)
