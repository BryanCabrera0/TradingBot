"""Options analysis engine — Greeks, probabilities, IV analysis, and strike selection."""

import math
import logging
from dataclasses import dataclass
from typing import Optional

from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class OptionMetrics:
    """Computed metrics for an options trade."""
    symbol: str
    expiration: str
    dte: int
    strike: float
    contract_type: str  # "CALL" or "PUT"
    bid: float
    ask: float
    mid: float
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float
    open_interest: int
    volume: int
    underlying_price: float
    # Computed
    moneyness: float = 0.0  # strike / underlying
    probability_otm: float = 0.0  # probability of expiring OTM
    probability_itm: float = 0.0
    expected_value: float = 0.0


@dataclass
class SpreadAnalysis:
    """Analysis results for an options spread."""
    symbol: str
    strategy: str
    expiration: str
    dte: int
    short_strike: float
    long_strike: float
    # For iron condors
    call_short_strike: Optional[float] = None
    call_long_strike: Optional[float] = None
    put_short_strike: Optional[float] = None
    put_long_strike: Optional[float] = None
    # Financials
    credit: float = 0.0
    max_loss: float = 0.0
    max_profit: float = 0.0
    risk_reward_ratio: float = 0.0
    credit_pct_of_width: float = 0.0
    # Probabilities
    probability_of_profit: float = 0.0
    expected_value: float = 0.0
    # Greeks
    net_delta: float = 0.0
    net_theta: float = 0.0
    net_gamma: float = 0.0
    net_vega: float = 0.0
    # Quality score (0-100)
    score: float = 0.0


def compute_probability_otm(
    underlying_price: float,
    strike: float,
    dte: int,
    iv: float,
    contract_type: str,
) -> float:
    """Estimate probability of an option expiring out of the money.

    Uses Black-Scholes log-normal distribution assumption.
    """
    if dte <= 0 or iv <= 0 or underlying_price <= 0:
        return 0.0

    # IV from API is typically annualized percentage
    sigma = iv / 100.0 if iv > 1 else iv
    t = dte / 365.0

    d2 = (math.log(underlying_price / strike) - 0.5 * sigma**2 * t) / (
        sigma * math.sqrt(t)
    )

    if contract_type.upper() in ("PUT", "P"):
        # Put is OTM when price > strike → P(S > K)
        return float(norm.cdf(d2))
    else:
        # Call is OTM when price < strike → P(S < K)
        return float(norm.cdf(-d2))


def compute_probability_of_profit_spread(
    underlying_price: float,
    short_strike: float,
    credit: float,
    dte: int,
    iv: float,
    contract_type: str,
) -> float:
    """Estimate probability of profit for a credit spread.

    Profit occurs when the underlying stays beyond the breakeven point.
    """
    if contract_type.upper() in ("PUT", "P"):
        breakeven = short_strike - credit
        # Need price to stay above breakeven
        return compute_probability_otm(
            underlying_price, breakeven, dte, iv, "PUT"
        )
    else:
        breakeven = short_strike + credit
        # Need price to stay below breakeven
        return compute_probability_otm(
            underlying_price, breakeven, dte, iv, "CALL"
        )


def compute_probability_of_profit_condor(
    underlying_price: float,
    put_short_strike: float,
    call_short_strike: float,
    credit: float,
    dte: int,
    iv: float,
) -> float:
    """Estimate probability of profit for an iron condor."""
    put_breakeven = put_short_strike - credit
    call_breakeven = call_short_strike + credit

    if dte <= 0 or iv <= 0 or underlying_price <= 0:
        return 0.0
    if put_breakeven <= 0 or call_breakeven <= 0 or put_breakeven >= call_breakeven:
        return 0.0

    sigma = iv / 100.0 if iv > 1 else iv
    t = dte / 365.0
    vol = sigma * math.sqrt(t)

    d2_lower = (math.log(underlying_price / put_breakeven) - 0.5 * sigma**2 * t) / vol
    d2_upper = (math.log(underlying_price / call_breakeven) - 0.5 * sigma**2 * t) / vol

    # Probability of being between the two breakevens:
    # P(S < upper) - P(S <= lower)
    pop = norm.cdf(-d2_upper) - norm.cdf(-d2_lower)
    return max(0.0, min(1.0, pop))


def analyze_credit_spread(
    underlying_price: float,
    short_option: dict,
    long_option: dict,
    contract_type: str,
) -> SpreadAnalysis:
    """Analyze a credit spread (bull put or bear call)."""
    credit = round(short_option["mid"] - long_option["mid"], 2)
    width = abs(short_option["strike"] - long_option["strike"])
    max_loss = round(width - credit, 2) if credit > 0 else width
    max_profit = credit

    if max_loss > 0:
        risk_reward = round(max_profit / max_loss, 3)
    else:
        risk_reward = 0.0

    credit_pct = round(credit / width, 3) if width > 0 else 0.0

    iv_avg = (short_option["iv"] + long_option["iv"]) / 2.0
    dte = short_option["dte"]

    pop = compute_probability_of_profit_spread(
        underlying_price,
        short_option["strike"],
        credit,
        dte,
        iv_avg,
        contract_type,
    )

    # Expected value = (POP * max_profit) - ((1-POP) * max_loss)
    ev = round(pop * max_profit * 100 - (1 - pop) * max_loss * 100, 2)

    # Score: weighted combination of probability, credit quality, and theta
    score = _score_spread(pop, credit_pct, risk_reward, short_option, long_option)

    strategy = "bull_put_spread" if contract_type.upper() in ("PUT", "P") else "bear_call_spread"

    return SpreadAnalysis(
        symbol=short_option.get("symbol", "").split(" ")[0] if short_option.get("symbol") else "",
        strategy=strategy,
        expiration=short_option.get("expiration", ""),
        dte=dte,
        short_strike=short_option["strike"],
        long_strike=long_option["strike"],
        credit=credit,
        max_loss=max_loss,
        max_profit=max_profit,
        risk_reward_ratio=risk_reward,
        credit_pct_of_width=credit_pct,
        probability_of_profit=round(pop, 4),
        expected_value=ev,
        # Position greeks for a credit spread are long-leg minus short-leg.
        net_delta=round(long_option["delta"] - short_option["delta"], 4),
        net_theta=round(long_option["theta"] - short_option["theta"], 4),
        net_gamma=round(
            float(long_option.get("gamma", 0.0)) - float(short_option.get("gamma", 0.0)),
            4,
        ),
        net_vega=round(long_option["vega"] - short_option["vega"], 4),
        score=score,
    )


def analyze_iron_condor(
    underlying_price: float,
    put_short: dict,
    put_long: dict,
    call_short: dict,
    call_long: dict,
) -> SpreadAnalysis:
    """Analyze an iron condor position."""
    put_credit = round(put_short["mid"] - put_long["mid"], 2)
    call_credit = round(call_short["mid"] - call_long["mid"], 2)
    total_credit = round(put_credit + call_credit, 2)

    put_width = abs(put_short["strike"] - put_long["strike"])
    call_width = abs(call_long["strike"] - call_short["strike"])
    max_width = max(put_width, call_width)
    max_loss = round(max(max_width - total_credit, 0.0), 2) if total_credit > 0 else max_width

    if max_loss > 0:
        risk_reward = round(total_credit / max_loss, 3)
    else:
        risk_reward = 0.0

    credit_pct = round(total_credit / max_width, 3) if max_width > 0 else 0.0
    dte = put_short["dte"]
    iv_avg = (
        put_short["iv"] + put_long["iv"] + call_short["iv"] + call_long["iv"]
    ) / 4.0

    pop = compute_probability_of_profit_condor(
        underlying_price,
        put_short["strike"],
        call_short["strike"],
        total_credit,
        dte,
        iv_avg,
    )

    ev = round(pop * total_credit * 100 - (1 - pop) * max_loss * 100, 2)

    net_delta = round(
        -put_short["delta"] + put_long["delta"] - call_short["delta"] + call_long["delta"], 4
    )
    net_theta = round(
        -put_short["theta"] + put_long["theta"] - call_short["theta"] + call_long["theta"], 4
    )
    net_gamma = round(
        -float(put_short.get("gamma", 0.0))
        + float(put_long.get("gamma", 0.0))
        - float(call_short.get("gamma", 0.0))
        + float(call_long.get("gamma", 0.0)),
        4,
    )
    net_vega = round(
        -put_short["vega"] + put_long["vega"] - call_short["vega"] + call_long["vega"], 4
    )

    score = _score_condor(pop, credit_pct, risk_reward, net_theta, dte)

    return SpreadAnalysis(
        symbol="",
        strategy="iron_condor",
        expiration=put_short.get("expiration", ""),
        dte=dte,
        short_strike=0,
        long_strike=0,
        put_short_strike=put_short["strike"],
        put_long_strike=put_long["strike"],
        call_short_strike=call_short["strike"],
        call_long_strike=call_long["strike"],
        credit=total_credit,
        max_loss=max_loss,
        max_profit=total_credit,
        risk_reward_ratio=risk_reward,
        credit_pct_of_width=credit_pct,
        probability_of_profit=round(pop, 4),
        expected_value=ev,
        net_delta=net_delta,
        net_theta=net_theta,
        net_gamma=net_gamma,
        net_vega=net_vega,
        score=score,
    )


def find_option_by_delta(
    options: list, target_delta: float, tolerance: float = 0.05
) -> Optional[dict]:
    """Find the option closest to a target delta."""
    if not options:
        return None

    best = None
    best_diff = float("inf")

    for opt in options:
        diff = abs(abs(opt["delta"]) - target_delta)
        if diff < best_diff:
            best_diff = diff
            best = opt

    if best and best_diff <= tolerance + target_delta * 0.5:
        return best
    return best  # Return closest even if outside tolerance


def find_option_by_strike(options: list, strike: float) -> Optional[dict]:
    """Find an option at a specific strike price."""
    for opt in options:
        if abs(opt["strike"] - strike) < 0.01:
            return opt
    return None


def find_spread_wing(
    options: list, anchor_strike: float, width: float, direction: str
) -> Optional[dict]:
    """Find the long leg of a spread given the short strike and width.

    direction: "lower" for puts (buy lower strike), "higher" for calls (buy higher strike)
    """
    if direction == "lower":
        target = anchor_strike - width
    else:
        target = anchor_strike + width
    return find_option_by_strike(options, target)


# ── Scoring ──────────────────────────────────────────────────────────

def _score_spread(
    pop: float,
    credit_pct: float,
    risk_reward: float,
    short_opt: dict,
    long_opt: dict,
) -> float:
    """Score a credit spread from 0-100 based on quality metrics."""
    score = 0.0

    # Probability of profit (40% weight) — higher is better
    score += min(pop, 1.0) * 40

    # Credit as % of width (25% weight) — higher is better, but cap at 50%
    score += min(credit_pct / 0.50, 1.0) * 25

    # Risk/reward (15% weight)
    score += min(risk_reward / 0.75, 1.0) * 15

    # Liquidity — open interest and volume (10% weight)
    avg_oi = (short_opt["open_interest"] + long_opt["open_interest"]) / 2
    avg_vol = (short_opt["volume"] + long_opt["volume"]) / 2
    liquidity = min(avg_oi / 500, 1.0) * 0.5 + min(avg_vol / 100, 1.0) * 0.5
    score += liquidity * 10

    # Bid-ask spread tightness (10% weight)
    short_spread = short_opt["ask"] - short_opt["bid"]
    long_spread = long_opt["ask"] - long_opt["bid"]
    avg_ba_spread = (short_spread + long_spread) / 2
    tightness = max(0, 1.0 - avg_ba_spread / 0.50)
    score += tightness * 10

    return round(min(score, 100), 1)


def _score_condor(
    pop: float,
    credit_pct: float,
    risk_reward: float,
    net_theta: float,
    dte: int,
) -> float:
    """Score an iron condor from 0-100."""
    score = 0.0

    score += min(pop, 1.0) * 40
    score += min(credit_pct / 0.50, 1.0) * 25
    score += min(risk_reward / 0.50, 1.0) * 15

    # Theta decay — positive theta is good
    if net_theta > 0:
        score += min(net_theta / 0.10, 1.0) * 10
    else:
        score += max(0, 1.0 + net_theta / 0.05) * 10

    # DTE sweet spot (30-45 ideal)
    if 25 <= dte <= 50:
        score += 10
    elif 15 <= dte < 25 or 50 < dte <= 60:
        score += 5

    return round(min(score, 100), 1)
