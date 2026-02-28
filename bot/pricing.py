"""
Mathematical models for pricing synthetic option chains.
Implements Black-Scholes-Merton and Geometric Brownian Motion (GBM).
"""

import math
from scipy.stats import norm
import random
import numpy as np
from typing import Literal

def black_scholes(
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    option_type: Literal["call", "put"] = "call"
) -> float:
    """
    Calculate the Black-Scholes fair value of an option.
    S: Underlying price
    K: Strike price
    T: Time to expiration (in years, e.g., 30/365)
    r: Risk-free interest rate (e.g., 0.05 for 5%)
    sigma: Annualized implied volatility (e.g., 0.20 for 20%)
    """
    if T <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return max(0.0, price)

def bs_greeks(
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    option_type: Literal["call", "put"] = "call"
) -> dict[str, float]:
    """Calculate the Black-Scholes Greeks."""
    if T <= 0:
        return {"delta": 1.0 if option_type == "call" and S > K else (
            -1.0 if option_type == "put" and S < K else 0.0
        ), "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Delta
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0

    # Gamma (same for both)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))

    # Vega (same for both, typically expressed as change per 1% change in IV)
    vega = (S * norm.pdf(d1) * math.sqrt(T)) / 100.0

    # Theta (typically expressed as decay per day)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    if option_type == "call":
        term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)
    
    theta = (term1 + term2) / 365.0

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega)
    }

def generate_gbm_path(
    S0: float, 
    mu: float, 
    sigma: float, 
    days: int, 
    dt: float = 1/252,
    inject_black_swan: bool = False
) -> list[float]:
    """
    Generate a price path using Geometric Brownian Motion (GBM) with optional Jump Diffusion.
    Used to simulate multiple days of market action for holding a trade.
    """
    prices = np.zeros(days)
    prices[0] = S0
    
    # Optional Poisson jump parameters for Black Swans
    # lambda_coeff = Expected jumps per year
    # jump_mean/vol = Distribution of the jump size (log normal)
    lambda_coeff = 2.0 if inject_black_swan else 0.0
    jump_mean = -0.10 # Heavy downside tail risk
    jump_vol = 0.20
    
    for t in range(1, days):
        # standard normal random shock
        Z = np.random.standard_normal()
        
        # Base GBM formula
        drift = (mu - 0.5 * sigma**2) * dt
        shock = sigma * math.sqrt(dt) * Z
        
        # Evaluate if a jump (Black Swan) occurs on this day
        jump_factor = 0.0
        if inject_black_swan:
            prob_jump = lambda_coeff * dt
            if random.random() < prob_jump:
                # Calculate size of the sudden gap
                jump_Z = np.random.standard_normal()
                jump_factor = jump_mean + jump_vol * jump_Z
                
        prices[t] = prices[t-1] * math.exp(drift + shock + jump_factor)
        
    return prices.tolist()
