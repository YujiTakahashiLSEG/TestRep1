"""
FX Forward Price Calculator based on No-Arbitrage Theory (Covered Interest Rate Parity).

The no-arbitrage condition states that the forward exchange rate must satisfy:

    F = S * (1 + r_d * T) / (1 + r_f * T)          [simple compounding]
    F = S * exp((r_d - r_f) * T)                     [continuous compounding]

where:
    F   = Forward exchange rate (domestic per unit of foreign currency)
    S   = Spot exchange rate (domestic per unit of foreign currency)
    r_d = Domestic risk-free interest rate (annualized)
    r_f = Foreign risk-free interest rate (annualized)
    T   = Time to maturity in years

If this condition is violated, a risk-free arbitrage profit is possible.
"""

import math
from dataclasses import dataclass
from enum import Enum


class CompoundingMethod(Enum):
    SIMPLE = "simple"
    CONTINUOUS = "continuous"


@dataclass
class FXForwardResult:
    spot_rate: float
    domestic_rate: float
    foreign_rate: float
    time_to_maturity: float
    forward_rate: float
    forward_points: float
    compounding: CompoundingMethod

    def __str__(self) -> str:
        return (
            f"FX Forward Pricing Result\n"
            f"{'=' * 40}\n"
            f"Spot Rate:          {self.spot_rate:.6f}\n"
            f"Domestic Rate:      {self.domestic_rate:.4%}\n"
            f"Foreign Rate:       {self.foreign_rate:.4%}\n"
            f"Maturity (years):   {self.time_to_maturity:.4f}\n"
            f"Compounding:        {self.compounding.value}\n"
            f"{'─' * 40}\n"
            f"Forward Rate:       {self.forward_rate:.6f}\n"
            f"Forward Points:     {self.forward_points:.6f}\n"
        )


def fx_forward_rate(
    spot: float,
    domestic_rate: float,
    foreign_rate: float,
    time_to_maturity: float,
    compounding: CompoundingMethod = CompoundingMethod.SIMPLE,
) -> FXForwardResult:
    """Calculate the FX forward rate using no-arbitrage pricing.

    Args:
        spot: Spot exchange rate (domestic per unit of foreign currency).
        domestic_rate: Annualized domestic risk-free interest rate (e.g., 0.05 for 5%).
        foreign_rate: Annualized foreign risk-free interest rate.
        time_to_maturity: Time to maturity in years (e.g., 0.25 for 3 months).
        compounding: Compounding method to use.

    Returns:
        FXForwardResult with the calculated forward rate and details.

    Raises:
        ValueError: If inputs are invalid.
    """
    if spot <= 0:
        raise ValueError(f"Spot rate must be positive, got {spot}")
    if time_to_maturity < 0:
        raise ValueError(f"Time to maturity must be non-negative, got {time_to_maturity}")

    if compounding == CompoundingMethod.SIMPLE:
        forward = spot * (1 + domestic_rate * time_to_maturity) / (1 + foreign_rate * time_to_maturity)
    else:
        forward = spot * math.exp((domestic_rate - foreign_rate) * time_to_maturity)

    forward_points = forward - spot

    return FXForwardResult(
        spot_rate=spot,
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
        time_to_maturity=time_to_maturity,
        forward_rate=forward,
        forward_points=forward_points,
        compounding=compounding,
    )


def check_arbitrage(
    spot: float,
    forward_market: float,
    domestic_rate: float,
    foreign_rate: float,
    time_to_maturity: float,
    compounding: CompoundingMethod = CompoundingMethod.SIMPLE,
) -> str:
    """Check whether an arbitrage opportunity exists given a market-quoted forward rate.

    Compares the theoretical no-arbitrage forward rate with the market forward rate
    and identifies any arbitrage strategy.

    Args:
        spot: Spot exchange rate (domestic per unit of foreign currency).
        forward_market: Market-quoted forward rate.
        domestic_rate: Annualized domestic risk-free interest rate.
        foreign_rate: Annualized foreign risk-free interest rate.
        time_to_maturity: Time to maturity in years.
        compounding: Compounding method.

    Returns:
        A description of the arbitrage opportunity, if any.
    """
    result = fx_forward_rate(spot, domestic_rate, foreign_rate, time_to_maturity, compounding)
    theoretical = result.forward_rate
    diff = forward_market - theoretical

    tolerance = 1e-8
    if abs(diff) < tolerance:
        return (
            f"No arbitrage opportunity.\n"
            f"Theoretical forward: {theoretical:.6f}\n"
            f"Market forward:      {forward_market:.6f}"
        )

    if forward_market > theoretical:
        strategy = (
            "Market forward is OVERPRICED relative to theoretical.\n"
            "Strategy:\n"
            "  1. Borrow domestic currency at the domestic rate\n"
            "  2. Convert to foreign currency at spot rate\n"
            "  3. Invest in foreign risk-free asset\n"
            "  4. Sell foreign currency forward at the market forward rate\n"
            "  -> Lock in risk-free profit"
        )
    else:
        strategy = (
            "Market forward is UNDERPRICED relative to theoretical.\n"
            "Strategy:\n"
            "  1. Borrow foreign currency at the foreign rate\n"
            "  2. Convert to domestic currency at spot rate\n"
            "  3. Invest in domestic risk-free asset\n"
            "  4. Buy foreign currency forward at the market forward rate\n"
            "  -> Lock in risk-free profit"
        )

    return (
        f"ARBITRAGE OPPORTUNITY DETECTED\n"
        f"{'=' * 40}\n"
        f"Theoretical forward: {theoretical:.6f}\n"
        f"Market forward:      {forward_market:.6f}\n"
        f"Mispricing:          {diff:.6f}\n"
        f"{'─' * 40}\n"
        f"{strategy}\n"
    )


if __name__ == "__main__":
    # Example 1: USD/JPY forward (USD domestic, JPY foreign perspective: JPY per USD)
    print("Example 1: USD/JPY 6-Month Forward")
    print("=" * 50)
    result = fx_forward_rate(
        spot=150.00,
        domestic_rate=0.005,   # JPY rate ~0.5%
        foreign_rate=0.053,    # USD rate ~5.3%
        time_to_maturity=0.5,  # 6 months
        compounding=CompoundingMethod.SIMPLE,
    )
    print(result)

    # Example 2: EUR/USD forward (USD per EUR)
    print("Example 2: EUR/USD 1-Year Forward")
    print("=" * 50)
    result = fx_forward_rate(
        spot=1.0800,
        domestic_rate=0.053,   # USD rate ~5.3%
        foreign_rate=0.04,     # EUR rate ~4.0%
        time_to_maturity=1.0,  # 1 year
        compounding=CompoundingMethod.SIMPLE,
    )
    print(result)

    # Example 3: Continuous compounding
    print("Example 3: EUR/USD 1-Year Forward (Continuous Compounding)")
    print("=" * 50)
    result = fx_forward_rate(
        spot=1.0800,
        domestic_rate=0.053,
        foreign_rate=0.04,
        time_to_maturity=1.0,
        compounding=CompoundingMethod.CONTINUOUS,
    )
    print(result)

    # Example 4: Arbitrage check
    print("Example 4: Arbitrage Check")
    print("=" * 50)
    print(check_arbitrage(
        spot=1.0800,
        forward_market=1.1000,
        domestic_rate=0.053,
        foreign_rate=0.04,
        time_to_maturity=1.0,
    ))
