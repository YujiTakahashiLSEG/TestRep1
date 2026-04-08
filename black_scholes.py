"""Black-Scholes Call Option Pricing Model."""

import math


def normal_cdf(x):
    """Cumulative distribution function for the standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(S, K, T, r, sigma):
    """Calculate European call option price using the Black-Scholes formula.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized, e.g. 0.05 for 5%)
        sigma: Volatility of the underlying asset (annualized, e.g. 0.2 for 20%)

    Returns:
        Call option price

    Raises:
        ValueError: If inputs are invalid
    """
    if S <= 0:
        raise ValueError("Stock price must be positive")
    if K <= 0:
        raise ValueError("Strike price must be positive")
    if T <= 0:
        raise ValueError("Time to expiration must be positive")
    if sigma <= 0:
        raise ValueError("Volatility must be positive")

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = S * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)
    return call_price


if __name__ == "__main__":
    # Example parameters
    S = 100.0      # Current stock price
    K = 105.0      # Strike price
    T = 1.0        # 1 year to expiration
    r = 0.05       # 5% risk-free rate
    sigma = 0.2    # 20% volatility

    price = black_scholes_call(S, K, T, r, sigma)

    print("Black-Scholes Call Option Pricing")
    print("=================================")
    print(f"Stock Price (S):    ${S:.2f}")
    print(f"Strike Price (K):   ${K:.2f}")
    print(f"Time to Expiry (T): {T:.2f} years")
    print(f"Risk-Free Rate (r): {r:.2%}")
    print(f"Volatility (σ):     {sigma:.2%}")
    print(f"---------------------------------")
    print(f"Call Option Price:  ${price:.4f}")
