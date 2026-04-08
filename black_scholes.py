"""Black-Scholes Option Pricing Model."""

import math


def normal_cdf(x):
    """Cumulative distribution function for the standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x):
    """Probability density function for the standard normal distribution."""
    return math.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)


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


def black_scholes_put(S, K, T, r, sigma):
    """Calculate European put option price using the Black-Scholes formula.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized, e.g. 0.05 for 5%)
        sigma: Volatility of the underlying asset (annualized, e.g. 0.2 for 20%)

    Returns:
        Put option price

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

    put_price = K * math.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)
    return put_price


def delta(S, K, T, r, sigma, option_type="call"):
    """Calculate the delta of an option (sensitivity to underlying price).

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility of the underlying asset
        option_type: 'call' or 'put'

    Returns:
        Delta value
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

    if option_type == "call":
        return normal_cdf(d1)
    elif option_type == "put":
        return normal_cdf(d1) - 1.0
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def gamma(S, K, T, r, sigma):
    """Calculate the gamma of an option (rate of change of delta).

    Gamma is the same for both call and put options.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility of the underlying asset

    Returns:
        Gamma value
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

    return normal_pdf(d1) / (S * sigma * math.sqrt(T))


def theta(S, K, T, r, sigma, option_type="call"):
    """Calculate the theta of an option (sensitivity to time decay).

    Returns the per-year rate of time decay. Divide by 365 for daily theta.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility of the underlying asset
        option_type: 'call' or 'put'

    Returns:
        Theta value (negative means the option loses value as time passes)
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

    common_term = -(S * normal_pdf(d1) * sigma) / (2.0 * math.sqrt(T))

    if option_type == "call":
        return common_term - r * K * math.exp(-r * T) * normal_cdf(d2)
    elif option_type == "put":
        return common_term + r * K * math.exp(-r * T) * normal_cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def vega(S, K, T, r, sigma):
    """Calculate the vega of an option (sensitivity to volatility).

    Vega is the same for both call and put options.
    Returns the change in option price for a 1-unit change in sigma.
    Divide by 100 for the per-1% change.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility of the underlying asset

    Returns:
        Vega value
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

    return S * normal_pdf(d1) * math.sqrt(T)


def rho(S, K, T, r, sigma, option_type="call"):
    """Calculate the rho of an option (sensitivity to interest rate).

    Returns the change in option price for a 1-unit change in r.
    Divide by 100 for the per-1% change.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility of the underlying asset
        option_type: 'call' or 'put'

    Returns:
        Rho value
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

    if option_type == "call":
        return K * T * math.exp(-r * T) * normal_cdf(d2)
    elif option_type == "put":
        return -K * T * math.exp(-r * T) * normal_cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


if __name__ == "__main__":
    # Example parameters
    S = 100.0      # Current stock price
    K = 105.0      # Strike price
    T = 1.0        # 1 year to expiration
    r = 0.05       # 5% risk-free rate
    sigma = 0.2    # 20% volatility

    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_put(S, K, T, r, sigma)

    print("Black-Scholes Option Pricing")
    print("=================================")
    print(f"Stock Price (S):    ${S:.2f}")
    print(f"Strike Price (K):   ${K:.2f}")
    print(f"Time to Expiry (T): {T:.2f} years")
    print(f"Risk-Free Rate (r): {r:.2%}")
    print(f"Volatility (σ):     {sigma:.2%}")
    print(f"---------------------------------")
    print(f"Call Option Price:  ${call_price:.4f}")
    print(f"Put Option Price:   ${put_price:.4f}")
    print(f"---------------------------------")
    print(f"Put-Call Parity Check:")
    print(f"  C - P:            ${call_price - put_price:.4f}")
    print(f"  S - K*e^(-rT):    ${S - K * math.exp(-r * T):.4f}")
    print()
    print("Greeks")
    print("=================================")
    print(f"{'Greek':<10} {'Call':>12} {'Put':>12}")
    print(f"---------------------------------")
    print(f"{'Delta':<10} {delta(S, K, T, r, sigma, 'call'):>12.6f} {delta(S, K, T, r, sigma, 'put'):>12.6f}")
    print(f"{'Gamma':<10} {gamma(S, K, T, r, sigma):>12.6f} {gamma(S, K, T, r, sigma):>12.6f}")
    print(f"{'Theta':<10} {theta(S, K, T, r, sigma, 'call'):>12.6f} {theta(S, K, T, r, sigma, 'put'):>12.6f}")
    print(f"{'Vega':<10} {vega(S, K, T, r, sigma):>12.6f} {vega(S, K, T, r, sigma):>12.6f}")
    print(f"{'Rho':<10} {rho(S, K, T, r, sigma, 'call'):>12.6f} {rho(S, K, T, r, sigma, 'put'):>12.6f}")
