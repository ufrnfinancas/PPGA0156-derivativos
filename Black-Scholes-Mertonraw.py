# Black-Scholes-Merton
import math

def black_scholes_call_put(S, K, T, r, sigma, option_type):
    """
    Calculate the price of a European call or put option using the Black-Scholes formula.
    
    :param S: Current price of the underlying asset
    :param K: Strike price of the option
    :param T: Time to expiration (in years)
    :param r: Risk-free interest rate (annual)
    :param sigma: Volatility of the underlying asset (annual)
    :param option_type: 'call' for call option, 'put' for put option
    :return: Option price
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == 'call':
        option_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    elif option_type == 'put':
        option_price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return option_price

def norm_cdf(x):
    """
    Calculate the cumulative distribution function (CDF) of the standard normal distribution.
    
    :param x: Value at which to calculate the CDF
    :return: CDF value
    """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

# Example usage:
S = 50  # Current price of the underlying asset
K = 52  # Strike price of the option
T = 100/252    # Time to expiration (year based)
r = 0.08 # Risk-free interest rate (5%)
sigma = 0.22 # Volatility (20%)

call_price = black_scholes_call_put(S, K, T, r, sigma, 'call')
put_price = black_scholes_call_put(S, K, T, r, sigma, 'put')

print(f"Call Option Price: {call_price:.2f}")
print(f"Put Option Price: {put_price:.2f}")
