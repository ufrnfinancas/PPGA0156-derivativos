# Binomial tree - call

import numpy as np

# Function to calculate option price using the binomial tree
def binomial_tree_option_pricing(S, K, T, r, sigma, n):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)

    # Initialize arrays to store option values at each node
    option_values = np.zeros((n + 1, n + 1))

    # Calculate option values at expiration (leaf nodes)
    for j in range(n + 1):
        option_values[n, j] = max(0, S * (u**j) * (d**(n - j)) - K)

    # Backward induction to calculate option values at earlier nodes
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_values[i, j] = np.exp(-r * dt) * (q * option_values[i + 1, j] + (1 - q) * option_values[i + 1, j + 1])

    # The option price is the value at the root node
    option_price = option_values[0, 0]
    
    return option_price

# Input parameters
S0 = 20  # Initial stock price
K = 22   # Strike price
T = 0.25   # Time to expiration
r = 0.12  # Risk-free interest rate
sigma = 0.2  # Volatility

# Specify the number of steps
n = int(2)

# Calculate the option price
option_price = binomial_tree_option_pricing(S0, K, T, r, sigma, n)

print(f"The option price with {n} steps is: {option_price:.2f}")
