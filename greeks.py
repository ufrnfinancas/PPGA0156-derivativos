# Greeks - Gregas

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parametros - exemplo
S = 33.51     # Preço spot da ação
K = 32.95     # Strike price
T = 26/252    # Tempo para o vencimento em anos
r = (1+0.125579)**(26/252)      # Taxa livre de risco
sigma = 0.33   # Volatilidade

### Delta
def black_scholes_delta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'call':
        delta = np.exp(-r * T) * norm.cdf(d1)
    elif option_type == 'put':
        delta = np.exp(-r * T) * (norm.cdf(d1) - 1)
    else:
        raise ValueError("Option type should be 'call' or 'put'.")
    
    return delta

# Calcula o delta
delta_call = black_scholes_delta(S, K, T, r, sigma, option_type='call')
delta_put = black_scholes_delta(S, K, T, r, sigma, option_type='put')
delta_call
delta_put


### Theta
def theta_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    theta = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)
    
    return theta

# Calcula o Theta da opção de compra
theta_call = theta_call(S, K, T, r, sigma)
theta_call

def theta_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    theta = -(S * norm.pdf(-d1) * sigma / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    return theta

# Calcula o Theta da opção de venda
theta_put = theta_put(S, K, T, r, sigma)
theta_put


### Gamma
def gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# Calcula a Gamma da opção de compra
gamma = gamma(S, K, T, r, sigma)
gamma


### Vega
def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return vega

# Calcula a Vega da opção de compra
vega = vega(S, K, T, r, sigma)
vega


### Rho
def rho_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    return rho

# Calcula o Rho da opção de compra
rho_call = rho_call(S, K, T, r, sigma)
rho_call

def rho_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    return rho

# Calcula o Rho da opção de venda
rho_put = rho_put(S, K, T, r, sigma)
rho_put

#### Gregas não convencionais
C = 1.50
P = 1.00
### Lambda
def lambda_call(S, K, T, r, sigma, C):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    lambda_value = (delta / C) * S
    return lambda_value

# Calcula a lambda da opção de compra
lambda_call_value = lambda_call(S, K, T, r, sigma, C)
lambda_call_value

def lambda_put(S, K, T, r, sigma, P):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta_put = -norm.cdf(-d1)
    lambda_value = (delta_put / P) * S
    return lambda_value

# Calcula a lambda da opção de venda
lambda_put_value = lambda_put(S, K, T, r, sigma, P)
lambda_put_value


### Vanna
t = 1/252
delta_t = T - t

def vanna_call(S, K, T, r, sigma, t, delta_call):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    
    vanna = -norm.pdf(d1) * (d2 / sigma) * t * delta_call
    return vanna

# Calcula a Vanna da opção de compra com base no tempo restante
vanna_call_value = vanna_call(S, K, T, r, sigma, t, delta_call)
vanna_call_value

def vanna_put(S, K, T, r, sigma, t, delta_put):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    
    vanna = norm.pdf(d1) * (d2 / sigma) * t * delta_put
    return vanna

# Calcula a Vanna da opção de venda com base no tempo restante
vanna_put_value = vanna_put(S, K, T, r, sigma, t, delta_put)
vanna_put_value


### Charm
# Função para calcular o Charm de uma opção de compra europeia
def charm_call(S, K, T, r, sigma, t):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)
    
    charm = -norm.pdf(d1) * (r * norm.cdf(d2) - (2 / (2 * (T - t))) * norm.cdf(d2))
    return charm

# Calcula o Charm da opção de compra com base no tempo restante
charm_call_value = charm_call(S, K, T, r, sigma, t)
charm_call_value

# Função para calcular o Charm de uma opção de venda europeia
def charm_put(S, K, T, r, sigma, t):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)
    
    charm = -norm.pdf(d1) * (r * norm.cdf(-d2) + (2 / (2 * (T - t))) * norm.cdf(-d2))
    return charm

# Calcula o Charm da opção de venda com base no tempo restante
charm_put_value = charm_put(S, K, T, r, sigma, t)
charm_put_value