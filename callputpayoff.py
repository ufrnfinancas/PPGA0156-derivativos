import numpy as np
import matplotlib.pyplot as plt

## Call PETR4 comprada

strike_price = 27          # Preço de exercício
premium = 7.20             # Prêmio pago pela opção
quantidade = 10000        # Quantidade de opções
preco_atual = 23.76        # Preço atual do ativo subjacente

# Geração de preços do ativo subjacente
s = np.linspace(0, 100, 100)  # Preços do ativo, de 0 a 100

# Cálculo do payoff
payoff_call = (np.maximum(s - strike_price, 0) - premium) * quantidade

# Plot
plt.figure(figsize=(10, 6))
plt.plot(s, payoff_call, label='Payoff da Call', color='b')
plt.axvline(preco_atual, color='red', linestyle='--', label=f'Preço Atual ($S$={preco_atual})')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Preço do Ativo Subjacente ($S$)')
plt.ylabel('Payoff')
plt.title('Payoff de uma Opção de Compra (Call)')
plt.legend()
plt.grid(True)
plt.show()

# Cálculo do payoff no preço atual
payoff_bruto = max(preco_atual - strike_price, 0) * quantidade

# Custo total da operação
custo_total = premium * quantidade

# Lucro ou prejuízo
lucro_prejuizo = payoff_bruto - custo_total

# Exibindo o resultado
print(f"Lucro ou Prejuízo da operação: R$ {lucro_prejuizo:,.2f}")


## Put PETR4 comprada

strike_price = 19.16    # Preço de exercício
premium = 0.20          # Prêmio recebido pela venda da opção
quantidade = 10000      # Quantidade de opções 
preco_atual = 24.04     # Preço atual do ativo subjacente

# Geração de preços do ativo subjacente
s = np.linspace(0, 100, 100)  

# Cálculo do payoff para a put
payoff_put = (np.maximum(strike_price - s, 0) - premium) * quantidade

# Plot
plt.figure(figsize=(10, 6))
plt.plot(s, payoff_put, label='Payoff da Put', color='b')
plt.axvline(preco_atual, color='red', linestyle='--', label=f'Preço Atual ($S$={preco_atual})')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Preço do Ativo Subjacente ($S$)')
plt.ylabel('Payoff')
plt.title('Payoff de uma Opção de Venda')
plt.legend()
plt.grid(True)
plt.show()

# Cálculo do payoff no preço atual
payoff_bruto = max(strike_price - preco_atual, 0) * quantidade

# Receita da operação (prêmio recebido)
receita_total = premium * quantidade

# Lucro ou prejuízo
lucro_prejuizo = -(payoff_bruto + receita_total)

# Exibindo o resultado
print(f"Lucro ou Prejuízo da operação: R$ {lucro_prejuizo:,.2f}")