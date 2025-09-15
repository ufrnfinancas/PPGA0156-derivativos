from bcb import sgs
import pandas as pd

# Configuração para evitar notação científica
pd.options.display.float_format = '{:,.2f}'.format

# Parâmetros iniciais
valor_inicial = 30000.00  # Valor inicial a ser atualizado
data_inicio = "2022-05-19"
data_fim = "2024-12-10"
remuneracao_cdi = 1.20  # Percentual do CDI (120%)

# Obtendo os dados do CDI diário (série 12 do SGS)
cdi_diario = sgs.get({"CDI Diário": 12}, start=data_inicio, end=data_fim)

# Criando o DataFrame para os cálculos
df = cdi_diario.copy()
df["Fator"] = 1 + (df["CDI Diário"] / 100)  # Transformando a taxa percentual em fator
df["Fator Acumulado"] = df["Fator"].cumprod()  # Calculando o fator acumulado
df["Valor Atualizado"] = valor_inicial * df["Fator Acumulado"].shift(fill_value=1)  # Usar o fator acumulado do dia anterior

# Cálculo para 120% do CDI
df["Fator 120%"] = 1 + (remuneracao_cdi * df["CDI Diário"] / 100)  # Aplicando 120% do CDI
df["Fator Acumulado 120%"] = df["Fator 120%"].cumprod()  # Calculando o fator acumulado
df["Valor Atualizado 120%"] = valor_inicial * df["Fator Acumulado 120%"].shift(fill_value=1)  # Usar o fator acumulado do dia anterior

# Exibindo os resultados
print(df[["CDI Diário", "Fator", "Fator Acumulado", "Valor Atualizado", "Fator Acumulado 120%", "Valor Atualizado 120%"]])