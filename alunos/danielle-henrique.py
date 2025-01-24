import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as bsm_iv

# Importa biblioteca para obter cotações do Yahoo Finance.
import yfinance as yf

# Importa funções para a otimização numérica (será usada no ajuste SVI).
from scipy.optimize import least_squares

def obter_cotacoes(subjacente, vencimento):
    """
    Faz uma requisição GET na API do site 'opcoes.net.br' para obter
    as cotações das opções de um ativo específico (subjacente) e um
    vencimento específico (Retorna o conteúdo em formato JSON).
    """
    url = (
        "https://opcoes.net.br/listaopcoes/completa"
        f"?idAcao={subjacente}"
        "&listarVencimentos=false"
        "&cotacoes=true"
        f"&vencimentos={vencimento}"
    )
    
    # Faz a requisição na URL montada.
    resp = requests.get(url)
    
    # Verifica se o status de retorno é 200 (OK).
    if resp.status_code != 200:
        print(f"Erro ao acessar API p/ {vencimento}: {resp.status_code}")
        return None
    
    # Retorna o JSON bruto com as informações.
    return resp.json()

def criar_dataframe_opcoes(subjacente, vencimento, dados_json):
    """
    Recebe o JSON retornado pela função 'obter_cotacoes' e o converte
    em um DF do pandas contendo informações relevantes:
    
    - Subjacente
    - Vencimento
    - Ativo
    - Tipo
    - Modelo
    - Strike
    - Preço
    - Negócios
    - Volume

    Faz limpeza de dados, removendo linhas inválidas ou com valores nulos.
    """
    if not dados_json:
        return pd.DataFrame()
    
    try:
        # Dentro do JSON, encontra a chave 'cotacoesOpcoes' em ['data'].
        cots = dados_json['data']['cotacoesOpcoes']
    except KeyError:
        # Se a estrutura do JSON não for a esperada, retorna um DF vazio.
        return pd.DataFrame()
    
    rows = []

    # 'c' é uma lista com várias colunas. Verifica se tem ao menos 11 colunas para garantir que há dados suficientes.
    for c in cots:
        if len(c) >= 11:
            # c[0] geralmente é algo como 'PETR3_2025-02-21', se existir underscore, faz split.
            ativo    = c[0].split('_')[0] if '_' in c[0] else c[0]
            tipo     = c[2]  # 'C' para Call, 'P' ou 'V' para Put, etc.
            modelo   = c[3]  # Modelo da opção (e.g., "EUROPEIA" ou "AMERICANA").
            strike   = c[5]  # Preço de exercício.
            preco    = c[8]  # Preço da opção.
            negocios = c[9]  # Número de negócios.
            volume   = c[10] # Volume financeiro ou quantidade negociada.
            
            rows.append([
                subjacente, vencimento, ativo, tipo, modelo,
                strike, preco, negocios, volume
            ])
    
    # Cria um DF com as colunas definidas.
    df = pd.DataFrame(rows, columns=[
        'subjacente','vencimento','ativo','tipo','modelo','strike',
        'preco','negocios','volume'
    ])
    
    # Remove linhas que tenham valores nulos nas colunas 'preco', 'negocios' e 'volume'.
    df.dropna(subset=['preco','negocios','volume'], inplace=True)
    
    # Converte as colunas numéricas para tipo numérico (float).
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    df['preco']  = pd.to_numeric(df['preco'],  errors='coerce')
    df['negocios'] = pd.to_numeric(df['negocios'], errors='coerce')
    df['volume']   = pd.to_numeric(df['volume'],   errors='coerce')
    
    # Mantém apenas as linhas cujo 'strike > 0' e 'preço > 0'.
    df = df[(df['strike']>0) & (df['preco']>0)]
    
    return df

def optionchaindate(subjacente, vencimento):
    """
    Função que obtém o JSON da API e, em seguida, chama 'criar_dataframe_opcoes' para retornar o DF consolidado.
    """
    dj = obter_cotacoes(subjacente, vencimento)
    return criar_dataframe_opcoes(subjacente, vencimento, dj)

def obter_spot_via_yahoo(ativo_yahoo):
    """
    Obtém a cotação de fechamento (spot) do ativo usando a biblioteca 'yfinance'.
    Retorna 'None' se houver falhas ou se a base estiver vazia.
    """
    try:
        tk = yf.Ticker(ativo_yahoo)
        # Pega o histórico de 1 dia.
        h = tk.history(period="1d")
        if h.empty:
            return None
        # Retorna o último valor de fechamento como float.
        return float(h['Close'].iloc[-1])
    except:
        return None

def valor_intrinseco(row, spot):
    """
    Calcula o valor intrínseco de uma opção, sem considerar dividendos.
    
    - No caso de calls, intrinseco = max(spot - strike, 0).
    - No caso de puts, intrinseco = max(strike - spot, 0).
    """
    tipo = str(row['tipo']).upper()
    if tipo.startswith('C'):
        # Call
        return max(spot - row['strike'], 0)
    else:
        # Put
        return max(row['strike'] - spot, 0)

def descartar_opcoes_inviaveis(df, spot):
    """
    Cria uma coluna 'intrinseco' em cada linha do DF, filtra (remove) as opções cujo preço de mercado
    esteja abaixo do valor intrínseco (inviáveis). Retorna um novo DF apenas com as opções viáveis.
    """
    df['intrinseco'] = df.apply(lambda r: valor_intrinseco(r, spot), axis=1)
    # Mantém linhas onde preco >= intrinseco.
    df_ok = df[df['preco'] >= df['intrinseco']].copy()
    # Remove a coluna 'intrinseco', pois não é mais necessária.
    df_ok.drop(columns='intrinseco', inplace=True)
    return df_ok

def calcular_tempo_ate_vencimento(data_venc, data_ref=None):
    """
    Calcula o tempo até o vencimento em anos, considerando dias úteis
    (convenção ACT/252, que é comum no mercado brasileiro).
    
    - data_venc: data de vencimento.
    - data_ref : data de referência (por padrão, a data atual).
    
    Retorna número de dias úteis/252. Se data_ref > data_venc, retorna 0.0.
    """
    if data_ref is None:
        data_ref = datetime.now()
    
    if data_ref > data_venc:
        # Se o vencimento já passou, tempo é zero.
        return 0.0
    
    # Cria uma série de datas com frequência 'B' (Business day -> dias úteis).
    busdays = pd.date_range(data_ref, data_venc, freq='B')
    
    # Divide a quantidade de dias úteis por 252 para ter fração de ano.
    return len(busdays) / 252.0

def calcular_iv_bsm(df, spot, r, q, T):
    """
    Calcula a volatilidade implícita (IV) de cada opção presente em 'df' usando a fórmula de Black-Scholes-Merton (py_vollib).
    
    - spot: preço do ativo subjacente
    - r: taxa de juros livre de risco (contínua)
    - q: taxa de dividendo contínuo
    - T: tempo até o vencimento em anos
    
    Adiciona a coluna 'iv' no DF com o valor da IV.
    """
    iv_list = []
    
    for idx, row in df.iterrows():
        price_opt = row['preco']       # preço de mercado da opção.
        K         = row['strike']      # preço de exercício.
        tipo_opt  = row['tipo'].upper()
        
        # Define o tipo para a função da py_vollib ('c' para call, 'p' para put).
        if tipo_opt.startswith('C'):
            flag = 'c'
        else:
            flag = 'p'
        
        try:
            # Chama a função de IV implícita (implied_volatility) do py_vollib.
            vol_ = bsm_iv(
                price_opt,
                spot,
                K,
                T,
                r,
                q,  # taxa de dividendo contínuo.
                flag
            )
            iv_list.append(vol_)
        except:
            # Se der erro (pode acontecer se o valor não é viável para IV), armazena NaN.
            iv_list.append(np.nan)
    
    # Cria a nova coluna no DF.
    df['iv'] = iv_list
    
    return df

## FUNÇÕES PARA SVI

# SVI (raw) => w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2 ) ]
def svi_raw(k, a, b, rho, m, sigma):
    """
    Fórmula base do SVI na parametrização 'raw'.
    
    - k = log-moneyness (ln(K/F)): a, b, rho, m, sigma são parâmetros a serem calibrados.
    - w(k) é a total implied variance (IV^2 * T).
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def erro_svi(params, k_array, w_array, wgt_array):
    """
    Função de erro para a minimização do SVI.
    Dado um conjunto de pontos (k_array, w_array) e pesos (wgt_array),
    calcula a diferença (resíduo) entre o w_array observado e o w_svi(k).
    """
    a, b, rho, m, sigma = params
    w_model = svi_raw(k_array, a, b, rho, m, sigma)
    # Retorna a diferença (w_array - w_model) * peso
    return (w_array - w_model) * wgt_array

def calibrar_svi(k_array, w_array, wgt_array=None):
    """
    Faz a calibração (ajuste) dos parâmetros SVI usando 'least_squares' do scipy, minimizando o erro definido em erro_svi.
    
    - k_array: valores de k = ln(K/F)
    - w_array: total implied variances observadas (IV^2 * T)
    - wgt_array: pesos para cada observação (opcional)
    
    Parâmetros com bounds: a > 0, b > 0, sigma > 0, -1 < rho < 1.
    """
    if wgt_array is None:
        wgt_array = np.ones_like(k_array)
    
    # Chute inicial heurístico.
    a0 = np.min(w_array)*0.5 if np.min(w_array) > 0 else 0.001
    b0 = 1.0
    rho0 = 0.0
    m0 = np.mean(k_array)
    sigma0 = 0.1
    
    guess = [a0, b0, rho0, m0, sigma0]
    
    # Definição dos limites (bounds) para cada parâmetro.
    lb = [1e-8, 1e-8, -0.999, -np.inf, 1e-8]
    ub = [np.inf, np.inf, 0.999,  np.inf, np.inf]
    
    # Realiza a otimização.
    res = least_squares(
        fun=erro_svi,
        x0=guess,
        bounds=(lb, ub),
        args=(k_array, w_array, wgt_array),
        method='trf'
    )
    
    # Retorna os parâmetros otimizados.
    return res.x  # [a, b, rho, m, sigma]

def plot_svi(df_otm, spot, r, q, T, subjacente, vencimento):
    """
    1) Calcula forward F = spot * e^((r-q)*T).
    2) Define k = ln(K / F).
    3) Define w_i = (iv_i^2) * T, que é a total implied variance.
    4) Define pesos (wgt_array) baseados no volume negociado (por ex, sqrt(volume)).
    5) Faz a calibração SVI e obtém os parâmetros a, b, rho, m, sigma.
    6) Gera um grid de k, calcula w_svi(k) e converte para IV_svi.
    7) Plota o smile de volatilidade observado e o ajustado (SVI).
    """
    # 1) Cálculo do forward
    F = spot * np.exp((r - q) * T)
    
    # Extrai arrays de strike, iv e volume
    K_arr = df_otm['strike'].values
    iv_arr = df_otm['iv'].values
    vol_arr = df_otm['volume'].values
    
    # Substitui valores 'NaN' em volume por 1.0 para evitar problemas
    vol_arr = np.nan_to_num(vol_arr, nan=1.0)
    
    # 2) total implied variance = IV^2 * T
    w_arr = (iv_arr**2) * T
    
    # 3) log-moneyness
    k_arr = np.log(K_arr / F)
    
    # 4) pesos = sqrt(volume), por exemplo
    wgt_arr = np.sqrt(vol_arr)
    
    # 5) Calibração dos parâmetros do SVI
    a, b, rho, m, sigma = calibrar_svi(k_arr, w_arr, wgt_arr)
    print(f"\nSVI params => a={a:.5f}, b={b:.5f}, rho={rho:.5f}, m={m:.5f}, sigma={sigma:.5f}")
    
    # 6) Gera um grid de k para plotar a curva SVI ajustada
    k_min, k_max = np.min(k_arr), np.max(k_arr)
    k_grid = np.linspace(k_min, k_max, 100)
    
    # w_svi(k) para cada ponto do grid
    w_svi = [svi_raw(kx, a, b, rho, m, sigma) for kx in k_grid]
    
    # Converte w_svi em IV_svi = sqrt(w_svi / T)
    iv_svi = [np.sqrt(wsvi / T) if wsvi > 0 else np.nan for wsvi in w_svi]
    
    # Converte k_grid -> strikes (K) usando K = F * e^k
    K_grid = F * np.exp(k_grid)
    
    # 7) Plotar
    plt.figure(figsize=(9, 6))
    
    # Plot dos pontos observados (OTM)
    plt.scatter(K_arr, iv_arr, color='darkblue', marker='o', s=40, alpha=0.8, label='Dados de Mercado (OTM)')
    
    # Plot da curva SVI ajustada
    plt.plot(K_grid, iv_svi, 'r--', color='red', linestyle='-', linewidth=1.75, label='Curva SVI Ajustada')
    
    plt.title(f"Smile de Volatilidade ({subjacente})\nVencimento em {vencimento}", fontsize=12, fontweight='bold')
    plt.xlabel("Strike (R$)", fontsize=11)
    plt.ylabel("Volatilidade Implícita", fontsize=11)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xlim(K_arr.min() - 1, K_arr.max() + 1)
    plt.ylim(0, 1.0)  # e.g., até 100% de vol
    plt.legend()
    plt.tight_layout()
    plt.show()

def filtrar_otm(df, spot, r, q, T):
    """
    Filtra apenas as opções OTM (out-of-the-money).

    - Calls OTM => strike > F
    - Puts OTM  => strike < F
    
    Onde F = spot * e^{(r-q)*T}, o forward price.
    
    Retorna um DF apenas com essas opções OTM, pois
    para calibrar a skew, costuma-se usar apenas OTM.
    """
    F = spot * np.exp((r - q) * T)
    
    # Separa calls e puts
    calls = df[df['tipo'].str.upper().str.startswith('C')]
    puts  = df[df['tipo'].str.upper().str.startswith('P') | df['tipo'].str.upper().str.startswith('V')]
    
    # Calls OTM => strike > F
    calls_otm = calls[calls['strike'] > F].copy()
    
    # Puts OTM => strike < F
    puts_otm  = puts[puts['strike'] < F].copy()
    
    # Concatena calls e puts OTM
    df_otm = pd.concat([calls_otm, puts_otm], ignore_index=True)
    
    # Remove linhas com IV nula
    df_otm.dropna(subset=['iv'], inplace=True)
    
    return df_otm

## FLUXO PRINCIPAL

if __name__ == "__main__":
    # Define o ativo subjacente e a data de vencimento desejada.
    subjacente = 'PETR4'
    vencimento = '2025-01-24'
    
    # 1) Coleta das opções via API.
    df_opcoes = optionchaindate(subjacente, vencimento)
    if df_opcoes.empty:
        print("Sem dados.")
        exit()
    
    # 2) Obtém preço de mercado (spot) via Yahoo Finance.
    spot = obter_spot_via_yahoo(f"{subjacente}.SA")
    if spot is None:
        print("Falha spot")
        exit()
    
    # Definição da taxa de juros e taxa de dividendo (contínuas).
    r = 0.13 # Taxa de juros.
    q = 0.04 # Taxa de dividendo anual.
    
    # 3) Calcula T (tempo até vencimento) em anos (dias úteis/252).
    ano, mes, dia = map(int, vencimento.split('-'))
    data_venc = datetime(ano, mes, dia)
    T = calcular_tempo_ate_vencimento(data_venc)
    if T <= 0:
        print("Venc expirado.")
        exit()
    
    # 4) Remove opções inviáveis (preço < intrínseco).
    df_opcoes = descartar_opcoes_inviaveis(df_opcoes, spot)
    df_opcoes.dropna(inplace=True)
    if df_opcoes.empty:
        print("Todas inviaveis")
        exit()
    
    # 5) Calcula a IV de cada opção via BSM c/ dividendo q.
    df_opcoes = calcular_iv_bsm(df_opcoes, spot, r, q, T)
    df_opcoes.dropna(subset=['iv'], inplace=True)
    if df_opcoes.empty:
        print("Nenhuma IV calculavel")
        exit()
    
    # 6) Separa apenas as opções OTM.
    df_otm = filtrar_otm(df_opcoes, spot, r, q, T)
    if df_otm.empty:
        print("Nenhuma OTM.")
        exit()
    
    # 7) Ajuste SVI e plot do smile.
    plot_svi(df_otm, spot, r, q, T, subjacente, vencimento)