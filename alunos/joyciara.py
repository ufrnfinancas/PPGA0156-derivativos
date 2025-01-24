# Este código tem como objetivo coletar dados de opções de ações de 10 empresas aleatórias,
# calcular e plotar gráficos conhecidos como "Smile de Volatilidade" para opções CALL e PUT.
# O "Smile de Volatilidade" é uma representação gráfica da volatilidade implícita das opções
# em diferentes níveis de preço de exercício (strike), usado para identificar a relação entre
# a volatilidade e o preço das opções.
import pandas as pd
import requests
import matplotlib.pyplot as plt
import random

# Função para obter a lista de empresas disponíveis no url do opções net.
def obter_empresas_aleatorias(url, num_empresas=10):
    """
    Obtém uma lista de empresas aleatórias a partir do link fornecido.
    """
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Erro ao acessar o site de empresas.")
        return []
    
    # Exemplo simplificado: Aqui estamos usando uma lista estática de empresas para simulação
    empresas = ['ABEV3', 'PETR4', 'ITUB4', 'VALE3', 'B3SA3', 'BRFS3', 'LREN3', 'MGLU3', 'SULA11', 'CSNA3']
    
    return random.sample(empresas, num_empresas)

# Função para obter as cotações de opções para uma empresa e vencimento específico
def obter_dados_opcoes(subjacente, vencimento):
    """
    Obtém as cotações das opções (CALL e PUT) para uma empresa e vencimento específico.
    """
    url = f'https://opcoes.net.br/listaopcoes/completa?idAcao={subjacente}&listarVencimentos=false&cotacoes=true&vencimentos={vencimento}'
    response = requests.get(url)
    
    # Verifica se a requisição foi bem-sucedida
    if response.status_code != 200:
        print(f"Erro ao acessar a API para o vencimento {vencimento}: {response.status_code}")
        return pd.DataFrame()
    
    try:
        data = response.json()
        cotacoes = data['data']['cotacoesOpcoes']
        rows = [
            [i[0].split('_')[0], i[2], i[3], i[5], i[8], i[9], i[10]]  # Dados relevantes como strike, tipo, volatilidade, etc.
            for i in cotacoes
        ]
        return pd.DataFrame(rows, columns=['ativo', 'tipo', 'modelo', 'strike', 'preco', 'negocios', 'volume'])
    except KeyError:
        print(f"Erro ao processar os dados das opções para o vencimento {vencimento}.")
        return pd.DataFrame()

# Função para plotar o Smile de Volatilidade
def plotar_smile_volatilidade(df_opcoes, tipo_opcao, all_strikes, all_volatilidades):
    """
    Plota o gráfico de Smile de Volatilidade para as opções CALL ou PUT de todas as empresas.
    """
    df_opcoes['strike'] = pd.to_numeric(df_opcoes['strike'], errors='coerce')
    df_opcoes['volatilidade'] = pd.to_numeric(df_opcoes['preco'], errors='coerce')  # Utilizando o preço como proxy da volatilidade
    df_opcoes = df_opcoes.dropna(subset=['strike', 'volatilidade'])

    # Adiciona os dados da empresa ao gráfico
    for _, row in df_opcoes.iterrows():
        all_strikes.append(row['strike'])
        all_volatilidades.append(row['volatilidade'])

    return all_strikes, all_volatilidades

# Função principal para gerar a análise para 10 empresas
def analisar_smile_volatilidade():
    url_empresas = 'https://opcoes.net.br/acoes'  # Link para pegar a lista de ações
    empresas_aleatorias = obter_empresas_aleatorias(url_empresas)

    # Listas para armazenar os dados de volatilidade para CALL e PUT
    all_strikes_call = []
    all_volatilidades_call = []
    all_strikes_put = []
    all_volatilidades_put = []

    # Vencimento das opções (exemplo fixo)
    vencimento = '2025-01-24'

    for empresa in empresas_aleatorias:
        print(f'Analisando empresa: {empresa}')
        df_opcoes = obter_dados_opcoes(empresa, vencimento)
        
        if not df_opcoes.empty:
            # Adiciona dados de CALL
            df_call = df_opcoes[df_opcoes['tipo'] == 'CALL']
            all_strikes_call, all_volatilidades_call = plotar_smile_volatilidade(df_call, 'CALL', all_strikes_call, all_volatilidades_call)

            # Adiciona dados de PUT
            df_put = df_opcoes[df_opcoes['tipo'] == 'PUT']
            all_strikes_put, all_volatilidades_put = plotar_smile_volatilidade(df_put, 'PUT', all_strikes_put, all_volatilidades_put)
        else:
            print(f"Nenhum dado encontrado para {empresa} no vencimento {vencimento}.")

    # Plotando os gráficos para CALL e PUT
    plt.figure(figsize=(12, 6))

    # Gráfico de CALL
    plt.subplot(1, 2, 1)
    plt.scatter(all_strikes_call, all_volatilidades_call, color='blue', label='CALL Options', alpha=0.5)
    plt.title('Smile de Volatilidade - CALL Options')
    plt.xlabel('Strike')
    plt.ylabel('Volatilidade Implícita (%)')
    plt.grid(True)
    plt.legend()

    # Gráfico de PUT
    plt.subplot(1, 2, 2)
    plt.scatter(all_strikes_put, all_volatilidades_put, color='red', label='PUT Options', alpha=0.5)
    plt.title('Smile de Volatilidade - PUT Options')
    plt.xlabel('Strike')
    plt.ylabel('Volatilidade Implícita (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Chama a função principal
analisar_smile_volatilidade()
# Estatísticas descritivas para volatilidade
print(df_opcoes['volatilidade'].describe())  # Estatísticas como média, desvio padrão, etc.

# Adicionando uma análise simples para as volatilidades
media_volatilidade_call = df_call['volatilidade'].mean()
media_volatilidade_put = df_put['volatilidade'].mean()

print(f'Média da volatilidade CALL: {media_volatilidade_call:.2f}%')
print(f'Média da volatilidade PUT: {media_volatilidade_put:.2f}%')

# Conclusão
if media_volatilidade_call > media_volatilidade_put:
    print("A volatilidade implícita das opções CALL está mais alta que das PUT.")
else:
    print("A volatilidade implícita das opções PUT está mais alta que das CALL.")
