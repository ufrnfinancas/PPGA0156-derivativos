# Obtendo dados de opções negociadas na B3 - Derivativos em Py #01

# Opções nos EUA
import yfinance as yf

# Defina o ticker
symbol = 'NVDA' 

# Crie o objeto
stock = yf.Ticker(symbol)

# Colete dados das opções - datas de vencimento
options = stock.options

# E, finalmente, a option chain
for option_symbol in options:
    option_chain = stock.option_chain(option_symbol)
    
    # Call options
    call_options = option_chain.calls
    print(f"Call Options para {option_symbol}:")
    print(call_options)

    # Put options
    put_options = option_chain.puts
    print(f"Put Options para {option_symbol}:")
    print(put_options)


# Opções na B3
import pandas as pd
import requests

subjacente = 'PETR4'   # não vem do Yfinance! esqueça o ".SA" ao final do ticker!

# Um vencimento
vencimento = '2024-11-14'      # YYYY-MM-DD
def optionchaindate(subjacente, vencimento):
    url = f'https://opcoes.net.br/listaopcoes/completa?idAcao={subjacente}&listarVencimentos=false&cotacoes=true&vencimentos={vencimento}'
    r = requests.get(url).json()
    x = ([subjacente, vencimento, i[0].split('_')[0], i[2], i[3], i[5], i[8], i[9], i[10]] for i in r['data']['cotacoesOpcoes'])
    return pd.DataFrame(x, columns=['subjacente', 'vencimento', 'ativo', 'tipo', 'modelo', 'strike', 'preco', 'negocios', 'volume'])

optionchaindate(subjacente, vencimento)

# Todos os vencimentos
def optionchain(subjacente):
    url2 = f'https://opcoes.net.br/listaopcoes/completa?idLista=ML&idAcao={subjacente}&listarVencimentos=true&cotacoes=true'
    r = requests.get(url2).json()
    vencimentos = [i['value'] for i in r['data']['vencimentos']]
    df = pd.concat([optionchaindate(subjacente, vencimento) for vencimento in vencimentos])
    return df

optionchain(subjacente)