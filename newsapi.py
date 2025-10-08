# News API
import requests
from datetime import datetime, timedelta

# Configurações da API
API_KEY = "CHAVE AQUI"
BASE_URL = "https://newsapi.org/v2"


def buscar_noticias(termo, idioma="pt", ordenar_por="publishedAt", limite=5):
    """Busca notícias sobre um tema específico."""
    url = f"{BASE_URL}/everything"
    data_inicial = (datetime.now() - timedelta(days=29)).strftime("%Y-%m-%d")
    
    parametros = {
        "q": termo,
        "from": data_inicial,
        "language": idioma,
        "sortBy": ordenar_por,
        "apiKey": API_KEY
    }
    
    resposta = requests.get(url, params=parametros)
    
    if resposta.status_code == 200:
        dados = resposta.json()
        print(f"\nEncontradas {dados['totalResults']} notícias sobre '{termo}':\n")
        
        for artigo in dados['articles'][:limite]:
            print(f"Título: {artigo['title']}")
            print(f"Fonte: {artigo['source']['name']}")
            print(f"Data: {artigo['publishedAt']}")
            print(f"URL: {artigo['url']}")
            print("-" * 80)
    else:
        print(f"Erro: {resposta.status_code}")
        print(resposta.json())


def top_headlines(pais="us", categoria=None, limite=5):
    """Busca principais manchetes de um país (plano gratuito: us, gb, ca, au)."""
    url = f"{BASE_URL}/top-headlines"
    
    parametros = {
        "country": pais,
        "apiKey": API_KEY
    }
    
    if categoria:
        parametros["category"] = categoria
    
    resposta = requests.get(url, params=parametros)
    
    if resposta.status_code == 200:
        dados = resposta.json()
        print(f"\nTop headlines - {pais.upper()}:\n")
        
        for artigo in dados['articles'][:limite]:
            print(f"Título: {artigo['title']}")
            print(f"Descrição: {artigo.get('description', 'N/A')}")
            print(f"URL: {artigo['url']}")
            print("-" * 80)
    else:
        print(f"Erro: {resposta.status_code}")
        print(resposta.json())


def noticias_brasil(limite=10):
    """Busca notícias em português sobre o Brasil."""
    url = f"{BASE_URL}/everything"
    
    parametros = {
        "q": "Brasil OR Brasília OR São Paulo",
        "language": "pt",
        "sortBy": "publishedAt",
        "apiKey": API_KEY
    }
    
    resposta = requests.get(url, params=parametros)
    
    if resposta.status_code == 200:
        dados = resposta.json()
        print(f"\nNotícias em português ({dados['totalResults']} encontradas):\n")
        
        for artigo in dados['articles'][:limite]:
            print(f"Título: {artigo['title']}")
            print(f"Fonte: {artigo['source']['name']}")
            print(f"Data: {artigo['publishedAt']}")
            print(f"URL: {artigo['url']}")
            print("-" * 80)
    else:
        print(f"Erro: {resposta.status_code}")
        print(resposta.json())


noticias_brasil()
top_headlines("us")
buscar_noticias("ufrn")