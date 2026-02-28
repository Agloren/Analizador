import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
import requests
import io
import anthropic
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Screener Fusión: Weinstein + BB + Momento", page_icon="🎯", layout="wide")

st.title("🎯 Screener Fusión: Weinstein + BB + Momento")
st.markdown("Analizador institucional: Identifica rupturas con confirmación de Volumen, Volatilidad (Gap BB 15%) y Fuerza (RSI/ADX). Lógica 100% TradingView.")

# --- FUNCIONES MATEMÁTICAS ---
@st.cache_data(ttl=3600)
def get_tickers(indice):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}
    urls = []
    if indice == "S&P 500":
        urls = ['https://en.wikipedia.org/wiki/List_of_S%26P_500_companies']
    elif indice == "S&P 1500":
        urls = [
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies',
            'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
        ]
    
    tickers = []
    for url in urls:
        try:
            res = requests.get(url, headers=headers)
            df_list = pd.read_html(io.StringIO(res.text))
            df = df_list[0]
            col = 'Symbol' if 'Symbol' in df.columns else ('Ticker symbol' if 'Ticker symbol' in df.columns else df.columns[0])
            tickers.extend(df[col].tolist())
        except Exception:
            pass
    return list(set([str(t).replace('.', '-') for t in tickers]))

def get_wma(serie, length):
    """Fórmula exacta del WMA de TradingView"""
    weights = np.arange(1, length + 1)
    return serie.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def get_rma(serie, length):
    """Réplica iterativa EXACTA del ta.rma de Pine Script"""
    rma = np.full_like(serie, np.nan, dtype=float)
    valid_idx = serie.first_valid_index()
    if valid_idx is None: return pd.Series(rma, index=serie.index)
    
    start_idx = serie.index.get_loc(valid_idx)
    if start_idx + length > len(serie): return pd.Series(rma, index=serie.index)
    
    rma[start_idx + length - 1] = np.mean(serie.iloc[start_idx : start_idx + length])
    
    for i in range(start_idx + length, len(serie)):
        rma[i] = (serie.iloc[i] + (length - 1) * rma[i-1]) / length
        
    return pd.Series(rma, index=serie.index)

def screener_fusion(ticker, ticker_ref="^GSPC"):
    try:
        # Descarga de datos
        df = yf.download(ticker, period="5y", interval="1wk", progress=False)
        spx = yf.download(ticker_ref, period="5y", interval="1wk", progress=False)
        
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if isinstance(spx.columns, pd.MultiIndex): spx.columns = spx.columns.droplevel(1)
        
        if df.empty or spx.empty or len(df) < 55: return None

        df.index = pd.to_datetime(df.index).tz_localize(None)
        spx.index = pd.to_datetime(spx.index).tz_localize(None)
        
        # Ajuste por dividendos (Clave para coincidir con TV)
        close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        spx_close_col = 'Adj Close' if 'Adj Close' in spx.columns else 'Close'

        data = pd.DataFrame({
            'High': df['High'], 'Low': df['Low'], 'Close': df[close_
