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

# --- FUNCIONES MATEMÁTICAS EXACTAS DE TRADINGVIEW ---
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
    """Fórmula exacta del RMA (Wilder's Smoothing) usado en RSI y ADX por TradingView"""
    return serie.ewm(alpha=1/length, min_periods=length, adjust=False).mean()

def screener_fusion(ticker, ticker_ref="^GSPC"):
    try:
        # 1. Descarga de datos
        df = yf.download(ticker, period="5y", interval="1wk", progress=False)
        spx = yf.download(ticker_ref, period="5y", interval="1wk", progress=False)
        
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if isinstance(spx.columns, pd.MultiIndex): spx.columns = spx.columns.droplevel(1)
        
        if df.empty or spx.empty or len(df) < 55: return None

        df.index = pd.to_datetime(df.index).tz_localize(None)
        spx.index = pd.to_datetime(spx.index).tz_localize(None)
        
        # Necesitamos High, Low, Close y Volume para el ADX y RSI
        data = pd.DataFrame({
            'High': df['High'], 'Low': df['Low'], 'Close': df['Close'], 'Volume': df['Volume']
        })
        spx_df = pd.DataFrame({'SPX_Close': spx['Close']})
        
        data = pd.merge_asof(data, spx_df, left_index=True, right_index=True, direction='backward')
        data = data.dropna()
        if len(data) < 55: return None

        # --- PARÁMETROS DEL CÓDIGO ---
        u_mansf1, distancia_max = -30.0, 15.0
        bb_len, bb_mult = 30, 1.0
        rsi_len, rsi_threshold = 14, 55.0
        adx_len, adx_threshold = 14, 15.0
        
        # --- 2. BANDAS DE BOLLINGER (Gap 15%) ---
        basis = data['Close'].rolling(bb_len).mean()
        dev = bb_mult * data['Close'].rolling(bb_len).std(ddof=0)
        upper = basis + dev
        lower = basis - dev
        data['gap_volatilidad'] = upper >= (lower * 1.15)

        # --- 3. MANSFIELD RS ---
        data['rs_line'] = (data['Close'] / data['SPX_Close']) * 100
        data['rs_ma'] = data['rs_line'].rolling(52).mean()
        data['mansfield'] = ((data['rs_line'] / data['rs_ma']) - 1) * 100
        data['mansfield_ok1'] = (data['mansfield'] > u_mansf1) & (data['mansfield'] > data['mansfield'].shift(1))

        # --- 4. MEDIAS MÓVILES Y SETUP BASE ---
        data['wma10'] = get_wma(data['Close'], 10)
        data['wma20'] = get_wma(data['Close'], 20)
        data['wma30'] = get_wma(data['Close'], 30)
        data['sma30'] = data['Close'].rolling(30).mean()

        data['distancia'] = ((data['Close'] - data['wma30']) / data['wma30']) * 100
        precio_ok = (data['Close'] >= data['wma30']) & (data['distancia'] <= distancia_max)
        
        medias_ok = (data['wma10'] > data['wma10'].shift(1)) & (data['wma20'] > data['wma20'].shift(1)) & (data['sma30'] < data['sma30'].shift(1))
        rompe_sma30 = data['Close'] > data['sma30']

        data['setup_1'] = precio_ok & data['mansfield_ok1'] & medias_ok & rompe_sma30

        # --- 5. VOLUMEN (VPM5) ---
        data['vol_avg'] = data['Volume'].rolling(52).mean()
        data['vol_std'] = data['Volume'].rolling(52).std(ddof=0) 
        data['vpm'] = (data['Volume'] - data['vol_avg']) / data['vol_std']
        data['vpm5'] = data['vpm'].rolling(5).mean()
        vpm_ok = data['vpm5'] > 0
        mansfield_ok2 = (data['mansfield'] > 0.0) & (data['mansfield'] > data['mansfield'].shift(1))

        # --- 6. MOMENTO: RSI (14) ---
        delta = data['Close'].diff()
        gain = get_rma(delta.where(delta > 0, 0), rsi_len)
        loss = -get_rma(delta.where(delta < 0, 0), rsi_len)
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        rsi_ok = data['rsi'] > rsi_threshold

        # --- 7. MOMENTO: ADX & DMI (14) ---
        up = data['High'].diff()
        down = -data['Low'].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        
        tr1 = data['High'] - data['Low']
        tr2 = abs(data['High'] - data['Close'].shift(1))
        tr3 = abs(data['Low'] - data['Close'].shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

        smoothed_tr = get_rma(tr, adx_len)
        data['di_plus'] = 100 * get_rma(pd.Series(plus_dm, index=data.index), adx_len) / smoothed_tr
        data['di_minus'] = 100 * get_rma(pd.Series(minus_dm, index=data.index), adx_len) / smoothed_tr
        
        dx = 100 * abs(data['di_plus'] - data['di_minus']) / (data['di_plus'] + data['di_minus'])
        data['adx'] = get_rma(dx, adx_len)
        
        adx_ok = (data['adx'] > adx_threshold) & (data['di_plus'] > data['di_minus'])

        # --- 8. CONFIRMACIÓN Y SEÑAL FINAL (Memoria 12 Semanas) ---
        data['confirmacion'] = vpm_ok & mansfield_ok2 & (data['Close'] > data['sma30']) & rsi_ok & adx_ok
        data['confirmacion_change'] = (data['confirmacion'] == True) & (data['confirmacion'].shift(1) == False)

        senal_final = []
        barras_setup = None
        
        for i in range(len(data)):
            if data['setup_1'].iloc[i]: barras_setup = i
            setup_activo = (barras_setup is not None) and (i - barras_setup <= 12)
            if not setup_activo: barras_setup = None
            
            # Lógica final de compra: Setup activo + Cambio confirmación + Gap BB
            senal_compra = setup_activo and data['confirmacion_change'].iloc[i] and data['gap_volatilidad'].iloc[i]
            senal_final.append(senal_compra)
            if senal_compra: barras_setup = None

        if senal_final[-1]:
            return {
                "Ticker": ticker, 
                "Precio ($)": round(data['Close'].iloc[-1], 2), 
                "RSI": round(data['rsi'].iloc[-1], 1),
                "ADX": round(data['adx'].iloc[-1], 1),
                "Mansfield RS": round(data['mansfield'].iloc[-1], 2)
            }
        return None
    except Exception:
        return None

# --- INTERFAZ STREAMLIT ---
with st.sidebar:
    st.header("⚙️ Configuración")
    indice_seleccionado = st.selectbox("Mercado a escanear:", ["S&P 500", "S&P 1500"])
    max_analisis = st.slider("Límite de acciones (Pruebas):", 50, 1500, 1500)
    iniciar = st.button("🚀 Iniciar Escaneo Fusión", type="primary")

if iniciar:
    tickers = get_tickers(indice_seleccionado)[:max_analisis]
    st.info(f"Escaneando {len(tickers)} acciones con filtros de Volatilidad y Momento...")
    
    progress_bar = st.progress(0)
    resultados = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(screener_fusion, t): t for t in tickers}
        completados = 0
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            completados += 1
            progress_bar.progress(completados / len(tickers))
            resultado = future.result()
            if resultado:
                resultados.append(resultado)
                
    if resultados:
        st.success(f"✅ ¡BINGO! Se encontraron {len(resultados)} acciones listas para el despegue.")
        df_resultados = pd.DataFrame(resultados)
        st.dataframe(df_resultados, use_container_width=True)
        
        st.subheader("🤖 Opinión de Claude AI")
        try:
            client = anthropic.Anthropic(api_key=st.secrets["CLAUDE_API_KEY"])
            tickers_ganadores = ", ".join(df_resultados['Ticker'].tolist())
            
            with st.spinner("Claude está elaborando el análisis fundamental..."):
                mensaje = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=500,
                    messages=[
                        {"role": "user", "content": f"Actúa como un trader institucional. Las acciones {tickers_ganadores} acaban de dar una señal técnica alcista fortísima (Ruptura Weinstein + Expansión de Bandas Bollinger + ADX alcista). Analiza brevemente de qué sectores son y si el entorno macroeconómico actual justifica esta explosión alcista."}
                    ]
                )
                st.write(mensaje.content[0].text)
        except Exception as e:
            st.warning("No se pudo conectar con Claude. Recuerda configurar tu 'CLAUDE_API_KEY' en los Secrets.")
    else:
        st.warning("Mercado en pausa: Ninguna acción cumplió con la exigente señal de Fusión esta semana.")
