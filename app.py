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
st.markdown("Analizador institucional: Identifica rupturas en la **ÚLTIMA VELA SEMANAL** con confirmación de Volumen, Volatilidad (Gap BB 15%) y Fuerza (RSI/ADX).")

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
    weights = np.arange(1, length + 1)
    return serie.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def get_rma(serie, length):
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
        # Descarga
        df = yf.download(ticker, period="5y", interval="1wk", progress=False)
        spx = yf.download(ticker_ref, period="5y", interval="1wk", progress=False)
        
        # Limpieza de MultiIndex (Para las nuevas versiones de yfinance)
        if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
        if isinstance(spx.columns, pd.MultiIndex): spx.columns = [c[0] for c in spx.columns]
        
        if df.empty or spx.empty or len(df) < 55: return None

        df.index = pd.to_datetime(df.index).tz_localize(None)
        spx.index = pd.to_datetime(spx.index).tz_localize(None)

        # SINCRONIZACIÓN MILIMÉTRICA POR SEMANA ISO (Evita desfases de festivos)
        df['yw'] = df.index.isocalendar().year.astype(str) + "-" + df.index.isocalendar().week.astype(str)
        spx['yw'] = spx.index.isocalendar().year.astype(str) + "-" + spx.index.isocalendar().week.astype(str)
        df = df.drop_duplicates(subset=['yw'], keep='last').set_index('yw')
        spx = spx.drop_duplicates(subset=['yw'], keep='last').set_index('yw')

        # CORRECCIÓN OHLC (El bug del ADX): Ajustar TODO en proporción a los dividendos
        close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        if close_col == 'Adj Close':
            adj_ratio = df['Adj Close'] / df['Close']
            data = pd.DataFrame({
                'High': df['High'] * adj_ratio,
                'Low': df['Low'] * adj_ratio,
                'Close': df['Adj Close'],
                'Volume': df['Volume']
            })
        else:
            data = pd.DataFrame({
                'High': df['High'], 'Low': df['Low'], 'Close': df['Close'], 'Volume': df['Volume']
            })
        
        spx_close_col = 'Adj Close' if 'Adj Close' in spx.columns else 'Close'
        spx_df = pd.DataFrame({'SPX_Close': spx[spx_close_col]})
        
        # Unimos las tablas y descartamos semanas a medias (volumen 0)
        data = data.join(spx_df, how='inner').dropna()
        data = data[data['Volume'] > 0]
        if len(data) < 55: return None

        # --- PARÁMETROS ---
        u_mansf1, distancia_max = -30.0, 15.0
        bb_len, bb_mult = 30, 1.0
        rsi_len, rsi_threshold = 14, 55.0
        adx_len, adx_threshold = 14, 15.0
        
        # Bollinger
        basis = data['Close'].rolling(bb_len).mean()
        dev = bb_mult * data['Close'].rolling(bb_len).std(ddof=0)
        data['gap_volatilidad'] = (basis + dev) >= ((basis - dev) * 1.15)

        # Mansfield
        data['rs_line'] = (data['Close'] / data['SPX_Close']) * 100
        data['rs_ma'] = data['rs_line'].rolling(52).mean()
        data['mansfield'] = ((data['rs_line'] / data['rs_ma']) - 1) * 100
        data['mansfield_ok1'] = (data['mansfield'] > u_mansf1) & (data['mansfield'] > data['mansfield'].shift(1))

        # Medias y Setup 1
        data['wma10'] = get_wma(data['Close'], 10)
        data['wma20'] = get_wma(data['Close'], 20)
        data['wma30'] = get_wma(data['Close'], 30)
        data['sma30'] = data['Close'].rolling(30).mean()

        data['distancia'] = ((data['Close'] - data['wma30']) / data['wma30']) * 100
        precio_ok = (data['Close'] >= data['wma30']) & (data['distancia'] <= distancia_max)
        medias_ok = (data['wma10'] > data['wma10'].shift(1)) & (data['wma20'] > data['wma20'].shift(1)) & (data['sma30'] < data['sma30'].shift(1))
        data['setup_1'] = precio_ok & data['mansfield_ok1'] & medias_ok & (data['Close'] > data['sma30'])

        # Volumen
        data['vol_avg'] = data['Volume'].rolling(52).mean()
        data['vol_std'] = data['Volume'].rolling(52).std(ddof=0) 
        data['vpm'] = (data['Volume'] - data['vol_avg']) / data['vol_std']
        data['vpm5'] = data['vpm'].rolling(5).mean()
        mansfield_ok2 = (data['mansfield'] > 0.0) & (data['mansfield'] > data['mansfield'].shift(1))

        # RSI
        delta = data['Close'].diff()
        gain = get_rma(delta.where(delta > 0, 0), rsi_len)
        loss = get_rma(abs(delta.where(delta < 0, 0)), rsi_len)
        data['rsi'] = np.where(loss == 0, 100, 100 - (100 / (1 + (gain / loss))))

        # ADX (Con TR Ajustado)
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

        # CONFIRMACIÓN
        data['confirmacion'] = (data['vpm5'] > 0) & mansfield_ok2 & (data['Close'] > data['sma30']) & (data['rsi'] > rsi_threshold) & ((data['adx'] > adx_threshold) & (data['di_plus'] > data['di_minus']))
        data['confirmacion_change'] = (data['confirmacion'] == True) & (data['confirmacion'].shift(1) == False)

        senal_final = []
        barras_setup = None
        
        # Bucle de memoria de 12 semanas idéntico al "var int semana_setup" de TradingView
        for i in range(len(data)):
            if data['setup_1'].iloc[i]: barras_setup = i
            setup_activo = (barras_setup is not None) and (i - barras_setup <= 12)
            if not setup_activo: barras_setup = None
            
            senal_compra = setup_activo and data['confirmacion_change'].iloc[i] and data['gap_volatilidad'].iloc[i]
            senal_final.append(senal_compra)
            if senal_compra: barras_setup = None

        # REGLA ESTRICTA: Solo devolver datos si la ÚLTIMA VELA es True
        if senal_final[-1] == True:
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
    st.info(f"Escaneando {len(tickers)} acciones. Buscando rupturas EXCLUSIVAS de esta semana...")
    
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
        st.success(f"✅ ¡BINGO! Se encontraron {len(resultados)} acciones con señal de compra en la última vela.")
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
                        {"role": "user", "content": f"Actúa como un trader institucional. Las acciones {tickers_ganadores} acaban de dar una señal técnica alcista fortísima (Ruptura Weinstein + Expansión BB + ADX alcista). Analiza de qué sectores son y si la macroeconomía actual justifica esto."}
                    ]
                )
                st.write(mensaje.content[0].text)
        except Exception as e:
            st.warning("No se pudo conectar con Claude. Recuerda configurar tu 'CLAUDE_API_KEY' en los Secrets.")
    else:
        st.warning("Mercado sin rupturas: Ninguna acción cumplió con la señal de Fusión en la vela actual.")
