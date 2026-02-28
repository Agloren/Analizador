import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import concurrent.futures
import requests
import io
import anthropic
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Screener Weinstein + IA", page_icon="📈", layout="wide")

st.title("📈 Screener Avanzado: Método Weinstein")
st.markdown("Busca señales de compra institucional e integra **Claude IA** para analizar los resultados.")

# --- FUNCIONES DEL SCREENER ---
@st.cache_data(ttl=3600) # Guarda en caché la lista por 1 hora para no saturar Wikipedia
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

def screener_weinstein(ticker, ticker_ref="^GSPC"):
    try:
        df = yf.download(ticker, period="5y", interval="1wk", progress=False)
        spx = yf.download(ticker_ref, period="5y", interval="1wk", progress=False)
        if df.empty or spx.empty or len(df) < 55: return None

        data = pd.DataFrame({
            'Close': df['Close'].squeeze(),
            'Volume': df['Volume'].squeeze(),
            'SPX_Close': spx['Close'].squeeze()
        }).dropna()

        u_mansf1, distancia_max = -30.0, 15.0

        # Cálculos Técnicos
        data['rs_line'] = (data['Close'] / data['SPX_Close']) * 100
        data['rs_ma'] = ta.sma(data['rs_line'], length=52)
        data['mansfield'] = ((data['rs_line'] / data['rs_ma']) - 1) * 100
        data['mansfield_ok1'] = (data['mansfield'] > u_mansf1) & (data['mansfield'] > data['mansfield'].shift(1))

        data['wma10'] = ta.wma(data['Close'], length=10)
        data['wma20'] = ta.wma(data['Close'], length=20)
        data['wma30'] = ta.wma(data['Close'], length=30)
        data['sma30'] = ta.sma(data['Close'], length=30)

        data['wma10_up'] = data['wma10'] > data['wma10'].shift(1)
        data['wma20_up'] = data['wma20'] > data['wma20'].shift(1)
        data['sma30_dn'] = data['sma30'] < data['sma30'].shift(1)

        data['distancia'] = ((data['Close'] - data['wma30']) / data['wma30']) * 100
        data['precio_ok'] = (data['Close'] >= data['wma30']) & (data['distancia'] <= distancia_max)
        
        crossover_sma30 = (data['Close'] > data['sma30']) & (data['Close'].shift(1) <= data['sma30'].shift(1))
        data['rompe_sma30'] = crossover_sma30 | (data['Close'] > data['sma30'])
        data['setup_1'] = (data['precio_ok'] & data['mansfield_ok1'] & data['wma10_up'] & data['wma20_up'] & data['sma30_dn'] & data['rompe_sma30'])

        data['vol_avg'] = ta.sma(data['Volume'], length=52)
        data['vol_std'] = ta.stdev(data['Volume'], length=52)
        data['vpm'] = (data['Volume'] - data['vol_avg']) / data['vol_std']
        data['vpm_ok'] = ta.sma(data['vpm'], length=5) > 0

        data['mansfield_ok2'] = (data['mansfield'] > 0.0) & (data['mansfield'] > data['mansfield'].shift(1))
        data['confirmacion'] = data['vpm_ok'] & data['mansfield_ok2'] & (data['Close'] > data['sma30'])
        data['confirmacion_change'] = (data['confirmacion'] == True) & (data['confirmacion'].shift(1) == False)

        senal_final = []
        barras_setup = None
        for i in range(len(data)):
            if data['setup_1'].iloc[i]: barras_setup = i
            setup_activo = (barras_setup is not None) and (i - barras_setup <= 12)
            if not setup_activo: barras_setup = None
            senal_compra = setup_activo and data['confirmacion_change'].iloc[i] and data['confirmacion'].iloc[i]
            senal_final.append(senal_compra)
            if senal_compra: barras_setup = None

        if senal_final[-1]:
            # Devolvemos datos para mostrarlos bonito en la tabla
            return {"Ticker": ticker, "Precio Actual": data['Close'].iloc[-1], "Volumen Relativo": data['vpm'].iloc[-1]}
        return None
    except Exception:
        return None

# --- INTERFAZ STREAMLIT ---
with st.sidebar:
    st.header("⚙️ Configuración")
    indice_seleccionado = st.selectbox("Selecciona el mercado:", ["S&P 500", "S&P 1500"])
    max_analisis = st.slider("Límite de acciones a analizar (para pruebas):", 50, 1500, 100)
    iniciar = st.button("🚀 Iniciar Escaneo Masivo", type="primary")

if iniciar:
    tickers = get_tickers(indice_seleccionado)[:max_analisis]
    st.info(f"Escaneando {len(tickers)} acciones. Por favor, espera...")
    
    progress_bar = st.progress(0)
    resultados = []
    
    # Análisis en paralelo
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(screener_weinstein, t): t for t in tickers}
        completados = 0
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            completados += 1
            progress_bar.progress(completados / len(tickers))
            resultado = future.result()
            if resultado:
                resultados.append(resultado)
                
    # --- RESULTADOS Y ANÁLISIS DE CLAUDE ---
    if resultados:
        st.success(f"✅ ¡Se encontraron {len(resultados)} acciones con señal de compra!")
        df_resultados = pd.DataFrame(resultados)
        st.dataframe(df_resultados, use_container_width=True)
        
        # Integración con Claude
        st.subheader("🤖 Análisis de Claude AI")
        try:
            # Llama a la API Key guardada en secreto de Streamlit
            client = anthropic.Anthropic(api_key=st.secrets["CLAUDE_API_KEY"])
            tickers_ganadores = ", ".join(df_resultados['Ticker'].tolist())
            
            with st.spinner("Claude está analizando el contexto de estas acciones..."):
                mensaje = client.messages.create(
                    model="claude-3-haiku-20240307", # Haiku es súper rápido y económico para esto
                    max_tokens=500,
                    messages=[
                        {"role": "user", "content": f"Actúa como un experto de Wall Street. Las siguientes acciones acaban de dar señal de compra técnica a largo plazo: {tickers_ganadores}. Dame un resumen de 2 párrafos indicando a qué sectores pertenecen, si hay alguna narrativa de mercado que las esté impulsando y si ves correlación entre ellas."}
                    ]
                )
                st.write(mensaje.content[0].text)
        except Exception as e:
            st.warning("No se pudo conectar con Claude. Revisa que tu API Key esté configurada en los Secrets de Streamlit.")
            st.error(e)
    else:
        st.warning("Ninguna acción dio señal de compra esta semana.")
