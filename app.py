def get_wma(serie, length):
    """Fórmula exacta del WMA de TradingView"""
    weights = np.arange(1, length + 1)
    return serie.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def get_rma(serie, length):
    """Réplica iterativa EXACTA del ta.rma de Pine Script (Evita variaciones de decimales en RSI/ADX)"""
    rma = np.full_like(serie, np.nan, dtype=float)
    valid_idx = serie.first_valid_index()
    if valid_idx is None: return pd.Series(rma, index=serie.index)
    
    start_idx = serie.index.get_loc(valid_idx)
    if start_idx + length > len(serie): return pd.Series(rma, index=serie.index)
    
    # TV inicializa el primer valor RMA como una media simple (SMA)
    rma[start_idx + length - 1] = np.mean(serie.iloc[start_idx : start_idx + length])
    
    # Y a partir de ahí usa la fórmula de Wilder
    for i in range(start_idx + length, len(serie)):
        rma[i] = (serie.iloc[i] + (length - 1) * rma[i-1]) / length
        
    return pd.Series(rma, index=serie.index)

def screener_fusion(ticker, ticker_ref="^GSPC"):
    try:
        # Descargamos los datos
        df = yf.download(ticker, period="5y", interval="1wk", progress=False)
        spx = yf.download(ticker_ref, period="5y", interval="1wk", progress=False)
        
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if isinstance(spx.columns, pd.MultiIndex): spx.columns = spx.columns.droplevel(1)
        
        if df.empty or spx.empty or len(df) < 55: return None

        df.index = pd.to_datetime(df.index).tz_localize(None)
        spx.index = pd.to_datetime(spx.index).tz_localize(None)
        
        # ¡CAMBIO CRUCIAL AQUÍ!: Usamos 'Adj Close' (si existe, sino 'Close') para igualar a TV
        close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        spx_close_col = 'Adj Close' if 'Adj Close' in spx.columns else 'Close'

        data = pd.DataFrame({
            'High': df['High'], 'Low': df['Low'], 'Close': df[close_col], 'Volume': df['Volume']
        })
        spx_df = pd.DataFrame({'SPX_Close': spx[spx_close_col]})
        
        data = pd.merge_asof(data, spx_df, left_index=True, right_index=True, direction='backward')
        data = data.dropna()
        if len(data) < 55: return None

        u_mansf1, distancia_max = -30.0, 15.0
        bb_len, bb_mult = 30, 1.0
        rsi_len, rsi_threshold = 14, 55.0
        adx_len, adx_threshold = 14, 15.0
        
        # --- BANDAS DE BOLLINGER ---
        basis = data['Close'].rolling(bb_len).mean()
        dev = bb_mult * data['Close'].rolling(bb_len).std(ddof=0)
        upper = basis + dev
        lower = basis - dev
        data['gap_volatilidad'] = upper >= (lower * 1.15)

        # --- MANSFIELD RS ---
        data['rs_line'] = (data['Close'] / data['SPX_Close']) * 100
        data['rs_ma'] = data['rs_line'].rolling(52).mean()
        data['mansfield'] = ((data['rs_line'] / data['rs_ma']) - 1) * 100
        data['mansfield_ok1'] = (data['mansfield'] > u_mansf1) & (data['mansfield'] > data['mansfield'].shift(1))

        # --- MEDIAS MÓVILES ---
        data['wma10'] = get_wma(data['Close'], 10)
        data['wma20'] = get_wma(data['Close'], 20)
        data['wma30'] = get_wma(data['Close'], 30)
        data['sma30'] = data['Close'].rolling(30).mean()

        data['distancia'] = ((data['Close'] - data['wma30']) / data['wma30']) * 100
        precio_ok = (data['Close'] >= data['wma30']) & (data['distancia'] <= distancia_max)
        
        medias_ok = (data['wma10'] > data['wma10'].shift(1)) & (data['wma20'] > data['wma20'].shift(1)) & (data['sma30'] < data['sma30'].shift(1))
        rompe_sma30 = data['Close'] > data['sma30']

        data['setup_1'] = precio_ok & data['mansfield_ok1'] & medias_ok & rompe_sma30

        # --- VOLUMEN ---
        data['vol_avg'] = data['Volume'].rolling(52).mean()
        data['vol_std'] = data['Volume'].rolling(52).std(ddof=0) 
        data['vpm'] = (data['Volume'] - data['vol_avg']) / data['vol_std']
        data['vpm5'] = data['vpm'].rolling(5).mean()
        vpm_ok = data['vpm5'] > 0
        mansfield_ok2 = (data['mansfield'] > 0.0) & (data['mansfield'] > data['mansfield'].shift(1))

        # --- MOMENTO: RSI ---
        delta = data['Close'].diff()
        gain = get_rma(delta.where(delta > 0, 0), rsi_len)
        loss = get_rma(abs(delta.where(delta < 0, 0)), rsi_len) # Absoluto para evitar divisiones con negativo
        rs = gain / loss
        data['rsi'] = np.where(loss == 0, 100, 100 - (100 / (1 + rs)))
        rsi_ok = data['rsi'] > rsi_threshold

        # --- MOMENTO: ADX ---
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

        # --- CONFIRMACIÓN Y MEMORIA ---
        data['confirmacion'] = vpm_ok & mansfield_ok2 & (data['Close'] > data['sma30']) & rsi_ok & adx_ok
        data['confirmacion_change'] = (data['confirmacion'] == True) & (data['confirmacion'].shift(1) == False)

        senal_final = []
        barras_setup = None
        
        for i in range(len(data)):
            if data['setup_1'].iloc[i]: barras_setup = i
            setup_activo = (barras_setup is not None) and (i - barras_setup <= 12)
            if not setup_activo: barras_setup = None
            
            senal_compra = setup_activo and data['confirmacion_change'].iloc[i] and data['gap_volatilidad'].iloc[i]
            senal_final.append(senal_compra)
            if senal_compra: barras_setup = None

        if senal_final[-1]:
            return {
                "Ticker": ticker, 
                "Precio Ajustado ($)": round(data['Close'].iloc[-1], 2), 
                "RSI": round(data['rsi'].iloc[-1], 1),
                "ADX": round(data['adx'].iloc[-1], 1),
                "Mansfield RS": round(data['mansfield'].iloc[-1], 2)
            }
        return None
    except Exception:
        return None
