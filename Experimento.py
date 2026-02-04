"""
experimento - Dashboard Financiero Pro Avanzado.

Dashboard interactivo para análisis financiero cuantitativo con optimización
robusta de portafolios, Ledoit-Wolf shrinkage, control de concentración y CVaR.
"""

from datetime import datetime, timedelta
from io import BytesIO

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

# ==========================================
# CONSTANTES
# ==========================================
ANNUAL_FACTOR = 252
TASA_LIBRE_RIESGO = 0.04
ESTRATEGIA_OPTIMIZACION = "Maximizar Sharpe Ratio"
NIVEL_CONFIANZA_CVAR = 0.05  # 95% de confianza

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Dashboard Financiero Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS personalizado - Esquema oscuro forzado
st.markdown(
    """
    <style>
    /* Forzar tema oscuro en toda la aplicación */
    .stApp {
        background-color: #0a0e27 !important;
        color: #e5e7eb !important;
    }
    
    /* Fondo de la barra lateral */
    [data-testid="stSidebar"] {
        background-color: #0d1117 !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: #0d1117 !important;
    }
    
    /* Fondo del contenido principal */
    .main .block-container {
        background-color: #0a0e27 !important;
        padding-top: 2rem;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d9ff;
        text-align: center;
        padding: 1rem 0;
        text-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1f3a 0%, #2d1b4e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(168, 85, 247, 0.2);
        color: white;
        text-align: center;
    }
    
    /* Estilo de métricas */
    .stMetric {
        background-color: #151a2e !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border: 1px solid rgba(0, 217, 255, 0.15) !important;
    }
    
    .stMetric label {
        color: #9ca3af !important;
        font-size: 0.9rem !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        color: #10b981 !important;
    }
    
    /* Tablas y DataFrames */
    .stDataFrame {
        background-color: #151a2e !important;
        border-radius: 8px !important;
        border: 1px solid rgba(0, 217, 255, 0.1) !important;
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background-color: #151a2e !important;
    }
    
    /* Inputs y selectboxes */
    .stTextInput input, .stSelectbox select, .stMultiSelect {
        background-color: #151a2e !important;
        color: #e5e7eb !important;
        border: 1px solid rgba(0, 217, 255, 0.2) !important;
        border-radius: 8px !important;
    }
    
    /* Botones */
    .stButton button {
        background: linear-gradient(135deg, #00d9ff 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 20px rgba(0, 217, 255, 0.4) !important;
    }
    
    /* Checkboxes */
    .stCheckbox {
        color: #e5e7eb !important;
    }
    
    /* Sliders */
    .stSlider {
        color: #e5e7eb !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #151a2e !important;
        color: #e5e7eb !important;
        border-radius: 8px !important;
        border: 1px solid rgba(0, 217, 255, 0.15) !important;
    }
    
    /* Headers y texto */
    h1, h2, h3, h4, h5, h6 {
        color: #f3f4f6 !important;
    }
    
    p, span, div {
        color: #e5e7eb !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #151a2e !important;
        border-radius: 8px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #9ca3af !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00d9ff !important;
        border-bottom-color: #00d9ff !important;
    }
    
    /* Mensajes de info/warning/success */
    .stAlert {
        background-color: #151a2e !important;
        border-left: 4px solid #00d9ff !important;
        color: #e5e7eb !important;
    }
    
    /* Divisores */
    hr {
        border-color: rgba(0, 217, 255, 0.2) !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #00d9ff !important;
    }
    </style>
""",
    unsafe_allow_html=True
)

st.markdown(
    '<h1 class="main-header">'
    '📈 Dashboard de Análisis Financiero Cuantitativo Pro'
    '</h1>',
    unsafe_allow_html=True
)

# ==========================================
# SESSION STATE
# ==========================================
if 'datos_descargados' not in st.session_state:
    st.session_state.datos_descargados = None
if 'tickers_analizados' not in st.session_state:
    st.session_state.tickers_analizados = []
if 'fecha_descarga' not in st.session_state:
    st.session_state.fecha_descarga = None


# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def calcular_pesos_dinamicos(num_activos):
    """
    Calcula pesos máximos y mínimos dinámicos basados en criterios robustos.

    Args:
        num_activos: Número de activos en el portafolio

    Returns:
        tuple: (max_weight, min_weight) calculados dinámicamente
    """
    # Peso mínimo: asegurar diversificación mínima (HHI approach)
    min_weight = max(0.02, 1.0 / (2 * num_activos))

    # Peso máximo: basado en el índice Herfindahl-Hirschman modificado
    # Para evitar concentración excesiva manteniendo flexibilidad
    if num_activos <= 3:
        max_weight = 0.50  # Hasta 50% con 3 activos o menos
    elif num_activos <= 5:
        max_weight = 0.35  # Hasta 35% con 4-5 activos
    elif num_activos <= 10:
        max_weight = 0.25  # Hasta 25% con 6-10 activos
    else:
        max_weight = 0.20  # Hasta 20% con más de 10 activos

    # Validar que los límites son factibles
    if min_weight * num_activos > 1.0:
        min_weight = 0.8 / num_activos

    return max_weight, min_weight


def estimar_matriz_covarianza_robusta(returns_df):
    """
    Estima la matriz de covarianza usando Ledoit-Wolf Shrinkage.

    Args:
        returns_df: DataFrame con retornos de los activos

    Returns:
        numpy.ndarray: Matriz de covarianza robusta anualizada
    """
    lw = LedoitWolf()
    lw.fit(returns_df.dropna())
    cov_matrix_robust = lw.covariance_ * ANNUAL_FACTOR
    return cov_matrix_robust


def calcular_cvar(returns, nivel_confianza=NIVEL_CONFIANZA_CVAR):
    """
    Calcula el Conditional Value at Risk (CVaR) o Expected Shortfall.

    Args:
        returns: Serie de retornos
        nivel_confianza: Nivel de confianza (default: 0.05 para 95%)

    Returns:
        float: CVaR al nivel de confianza especificado
    """
    var = np.percentile(returns, nivel_confianza * 100)
    cvar = returns[returns <= var].mean()
    return cvar


def calcular_metricas_riesgo(returns, factor_anual=ANNUAL_FACTOR):
    """
    Calcula métricas de riesgo con precisión matemática incluyendo CVaR.

    Args:
        returns: Serie de retornos logarítmicos
        factor_anual: Factor de anualización

    Returns:
        dict: Diccionario con métricas calculadas
    """
    rendimiento_anual = returns.mean() * factor_anual
    volatilidad_anual = returns.std() * np.sqrt(factor_anual)

    if volatilidad_anual != 0:
        ratio_sharpe = (rendimiento_anual - TASA_LIBRE_RIESGO) / \
            volatilidad_anual
    else:
        ratio_sharpe = 0

    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std() * np.sqrt(factor_anual)
    else:
        downside_std = volatilidad_anual

    if downside_std != 0:
        ratio_sortino = (rendimiento_anual - TASA_LIBRE_RIESGO) / downside_std
    else:
        ratio_sortino = 0

    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    var_95 = np.percentile(returns, 5)
    cvar_95 = calcular_cvar(returns, 0.05)

    return {
        'rendimiento_anual': rendimiento_anual,
        'volatilidad_anual': volatilidad_anual,
        'sharpe_ratio': ratio_sharpe,
        'sortino_ratio': ratio_sortino,
        'max_drawdown': max_drawdown,
        'var_95': var_95 * np.sqrt(factor_anual),
        'cvar_95': cvar_95 * np.sqrt(factor_anual)
    }


def calcular_beta(returns_activo, returns_mercado):
    """Calcula el Beta del activo respecto al mercado."""
    df_temp = pd.DataFrame(
        {'activo': returns_activo, 'mercado': returns_mercado}).dropna()

    if len(df_temp) < 30:
        return None, None

    x_values = df_temp['mercado'].values
    y_values = df_temp['activo'].values

    coef = np.polyfit(x_values, y_values, 1)
    slope = coef[0]

    y_pred = np.polyval(coef, x_values)
    ss_res = np.sum((y_values - y_pred) ** 2)
    ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
    coef_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return slope, coef_r2


def calcular_rendimiento_portafolio(pesos, rendimientos_medios):
    """Calcula el rendimiento esperado del portafolio."""
    return np.sum(pesos * rendimientos_medios)


def calcular_volatilidad_portafolio(pesos, matriz_covarianza):
    """Calcula la volatilidad (riesgo) del portafolio."""
    return np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos)))


def sharpe_ratio_negativo_regularizado(pesos, rendimientos_medios, matriz_covarianza,
                                       libre_riesgo, gamma=0.1):
    """
    Calcula Sharpe Ratio negativo con regularización L2.

    Args:
        pesos: Array de pesos
        rendimientos_medios: Rendimientos medios
        matriz_covarianza: Matriz de covarianza
        libre_riesgo: Tasa libre de riesgo
        gamma: Parámetro de regularización L2

    Returns:
        float: Sharpe negativo regularizado
    """
    rendimiento = calcular_rendimiento_portafolio(pesos, rendimientos_medios)
    volatilidad = calcular_volatilidad_portafolio(pesos, matriz_covarianza)

    if volatilidad == 0:
        return 0

    ratio_sharpe_calc = (rendimiento - libre_riesgo) / volatilidad

    # Añadir penalización L2 para suavizar distribución
    penalizacion_l2 = gamma * np.sum(pesos ** 2)

    return -ratio_sharpe_calc + penalizacion_l2


def optimizar_portafolio(dataframe_returns, estrategia=ESTRATEGIA_OPTIMIZACION,
                         pesos_manuales=None, usar_shrinkage=True, gamma_reg=0.1):
    """
    Optimiza el portafolio con características avanzadas.

    Args:
        dataframe_returns: DataFrame con retornos
        estrategia: Tipo de optimización
        pesos_manuales: Dict con pesos fijos del usuario (opcional)
        usar_shrinkage: Si usar Ledoit-Wolf shrinkage
        gamma_reg: Parámetro de regularización L2

    Returns:
        dict: Diccionario con pesos óptimos y métricas
    """
    num_activos = len(dataframe_returns.columns)
    rendimientos_medios = dataframe_returns.mean() * ANNUAL_FACTOR

    # Usar matriz de covarianza robusta (Ledoit-Wolf)
    if usar_shrinkage:
        matriz_cov = estimar_matriz_covarianza_robusta(dataframe_returns)
    else:
        matriz_cov = dataframe_returns.cov().values * ANNUAL_FACTOR

    # Calcular pesos dinámicos
    max_weight, min_weight = calcular_pesos_dinamicos(num_activos)

    # Configurar restricciones y límites
    activos_libres = list(dataframe_returns.columns)
    pesos_fijos = {}

    if pesos_manuales:
        # Modo híbrido: algunos pesos fijos, otros optimizar
        pesos_fijos = {k: v for k, v in pesos_manuales.items() if v > 0}
        activos_libres = [col for col in dataframe_returns.columns
                          if col not in pesos_fijos]
        capital_restante = 1.0 - sum(pesos_fijos.values())
    else:
        capital_restante = 1.0

    # Restricción: suma de pesos libres = capital_restante
    def restriccion_suma(pesos_libres):
        if pesos_manuales:
            return np.sum(pesos_libres) - capital_restante
        return np.sum(pesos_libres) - 1.0

    restricciones = {'type': 'eq', 'fun': restriccion_suma}

    # Límites con control de concentración dinámico
    num_libres = len(activos_libres)
    if num_libres > 0:
        limites = tuple((min_weight, min(max_weight, capital_restante))
                        for _ in range(num_libres))
        pesos_iniciales = np.array([capital_restante/num_libres] * num_libres)

        # Mapear índices
        idx_map = {col: i for i, col in enumerate(dataframe_returns.columns)}
        idx_libres = [idx_map[col] for col in activos_libres]

        # Función objetivo que maneja pesos parciales
        def objetivo(pesos_libres):
            pesos_completos = np.zeros(num_activos)
            # Asignar pesos fijos
            for col, peso in pesos_fijos.items():
                pesos_completos[idx_map[col]] = peso
            # Asignar pesos optimizados
            for i, idx in enumerate(idx_libres):
                pesos_completos[idx] = pesos_libres[i]

            if estrategia == "Maximizar Sharpe Ratio":
                return sharpe_ratio_negativo_regularizado(
                    pesos_completos, rendimientos_medios.values,
                    matriz_cov, TASA_LIBRE_RIESGO, gamma_reg
                )
            else:
                return calcular_volatilidad_portafolio(pesos_completos, matriz_cov)

        # Optimización
        resultado = minimize(
            objetivo,
            pesos_iniciales,
            method='SLSQP',
            bounds=limites,
            constraints=restricciones,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        # Reconstruir pesos completos
        pesos_optimos = np.zeros(num_activos)
        for col, peso in pesos_fijos.items():
            pesos_optimos[idx_map[col]] = peso
        for i, idx in enumerate(idx_libres):
            pesos_optimos[idx] = resultado.x[i]

    else:
        # Todos los pesos son manuales
        pesos_optimos = np.array([pesos_manuales.get(col, 0)
                                  for col in dataframe_returns.columns])

    # Calcular métricas del portafolio óptimo
    rendimiento_opt = calcular_rendimiento_portafolio(
        pesos_optimos, rendimientos_medios.values)
    volatilidad_opt = calcular_volatilidad_portafolio(
        pesos_optimos, matriz_cov)
    sharpe_opt = (rendimiento_opt - TASA_LIBRE_RIESGO) / \
        volatilidad_opt if volatilidad_opt != 0 else 0

    # Calcular CVaR del portafolio
    retornos_portafolio = (dataframe_returns.values *
                           pesos_optimos).sum(axis=1)
    cvar_portafolio = calcular_cvar(
        retornos_portafolio) * np.sqrt(ANNUAL_FACTOR)

    return {
        'pesos': dict(zip(dataframe_returns.columns, pesos_optimos)),
        'rendimiento': rendimiento_opt,
        'volatilidad': volatilidad_opt,
        'sharpe': sharpe_opt,
        'cvar': cvar_portafolio,
        'max_weight': max_weight,
        'min_weight': min_weight
    }


def generar_frontera_eficiente(dataframe_returns, num_portafolios=5000,
                               usar_shrinkage=True):
    """Genera la frontera eficiente con matriz robusta."""
    num_activos = len(dataframe_returns.columns)
    rendimientos_medios = dataframe_returns.mean() * ANNUAL_FACTOR

    if usar_shrinkage:
        matriz_cov = estimar_matriz_covarianza_robusta(dataframe_returns)
    else:
        matriz_cov = dataframe_returns.cov().values * ANNUAL_FACTOR

    max_weight, min_weight = calcular_pesos_dinamicos(num_activos)

    resultados = {
        'rendimiento': [],
        'volatilidad': [],
        'sharpe': [],
        'pesos': []
    }

    for _ in range(num_portafolios):
        # Generar pesos aleatorios respetando límites
        pesos = np.random.uniform(min_weight, max_weight, num_activos)
        pesos /= np.sum(pesos)  # Normalizar

        rend = calcular_rendimiento_portafolio(
            pesos, rendimientos_medios.values)
        vol = calcular_volatilidad_portafolio(pesos, matriz_cov)
        ratio_sharpe_port = (rend - TASA_LIBRE_RIESGO) / vol if vol != 0 else 0

        resultados['rendimiento'].append(rend)
        resultados['volatilidad'].append(vol)
        resultados['sharpe'].append(ratio_sharpe_port)
        resultados['pesos'].append(pesos)

    return pd.DataFrame(resultados)


def calcular_metricas_portafolio(dataframe_returns, pesos):
    """Calcula métricas de riesgo para un portafolio con pesos dados."""
    pesos_array = np.array([pesos.get(col, 0)
                           for col in dataframe_returns.columns])
    retornos_portafolio = (dataframe_returns * pesos_array).sum(axis=1)
    metricas_resultado = calcular_metricas_riesgo(retornos_portafolio)
    return metricas_resultado


def crear_excel_rendimientos(dataframe_returns, adj_close_df, tickers_list,
                             pesos_optimos=None):
    """
    Crea un archivo Excel con análisis completo de rendimientos.

    Args:
        dataframe_returns: DataFrame con retornos
        adj_close_df: DataFrame con precios ajustados
        tickers_list: Lista de tickers
        pesos_optimos: Pesos del portafolio (opcional)

    Returns:
        BytesIO: Archivo Excel en memoria
    """
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Hoja 1: Retornos diarios
        returns_export = dataframe_returns.copy()
        returns_export.index = returns_export.index.strftime('%Y-%m-%d')
        returns_export.to_excel(writer, sheet_name='Retornos Diarios')

        # Hoja 2: Precios ajustados
        precios_export = adj_close_df.copy()
        precios_export.index = precios_export.index.strftime('%Y-%m-%d')
        precios_export.to_excel(writer, sheet_name='Precios')

        # Hoja 3: Retornos acumulados
        retornos_acum = (1 + dataframe_returns).cumprod() - 1
        retornos_acum.index = retornos_acum.index.strftime('%Y-%m-%d')
        retornos_acum.to_excel(writer, sheet_name='Retornos Acumulados')

        # Hoja 4: Estadísticas
        stats = pd.DataFrame({
            'Ticker': tickers_list,
            'Retorno Anual': [dataframe_returns[t].mean() * ANNUAL_FACTOR for t in tickers_list],
            'Volatilidad Anual': [dataframe_returns[t].std() * np.sqrt(ANNUAL_FACTOR)
                                  for t in tickers_list],
            'Sharpe Ratio': [(dataframe_returns[t].mean() * ANNUAL_FACTOR - TASA_LIBRE_RIESGO) /
                             (dataframe_returns[t].std()
                              * np.sqrt(ANNUAL_FACTOR))
                             for t in tickers_list],
            'Max Drawdown': [(((1 + dataframe_returns[t]).cumprod() /
                              (1 + dataframe_returns[t]).cumprod().expanding().max()) - 1).min()
                             for t in tickers_list]
        })
        stats.to_excel(writer, sheet_name='Estadísticas', index=False)

        # Hoja 5: Portafolio (si hay pesos)
        if pesos_optimos:
            portafolio_df = pd.DataFrame({
                'Ticker': list(pesos_optimos.keys()),
                'Peso': list(pesos_optimos.values())
            })
            portafolio_df['Peso %'] = portafolio_df['Peso'] * 100

            # Calcular retornos del portafolio
            pesos_array = np.array([pesos_optimos.get(col, 0)
                                   for col in dataframe_returns.columns])
            retornos_port = (dataframe_returns * pesos_array).sum(axis=1)
            retornos_port_df = pd.DataFrame({
                'Fecha': retornos_port.index.strftime('%Y-%m-%d'),
                'Retorno Portafolio': retornos_port.values,
                'Retorno Acumulado': ((1 + retornos_port).cumprod() - 1).values
            })

            portafolio_df.to_excel(
                writer, sheet_name='Portafolio', index=False)
            retornos_port_df.to_excel(writer, sheet_name='Retornos Portafolio',
                                      index=False, startrow=len(portafolio_df) + 3)

    output.seek(0)
    return output


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Configuración del Análisis")

tickers_input = st.sidebar.text_input(
    "🔍 Tickers a analizar",
    "AAPL,MSFT,GOOGL,AMZN,TSLA",
    help="Separa múltiples tickers con comas"
)

benchmark_ticker = st.sidebar.text_input(
    "📊 Benchmark (opcional)",
    "SPY",
    help="Ticker del índice de referencia"
)

# Selector de fechas personalizado
st.sidebar.subheader("📅 Período de Análisis")
usar_fechas_custom = st.sidebar.checkbox(
    "Usar fechas personalizadas", value=False)

if usar_fechas_custom:
    col_fecha1, col_fecha2 = st.sidebar.columns(2)
    with col_fecha1:
        fecha_inicio = st.date_input(
            "Fecha inicio",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now()
        )
    with col_fecha2:
        fecha_fin = st.date_input(
            "Fecha fin",
            value=datetime.now(),
            max_value=datetime.now()
        )
    periodo = None
else:
    periodo = st.sidebar.selectbox(
        "Período predefinido",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3
    )
    fecha_inicio = None
    fecha_fin = None

capital_inversion = st.sidebar.number_input(
    "💰 Capital de Inversión (USD)",
    min_value=100,
    max_value=10000000,
    value=10000,
    step=100
)

st.sidebar.subheader("🎯 Estrategia de Optimización")
estrategia_opt = st.sidebar.radio(
    "Tipo de optimización",
    ["Maximizar Sharpe Ratio", "Minimizar Riesgo", "Pesos Personalizados"]
)

# Configuración avanzada
with st.sidebar.expander("⚙️ Configuración Avanzada"):
    usar_shrinkage = st.checkbox(
        "Usar Ledoit-Wolf Shrinkage",
        value=True,
        help="Estimación robusta de covarianza para reducir ruido estadístico"
    )

    gamma_regularizacion = st.slider(
        "Regularización L2 (γ)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Penalización para suavizar distribución de pesos"
    )

analizar = st.sidebar.button("🚀 Analizar Activos", type="primary")

# ==========================================
# PROCESAMIENTO
# ==========================================
if analizar:
    tickers_list = [t.strip().upper() for t in tickers_input.split(",")]

    with st.spinner("📡 Descargando datos..."):
        try:
            all_tickers = tickers_list.copy()
            if benchmark_ticker and benchmark_ticker.strip():
                all_tickers.append(benchmark_ticker.strip().upper())

            # Descargar según el modo seleccionado
            if usar_fechas_custom and fecha_inicio and fecha_fin:
                data = yf.download(
                    all_tickers,
                    start=fecha_inicio,
                    end=fecha_fin,
                    progress=False,
                    auto_adjust=False
                )
            else:
                data = yf.download(
                    all_tickers,
                    period=periodo,
                    progress=False,
                    auto_adjust=False
                )

            if data.empty:
                st.error("❌ No se pudieron descargar los datos.")
                st.stop()

            st.session_state.datos_descargados = data
            st.session_state.tickers_analizados = tickers_list
            st.session_state.fecha_descarga = datetime.now()

            st.success("✅ Datos descargados correctamente!")

        except (ValueError, KeyError, ConnectionError) as error:
            st.error(f"❌ Error: {str(error)}")
            st.stop()

if st.session_state.datos_descargados is not None:
    data = st.session_state.datos_descargados
    tickers_list = st.session_state.tickers_analizados

    if len(tickers_list) == 1:
        adj_close_df = pd.DataFrame({tickers_list[0]: data['Adj Close']})
        close_df = pd.DataFrame({tickers_list[0]: data['Close']})
        volume_df = pd.DataFrame({tickers_list[0]: data['Volume']})
        high_df = pd.DataFrame({tickers_list[0]: data['High']})
        low_df = pd.DataFrame({tickers_list[0]: data['Low']})
        open_df = pd.DataFrame({tickers_list[0]: data['Open']})
    else:
        adj_close_df = data['Adj Close'][tickers_list]
        close_df = data['Close'][tickers_list]
        volume_df = data['Volume'][tickers_list]
        high_df = data['High'][tickers_list]
        low_df = data['Low'][tickers_list]
        open_df = data['Open'][tickers_list]

    benchmark_data_local = None
    if benchmark_ticker and benchmark_ticker.strip():
        bench_tick = benchmark_ticker.strip().upper()
        if bench_tick in data['Adj Close'].columns:
            benchmark_data_local = data['Adj Close'][bench_tick]

    # ==========================================
    # OPTIMIZACIÓN DE PORTAFOLIO
    # ==========================================
    if len(tickers_list) >= 2:
        st.header("💼 1. Optimización Avanzada de Portafolio")

        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.info("🔬 **Ledoit-Wolf Shrinkage**: Matriz de covarianza robusta")
        with info_col2:
            st.info("📊 **Control de Concentración**: Límites dinámicos automáticos")
        with info_col3:
            st.info("🎯 **Regularización L2**: Distribución suavizada de pesos")

        returns_df = adj_close_df.pct_change().dropna()

        # Manejo de pesos personalizados
        pesos_manuales = None
        if estrategia_opt == "Pesos Personalizados":
            st.subheader("⚖️ Define tus Pesos Personalizados")
            st.markdown("""
            **Instrucciones:** 
            - Define los pesos que desees (en %).
            - Los pesos deben sumar 100%.
            - Deja en 0 los activos que quieras que el sistema optimice automáticamente.
            """)

            cols_pesos = st.columns(min(len(tickers_list), 4))
            pesos_input = {}

            for i, ticker in enumerate(tickers_list):
                with cols_pesos[i % len(cols_pesos)]:
                    peso = st.number_input(
                        f"{ticker} (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=1.0,
                        key=f"peso_{ticker}"
                    )
                    pesos_input[ticker] = peso / 100.0

            suma_pesos = sum(pesos_input.values())

            if suma_pesos > 0:
                st.metric("Suma de pesos", f"{suma_pesos*100:.1f}%")

                if suma_pesos <= 1.0:
                    pesos_manuales = {k: v for k,
                                      v in pesos_input.items() if v > 0}
                    if suma_pesos < 1.0:
                        st.info(
                            f"💡 El {(1-suma_pesos)*100:.1f}% restante será optimizado automáticamente")
                else:
                    st.error("❌ La suma de pesos no puede exceder 100%")
                    st.stop()

        with st.spinner("🔄 Optimizando con algoritmos avanzados..."):
            if estrategia_opt == "Pesos Personalizados" and pesos_manuales:
                resultado_opt = optimizar_portafolio(
                    returns_df,
                    "Maximizar Sharpe Ratio",
                    pesos_manuales=pesos_manuales,
                    usar_shrinkage=usar_shrinkage,
                    gamma_reg=gamma_regularizacion
                )
            else:
                resultado_opt = optimizar_portafolio(
                    returns_df,
                    estrategia_opt,
                    usar_shrinkage=usar_shrinkage,
                    gamma_reg=gamma_regularizacion
                )

        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📈 Rendimiento Esperado",
                f"{resultado_opt['rendimiento']*100:.2f}%",
                help="Rendimiento anualizado esperado"
            )

        with col2:
            st.metric(
                "📊 Volatilidad",
                f"{resultado_opt['volatilidad']*100:.2f}%",
                help="Desviación estándar anualizada"
            )

        with col3:
            sharpe_value = resultado_opt['sharpe']
            emoji_sharpe = "🟢" if sharpe_value > 1 else "🟡" if sharpe_value > 0.5 else "🔴"
            st.metric(
                "⚡ Sharpe Ratio",
                f"{sharpe_value:.3f} {emoji_sharpe}",
                help="Rendimiento ajustado por riesgo"
            )

        with col4:
            st.metric(
                "🔻 CVaR (95%)",
                f"{resultado_opt['cvar']*100:.2f}%",
                help="Pérdida esperada en el 5% peor de los casos",
                delta_color="inverse"
            )

        # Info de límites de concentración
        st.info(
            f"📏 **Límites Dinámicos Aplicados**: "
            f"Peso Mínimo = {resultado_opt['min_weight']*100:.1f}%, "
            f"Peso Máximo = {resultado_opt['max_weight']*100:.1f}% "
            f"(calculados automáticamente según {len(tickers_list)} activos)"
        )

        st.markdown("---")

        # Asignación de pesos
        st.subheader("💼 Asignación Óptima del Portafolio")

        col_pesos1, col_pesos2 = st.columns([3, 2])

        with col_pesos1:
            pesos_df = pd.DataFrame({
                'Ticker': list(resultado_opt['pesos'].keys()),
                'Peso (%)': [v*100 for v in resultado_opt['pesos'].values()],
                'Inversión (USD)': [v*capital_inversion for v in resultado_opt['pesos'].values()]
            })
            pesos_df = pesos_df.sort_values('Peso (%)', ascending=False)

            # Marcar pesos fijos si los hay
            if pesos_manuales:
                pesos_df['Tipo'] = pesos_df['Ticker'].apply(
                    lambda x: '🔒 Fijo' if x in pesos_manuales else '🔄 Optimizado'
                )

            fig_pie = go.Figure(data=[go.Pie(
                labels=pesos_df['Ticker'],
                values=pesos_df['Peso (%)'],
                hole=0.4,
                textinfo='label+percent',
                textposition='auto',
                marker={'line': {'color': 'white', 'width': 2}}
            )])
            fig_pie.update_layout(
                title="Distribución del Portafolio Optimizado",
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_pesos2:
            st.markdown("#### 📋 Detalles de Inversión:")
            display_cols = ['Ticker', 'Peso (%)', 'Inversión (USD)']
            if 'Tipo' in pesos_df.columns:
                display_cols.append('Tipo')

            st.dataframe(
                pesos_df[display_cols].style.format({
                    'Peso (%)': '{:.2f}%',
                    'Inversión (USD)': '${:,.2f}'
                }),
                use_container_width=True,
                height=300
            )

            total_invertido = pesos_df['Inversión (USD)'].sum()
            st.markdown(f"**💰 Capital Total:** ${total_invertido:,.2f}")

            retorno_anual_usd = capital_inversion * \
                resultado_opt['rendimiento']
            st.markdown(
                f"**📈 Retorno Esperado (anual):** ${retorno_anual_usd:,.2f}")

            # CVaR en dólares
            cvar_usd = capital_inversion * abs(resultado_opt['cvar'])
            st.markdown(f"**🔻 CVaR en USD:** ${cvar_usd:,.2f}")
            st.caption("Pérdida esperada en el peor 5% de escenarios")

        st.markdown("---")

        # Frontera Eficiente
        st.subheader("📊 Frontera Eficiente con Estimación Robusta")

        with st.spinner("🔄 Generando frontera eficiente..."):
            frontera_df = generar_frontera_eficiente(
                returns_df, 3000, usar_shrinkage=usar_shrinkage
            )

        fig_frontera = go.Figure()

        # Scatter de todos los portafolios
        fig_frontera.add_trace(go.Scatter(
            x=frontera_df['volatilidad'] * 100,
            y=frontera_df['rendimiento'] * 100,
            mode='markers',
            marker={
                'size': 5,
                'color': frontera_df['sharpe'],
                'colorscale': 'Viridis',
                'showscale': True,
                'colorbar': {'title': "Sharpe<br>Ratio"}
            },
            name='Portafolios Simulados',
            text=[f"Sharpe: {s:.2f}" for s in frontera_df['sharpe']],
            hovertemplate='<b>Retorno:</b> %{y:.2f}%<br>' +
                          '<b>Riesgo:</b> %{x:.2f}%<br>%{text}<extra></extra>'
        ))

        # Marcar portafolio óptimo
        fig_frontera.add_trace(go.Scatter(
            x=[resultado_opt['volatilidad'] * 100],
            y=[resultado_opt['rendimiento'] * 100],
            mode='markers',
            marker={
                'size': 20,
                'color': 'red',
                'symbol': 'star',
                'line': {'color': 'white', 'width': 2}
            },
            name='Portafolio Óptimo',
            hovertemplate=f'<b>Óptimo</b><br>Retorno: {resultado_opt["rendimiento"]*100:.2f}%<br>' +
                          f'Riesgo: {resultado_opt["volatilidad"]*100:.2f}%<br>' +
                          f'Sharpe: {resultado_opt["sharpe"]:.3f}<extra></extra>'
        ))

        fig_frontera.update_layout(
            title="Frontera Eficiente (con Ledoit-Wolf Shrinkage y Límites Dinámicos)",
            xaxis_title="Volatilidad (Riesgo) %",
            yaxis_title="Rendimiento Esperado %",
            template="plotly_dark",
            height=500,
            hovermode='closest'
        )

        st.plotly_chart(fig_frontera, use_container_width=True)

        st.success(
            "✅ **Optimización Robusta Completada**: El portafolio ha sido optimizado usando "
            "estimación Ledoit-Wolf, control de concentración dinámico y regularización L2."
        )

        st.markdown("---")

    # ==========================================
    # RESUMEN EJECUTIVO
    # ==========================================
    st.header("📊 2. Resumen Ejecutivo con CVaR")
    returns_df = adj_close_df.pct_change().dropna()

    if len(tickers_list) >= 2:
        st.subheader("💼 Métricas del Portafolio Optimizado")
        metricas_port = calcular_metricas_portafolio(
            returns_df, resultado_opt['pesos'])

        col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)

        with col_p1:
            st.metric("Rendimiento Anual",
                      f"{metricas_port['rendimiento_anual']*100:.2f}%")
        with col_p2:
            st.metric("Volatilidad Anual",
                      f"{metricas_port['volatilidad_anual']*100:.2f}%")
        with col_p3:
            sharpe_port = metricas_port['sharpe_ratio']
            emoji_sp = "🟢" if sharpe_port > 1 else "🟡" if sharpe_port > 0.5 else "🔴"
            st.metric("Sharpe Ratio", f"{sharpe_port:.3f} {emoji_sp}")
        with col_p4:
            st.metric("Max Drawdown", f"{metricas_port['max_drawdown']*100:.2f}%",
                      delta_color="inverse")
        with col_p5:
            st.metric("CVaR (95%)", f"{metricas_port['cvar_95']*100:.2f}%",
                      help="Pérdida esperada en el peor 5% de escenarios",
                      delta_color="inverse")

        st.markdown("---")

    # Métricas individuales
    st.subheader("📈 Métricas Individuales por Activo (con CVaR)")
    metricas_summary = []

    for ticker in tickers_list:
        try:
            ret_ticker = returns_df[ticker].dropna()
            metricas_calc = calcular_metricas_riesgo(ret_ticker)

            beta_str = "N/A"
            r2_str = "N/A"
            if benchmark_data_local is not None:
                benchmark_returns = benchmark_data_local.pct_change().dropna()
                beta_calc, r_squared_calc = calcular_beta(
                    ret_ticker, benchmark_returns)
                if beta_calc is not None:
                    beta_str = f"{beta_calc:.3f}"
                    r2_str = f"{r_squared_calc:.3f}"

            metricas_summary.append({
                'Ticker': ticker,
                'Rendimiento Anual': f"{metricas_calc['rendimiento_anual']*100:.2f}%",
                'Volatilidad': f"{metricas_calc['volatilidad_anual']*100:.2f}%",
                'Sharpe Ratio': f"{metricas_calc['sharpe_ratio']:.3f}",
                'Sortino Ratio': f"{metricas_calc['sortino_ratio']:.3f}",
                'Max Drawdown': f"{metricas_calc['max_drawdown']*100:.2f}%",
                'CVaR (95%)': f"{metricas_calc['cvar_95']*100:.2f}%",
                'Beta': beta_str,
                'R²': r2_str
            })
        except (KeyError, ValueError):
            continue

    if metricas_summary:
        df_metricas = pd.DataFrame(metricas_summary)
        st.dataframe(df_metricas, use_container_width=True, height=400)

        # Botón de descarga en Excel
        st.markdown("### 📥 Descargar Análisis Completo")

        if len(tickers_list) >= 2 and 'resultado_opt' in locals():
            excel_data = crear_excel_rendimientos(
                returns_df, adj_close_df, tickers_list, resultado_opt['pesos']
            )
        else:
            excel_data = crear_excel_rendimientos(
                returns_df, adj_close_df, tickers_list
            )

        st.download_button(
            label="📊 Descargar Excel con Rendimientos",
            data=excel_data,
            file_name=f"analisis_financiero_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Gráficos comparativos
        col_comp1, col_comp2 = st.columns(2)

        with col_comp1:
            sharpe_vals = [float(m['Sharpe Ratio']) for m in metricas_summary]
            tickers_sharpe = [m['Ticker'] for m in metricas_summary]

            fig_sharpe = go.Figure(data=[go.Bar(
                x=tickers_sharpe,
                y=sharpe_vals,
                text=[f"{v:.2f}" for v in sharpe_vals],
                textposition='auto',
                marker={'color': sharpe_vals,
                        'colorscale': 'RdYlGn', 'showscale': False}
            )])
            fig_sharpe.update_layout(
                title="Comparación Sharpe Ratio",
                xaxis_title="Ticker",
                yaxis_title="Sharpe",
                template="plotly_dark",
                height=350
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)

        with col_comp2:
            rend_vals = [float(m['Rendimiento Anual'].replace('%', ''))
                         for m in metricas_summary]
            vol_vals = [float(m['Volatilidad'].replace('%', ''))
                        for m in metricas_summary]

            fig_scatter = go.Figure(data=[go.Scatter(
                x=vol_vals,
                y=rend_vals,
                mode='markers+text',
                text=tickers_sharpe,
                textposition='top center',
                marker={
                    'size': 15,
                    'color': sharpe_vals,
                    'colorscale': 'RdYlGn',
                    'showscale': True,
                    'colorbar': {'title': "Sharpe"}
                }
            )])
            fig_scatter.update_layout(
                title="Riesgo vs Retorno",
                xaxis_title="Volatilidad %",
                yaxis_title="Rendimiento %",
                template="plotly_dark",
                height=350
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # ANÁLISIS DE PRECIOS
    # ==========================================
    st.header("📈 3. Análisis de Precios y Rendimientos")

    ticker_seleccionado = st.selectbox("Selecciona un activo:", tickers_list)

    precio_actual = adj_close_df[ticker_seleccionado].iloc[-1]
    precio_inicial = adj_close_df[ticker_seleccionado].iloc[0]
    cambio_total = ((precio_actual - precio_inicial) / precio_inicial) * 100

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.metric("Precio Actual", f"${precio_actual:.2f}")
    with col_m2:
        delta_color = "normal" if cambio_total >= 0 else "inverse"
        st.metric("Cambio Total", f"{cambio_total:.2f}%",
                  delta=f"{cambio_total:.2f}%", delta_color=delta_color)
    with col_m3:
        st.metric("Máximo", f"${adj_close_df[ticker_seleccionado].max():.2f}")
    with col_m4:
        st.metric("Mínimo", f"${adj_close_df[ticker_seleccionado].min():.2f}")

    st.subheader("📊 Gráfico de Velas Japonesas")

    fig_candle = go.Figure(data=[go.Candlestick(
        x=close_df.index,
        open=open_df[ticker_seleccionado],
        high=high_df[ticker_seleccionado],
        low=low_df[ticker_seleccionado],
        close=close_df[ticker_seleccionado],
        name=ticker_seleccionado
    )])

    sma_20 = close_df[ticker_seleccionado].rolling(window=20).mean()
    sma_50 = close_df[ticker_seleccionado].rolling(window=50).mean()

    fig_candle.add_trace(go.Scatter(
        x=close_df.index, y=sma_20, mode='lines', name='SMA 20',
        line={'color': 'orange', 'width': 1}
    ))
    fig_candle.add_trace(go.Scatter(
        x=close_df.index, y=sma_50, mode='lines', name='SMA 50',
        line={'color': 'blue', 'width': 1}
    ))

    fig_candle.update_layout(
        title=f"{ticker_seleccionado} - Precio con Medias Móviles",
        yaxis_title="Precio USD",
        xaxis_title="Fecha",
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    st.subheader("📊 Análisis de Volumen")

    fig_volume = go.Figure()
    colors_vol = ['red' if close_df[ticker_seleccionado].iloc[i] <
                  open_df[ticker_seleccionado].iloc[i]
                  else 'green' for i in range(len(close_df))]

    fig_volume.add_trace(go.Bar(
        x=volume_df.index,
        y=volume_df[ticker_seleccionado],
        marker_color=colors_vol
    ))
    fig_volume.update_layout(
        title=f"{ticker_seleccionado} - Volumen de Operaciones",
        yaxis_title="Volumen",
        template="plotly_dark",
        height=300
    )
    st.plotly_chart(fig_volume, use_container_width=True)

    if len(tickers_list) > 1:
        st.subheader("📈 Comparación de Rendimientos Acumulados")
        normalized_prices = (adj_close_df / adj_close_df.iloc[0]) * 100

        fig_comp = go.Figure()
        for ticker_comp in tickers_list:
            fig_comp.add_trace(go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[ticker_comp],
                mode='lines',
                name=ticker_comp
            ))

        if benchmark_data_local is not None:
            normalized_benchmark = (
                benchmark_data_local / benchmark_data_local.iloc[0]) * 100
            fig_comp.add_trace(go.Scatter(
                x=normalized_benchmark.index,
                y=normalized_benchmark,
                mode='lines',
                name=f'{benchmark_ticker} (Benchmark)',
                line={'dash': 'dash', 'width': 2, 'color': 'gray'}
            ))

        fig_comp.update_layout(
            title="Rendimiento Relativo (Base 100)",
            yaxis_title="Valor Indexado",
            template="plotly_dark",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # ANÁLISIS TÉCNICO
    # ==========================================
    st.header("🔧 4. Análisis Técnico")

    col_tech1, col_tech2 = st.columns(2)

    with col_tech1:
        st.subheader("📊 RSI - Relative Strength Index")
        precios = close_df[ticker_seleccionado]
        delta = precios.diff()
        ganancia = delta.where(delta > 0, 0)
        perdida = -delta.where(delta < 0, 0)
        avg_gain = ganancia.rolling(window=14).mean()
        avg_loss = perdida.rolling(window=14).mean()
        rs_index = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs_index))

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=rsi.index, y=rsi, mode='lines', name='RSI',
            line={'color': 'purple', 'width': 2}
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",
                          annotation_text="Sobrecompra")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                          annotation_text="Sobreventa")
        fig_rsi.update_layout(
            title=f"{ticker_seleccionado} - RSI (14 períodos)",
            yaxis_title="RSI",
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

        rsi_actual = rsi.iloc[-1]
        if rsi_actual > 70:
            texto_interpretacion = "🔴 **Sobrecompra** - Posible sobrevaloración"
        elif rsi_actual < 30:
            texto_interpretacion = "🟢 **Sobreventa** - Posible subvaloración"
        else:
            texto_interpretacion = "🟡 **Neutral** - Rango normal"

        st.markdown(f"**RSI Actual:** {rsi_actual:.2f}")
        st.markdown(texto_interpretacion)

    with col_tech2:
        st.subheader("📊 Bandas de Bollinger")
        sma_bb = close_df[ticker_seleccionado].rolling(window=20).mean()
        std_bb = close_df[ticker_seleccionado].rolling(window=20).std()
        upper_band = sma_bb + (std_bb * 2)
        lower_band = sma_bb - (std_bb * 2)

        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(
            x=close_df.index, y=close_df[ticker_seleccionado],
            mode='lines', name='Precio', line={'color': 'blue', 'width': 1}
        ))
        fig_bb.add_trace(go.Scatter(
            x=close_df.index, y=upper_band, mode='lines',
            name='Banda Superior', line={'color': 'red', 'width': 1, 'dash': 'dash'}
        ))
        fig_bb.add_trace(go.Scatter(
            x=close_df.index, y=sma_bb, mode='lines',
            name='SMA 20', line={'color': 'orange', 'width': 1}
        ))
        fig_bb.add_trace(go.Scatter(
            x=close_df.index, y=lower_band, mode='lines',
            name='Banda Inferior',
            line={'color': 'green', 'width': 1, 'dash': 'dash'},
            fill='tonexty'
        ))
        fig_bb.update_layout(
            title=f"{ticker_seleccionado} - Bandas de Bollinger (20, 2)",
            yaxis_title="Precio",
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig_bb, use_container_width=True)

        precio_act = close_df[ticker_seleccionado].iloc[-1]
        banda_sup = upper_band.iloc[-1]
        banda_inf = lower_band.iloc[-1]

        if precio_act > banda_sup:
            texto_interp_bb = "🔴 **Sobre banda superior** - Posible sobrecompra"
        elif precio_act < banda_inf:
            texto_interp_bb = "🟢 **Bajo banda inferior** - Posible sobreventa"
        else:
            texto_interp_bb = "🟡 **Dentro de bandas** - Normal"

        st.markdown(texto_interp_bb)

    st.markdown("---")

    # ==========================================
    # CORRELACIÓN
    # ==========================================
    st.header("🔗 5. Matriz de Correlación")

    if len(tickers_list) > 1:
        returns_corr_df = adj_close_df.pct_change().dropna()
        corr_matrix = returns_corr_df.corr()

        col_corr1, col_corr2 = st.columns([2, 1])

        with col_corr1:
            fig_corr = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                aspect="auto",
                title="Correlación de Retornos",
                labels={'color': "Correlación"}
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)

        with col_corr2:
            st.markdown("### Interpretación:")
            st.markdown("""
            - **1.0**: Correlación perfecta positiva
            - **0.0**: Sin correlación
            - **-1.0**: Correlación perfecta negativa
            
            #### Diversificación:
            - < 0.3: Buena
            - 0.3-0.7: Moderada
            - > 0.7: Poca
            """)

            corr_flat = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            ).stack().sort_values()

            st.markdown("#### Menor Correlación:")
            min_pair = corr_flat.index[0]
            min_val = corr_flat.iloc[0]
            st.text(f"{min_pair[0]} - {min_pair[1]}: {min_val:.3f}")

            st.markdown("#### Mayor Correlación:")
            max_pair = corr_flat.index[-1]
            max_val = corr_flat.iloc[-1]
            st.text(f"{max_pair[0]} - {max_pair[1]}: {max_val:.3f}")
    else:
        st.info("ℹ️ Se necesitan 2+ activos para correlación.")

    st.markdown("---")

    # ==========================================
    # DATOS FUNDAMENTALES
    # ==========================================
    st.header("💼 6. Datos Fundamentales")

    if st.checkbox("📊 Mostrar Fundamentales"):
        with st.spinner("Obteniendo datos..."):
            fund_data = []

            for ticker_fund in tickers_list:
                try:
                    ticker_obj = yf.Ticker(ticker_fund)
                    info = ticker_obj.info

                    market_cap = info.get('marketCap')
                    cap_display = f"${market_cap/1e9:.2f}B" if market_cap else 'N/A'

                    fund_data.append({
                        "Ticker": ticker_fund,
                        "Nombre": info.get('longName', 'N/A'),
                        "Sector": info.get('sector', 'N/A'),
                        "Industria": info.get('industry', 'N/A'),
                        "Precio": f"${info.get('currentPrice', 0):.2f}",
                        "Market Cap": cap_display,
                        "P/E": round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else 'N/A',
                        "P/B": round(info.get('priceToBook', 0), 2) if info.get('priceToBook') else 'N/A',
                        "Div Yield %": (
                            round(info.get('dividendYield', 0) * 100, 2)
                            if info.get('dividendYield') else 'N/A'
                        ),
                        "Beta": round(info.get('beta', 0), 3) if info.get('beta') else 'N/A',
                        "EPS": round(info.get('trailingEps', 0), 2) if info.get('trailingEps') else 'N/A'
                    })
                except (ValueError, KeyError):
                    st.warning(f"⚠️ No se obtuvieron datos para {ticker_fund}")
                    continue

            if fund_data:
                df_fund = pd.DataFrame(fund_data)
                st.dataframe(df_fund, use_container_width=True)

                if len(fund_data) > 1:
                    st.subheader("📊 Comparación P/E Ratio")
                    pe_data = [(d['Ticker'], d['P/E']) for d in fund_data
                               if isinstance(d['P/E'], (int, float))]

                    if pe_data:
                        tickers_pe = [x[0] for x in pe_data]
                        valores_pe = [x[1] for x in pe_data]

                        fig_pe = go.Figure(data=[go.Bar(
                            x=tickers_pe,
                            y=valores_pe,
                            text=[f"{x[1]:.1f}" for x in pe_data],
                            textposition='auto'
                        )])
                        fig_pe.update_layout(
                            title="Comparación P/E Ratio",
                            xaxis_title="Ticker",
                            yaxis_title="P/E",
                            template="plotly_dark",
                            height=400
                        )
                        st.plotly_chart(fig_pe, use_container_width=True)
            else:
                st.warning("⚠️ No se obtuvieron datos fundamentales.")

else:
    st.info("👈 Configura los parámetros y presiona **'Analizar Activos'**")
    st.markdown("""
    ## 🎯 Características Profesionales:
    
    ### 💼 Optimización Robusta de Portafolio
    - **Ledoit-Wolf Shrinkage**: Estimación robusta de covarianza
    - **Control de Concentración Dinámico**: Límites calculados automáticamente
    - **Regularización L2**: Distribución suavizada de pesos
    - **Modo Híbrido**: Define pesos manuales y optimiza el resto
    - **CVaR (95%)**: Análisis de riesgo de cola
    
    ### 📅 Análisis Flexible
    - Fechas personalizadas o períodos predefinidos
    - Descarga de análisis completo en Excel
    - Múltiples hojas: precios, retornos, estadísticas, portafolio
    
    ### 📊 Análisis Completo
    - Métricas avanzadas: Volatilidad, VaR, CVaR
    - Ratios: Sharpe, Sortino
    - Beta y correlación con benchmarks
    
    ### 📈 Análisis Técnico
    - Gráficos de velas japonesas
    - RSI y Bandas de Bollinger
    - Medias móviles (SMA 20, SMA 50)
    
    ### 💼 Datos Fundamentales
    - P/E, P/B, Market Cap
    - Información sectorial
    - Dividendos y EPS
    """)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
texto_footer = """
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p><strong>Dashboard Financiero Pro Avanzado</strong> | Análisis Cuantitativo con Técnicas Institucionales</p>
    <p style='font-size: 0.8rem;'>
    Optimización robusta con Ledoit-Wolf Shrinkage, CVaR y control de concentración dinámico<br>
    Datos: Yahoo Finance | Solo información educativa, no recomendaciones de inversión
    </p>
</div>
"""
st.markdown(texto_footer, unsafe_allow_html=True)
