"""
Experimento.py — Dashboard Financiero Cuantitativo.

Interfaz Streamlit. Todo el cálculo financiero vive en core_quant.py
(funciones puras, verificadas con tests en test_core_quant.py).

Ejecutar:  streamlit run Experimento.py
"""
from datetime import datetime, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import core_quant as cq

VERSION = "2.0"
TICKERS_DEMO = "AAPL,MSFT,GOOGL,AMZN,TSLA"

ETIQUETA_ESTRATEGIA = {
    "Maximizar Sharpe (retorno ajustado por riesgo)": cq.EST_SHARPE,
    "Mínima Volatilidad (el portafolio más estable)": cq.EST_MINVOL,
    "Mínimo CVaR (proteger contra pérdidas extremas)": cq.EST_MINCVAR,
    "Pesos personalizados (híbrido)": "PERSONALIZADO",
}

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO
# ==========================================
st.set_page_config(
    page_title="Dashboard Financiero Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

# El tema oscuro vive en .streamlit/config.toml; aquí solo acentos puntuales.
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #00d9ff;
        text-align: center; padding: 0.5rem 0;
    }
    .stMetric {
        background-color: rgba(21, 26, 46, 0.6);
        padding: 1rem !important; border-radius: 12px !important;
        border: 1px solid rgba(0, 217, 255, 0.15) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="main-header">📈 Dashboard de Análisis Financiero Cuantitativo</h1>',
    unsafe_allow_html=True,
)


# ==========================================
# HELPERS DE FORMATO
# ==========================================
def fmt_pct(x, dec: int = 2) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:.{dec}f}%"


def fmt_usd(x) -> str:
    return "—" if x is None or pd.isna(x) else f"${x:,.2f}"


def fmt_num(x, dec: int = 3) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.{dec}f}"


def etiqueta_sharpe(s: float) -> str:
    if pd.isna(s):
        return "sin datos"
    if s > 1:
        return "🟢 Bueno"
    if s > 0.5:
        return "🟡 Aceptable"
    return "🔴 Bajo"


# ==========================================
# DESCARGA Y PREPARACIÓN DE DATOS (con caché)
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def descargar_datos(tickers: tuple, periodo: str | None,
                    inicio: str | None, fin: str | None) -> pd.DataFrame:
    kwargs = {"progress": False, "auto_adjust": False, "threads": True}
    if inicio and fin:
        data = yf.download(list(tickers), start=inicio, end=fin, **kwargs)
    else:
        data = yf.download(list(tickers), period=periodo, **kwargs)
    if data is None:
        return pd.DataFrame()
    if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    return data


def extraer_campo(data: pd.DataFrame, campo: str, tickers: list) -> pd.DataFrame:
    """Devuelve siempre un DataFrame ticker->columna, con o sin MultiIndex."""
    if isinstance(data.columns, pd.MultiIndex):
        df = data[campo]
        cols = [t for t in tickers if t in df.columns]
        return df[cols].copy()
    return data[[campo]].rename(columns={campo: tickers[0]})


@st.cache_data(ttl=86400, show_spinner=False)
def obtener_fundamentales(tickers: tuple) -> list:
    filas = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            filas.append({"Ticker": t, "Nombre": "Error al consultar"})
            continue

        market_cap = info.get("marketCap")
        if market_cap and market_cap >= 1e12:
            cap = f"${market_cap / 1e12:.2f}T"
        elif market_cap:
            cap = f"${market_cap / 1e9:.2f}B"
        else:
            cap = None

        # yfinance cambió las unidades de dividendYield entre versiones:
        # algunas devuelven 0.0045 y otras 0.45 para un yield de 0.45%.
        dy = info.get("dividendYield")
        if dy is not None:
            dy = dy if dy > 1 else dy * 100

        filas.append({
            "Ticker": t,
            "Nombre": info.get("longName"),
            "Sector": info.get("sector"),
            "Industria": info.get("industry"),
            "Precio": info.get("currentPrice"),
            "Market Cap": cap,
            "P/E": info.get("trailingPE"),
            "P/B": info.get("priceToBook"),
            "Div Yield %": dy,
            "Beta (yf)": info.get("beta"),
            "EPS": info.get("trailingEps"),
        })
    return filas


@st.cache_data(show_spinner=False)
def calcular_frontera(returns_df: pd.DataFrame, rf: float, shrinkage: bool,
                      exclusion: bool, mu: pd.Series) -> pd.DataFrame:
    return cq.frontera_eficiente(returns_df, rf, shrinkage, exclusion, mu=mu)


@st.cache_data(show_spinner=False)
def calcular_nube(returns_df: pd.DataFrame, rf: float, shrinkage: bool,
                  mu: pd.Series) -> pd.DataFrame:
    return cq.nube_portafolios(returns_df, rf, shrinkage, mu=mu)


@st.cache_data(show_spinner=False)
def comparar_estrategias(returns_df: pd.DataFrame, rf: float, shrinkage: bool,
                         gamma: float, exclusion: bool,
                         mu: pd.Series) -> pd.DataFrame:
    """Tabla de las 3 estrategias + 1/N: μ esperado forward y métricas
    históricas realizadas de cada mezcla sobre toda la muestra."""
    filas = []
    n = returns_df.shape[1]
    carteras = {}
    for est in cq.ESTRATEGIAS:
        res = cq.optimizar_portafolio(returns_df, est, rf, usar_shrinkage=shrinkage,
                                      gamma=gamma, permitir_exclusion=exclusion,
                                      mu=mu)
        carteras[est] = res.pesos
    carteras["Equiponderado (1/N)"] = {c: 1.0 / n for c in returns_df.columns}

    for nombre, pesos in carteras.items():
        m = cq.metricas_riesgo(cq.serie_portafolio(returns_df, pesos), rf)
        w = np.array([pesos.get(c, 0.0) for c in returns_df.columns])
        filas.append({
            "Estrategia": nombre,
            "μ esperado": float(w @ mu.reindex(returns_df.columns).values),
            "CAGR histórico": m["cagr"], "Volatilidad": m["vol"],
            "Sharpe (hist.)": m["sharpe"], "Sortino (hist.)": m["sortino"],
            "CVaR diario (95%)": m["cvar_d"], "Max Drawdown": m["mdd"],
        })
    return pd.DataFrame(filas)


@st.cache_data(show_spinner=False)
def calcular_simulacion(returns_port: pd.Series, capital: float,
                        horizonte: int, n_sims: int,
                        drift: float | None) -> dict:
    return cq.simular_bootstrap(returns_port, capital, horizonte, n_sims,
                                drift_anual=drift)


@st.cache_data(show_spinner=False)
def calcular_backtest(returns_df: pd.DataFrame, rf: float, shrinkage: bool,
                      gamma: float, exclusion: bool):
    return cq.backtest_train_test(returns_df, rf, usar_shrinkage=shrinkage,
                                  gamma=gamma, permitir_exclusion=exclusion)


@st.cache_data(show_spinner=False)
def crear_excel(returns_df: pd.DataFrame, precios_df: pd.DataFrame,
                pesos: dict | None, metadatos: dict) -> bytes:
    """Excel con hoja de metadatos para auditoría y reproducibilidad."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        meta_df = pd.DataFrame(
            {"Parámetro": list(metadatos.keys()), "Valor": list(metadatos.values())}
        )
        meta_df.to_excel(writer, sheet_name="Metadatos", index=False)

        for nombre, df in (("Retornos Diarios", returns_df),
                           ("Precios Ajustados", precios_df),
                           ("Retornos Acumulados", (1 + returns_df).cumprod() - 1)):
            export = df.copy()
            export.index = export.index.strftime("%Y-%m-%d")
            export.to_excel(writer, sheet_name=nombre)

        stats_filas = []
        for t in returns_df.columns:
            m = cq.metricas_riesgo(returns_df[t], metadatos["Tasa libre de riesgo"])
            stats_filas.append({
                "Ticker": t, "CAGR": m["cagr"], "Volatilidad Anual": m["vol"],
                "Sharpe": m["sharpe"], "Sortino": m["sortino"],
                "Max Drawdown": m["mdd"], "VaR diario 95%": m["var_d"],
                "CVaR diario 95%": m["cvar_d"], "Observaciones": m["n"],
            })
        pd.DataFrame(stats_filas).to_excel(writer, sheet_name="Estadísticas", index=False)

        if pesos:
            port_df = pd.DataFrame({
                "Ticker": list(pesos.keys()),
                "Peso": list(pesos.values()),
                "Peso %": [v * 100 for v in pesos.values()],
            })
            port_df.to_excel(writer, sheet_name="Portafolio", index=False)
            rp = cq.serie_portafolio(returns_df, pesos)
            rp_df = pd.DataFrame({
                "Fecha": rp.index.strftime("%Y-%m-%d"),
                "Retorno Diario": rp.values,
                "Retorno Acumulado": ((1 + rp).cumprod() - 1).values,
            })
            rp_df.to_excel(writer, sheet_name="Retornos Portafolio", index=False)
    return output.getvalue()


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Configuración del Análisis")

tickers_input = st.sidebar.text_input(
    "🔍 Tickers a analizar", TICKERS_DEMO,
    help="Separa múltiples tickers con comas. Ej: AAPL,MSFT,GOOGL",
)
benchmark_input = st.sidebar.text_input(
    "📊 Benchmark (opcional)", "SPY",
    help="Índice de referencia para beta, alfa y correlación en estrés.",
)

st.sidebar.subheader("📅 Período de Análisis")
usar_fechas_custom = st.sidebar.checkbox("Usar fechas personalizadas", value=False)
if usar_fechas_custom:
    col_f1, col_f2 = st.sidebar.columns(2)
    fecha_inicio = col_f1.date_input(
        "Inicio", value=datetime.now() - timedelta(days=730),
        max_value=datetime.now(),
    )
    fecha_fin = col_f2.date_input(
        "Fin", value=datetime.now(), max_value=datetime.now(),
    )
    periodo = None
else:
    periodo = st.sidebar.selectbox(
        "Período predefinido", ["6mo", "1y", "2y", "5y", "10y", "max"], index=2,
        help="Se recomienda 2y+ para que las métricas de riesgo sean confiables.",
    )
    fecha_inicio = fecha_fin = None

capital_inversion = st.sidebar.number_input(
    "💰 Capital de Inversión (USD)", min_value=100, max_value=100_000_000,
    value=10_000, step=100,
)
tasa_rf_pct = st.sidebar.number_input(
    "🏦 Tasa libre de riesgo anual (%)", min_value=0.0, max_value=15.0,
    value=4.0, step=0.25,
    help="Usa la T-bill de 3 meses o la tasa de referencia vigente.",
)
TASA_RF = tasa_rf_pct / 100.0

st.sidebar.subheader("📐 Retornos Esperados (μ)")
estimador_mu = st.sidebar.selectbox(
    "Estimador del rendimiento esperado",
    [cq.RE_CAPM, cq.RE_BAYES_STEIN, cq.RE_HISTORICO],
    index=0,
    help="El insumo más ruidoso de Markowitz son las medias (Merton 1980); "
         "el optimizador amplifica ese ruido (Michaud 1989). CAPM usa el "
         "equilibrio de mercado (prior de Black-Litterman); Bayes-Stein "
         "(Jorion 1986) contrae las medias históricas hacia su centro. El "
         "histórico puro solo describe el período observado.",
)
if estimador_mu == cq.RE_CAPM:
    erp_pct = st.sidebar.number_input(
        "Prima de riesgo de mercado — ERP (%)", min_value=1.0, max_value=10.0,
        value=5.0, step=0.25,
        help="Exceso esperado del mercado sobre la tasa libre de riesgo. "
             "La histórica de largo plazo ronda 4–6% (Damodaran).",
    )
    ERP = erp_pct / 100.0
else:
    ERP = 0.05

st.sidebar.subheader("🎯 Estrategia de Optimización")
estrategia_radio = st.sidebar.radio(
    "Tipo de optimización", list(ETIQUETA_ESTRATEGIA.keys()),
)
estrategia_core = ETIQUETA_ESTRATEGIA[estrategia_radio]

with st.sidebar.expander("⚙️ Configuración Avanzada"):
    usar_shrinkage = st.checkbox(
        "Ledoit-Wolf Shrinkage", value=True,
        help="Estimación robusta de covarianza: reduce el ruido estadístico "
             "de la matriz muestral.",
    )
    gamma_reg = st.slider(
        "Regularización L2 (γ)", 0.0, 1.0, 0.1, 0.05,
        help="Suaviza la distribución de pesos. Nota: con γ alto el portafolio "
             "mostrado ya NO es el de máximo Sharpe puro.",
    )
    permitir_exclusion = st.checkbox(
        "Permitir excluir activos (peso mínimo 0%)", value=False,
        help="Si se desactiva, todo activo recibe un peso mínimo automático.",
    )
    n_sims = st.select_slider("Simulaciones Monte Carlo",
                              options=[1000, 2000, 5000], value=2000)
    horizonte_sim = st.select_slider(
        "Horizonte de proyección (días hábiles)",
        options=[126, 252, 504], value=252,
        help="126 ≈ 6 meses, 252 ≈ 1 año, 504 ≈ 2 años.",
    )

analizar = st.sidebar.button("🚀 Analizar Activos", type="primary")


# ==========================================
# DESCARGA Y VALIDACIÓN
# ==========================================
def parsear_tickers(texto: str) -> list:
    """Limpia, deduplica y valida formato básico de los tickers."""
    crudos = [t.strip().upper() for t in texto.split(",") if t.strip()]
    unicos = list(dict.fromkeys(crudos))
    validos = [t for t in unicos
               if all(c.isalnum() or c in ".-^=" for c in t) and len(t) <= 12]
    return validos


if analizar:
    tickers_list = parsear_tickers(tickers_input)
    bench = benchmark_input.strip().upper() if benchmark_input.strip() else None

    if not tickers_list:
        st.error("❌ No se reconoció ningún ticker válido. Ejemplo: AAPL,MSFT,GOOGL")
        st.stop()
    if usar_fechas_custom and fecha_inicio >= fecha_fin:
        st.error("❌ La fecha de inicio debe ser anterior a la fecha de fin.")
        st.stop()

    todos = list(dict.fromkeys(tickers_list + ([bench] if bench else [])))

    with st.spinner("📡 Descargando datos de Yahoo Finance..."):
        try:
            data = descargar_datos(
                tuple(todos), periodo,
                str(fecha_inicio) if usar_fechas_custom else None,
                str(fecha_fin) if usar_fechas_custom else None,
            )
        except Exception as err:
            st.error(
                f"❌ Falló la descarga de datos: {err}\n\n"
                "Posibles causas: sin conexión, ticker inexistente o límite "
                "temporal de Yahoo Finance. Espera un minuto y reintenta."
            )
            st.stop()

    if data is None or data.empty:
        st.error("❌ Yahoo Finance no devolvió datos para esos tickers/período.")
        st.stop()

    # Validar ticker por ticker: columnas inexistentes o completamente vacías
    adj_check = extraer_campo(data, "Adj Close", todos)
    invalidos = [t for t in tickers_list
                 if t not in adj_check.columns or adj_check[t].dropna().empty]
    tickers_ok = [t for t in tickers_list if t not in invalidos]

    if invalidos:
        st.warning(f"⚠️ Sin datos para: **{', '.join(invalidos)}**. "
                   "Se excluyeron del análisis (verifica el símbolo).")
    if not tickers_ok:
        st.error("❌ Ningún ticker tiene datos. Verifica los símbolos.")
        st.stop()

    bench_ok = bench if (bench and bench in adj_check.columns
                         and not adj_check[bench].dropna().empty) else None
    if bench and not bench_ok:
        st.warning(f"⚠️ El benchmark **{bench}** no tiene datos; "
                   "se omiten beta y alfa.")

    st.session_state["analisis"] = {
        "data": data,
        "tickers": tickers_ok,
        "benchmark": bench_ok,
        "periodo": periodo if not usar_fechas_custom else
                   f"{fecha_inicio} → {fecha_fin}",
        "fecha_descarga": datetime.now(),
    }


# ==========================================
# CUERPO PRINCIPAL
# ==========================================
if "analisis" in st.session_state:
    A = st.session_state["analisis"]
    data, tickers_list, bench_tick = A["data"], A["tickers"], A["benchmark"]

    todos_descargados = tickers_list + ([bench_tick] if bench_tick else [])
    adj_close_df = extraer_campo(data, "Adj Close", todos_descargados)[tickers_list]
    close_df = extraer_campo(data, "Close", todos_descargados)[tickers_list]
    open_df = extraer_campo(data, "Open", todos_descargados)[tickers_list]
    high_df = extraer_campo(data, "High", todos_descargados)[tickers_list]
    low_df = extraer_campo(data, "Low", todos_descargados)[tickers_list]
    volume_df = extraer_campo(data, "Volume", todos_descargados)[tickers_list]

    bench_series = None
    bench_returns = None
    if bench_tick:
        bench_series = extraer_campo(data, "Adj Close", todos_descargados)[bench_tick]
        bench_returns = bench_series.pct_change().dropna()

    # Retornos: se calcula UNA vez. dropna() recorta a la historia común.
    returns_df = adj_close_df.pct_change().dropna()
    n_obs = len(returns_df)

    # --- Vector de retornos esperados según el estimador elegido ---
    estimador_efectivo = estimador_mu
    if estimador_mu == cq.RE_CAPM and bench_returns is None:
        estimador_efectivo = cq.RE_BAYES_STEIN
        st.warning("⚠️ CAPM requiere un benchmark con datos; se usó "
                   "**Bayes-Stein** como estimador de retornos esperados.")
    if estimador_efectivo == cq.RE_CAPM:
        mu_esperado = cq.retornos_esperados_capm(returns_df, bench_returns,
                                                 TASA_RF, ERP)
    elif estimador_efectivo == cq.RE_BAYES_STEIN:
        mu_esperado = cq.retornos_esperados_bayes_stein(returns_df)
    else:
        mu_esperado = cq.retornos_esperados(returns_df)
    mu_historico = cq.retornos_esperados(returns_df)

    # --- Sello de auditoría de los datos ---
    inicio_efectivo = returns_df.index[0].date() if n_obs else "—"
    fin_efectivo = returns_df.index[-1].date() if n_obs else "—"
    st.caption(
        f"📌 **Datos del análisis:** {', '.join(tickers_list)}"
        + (f" · Benchmark: {bench_tick}" if bench_tick else "")
        + f" · Período efectivo: {inicio_efectivo} → {fin_efectivo}"
        f" ({n_obs} días hábiles) · Descargado: "
        f"{A['fecha_descarga']:%Y-%m-%d %H:%M}"
    )

    # Avisar si la historia común recorta a algún activo (sesgo de muestra)
    primeras_fechas = adj_close_df.apply(lambda s: s.first_valid_index())
    if n_obs and (primeras_fechas.max() - primeras_fechas.min()).days > 30:
        recorte = primeras_fechas.idxmax()
        st.warning(
            f"⚠️ **{recorte}** tiene historia más corta (desde "
            f"{primeras_fechas.max().date()}). Todos los activos se "
            "recortaron a la historia común para poder compararlos."
        )
    if n_obs < cq.MIN_OBS_CONFIABLE:
        st.warning(
            f"⚠️ Solo hay **{n_obs} observaciones** (< {cq.MIN_OBS_CONFIABLE}). "
            "El CVaR, la optimización y el backtest pierden confiabilidad con "
            "muestras cortas — considera un período de 2 años o más."
        )

    with st.expander("📚 Glosario — entiende cada métrica antes de usarla"):
        st.markdown("""
| Métrica | Qué significa en la práctica |
|---|---|
| **μ esperado** | Rendimiento anual estimado **hacia adelante** según el estimador del sidebar. Es la cifra para decidir; el CAGR histórico solo describe el pasado. |
| **CAPM / Equilibrio** | E[r] = rf + β × prima de mercado (Sharpe 1964). Es el prior de Black-Litterman: lo que el activo "debería" rendir por su riesgo sistemático. |
| **Bayes-Stein** | Contrae las medias históricas hacia su centro común (Jorion 1986): reduce el error de estimación que Markowitz amplifica. |
| **CAGR** | Crecimiento anual compuesto real de tu dinero (geométrico, no promedio simple). |
| **Volatilidad** | Qué tanto "se sacude" el valor en un año típico. Mayor = más sobresaltos. |
| **Sharpe** | Retorno extra por unidad de riesgo total. >1 bueno, 0.5–1 aceptable, <0.5 bajo. |
| **Sortino** | Como Sharpe, pero solo castiga la volatilidad a la baja (la que duele). |
| **Max Drawdown** | La peor caída desde un máximo histórico. Lo máximo que habrías visto perder. |
| **VaR diario 95%** | En el 95% de los días no perderás más que esto. |
| **CVaR diario 95%** | Cuando ocurre ese 5% peor, esta es la pérdida promedio. Se reporta **diario**: anualizarlo con √252 es matemáticamente inválido. |
| **Beta / R²** | Sensibilidad al mercado y qué tan confiable es esa estimación (R² bajo = beta poco informativo). |
| **Alfa de Jensen** | Retorno por encima de lo que justifica el riesgo de mercado asumido. |
| **Ledoit-Wolf** | Técnica que estabiliza la matriz de covarianza cuando hay ruido estadístico. |
| **Frontera eficiente** | Para cada nivel de riesgo, el mejor retorno alcanzable. Cada punto es una optimización real. |
        """)

    # ------------------------------------------
    # PESOS PERSONALIZADOS (antes de optimizar)
    # ------------------------------------------
    pesos_manuales = None
    if estrategia_core == "PERSONALIZADO" and len(tickers_list) >= 2:
        st.subheader("⚖️ Define tus Pesos Personalizados")
        st.markdown(
            "Fija el porcentaje de los activos que quieras controlar y deja en "
            "**0** los que el sistema deba optimizar (el restante se asigna "
            "maximizando Sharpe)."
        )
        cols_pesos = st.columns(min(len(tickers_list), 4))
        pesos_input = {}
        for i, t in enumerate(tickers_list):
            with cols_pesos[i % len(cols_pesos)]:
                pesos_input[t] = st.number_input(
                    f"{t} (%)", 0.0, 100.0, 0.0, 1.0, key=f"peso_{t}"
                ) / 100.0
        suma = sum(pesos_input.values())
        if suma > 1.0 + 1e-9:
            st.error(f"❌ La suma de pesos ({suma:.0%}) no puede exceder 100%.")
            st.stop()
        if suma > 0:
            st.metric("Suma de pesos fijos", fmt_pct(suma, 1))
            pesos_manuales = {k: v for k, v in pesos_input.items() if v > 0}
            if suma < 1.0:
                st.info(f"💡 El {fmt_pct(1 - suma, 1)} restante se optimizará "
                        "automáticamente (máximo Sharpe).")

    # ------------------------------------------
    # OPTIMIZACIÓN
    # ------------------------------------------
    resultado_opt = None
    if len(tickers_list) >= 2:
        estrategia_efectiva = (cq.EST_SHARPE if estrategia_core == "PERSONALIZADO"
                               else estrategia_core)
        with st.spinner("🔄 Optimizando portafolio..."):
            try:
                resultado_opt = cq.optimizar_portafolio(
                    returns_df, estrategia_efectiva, TASA_RF,
                    pesos_fijos=pesos_manuales, usar_shrinkage=usar_shrinkage,
                    gamma=gamma_reg, permitir_exclusion=permitir_exclusion,
                    mu=mu_esperado,
                )
            except ValueError as err:
                st.error(f"❌ No se pudo optimizar: {err}")
    else:
        # Un solo activo: el "portafolio" es 100% ese activo
        resultado_opt = None

    pesos_port = (resultado_opt.pesos if resultado_opt
                  else {tickers_list[0]: 1.0})
    serie_port = cq.serie_portafolio(returns_df, pesos_port)

    # ------------------------------------------
    # PESTAÑAS
    # ------------------------------------------
    tabs = st.tabs([
        "💼 Portafolio", "🎲 Proyección", "✅ Validación", "📊 Activos",
        "📈 Técnico", "🔗 Correlación", "🏢 Fundamentales", "📥 Reporte",
    ])
    (tab_port, tab_sim, tab_val, tab_activos,
     tab_tecnico, tab_corr, tab_fund, tab_rep) = tabs

    # ==========================================
    # TAB 1 — PORTAFOLIO
    # ==========================================
    with tab_port:
        if resultado_opt is None:
            st.info("ℹ️ Agrega 2 o más activos para optimizar un portafolio. "
                    "Las demás pestañas funcionan con un solo activo.")
        else:
            m = resultado_opt.metricas

            if not resultado_opt.exito:
                st.error("⚠️ **El optimizador no convergió por completo.** Se "
                         "muestra la mejor solución encontrada; interpreta con "
                         "cautela o cambia parámetros.")
            for nota in resultado_opt.notas:
                st.warning(f"⚠️ {nota}")

            mu_port = resultado_opt.rendimiento_esperado
            sharpe_esp = ((mu_port - TASA_RF) / m["vol"]
                          if m["vol"] and m["vol"] > 0 else np.nan)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("🎯 Rendimiento esperado (μ)", fmt_pct(mu_port),
                      help=f"Estimador: {estimador_efectivo}. Es la cifra para "
                           "decisiones hacia adelante — no el rendimiento del "
                           "pasado.")
            c2.metric("📊 Volatilidad", fmt_pct(m["vol"]),
                      help="Desviación estándar anualizada de la mezcla.")
            c3.metric("⚡ Sharpe esperado", f"{fmt_num(sharpe_esp)}",
                      delta=etiqueta_sharpe(sharpe_esp),
                      help="(μ esperado − tasa libre de riesgo) / volatilidad.")
            c4.metric("🔻 CVaR diario (95%)", fmt_pct(m["cvar_d"]),
                      help="Pérdida promedio en el peor 5% de los días. "
                           "Horizonte diario — no se anualiza.")
            c5.metric("📉 Max Drawdown", fmt_pct(m["mdd"]),
                      help=f"Peor caída histórica desde un máximo. Duración "
                           f"máxima bajo el agua: {m['dd_dur']} días hábiles.")

            # Contraste explícito esperado vs histórico (anti-extrapolación)
            brecha = m["cagr"] - mu_port
            if estimador_efectivo != cq.RE_HISTORICO and brecha > 0.10:
                st.warning(
                    f"📜 **Contexto histórico:** esta mezcla rindió "
                    f"**{fmt_pct(m['cagr'])} anual** en el período analizado, "
                    f"muy por encima del esperado de equilibrio "
                    f"({fmt_pct(mu_port)}). Esa diferencia de "
                    f"{fmt_pct(brecha)} corresponde a un período excepcional "
                    "que no es prudente extrapolar (Michaud 1989: el "
                    "optimizador amplifica los errores de estimación de las "
                    "medias)."
                )
            elif estimador_efectivo != cq.RE_HISTORICO:
                st.info(f"📜 CAGR histórico de esta mezcla en el período: "
                        f"**{fmt_pct(m['cagr'])}** (vs. μ esperado "
                        f"{fmt_pct(mu_port)}).")
            else:
                st.warning(
                    "📜 Estás usando el **CAGR histórico puro** como retorno "
                    "esperado: el optimizador sobrepondera a los ganadores del "
                    "período y la cifra resultante no es extrapolable. Para "
                    "decisiones de capital usa CAPM o Bayes-Stein (sidebar → "
                    "Retornos Esperados)."
                )

            st.info(
                f"📏 **Límites aplicados:** peso mínimo "
                f"{fmt_pct(resultado_opt.min_w, 1)} · peso máximo "
                f"{fmt_pct(resultado_opt.max_w, 1)} "
                f"(automáticos para {len(tickers_list)} activos) · "
                f"Estrategia: **{resultado_opt.estrategia}**"
                + (" + pesos fijos manuales" if pesos_manuales else "")
            )

            # --- Asignación ---
            st.subheader("💼 Asignación del Portafolio")
            precios_ult = close_df.iloc[-1]
            pesos_df = pd.DataFrame({
                "Ticker": list(pesos_port.keys()),
                "Peso (%)": [v * 100 for v in pesos_port.values()],
                "Inversión (USD)": [v * capital_inversion for v in pesos_port.values()],
            })
            pesos_df["Precio"] = pesos_df["Ticker"].map(precios_ult)
            pesos_df["Acciones (enteras)"] = np.floor(
                pesos_df["Inversión (USD)"] / pesos_df["Precio"]
            ).astype(int)
            pesos_df["Inversión efectiva"] = (
                pesos_df["Acciones (enteras)"] * pesos_df["Precio"]
            )
            if pesos_manuales:
                pesos_df["Tipo"] = pesos_df["Ticker"].apply(
                    lambda x: "🔒 Fijo" if x in pesos_manuales else "🔄 Optimizado")
            pesos_df = pesos_df.sort_values("Peso (%)", ascending=False)

            col_g, col_t = st.columns([2, 3])
            with col_g:
                visibles = pesos_df[pesos_df["Peso (%)"] > 0.01]
                fig_pie = go.Figure(go.Pie(
                    labels=visibles["Ticker"], values=visibles["Peso (%)"],
                    hole=0.45, textinfo="label+percent",
                    marker={"line": {"color": "#0a0e27", "width": 2}},
                ))
                fig_pie.update_layout(template="plotly_dark", height=380,
                                      title="Distribución del capital",
                                      margin=dict(t=50, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_t:
                st.dataframe(
                    pesos_df.style.format({
                        "Peso (%)": "{:.2f}%", "Inversión (USD)": "${:,.2f}",
                        "Precio": "${:,.2f}", "Inversión efectiva": "${:,.2f}",
                    }),
                    use_container_width=True, height=320, hide_index=True,
                )
                st.caption(
                    "Las acciones enteras se calculan con el último precio de "
                    "cierre; si tu bróker permite fracciones, usa la columna "
                    "Inversión (USD)."
                )

            # --- Interpretación en palabras ---
            top_ticker = max(pesos_port, key=pesos_port.get)
            cvar_usd = abs(m["cvar_d"]) * capital_inversion
            mdd_usd = abs(m["mdd"]) * capital_inversion
            st.markdown(
                f"""
**🗣️ En palabras simples:** con **{fmt_usd(capital_inversion)}**, la mayor
posición es **{top_ticker}** ({fmt_pct(pesos_port[top_ticker], 1)}). El
rendimiento esperado de esta mezcla es **{fmt_pct(mu_port)} anual**
(estimador {estimador_efectivo}); en el período analizado rindió
{fmt_pct(m['cagr'])}, pero esa cifra describe el pasado. En un día realmente
malo (el peor 5%), la pérdida promedio rondaría **{fmt_usd(cvar_usd)}**; y la
peor racha histórica habría significado ver tu cuenta
**{fmt_usd(mdd_usd)}** abajo de su máximo durante hasta
**{m['dd_dur']} días hábiles** antes de recuperarse.
                """
            )
            st.caption(
                "Los estimados forward dependen de los supuestos del sidebar "
                "(estimador de μ, ERP, tasa libre de riesgo); todos quedan "
                "documentados en la hoja de Metadatos del reporte."
            )

            st.markdown("---")

            # --- Contribución al riesgo ---
            st.subheader("🧯 ¿De dónde viene el riesgo?")
            st.caption(
                "Compara cuánto capital pones en cada activo vs. cuánto riesgo "
                "aporta realmente. Un activo puede ser 20% del capital pero 40% "
                "del riesgo."
            )
            cov_anual, _ = cq.matriz_covarianza(returns_df, usar_shrinkage)
            rc = cq.contribucion_riesgo(resultado_opt.w, cov_anual)
            fig_rc = go.Figure()
            fig_rc.add_bar(name="Peso (capital)", x=tickers_list,
                           y=[pesos_port[t] * 100 for t in tickers_list],
                           marker_color="#00d9ff")
            fig_rc.add_bar(name="Contribución al riesgo", x=tickers_list,
                           y=rc * 100, marker_color="#a855f7")
            fig_rc.update_layout(template="plotly_dark", barmode="group",
                                 height=350, yaxis_title="%",
                                 legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_rc, use_container_width=True)

            st.markdown("---")

            # --- Comparación de estrategias ---
            st.subheader("⚖️ Comparación de Estrategias (misma muestra)")
            st.caption(
                "El equiponderado 1/N es la referencia honesta: si una "
                "estrategia 'óptima' no lo supera claramente, la optimización "
                "está ajustando ruido."
            )
            tabla_comp = comparar_estrategias(returns_df, TASA_RF, usar_shrinkage,
                                              gamma_reg, permitir_exclusion,
                                              mu_esperado)
            st.dataframe(
                tabla_comp.style.format({
                    "μ esperado": "{:.2%}", "CAGR histórico": "{:.2%}",
                    "Volatilidad": "{:.2%}", "Sharpe (hist.)": "{:.3f}",
                    "Sortino (hist.)": "{:.3f}", "CVaR diario (95%)": "{:.2%}",
                    "Max Drawdown": "{:.2%}",
                }, na_rep="—"),
                use_container_width=True, hide_index=True,
            )

            st.markdown("---")

            # --- Frontera eficiente ---
            st.subheader("📊 Frontera Eficiente")
            st.caption(
                "La línea cian es la frontera real (cada punto resuelve una "
                "optimización con tus mismos límites de concentración). Los "
                "puntos grises son portafolios aleatorios de referencia."
            )
            with st.spinner("Calculando frontera eficiente..."):
                frontera = calcular_frontera(returns_df, TASA_RF, usar_shrinkage,
                                             permitir_exclusion, mu_esperado)
                nube = calcular_nube(returns_df, TASA_RF, usar_shrinkage,
                                     mu_esperado)

            mu_act = mu_esperado.reindex(tickers_list)
            vol_act = returns_df.std() * np.sqrt(cq.PERIODOS_ANUALES)
            w_eq = {c: 1 / len(tickers_list) for c in tickers_list}
            m_eq = cq.metricas_riesgo(cq.serie_portafolio(returns_df, w_eq), TASA_RF)
            mu_eq = float(np.mean(mu_act.values))

            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(
                x=nube["volatilidad"] * 100, y=nube["rendimiento"] * 100,
                mode="markers", name="Aleatorios (referencia)",
                marker={"size": 4, "color": "rgba(150,150,150,0.25)"},
                hoverinfo="skip",
            ))
            if not frontera.empty:
                fig_f.add_trace(go.Scatter(
                    x=frontera["volatilidad"] * 100,
                    y=frontera["rendimiento"] * 100,
                    mode="lines", name="Frontera eficiente",
                    line={"color": "#00d9ff", "width": 3},
                    hovertemplate="Riesgo: %{x:.2f}%<br>Retorno: %{y:.2f}%<extra></extra>",
                ))
            fig_f.add_trace(go.Scatter(
                x=vol_act * 100, y=mu_act * 100, mode="markers+text",
                text=tickers_list, textposition="top center",
                name="Activos individuales",
                marker={"size": 9, "color": "#fbbf24", "symbol": "circle"},
            ))
            fig_f.add_trace(go.Scatter(
                x=[m_eq["vol"] * 100], y=[mu_eq * 100], mode="markers",
                name="Equiponderado 1/N",
                marker={"size": 13, "color": "white", "symbol": "diamond"},
            ))
            fig_f.add_trace(go.Scatter(
                x=[m["vol"] * 100], y=[mu_port * 100], mode="markers",
                name="Tu portafolio",
                marker={"size": 20, "color": "red", "symbol": "star",
                        "line": {"color": "white", "width": 2}},
            ))
            fig_f.update_layout(
                template="plotly_dark", height=520,
                xaxis_title="Volatilidad anual (%)",
                yaxis_title=f"Rendimiento esperado anual (%) — {estimador_efectivo}",
                hovermode="closest", legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_f, use_container_width=True)

    # ==========================================
    # TAB 2 — PROYECCIÓN MONTE CARLO
    # ==========================================
    with tab_sim:
        st.subheader("🎲 Proyección del Portafolio — Bootstrap Histórico")
        st.caption(
            "Se re-muestrean miles de veces los retornos diarios reales del "
            "portafolio (con reemplazo) para proyectar la distribución de "
            "resultados. Es no-paramétrico: conserva la volatilidad, asimetría "
            "y colas gordas de los datos reales."
        )
        mu_proyeccion = (resultado_opt.rendimiento_esperado if resultado_opt
                         else float(mu_esperado.get(tickers_list[0], np.nan)))
        centrar = st.checkbox(
            f"Centrar la deriva en el μ esperado ({fmt_pct(mu_proyeccion)}) "
            "— recomendado",
            value=True,
            help="Sin centrar, la proyección extrapola la deriva histórica del "
                 "período analizado, lo cual es peligroso tras un período "
                 "excepcional. Centrar preserva la forma de la distribución "
                 "(volatilidad y colas) pero ancla la tendencia al estimador "
                 "forward.",
        )
        try:
            sim = calcular_simulacion(
                serie_port, float(capital_inversion), horizonte_sim, n_sims,
                mu_proyeccion if centrar and not pd.isna(mu_proyeccion) else None,
            )
        except ValueError as err:
            sim = None
            st.info(f"ℹ️ {err}")

        if sim:
            pct = sim["percentiles"]
            terminal = sim["terminal"]
            x_dias = np.arange(1, horizonte_sim + 1)

            cs1, cs2, cs3, cs4 = st.columns(4)
            cs1.metric("Escenario pesimista (P5)", fmt_usd(np.percentile(terminal, 5)),
                       help="El 95% de las simulaciones terminó por encima de esto.")
            cs2.metric("Escenario central (mediana)", fmt_usd(np.median(terminal)))
            cs3.metric("Escenario optimista (P95)", fmt_usd(np.percentile(terminal, 95)))
            cs4.metric("Probabilidad de pérdida", fmt_pct(sim["prob_perdida"], 1),
                       help="Fracción de simulaciones que terminan por debajo "
                            "del capital inicial.")

            fig_fan = go.Figure()
            fig_fan.add_trace(go.Scatter(
                x=x_dias, y=pct[95], line={"width": 0}, showlegend=False,
                hoverinfo="skip"))
            fig_fan.add_trace(go.Scatter(
                x=x_dias, y=pct[5], fill="tonexty", name="Rango P5–P95",
                fillcolor="rgba(0,217,255,0.12)", line={"width": 0}))
            fig_fan.add_trace(go.Scatter(
                x=x_dias, y=pct[75], line={"width": 0}, showlegend=False,
                hoverinfo="skip"))
            fig_fan.add_trace(go.Scatter(
                x=x_dias, y=pct[25], fill="tonexty", name="Rango P25–P75",
                fillcolor="rgba(0,217,255,0.25)", line={"width": 0}))
            fig_fan.add_trace(go.Scatter(
                x=x_dias, y=pct[50], name="Mediana",
                line={"color": "#00d9ff", "width": 2.5}))
            fig_fan.add_hline(y=capital_inversion, line_dash="dash",
                              line_color="gray",
                              annotation_text="Capital inicial")
            fig_fan.update_layout(
                template="plotly_dark", height=450,
                title=f"Evolución proyectada de {fmt_usd(capital_inversion)} "
                      f"({horizonte_sim} días hábiles, {n_sims:,} simulaciones)",
                xaxis_title="Días hábiles", yaxis_title="Valor del portafolio (USD)",
            )
            st.plotly_chart(fig_fan, use_container_width=True)

            fig_hist = px.histogram(
                x=terminal, nbins=60,
                labels={"x": "Valor terminal (USD)"},
                title="Distribución del valor final",
            )
            fig_hist.add_vline(x=capital_inversion, line_dash="dash",
                               line_color="white",
                               annotation_text="Capital inicial")
            fig_hist.update_layout(template="plotly_dark", height=350,
                                   showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

    # ==========================================
    # TAB 3 — VALIDACIÓN OUT-OF-SAMPLE
    # ==========================================
    with tab_val:
        st.subheader("✅ Validación Out-of-Sample (la prueba honesta)")
        st.caption(
            "Las métricas 'in-sample' de un portafolio optimizado casi siempre "
            "están infladas: el optimizador vio esos mismos datos. Aquí se "
            "optimiza con el primer 70% de la muestra y se evalúa en el 30% "
            "final, que el optimizador nunca vio."
        )
        if len(tickers_list) < 2:
            st.info("ℹ️ Se requieren 2+ activos para validar estrategias.")
        else:
            try:
                tabla_oos, curvas_oos, fecha_corte = calcular_backtest(
                    returns_df, TASA_RF, usar_shrinkage, gamma_reg,
                    permitir_exclusion)
            except ValueError as err:
                tabla_oos = None
                st.info(f"ℹ️ {err}")

            if tabla_oos is not None:
                st.markdown(f"**Período de evaluación (nunca visto):** "
                            f"{fecha_corte.date()} → {fin_efectivo}")
                st.dataframe(
                    tabla_oos.style.format({
                        "CAGR": "{:.2%}", "Volatilidad": "{:.2%}",
                        "Sharpe": "{:.3f}", "CVaR diario (95%)": "{:.2%}",
                        "Max Drawdown": "{:.2%}",
                    }, na_rep="—"),
                    use_container_width=True, hide_index=True,
                )

                fig_oos = go.Figure()
                for col in curvas_oos.columns:
                    fig_oos.add_trace(go.Scatter(
                        x=curvas_oos.index, y=curvas_oos[col] * 100,
                        mode="lines", name=col,
                    ))
                fig_oos.update_layout(
                    template="plotly_dark", height=450,
                    title="Crecimiento de $100 en el período de evaluación",
                    yaxis_title="Valor (base 100)", hovermode="x unified",
                )
                st.plotly_chart(fig_oos, use_container_width=True)
                st.caption(
                    "💡 Si el 1/N gana o empata a las estrategias optimizadas, "
                    "es señal de que la muestra es corta o el mercado cambió de "
                    "régimen — no asignes capital basándote solo en la "
                    "optimización in-sample."
                )

    # ==========================================
    # TAB 4 — ACTIVOS INDIVIDUALES
    # ==========================================
    with tab_activos:
        st.subheader("📊 Métricas por Activo")
        filas = []
        betas = {}
        for t in tickers_list:
            mt = cq.metricas_riesgo(returns_df[t], TASA_RF)
            fila = {
                "Ticker": t, "CAGR": mt["cagr"], "Volatilidad": mt["vol"],
                "Sharpe": mt["sharpe"], "Sortino": mt["sortino"],
                "Max DD": mt["mdd"], "VaR d. 95%": mt["var_d"],
                "CVaR d. 95%": mt["cvar_d"], "Asimetría": mt["skew"],
                "Curtosis": mt["kurt"],
            }
            if bench_returns is not None:
                b = cq.beta_regresion(returns_df[t], bench_returns, TASA_RF)
                if b:
                    betas[t] = b
                    fila["Beta"] = b["beta"]
                    fila["α Jensen"] = b["alpha_jensen"]
                    fila["R²"] = b["r2"]
            filas.append(fila)

        df_act = pd.DataFrame(filas)
        formato = {
            "CAGR": "{:.2%}", "Volatilidad": "{:.2%}", "Sharpe": "{:.3f}",
            "Sortino": "{:.3f}", "Max DD": "{:.2%}", "VaR d. 95%": "{:.2%}",
            "CVaR d. 95%": "{:.2%}", "Asimetría": "{:.2f}", "Curtosis": "{:.2f}",
            "Beta": "{:.3f}", "α Jensen": "{:.2%}", "R²": "{:.2f}",
        }
        st.dataframe(
            df_act.style.format(
                {k: v for k, v in formato.items() if k in df_act.columns},
                na_rep="—"),
            use_container_width=True, hide_index=True,
        )
        if betas:
            poco_fiables = [t for t, b in betas.items() if b["r2"] < 0.1]
            if poco_fiables:
                st.caption(
                    f"⚠️ Beta poco informativo (R² < 0.10) para: "
                    f"{', '.join(poco_fiables)} — esos activos se mueven casi "
                    f"independientes de {bench_tick}."
                )

        col_a, col_b = st.columns(2)
        with col_a:
            fig_sh = go.Figure(go.Bar(
                x=df_act["Ticker"], y=df_act["Sharpe"],
                text=[f"{v:.2f}" for v in df_act["Sharpe"]],
                textposition="auto",
                marker={"color": df_act["Sharpe"], "colorscale": "RdYlGn"},
            ))
            fig_sh.update_layout(template="plotly_dark", height=350,
                                 title="Sharpe por activo", yaxis_title="Sharpe")
            st.plotly_chart(fig_sh, use_container_width=True)
        with col_b:
            fig_rr = go.Figure(go.Scatter(
                x=df_act["Volatilidad"] * 100, y=df_act["CAGR"] * 100,
                mode="markers+text", text=df_act["Ticker"],
                textposition="top center",
                marker={"size": 14, "color": df_act["Sharpe"],
                        "colorscale": "RdYlGn", "showscale": True,
                        "colorbar": {"title": "Sharpe"}},
            ))
            fig_rr.update_layout(template="plotly_dark", height=350,
                                 title="Riesgo vs Retorno",
                                 xaxis_title="Volatilidad %",
                                 yaxis_title="CAGR %")
            st.plotly_chart(fig_rr, use_container_width=True)

    # ==========================================
    # TAB 5 — ANÁLISIS TÉCNICO
    # ==========================================
    with tab_tecnico:
        ticker_sel = st.selectbox("Selecciona un activo:", tickers_list)

        serie_precio = adj_close_df[ticker_sel].dropna()
        precio_actual = serie_precio.iloc[-1]
        cambio_total = serie_precio.iloc[-1] / serie_precio.iloc[0] - 1
        cambio_dia = serie_precio.iloc[-1] / serie_precio.iloc[-2] - 1 \
            if len(serie_precio) > 1 else np.nan

        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("Precio actual (ajustado)", fmt_usd(precio_actual),
                   delta=f"{fmt_pct(cambio_dia)} hoy")
        cm2.metric("Cambio en el período", fmt_pct(cambio_total))
        cm3.metric("Máximo del período", fmt_usd(serie_precio.max()))
        cm4.metric("Mínimo del período", fmt_usd(serie_precio.min()))

        st.caption(
            "Las velas, RSI y Bollinger usan precios de cotización (Close); las "
            "métricas de retorno usan precios ajustados por dividendos/splits."
        )

        # --- Velas con medias móviles ---
        cierre = close_df[ticker_sel]
        fig_candle = go.Figure(go.Candlestick(
            x=cierre.index, open=open_df[ticker_sel], high=high_df[ticker_sel],
            low=low_df[ticker_sel], close=cierre, name=ticker_sel,
        ))
        for ventana, color in ((20, "orange"), (50, "#60a5fa")):
            fig_candle.add_trace(go.Scatter(
                x=cierre.index, y=cierre.rolling(ventana).mean(),
                mode="lines", name=f"SMA {ventana}",
                line={"color": color, "width": 1.2},
            ))
        fig_candle.update_layout(
            template="plotly_dark", height=480,
            title=f"{ticker_sel} — Velas con medias móviles",
            yaxis_title="Precio USD", xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        # --- Volumen (vectorizado) ---
        colores_vol = np.where(cierre < open_df[ticker_sel], "#ef4444", "#22c55e")
        fig_vol = go.Figure(go.Bar(
            x=volume_df.index, y=volume_df[ticker_sel],
            marker_color=colores_vol, name="Volumen",
        ))
        fig_vol.update_layout(template="plotly_dark", height=250,
                              title=f"{ticker_sel} — Volumen",
                              yaxis_title="Volumen")
        st.plotly_chart(fig_vol, use_container_width=True)

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.subheader("📊 RSI (suavizado de Wilder)")
            delta_p = cierre.diff()
            ganancia = delta_p.clip(lower=0)
            perdida = -delta_p.clip(upper=0)
            # Wilder usa media exponencial alpha=1/14, no SMA
            avg_gain = ganancia.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
            avg_loss = perdida.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - 100 / (1 + rs)

            fig_rsi = go.Figure(go.Scatter(
                x=rsi.index, y=rsi, mode="lines", name="RSI",
                line={"color": "#a855f7", "width": 2},
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",
                              annotation_text="Sobrecompra (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                              annotation_text="Sobreventa (30)")
            fig_rsi.update_layout(template="plotly_dark", height=300,
                                  yaxis_title="RSI", yaxis_range=[0, 100])
            st.plotly_chart(fig_rsi, use_container_width=True)

            rsi_actual = rsi.dropna().iloc[-1] if rsi.notna().any() else np.nan
            if pd.isna(rsi_actual):
                st.markdown("RSI: datos insuficientes.")
            elif rsi_actual > 70:
                st.markdown(f"**RSI {rsi_actual:.1f} — Sobrecompra** (🔴): el "
                            "precio subió rápido; históricamente aumenta la "
                            "probabilidad de pausa o retroceso.")
            elif rsi_actual < 30:
                st.markdown(f"**RSI {rsi_actual:.1f} — Sobreventa** (🟢): caída "
                            "acelerada; puede haber rebote técnico.")
            else:
                st.markdown(f"**RSI {rsi_actual:.1f} — Neutral** (🟡): sin "
                            "señal de extremos.")

        with col_t2:
            st.subheader("📊 Bandas de Bollinger (20, 2σ)")
            sma_bb = cierre.rolling(20).mean()
            std_bb = cierre.rolling(20).std()
            banda_sup, banda_inf = sma_bb + 2 * std_bb, sma_bb - 2 * std_bb

            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(
                x=cierre.index, y=banda_sup, mode="lines", name="Superior",
                line={"color": "rgba(239,68,68,0.7)", "width": 1, "dash": "dash"}))
            fig_bb.add_trace(go.Scatter(
                x=cierre.index, y=banda_inf, mode="lines", name="Inferior",
                fill="tonexty", fillcolor="rgba(0,217,255,0.07)",
                line={"color": "rgba(34,197,94,0.7)", "width": 1, "dash": "dash"}))
            fig_bb.add_trace(go.Scatter(
                x=cierre.index, y=sma_bb, mode="lines", name="SMA 20",
                line={"color": "orange", "width": 1}))
            fig_bb.add_trace(go.Scatter(
                x=cierre.index, y=cierre, mode="lines", name="Precio",
                line={"color": "#00d9ff", "width": 1.5}))
            fig_bb.update_layout(template="plotly_dark", height=300,
                                 yaxis_title="Precio")
            st.plotly_chart(fig_bb, use_container_width=True)

            if cierre.iloc[-1] > banda_sup.iloc[-1]:
                st.markdown("**Sobre la banda superior** (🔴): movimiento "
                            "estadísticamente estirado al alza.")
            elif cierre.iloc[-1] < banda_inf.iloc[-1]:
                st.markdown("**Bajo la banda inferior** (🟢): movimiento "
                            "estadísticamente estirado a la baja.")
            else:
                st.markdown("**Dentro de las bandas** (🟡): comportamiento "
                            "normal.")

        st.caption(
            "⚠️ RSI y Bollinger son descripciones estadísticas del precio, no "
            "señales con valor predictivo comprobado en este dashboard."
        )

        # --- Comparación base 100 ---
        if len(tickers_list) > 1 or bench_series is not None:
            st.subheader("📈 Rendimiento relativo (base 100)")
            norm = adj_close_df.div(adj_close_df.iloc[0]) * 100
            fig_comp = go.Figure()
            for t in tickers_list:
                fig_comp.add_trace(go.Scatter(
                    x=norm.index, y=norm[t], mode="lines", name=t))
            if bench_series is not None:
                bn = bench_series / bench_series.iloc[0] * 100
                fig_comp.add_trace(go.Scatter(
                    x=bn.index, y=bn, mode="lines",
                    name=f"{bench_tick} (benchmark)",
                    line={"dash": "dash", "width": 2, "color": "gray"}))
            fig_comp.update_layout(template="plotly_dark", height=450,
                                   yaxis_title="Valor indexado (base 100)",
                                   hovermode="x unified")
            st.plotly_chart(fig_comp, use_container_width=True)

    # ==========================================
    # TAB 6 — CORRELACIÓN
    # ==========================================
    with tab_corr:
        if len(tickers_list) < 2:
            st.info("ℹ️ Se necesitan 2+ activos para analizar correlación.")
        else:
            corr = returns_df.corr()
            n_act = len(tickers_list)
            prom_offdiag = (corr.values.sum() - n_act) / (n_act * (n_act - 1))

            col_c1, col_c2 = st.columns([2, 1])
            with col_c1:
                fig_corr = px.imshow(
                    corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto",
                    title="Correlación de retornos diarios",
                )
                fig_corr.update_layout(template="plotly_dark", height=480)
                st.plotly_chart(fig_corr, use_container_width=True)
            with col_c2:
                st.metric("Correlación promedio", f"{prom_offdiag:.2f}",
                          help="Promedio de todos los pares. <0.3 buena "
                               "diversificación; >0.7 poca.")
                pares = corr.where(
                    np.triu(np.ones(corr.shape), k=1).astype(bool)
                ).stack().sort_values()
                st.markdown("**Par menos correlacionado** (mejor diversificador):")
                st.text(f"{pares.index[0][0]} – {pares.index[0][1]}: "
                        f"{pares.iloc[0]:.3f}")
                st.markdown("**Par más correlacionado** (riesgo duplicado):")
                st.text(f"{pares.index[-1][0]} – {pares.index[-1][1]}: "
                        f"{pares.iloc[-1]:.3f}")
                st.markdown("""
**Guía rápida:**
- < 0.3 → diversifican bien
- 0.3 – 0.7 → diversificación moderada
- > 0.7 → se mueven casi juntos
                """)

            # --- Correlación en estrés ---
            if bench_returns is not None:
                st.subheader("🌧️ Correlación en días de estrés")
                st.caption(
                    "Las correlaciones suben justo cuando más importan. Aquí se "
                    f"recalculan usando solo el peor 10% de días de {bench_tick}."
                )
                bench_alineado = bench_returns.reindex(returns_df.index).dropna()
                comunes = returns_df.index.intersection(bench_alineado.index)
                umbral = bench_alineado.loc[comunes].quantile(0.10)
                dias_estres = comunes[bench_alineado.loc[comunes] <= umbral]
                if len(dias_estres) >= 25:
                    corr_estres = returns_df.loc[dias_estres].corr()
                    prom_estres = ((corr_estres.values.sum() - n_act)
                                   / (n_act * (n_act - 1)))
                    ce1, ce2 = st.columns(2)
                    ce1.metric("Correlación promedio (normal)",
                               f"{prom_offdiag:.2f}")
                    ce2.metric("Correlación promedio (estrés)",
                               f"{prom_estres:.2f}",
                               delta=f"{prom_estres - prom_offdiag:+.2f}",
                               delta_color="inverse")
                    if prom_estres > prom_offdiag + 0.1:
                        st.warning(
                            "⚠️ Tu diversificación se debilita en caídas de "
                            "mercado: los activos caen más juntos de lo que la "
                            "correlación normal sugiere."
                        )
                else:
                    st.info("ℹ️ Muy pocos días de estrés en la muestra para un "
                            "cálculo confiable (se requieren 25+).")

    # ==========================================
    # TAB 7 — FUNDAMENTALES
    # ==========================================
    with tab_fund:
        st.subheader("🏢 Datos Fundamentales")
        if st.checkbox("📊 Consultar fundamentales (1 petición por ticker)"):
            with st.spinner("Consultando Yahoo Finance..."):
                fund = obtener_fundamentales(tuple(tickers_list))
            if fund:
                df_fund = pd.DataFrame(fund)
                st.dataframe(
                    df_fund.style.format({
                        "Precio": "${:,.2f}", "P/E": "{:.2f}", "P/B": "{:.2f}",
                        "Div Yield %": "{:.2f}", "Beta (yf)": "{:.2f}",
                        "EPS": "{:.2f}",
                    }, na_rep="—"),
                    use_container_width=True, hide_index=True,
                )
                pe = df_fund.dropna(subset=["P/E"]) if "P/E" in df_fund else None
                if pe is not None and len(pe) > 1:
                    fig_pe = go.Figure(go.Bar(
                        x=pe["Ticker"], y=pe["P/E"],
                        text=[f"{v:.1f}" for v in pe["P/E"]],
                        textposition="auto", marker_color="#00d9ff",
                    ))
                    fig_pe.update_layout(template="plotly_dark", height=350,
                                         title="P/E Ratio", yaxis_title="P/E")
                    st.plotly_chart(fig_pe, use_container_width=True)
                st.caption(
                    "Los fundamentales se cachean por 24 h. 'Beta (yf)' es el "
                    "que reporta Yahoo (mensual, 5 años) y puede diferir del "
                    "beta diario de la pestaña Activos."
                )

    # ==========================================
    # TAB 8 — REPORTE
    # ==========================================
    with tab_rep:
        st.subheader("📥 Reporte en Excel")
        st.markdown(
            "El archivo incluye una hoja de **Metadatos** (fecha, período, "
            "supuestos) para que cualquier número del reporte sea auditable y "
            "reproducible."
        )
        metadatos = {
            "Generado": f"{datetime.now():%Y-%m-%d %H:%M}",
            "Versión del dashboard": VERSION,
            "Tickers": ", ".join(tickers_list),
            "Benchmark": bench_tick or "—",
            "Período solicitado": A["periodo"],
            "Período efectivo": f"{inicio_efectivo} → {fin_efectivo}",
            "Observaciones": n_obs,
            "Tasa libre de riesgo": TASA_RF,
            "Estimador de retornos esperados": estimador_efectivo,
            "ERP (si CAPM)": ERP if estimador_efectivo == cq.RE_CAPM else "—",
            "Estrategia": estrategia_radio,
            "Ledoit-Wolf": usar_shrinkage,
            "Gamma L2": gamma_reg,
            "Nota": "CAGR geométrico; VaR/CVaR en horizonte diario; "
                    "μ esperado forward según estimador documentado arriba.",
        }
        excel_bytes = crear_excel(
            returns_df, adj_close_df,
            resultado_opt.pesos if resultado_opt else None, metadatos,
        )
        st.download_button(
            "📊 Descargar Excel completo",
            data=excel_bytes,
            file_name=f"analisis_financiero_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

else:
    # ==========================================
    # PANTALLA DE BIENVENIDA
    # ==========================================
    st.info("👈 Configura los parámetros y presiona **'🚀 Analizar Activos'**. "
            f"Los valores por defecto ({TICKERS_DEMO} vs SPY, 2 años) "
            "funcionan como demo inmediata.")
    st.markdown("""
## 🎯 Qué hace este dashboard

### 💼 Optimización verificable de portafolio
- **Tres estrategias**: máximo Sharpe, mínima volatilidad y **mínimo CVaR**
  (programación lineal de Rockafellar-Uryasev).
- **Covarianza Ledoit-Wolf** y regularización L2 opcionales.
- **Convergencia verificada** con arranques múltiples: si el optimizador
  no converge, lo verás en pantalla.
- **Modo híbrido**: fija pesos manuales y optimiza el resto.

### 🎲 Proyección y validación honesta
- **Monte Carlo bootstrap**: distribución del valor de tu capital a 6–24
  meses con escenarios pesimista/central/optimista.
- **Backtest out-of-sample**: la estrategia se evalúa en datos que el
  optimizador nunca vio, contra el equiponderado 1/N.

### 📐 Metodología de nivel institucional
- **Retornos esperados defendibles**: CAPM/equilibrio (prior de
  Black-Litterman) o shrinkage Bayes-Stein (Jorion 1986) — el CAGR
  histórico puro sobrepondera a los ganadores del período y produce
  expectativas infladas ("error maximization", Michaud 1989).
- Rendimientos anuales como **CAGR geométrico** (la media aritmética × 252
  sobreestima).
- **VaR/CVaR en horizonte diario** — anualizarlos con √252 es inválido.
- **Sortino con downside deviation** contra la tasa libre de riesgo.
- **Frontera eficiente real** (optimización por punto, no nube aleatoria).
- Beta con **R² y significancia**, alfa de Jensen, correlación en estrés.

### 📊 Además
- Velas, **RSI de Wilder**, Bollinger, volumen, comparación base 100.
- Fundamentales (P/E, P/B, yield) con caché.
- **Excel auditable** con hoja de metadatos y supuestos.
    """)

# ==========================================
# FOOTER / DISCLAIMER
# ==========================================
st.markdown("---")
st.markdown(
    f"""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p><strong>Dashboard Financiero Cuantitativo v{VERSION}</strong> —
    Herramienta propietaria de gestión de portafolio personal</p>
    <p style='font-size: 0.85rem;'>
    Metodología: μ esperado por CAPM/Bayes-Stein (configurable), covarianza
    Ledoit-Wolf, CVaR histórico diario, validación out-of-sample.
    Referencias: Markowitz (1952), Sharpe (1964), Jorion (1986),
    Michaud (1989), Black-Litterman (1992), Ledoit-Wolf (2004),
    Rockafellar-Uryasev (2000).<br>
    Datos: Yahoo Finance (sujetos a errores u omisiones). Todo estimado
    depende de los supuestos documentados en la hoja de Metadatos del
    reporte; el rendimiento histórico no determina el futuro.
    </p>
</div>
""",
    unsafe_allow_html=True,
)
