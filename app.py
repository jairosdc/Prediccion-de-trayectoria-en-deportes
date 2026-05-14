"""
app.py — Interfaz web con Streamlit para Predicción de trayectoria en deportes.
Ejecutar: python -m streamlit run app.py
"""

import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.preprocessing import cargar_datos, preparar_dataframe, COLUMNAS_MODELO
from src.train_model import entrenar_y_evaluar, predecir_probabilidades
from src.pressure import calcular_pressure_index, clasificar_presion
from src.analysis import (
    decisiones_por_deporte, decisiones_por_presion,
    decisiones_por_lado_dominante, presion_media_por_deporte
)

# ============================================================
# Config
# ============================================================
RUTA_DATASET = "data/synthetic_multisport_decisions.csv"

st.set_page_config(
    page_title="Predicción de trayectoria en deportes",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# CSS — Tema profesional claro y neutral
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #f8fafc;
    --bg-card: #ffffff;
    --accent: #2563eb;
    --accent-light: #dbeafe;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border-color: #e2e8f0;
    --success: #16a34a;
    --warning: #b45309;
    --danger: #dc2626;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stApp"] {
    background: #f8fafc;
}

[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1e293b;
    animation: fadeInUp 0.6s ease-out;
    margin-bottom: 0;
    letter-spacing: -0.5px;
}

.hero-sub {
    font-size: 1rem;
    color: #64748b;
    animation: fadeInUp 0.8s ease-out;
    margin-top: 4px;
    margin-bottom: 24px;
}

.glass-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 20px;
    animation: fadeInUp 0.6s ease-out;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s ease;
}

.glass-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    animation: fadeInUp 0.7s ease-out;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s ease;
}

.metric-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e293b;
}

.metric-label {
    font-size: 0.78rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 4px;
}

.pressure-bar {
    width: 100%;
    height: 10px;
    background: #f1f5f9;
    border-radius: 5px;
    overflow: hidden;
    margin: 10px 0;
}

.pressure-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 1s ease-out;
    background: linear-gradient(90deg, #16a34a, #d97706, #dc2626);
}

.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.3px;
}

.badge-baja { background: #dcfce7; color: #16a34a; border: 1px solid #bbf7d0; }
.badge-media { background: #fef9c3; color: #a16207; border: 1px solid #fde68a; }
.badge-alta  { background: #fee2e2; color: #dc2626; border: 1px solid #fecaca; }

.result-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    animation: fadeInUp 0.8s ease-out;
}

.result-decision {
    font-size: 1.6rem;
    font-weight: 700;
    color: #15803d;
}

.result-prob {
    font-size: 1rem;
    color: #64748b;
    margin-top: 4px;
}

.sport-icon { font-size: 1.5rem; margin-right: 6px; }

.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1.5px solid #e2e8f0;
    animation: fadeInUp 0.5s ease-out;
}

.info-banner {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #2563eb;
    border-radius: 6px;
    padding: 12px 16px;
    color: #475569;
    font-size: 0.9rem;
    animation: fadeInUp 1.0s ease-out;
    margin-bottom: 24px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f1f5f9;
    border-radius: 8px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    color: #64748b;
    font-weight: 500;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Carga y entrenamiento cacheados
# ============================================================
@st.cache_data
def cargar_y_preparar_dataset():
    if not os.path.exists(RUTA_DATASET):
        return None, None
    df = cargar_datos(RUTA_DATASET)
    return df, preparar_dataframe(df)


@st.cache_resource
def entrenar_modelo_cacheado():
    df = cargar_datos(RUTA_DATASET)
    return entrenar_y_evaluar(df)


# ============================================================
# Funciones auxiliares
# ============================================================
SPORT_ICONS = {
    "football": "⚽", "tennis": "🎾", "basketball": "🏀",
    "handball": "🤾", "hockey": "🏒", "esports": "🎮"
}

SPORT_COLORS = {
    "football": "#2563eb", "tennis": "#d97706", "basketball": "#dc2626",
    "handball": "#7c3aed", "hockey": "#0891b2", "esports": "#be185d"
}

SPORT_DESC = {
    "football": "Penalti o acción de lanzamiento en fútbol.",
    "tennis": "Saque o punto importante en tenis.",
    "basketball": "Tiro o decisión ofensiva en baloncesto.",
    "handball": "Lanzamiento de 7 metros en balonmano.",
    "hockey": "Shootout contra el portero en hockey.",
    "esports": "Duelo 1v1 con decisión táctica bajo presión."
}

DECISION_NAMES = {
    "football": "zona del lanzamiento", "tennis": "zona del saque",
    "basketball": "tipo o zona de tiro", "handball": "zona del lanzamiento",
    "hockey": "tipo de finalización", "esports": "decisión táctica"
}


def obtener_opciones(df, col):
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().unique().tolist())


def filtrar_df(df, sport=None, action_type=None, competition=None):
    f = df.copy()
    if sport: f = f[f["sport"] == sport]
    if action_type: f = f[f["action_type"] == action_type]
    if competition: f = f[f["competition"] == competition]
    return f


def crear_caso(sport, action_type, competition, match_time, score_difference,
               is_home_player, player, opponent, dominant_side, fatigue_level,
               competition_importance, pressure_index, pressure_level):
    caso = pd.DataFrame([{
        "sport": sport, "action_type": action_type, "competition": competition,
        "match_time": match_time, "score_difference": score_difference,
        "is_home_player": is_home_player, "player": player, "opponent": opponent,
        "dominant_side": dominant_side, "fatigue_level": fatigue_level,
        "competition_importance": competition_importance,
        "pressure_index": pressure_index, "pressure_level": pressure_level,
    }])
    return caso[COLUMNAS_MODELO]


def crear_gauge_presion(value, level):
    colors = {"baja": "#16a34a", "media": "#d97706", "alta": "#dc2626"}
    color = colors.get(level, "#2563eb")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 38, "color": color}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#cbd5e1",
                     "tickfont": {"color": "#64748b"}},
            "bar": {"color": color, "thickness": 0.65},
            "bgcolor": "#f1f5f9",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 0.35], "color": "rgba(22,163,74,0.10)"},
                {"range": [0.35, 0.70], "color": "rgba(217,119,6,0.10)"},
                {"range": [0.70, 1.0], "color": "rgba(220,38,38,0.10)"},
            ],
            "threshold": {
                "line": {"color": "#1e293b", "width": 2},
                "thickness": 0.75, "value": value
            }
        }
    ))
    fig.update_layout(
        height=220, margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#64748b"}
    )
    return fig


def crear_chart_probabilidades(probs, sport):
    color = SPORT_COLORS.get(sport, "#3b82f6")
    fig = go.Figure(go.Bar(
        x=probs["probability"] * 100,
        y=probs["decision_zone"],
        orientation="h",
        marker=dict(
            color=[f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},{0.3 + 0.7*p})"
                   for p in probs["probability"]],
            line=dict(color=color, width=1.5),
        ),
        text=[f"{p:.1f}%" for p in probs["probability"] * 100],
        textposition="outside",
        textfont=dict(color="#1e293b", size=13, family="Inter"),
    ))
    fig.update_layout(
        height=max(250, len(probs) * 50),
        margin=dict(l=10, r=60, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.8)",
                   tickfont=dict(color="#64748b"),
                   title=dict(text="Probabilidad (%)", font=dict(color="#64748b"))),
        yaxis=dict(tickfont=dict(color="#1e293b", size=12), autorange="reversed"),
        font=dict(family="Inter"),
    )
    return fig


# ============================================================
# Comprobar dataset
# ============================================================
if not os.path.exists(RUTA_DATASET):
    st.error("No se encontró el dataset. Coloca synthetic_multisport_decisions.csv en data/.")
    st.stop()

df_original, df_preparado = cargar_y_preparar_dataset()
modelo, resultados, df_modelo = entrenar_modelo_cacheado()

# ============================================================
# Header
# ============================================================
st.markdown('<div class="hero-title">Predicción de trayectoria en deportes</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Modelo interactivo para estimar decisiones deportivas bajo presión en situaciones 1v1</div>', unsafe_allow_html=True)
st.markdown("""<div class="info-banner">
    <strong>Dataset sintético multideporte</strong> — Esta aplicación demuestra el funcionamiento del framework:
    carga datos, calcula presión, entrena un modelo y devuelve probabilidades de decisión.
</div>""", unsafe_allow_html=True)

# ============================================================
# Sidebar
# ============================================================
st.sidebar.markdown("## Panel de simulación")
st.sidebar.markdown("Elige un deporte y configura la situación.")

sports = obtener_opciones(df_preparado, "sport")
sport = st.sidebar.selectbox("Deporte", sports,
    format_func=lambda s: f"{SPORT_ICONS.get(s,'')} {s.capitalize()}")

df_sport = filtrar_df(df_preparado, sport=sport)
action_types = obtener_opciones(df_sport, "action_type")
action_type = st.sidebar.selectbox("Tipo de acción", action_types)

df_action = filtrar_df(df_preparado, sport=sport, action_type=action_type)
competitions = obtener_opciones(df_action, "competition")
competition = st.sidebar.selectbox("Competición", competitions)

df_contexto = filtrar_df(df_preparado, sport=sport, action_type=action_type, competition=competition)

# ============================================================
# Layout principal
# ============================================================
col_info, col_modelo = st.columns([1.1, 1])

with col_info:
    icon = SPORT_ICONS.get(sport, "🏅")
    st.markdown(f'<div class="section-title">Situación elegida</div>',
                unsafe_allow_html=True)
    st.write(SPORT_DESC.get(sport, "Situación deportiva bajo presión."))
    st.write(f"El modelo estimará la **{DECISION_NAMES.get(sport, 'decisión')}** más probable.")

    st.markdown("#### Contexto del evento")
    col_a, col_b = st.columns(2)
    with col_a:
        match_time = st.slider("Momento del partido", 0, 120, 80,
            help="Avance del evento. Valores altos = más cerca del final.")
        score_difference = st.slider("Diferencia de marcador", -5, 5, 0)
        is_home_player = st.selectbox("¿Juega como local?", [1, 0],
            format_func=lambda x: "Sí" if x == 1 else "No")
    with col_b:
        fatigue_level = st.slider("Nivel de fatiga", 0.0, 1.0, 0.55, 0.05)
        comp_imp_default = float(round(df_contexto["competition_importance"].mean(), 2))
        competition_importance = st.slider("Importancia competición", 0.0, 1.0, comp_imp_default, 0.05)
        critical_moment = st.selectbox("¿Momento crítico?", [0, 1],
            format_func=lambda x: "Sí" if x == 1 else "No")

with col_modelo:
    st.markdown('<div class="section-title">Jugador y rival</div>', unsafe_allow_html=True)

    players = obtener_opciones(df_contexto, "player")
    opponents = obtener_opciones(df_contexto, "opponent")
    dominant_sides = obtener_opciones(df_contexto, "dominant_side")

    if not players or not opponents:
        st.error("No hay datos para este filtro.")
        st.stop()

    player = st.selectbox("Jugador", players)
    opponent = st.selectbox("Rival / oponente", opponents)
    dominant_side = st.selectbox("Lado dominante", dominant_sides,
        format_func=lambda s: s.capitalize())

    pressure_index = calcular_pressure_index(
        match_time, score_difference, competition_importance,
        fatigue_level, critical_moment)
    pressure_level = clasificar_presion(pressure_index)

    st.markdown('<div class="section-title">Presión calculada</div>', unsafe_allow_html=True)
    st.plotly_chart(crear_gauge_presion(pressure_index, pressure_level),
                    use_container_width=True, key="gauge")

    badge_class = f"badge-{pressure_level}"
    st.markdown(f"""<div style="text-align:center;margin-top:-10px;">
        <span class="badge {badge_class}">Nivel: {pressure_level.upper()}</span>
    </div>""", unsafe_allow_html=True)

# ============================================================
# Predicción
# ============================================================
st.markdown("---")
st.markdown('<div class="hero-title" style="font-size:1.6rem;">Predicción del modelo</div>',
            unsafe_allow_html=True)

caso = crear_caso(sport, action_type, competition, match_time, score_difference,
                  is_home_player, player, opponent, dominant_side, fatigue_level,
                  competition_importance, pressure_index, pressure_level)

probabilidades = predecir_probabilidades(modelo, caso)

col_pred, col_chart = st.columns([1, 1.2])

with col_pred:
    decision_top = probabilidades.iloc[0]["decision_zone"]
    prob_top = probabilidades.iloc[0]["probability"]

    st.markdown(f"""<div class="result-box">
        <div style="font-size:0.85rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1.5px;">
            Decisión más probable</div>
        <div class="result-decision">{decision_top.upper()}</div>
        <div class="result-prob">Probabilidad: {prob_top:.1%}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    prob_show = probabilidades.copy()
    prob_show["probability"] = (prob_show["probability"] * 100).round(2)
    prob_show = prob_show.rename(columns={
        "decision_zone": "Decisión", "probability": "Probabilidad (%)"})
    st.dataframe(prob_show, width=500, hide_index=True)

with col_chart:
    st.plotly_chart(crear_chart_probabilidades(probabilidades, sport),
                    use_container_width=True, key="probs_chart")

# ============================================================
# Dinámica con el público
# ============================================================
st.markdown("---")
st.markdown('<div class="section-title">Dinámica para la presentación</div>',
            unsafe_allow_html=True)
st.write("Antes de enseñar la predicción, pide al público que vote.")

opciones = probabilidades["decision_zone"].tolist()
voto = st.radio("Voto del público", opciones, horizontal=True)

if st.button("Comparar voto con el modelo"):
    if voto == decision_top:
        st.success(f"El público coincide con el modelo. Ambos eligieron **{decision_top}**.")
    else:
        st.warning(f"El público eligió **{voto}**, pero el modelo considera más probable **{decision_top}** ({prob_top:.1%}).")

# ============================================================
# Análisis del dataset
# ============================================================
st.markdown("---")
st.markdown('<div class="section-title">Análisis del dataset</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Presión", "Decisiones", "Modelo"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Eventos", df_preparado.shape[0])
    col2.metric("Deportes", df_preparado["sport"].nunique())
    col3.metric("Decisiones distintas", df_preparado["decision_zone"].nunique())
    st.dataframe(df_preparado.head(50), use_container_width=True, hide_index=True)

with tab2:
    tabla_pm = presion_media_por_deporte(df_preparado)
    fig_p = go.Figure(go.Bar(
        x=tabla_pm.index, y=tabla_pm.values,
        marker=dict(color=[SPORT_COLORS.get(s, "#3b82f6") for s in tabla_pm.index],
                    line=dict(width=0)),
        text=[f"{v:.3f}" for v in tabla_pm.values], textposition="outside",
        textfont=dict(color="#1e293b"),
    ))
    fig_p.update_layout(
        title="Presión media por deporte",
        height=350, margin=dict(l=40,r=40,t=50,b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="#1e293b")),
        yaxis=dict(tickfont=dict(color="#64748b"), gridcolor="rgba(226,232,240,0.8)"),
        font=dict(family="Inter", color="#1e293b"),
    )
    st.plotly_chart(fig_p, use_container_width=True)

with tab3:
    st.subheader("Decisiones por deporte")
    st.dataframe(decisiones_por_deporte(df_preparado), use_container_width=True)
    st.subheader("Decisiones por presión")
    st.dataframe(decisiones_por_presion(df_preparado), use_container_width=True)
    st.subheader("Decisiones por lado dominante")
    st.dataframe(decisiones_por_lado_dominante(df_preparado), use_container_width=True)

with tab4:
    st.metric("Accuracy", round(resultados["accuracy"], 4))
    st.write("Clases del modelo:")
    st.write(list(resultados["classes"]))
    st.write("Matriz de confusión:")
    matriz = pd.DataFrame(
        resultados["confusion_matrix"],
        index=resultados["classes"], columns=resultados["classes"])
    st.dataframe(matriz, use_container_width=True)
    st.write("Reporte de clasificación:")
    st.text(resultados["classification_report"])

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("""<div style="text-align:center;color:#94a3b8;font-size:0.82rem;padding:20px 0;">
    Los datos usados son sintéticos. La aplicación demuestra el funcionamiento del pipeline,
    sin extraer conclusiones empíricas reales.<br>
    <span style="color:#64748b;">Predicción de trayectoria en deportes</span> &middot; 2025
</div>""", unsafe_allow_html=True)