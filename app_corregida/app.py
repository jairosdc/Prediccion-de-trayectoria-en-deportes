"""
app.py — Interfaz web con Streamlit para Predicción de trayectoria en deportes.
VERSIÓN CORREGIDA Y MEJORADA.
Ejecutar: python -m streamlit run app.py
"""

import os
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.preprocessing import cargar_datos, preparar_dataframe, COLUMNAS_MODELO
from src.train_model import entrenar_y_evaluar, predecir_probabilidades, COLUMNAS_MODELO_SPORT
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
    page_icon="🎯",
    layout="wide"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background: #0f0f13 !important;
    color: #e2e8f0;
}

[data-testid="stApp"] { background: #0f0f13 !important; }

[data-testid="stSidebar"] {
    background: #16161e !important;
    border-right: 1px solid #2a2a3a !important;
}

[data-testid="stSidebar"] * { color: #c4c4d4 !important; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(99,102,241,0.3); }
    50%       { box-shadow: 0 0 40px rgba(99,102,241,0.6); }
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: fadeUp 0.6s ease-out;
    letter-spacing: -1px;
    line-height: 1.1;
}

.hero-sub {
    font-size: 1rem;
    color: #64748b;
    animation: fadeUp 0.8s ease-out;
    margin-top: 6px;
    margin-bottom: 28px;
}

.card {
    background: #16161e;
    border: 1px solid #2a2a3a;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 18px;
    animation: fadeUp 0.6s ease-out;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.card:hover {
    border-color: #6366f1;
    transform: translateY(-1px);
}

.section-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2a3a;
}

.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.78rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.5px;
}
.badge-baja  { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-media { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-alta  { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

.result-box {
    background: linear-gradient(135deg, #16161e, #1e1e2e);
    border: 1px solid #6366f1;
    border-radius: 14px;
    padding: 28px 22px;
    text-align: center;
    animation: pulse-glow 3s ease-in-out infinite;
}

.result-decision {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 3px;
}

.result-prob {
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 6px;
}

.sport-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid;
    cursor: default;
}

.vote-section {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 2px solid #4f46e5;
    border-radius: 16px;
    padding: 30px;
    margin: 20px 0;
    text-align: center;
}

.vote-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #a78bfa;
    margin-bottom: 6px;
}

.vote-subtitle {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 20px;
}

.stat-row {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
}

.stat-item {
    flex: 1;
    background: #1e1e2e;
    border: 1px solid #2a2a3a;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}

.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #a78bfa;
    font-family: 'DM Mono', monospace;
}

.stat-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

.info-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #94a3b8;
    margin-bottom: 20px;
}

.reveal-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(167,139,250,0.1));
    border: 2px solid #6366f1;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin-top: 16px;
}

.vote-bar-container {
    margin: 8px 0;
    text-align: left;
}

.vote-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: #94a3b8;
    margin-bottom: 4px;
    font-family: 'DM Mono', monospace;
}

.vote-bar-bg {
    background: #2a2a3a;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}

.vote-bar-fill {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    transition: width 0.8s ease-out;
}

[data-testid="stMetric"] {
    background: #16161e !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}

[data-testid="stMetricValue"] { color: #a78bfa !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #16161e;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: #6366f1 !important;
    color: white !important;
}

div[data-testid="stRadio"] > label { color: #c4c4d4 !important; }

footer { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Cache
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
# Constantes
# ============================================================
SPORT_ICONS = {
    "football": "⚽", "tennis": "🎾", "basketball": "🏀",
    "handball": "🤾", "hockey": "🏒", "esports": "🎮"
}

SPORT_COLORS = {
    "football": "#3b82f6", "tennis": "#f59e0b", "basketball": "#ef4444",
    "handball": "#8b5cf6", "hockey": "#06b6d4", "esports": "#ec4899"
}

SPORT_DESC = {
    "football":   "Penalti: el portero debe decidir a qué lado tirarse y el jugador a dónde lanzar.",
    "tennis":     "Saque decisivo: el tenista elige zona bajo máxima presión.",
    "basketball": "Tiro decisivo: ¿mid-range, triple o penetración?",
    "handball":   "Lanzamiento de 7 metros: el portero y el lanzador se leen mutuamente.",
    "hockey":     "Shootout: el jugador enfrenta al portero en un duelo 1v1.",
    "esports":    "Duelo 1v1: decisión táctica en el momento más tenso.",
}

DECISION_NAMES = {
    "football": "zona del lanzamiento", "tennis": "zona del saque",
    "basketball": "tipo de tiro", "handball": "zona del lanzamiento",
    "hockey": "tipo de finalización", "esports": "táctica",
}

# Inicializar estado de votos
if "votos" not in st.session_state:
    st.session_state.votos = {}
if "voto_enviado" not in st.session_state:
    st.session_state.voto_enviado = False
if "mostrar_resultado" not in st.session_state:
    st.session_state.mostrar_resultado = False
if "sport_actual" not in st.session_state:
    st.session_state.sport_actual = None


def obtener_opciones(df, col):
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().unique().tolist())


def filtrar_df(df, sport=None, action_type=None, competition=None):
    f = df.copy()
    if sport:       f = f[f["sport"] == sport]
    if action_type: f = f[f["action_type"] == action_type]
    if competition: f = f[f["competition"] == competition]
    return f


def crear_caso(sport, action_type, competition, match_time, score_difference,
               is_home_player, player, opponent, dominant_side, fatigue_level,
               competition_importance, pressure_index, pressure_level):
    return pd.DataFrame([{
        "sport": sport, "action_type": action_type, "competition": competition,
        "match_time": match_time, "score_difference": score_difference,
        "is_home_player": is_home_player, "player": player, "opponent": opponent,
        "dominant_side": dominant_side, "fatigue_level": fatigue_level,
        "competition_importance": competition_importance,
        "pressure_index": pressure_index, "pressure_level": pressure_level,
    }])


def crear_gauge_presion(value, level):
    colors = {"baja": "#10b981", "media": "#f59e0b", "alta": "#ef4444"}
    color = colors.get(level, "#6366f1")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 36, "color": color, "family": "DM Mono"}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#4a4a5a",
                     "tickfont": {"color": "#64748b", "size": 10}},
            "bar": {"color": color, "thickness": 0.6},
            "bgcolor": "#1e1e2e",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 0.35], "color": "rgba(16,185,129,0.12)"},
                {"range": [0.35, 0.70], "color": "rgba(245,158,11,0.12)"},
                {"range": [0.70, 1.0], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 2}, "thickness": 0.75, "value": value}
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=30, r=30, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#64748b"}
    )
    return fig


def crear_chart_probabilidades(probs, sport):
    color = SPORT_COLORS.get(sport, "#6366f1")
    fig = go.Figure(go.Bar(
        x=probs["probability"] * 100,
        y=probs["decision_zone"],
        orientation="h",
        marker=dict(
            color=[f"rgba(99,102,241,{0.2 + 0.8 * p})" for p in probs["probability"]],
            line=dict(color="#6366f1", width=1),
        ),
        text=[f"{p:.1f}%" for p in probs["probability"] * 100],
        textposition="outside",
        textfont=dict(color="#c4c4d4", size=12, family="DM Mono"),
    ))
    fig.update_layout(
        height=max(220, len(probs) * 48),
        margin=dict(l=10, r=70, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(42,42,58,0.8)",
                   tickfont=dict(color="#64748b", size=11),
                   title=dict(text="Probabilidad (%)", font=dict(color="#64748b"))),
        yaxis=dict(tickfont=dict(color="#e2e8f0", size=13, family="DM Mono"),
                   autorange="reversed"),
        font=dict(family="Space Grotesk"),
    )
    return fig


# ============================================================
# Comprobar dataset
# ============================================================
if not os.path.exists(RUTA_DATASET):
    st.error("No se encontró el dataset. Coloca synthetic_multisport_decisions.csv en data/.")
    st.stop()

df_original, df_preparado = cargar_y_preparar_dataset()

with st.spinner("Entrenando modelos por deporte..."):
    modelos, resultados, df_modelo = entrenar_modelo_cacheado()

# ============================================================
# Header
# ============================================================
st.markdown('<div class="hero-title">🎯 Predicción de Trayectoria en Deportes</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Modelo interactivo — decisiones deportivas bajo presión · Dataset sintético multideporte</div>', unsafe_allow_html=True)

# ============================================================
# Sidebar
# ============================================================
st.sidebar.markdown("## ⚙️ Configuración")
st.sidebar.markdown("---")

sports = obtener_opciones(df_preparado, "sport")
sport = st.sidebar.selectbox("🏅 Deporte", sports,
    format_func=lambda s: f"{SPORT_ICONS.get(s, '')} {s.capitalize()}")

# Resetear votos si cambia el deporte
if st.session_state.sport_actual != sport:
    st.session_state.votos = {}
    st.session_state.voto_enviado = False
    st.session_state.mostrar_resultado = False
    st.session_state.sport_actual = sport

df_sport = filtrar_df(df_preparado, sport=sport)
action_types = obtener_opciones(df_sport, "action_type")
action_type = st.sidebar.selectbox("🎬 Tipo de acción", action_types)

df_action = filtrar_df(df_preparado, sport=sport, action_type=action_type)
competitions = obtener_opciones(df_action, "competition")
competition = st.sidebar.selectbox("🏆 Competición", competitions)

df_contexto = filtrar_df(df_preparado, sport=sport, action_type=action_type, competition=competition)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Situación")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    match_time = st.sidebar.slider("⏱ Minuto del partido", 0, 120, 80)
    score_difference = st.sidebar.slider("⚖️ Diferencia marcador", -5, 5, 0)
with col_b:
    fatigue_level = st.sidebar.slider("😓 Fatiga", 0.0, 1.0, 0.55, 0.05)
    competition_importance = st.sidebar.slider("🌟 Import. competición", 0.0, 1.0,
        float(round(df_contexto["competition_importance"].mean(), 2)), 0.05)

is_home_player = st.sidebar.selectbox("🏠 ¿Local?", [1, 0],
    format_func=lambda x: "Sí" if x == 1 else "No")
critical_moment = st.sidebar.selectbox("⚡ ¿Momento crítico?", [0, 1],
    format_func=lambda x: "Sí" if x == 1 else "No")

# ============================================================
# Layout principal
# ============================================================
col_info, col_modelo = st.columns([1.1, 1])

with col_info:
    st.markdown(f'<div class="section-title">Situación elegida</div>', unsafe_allow_html=True)
    icon = SPORT_ICONS.get(sport, "🏅")
    color = SPORT_COLORS.get(sport, "#6366f1")
    st.markdown(f"""<span class="sport-pill" style="color:{color};border-color:{color}40;background:{color}12;">
        {icon} {sport.upper()}</span>""", unsafe_allow_html=True)
    st.markdown(f"<br><p style='color:#94a3b8;font-size:0.95rem'>{SPORT_DESC.get(sport, '')}</p>",
                unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748b;font-size:0.88rem'>El modelo estimará la <strong style='color:#a78bfa'>{DECISION_NAMES.get(sport, 'decisión')}</strong> más probable.</p>",
                unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:20px;">Jugador y rival</div>', unsafe_allow_html=True)
    players = obtener_opciones(df_contexto, "player")
    opponents = obtener_opciones(df_contexto, "opponent")
    dominant_sides = obtener_opciones(df_contexto, "dominant_side")

    if not players or not opponents:
        st.error("No hay datos para este filtro. Elige otra combinación.")
        st.stop()

    player = st.selectbox("👤 Jugador", players)
    opponent = st.selectbox("🥅 Rival / oponente", opponents)
    dominant_side = st.selectbox("✋ Lado dominante", dominant_sides,
        format_func=lambda s: s.capitalize())

with col_modelo:
    pressure_index = calcular_pressure_index(
        match_time, score_difference, competition_importance,
        fatigue_level, critical_moment)
    pressure_level = clasificar_presion(pressure_index)

    st.markdown('<div class="section-title">Índice de presión</div>', unsafe_allow_html=True)
    st.plotly_chart(crear_gauge_presion(pressure_index, pressure_level),
                    use_container_width=True, key="gauge")

    badge_class = f"badge-{pressure_level}"
    labels = {"baja": "🟢 PRESIÓN BAJA", "media": "🟡 PRESIÓN MEDIA", "alta": "🔴 PRESIÓN ALTA"}
    st.markdown(f"""<div style="text-align:center;margin-top:-8px;">
        <span class="badge {badge_class}">{labels.get(pressure_level, pressure_level.upper())}</span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="info-box" style="margin-top:16px;">
        <strong>Componentes:</strong> Temporal {match_time/120:.0%} · 
        Marcador {1/(1+abs(score_difference)):.0%} · 
        Comp. {competition_importance:.0%} · 
        Fatiga {fatigue_level:.0%} · 
        Crítico {'Sí' if critical_moment else 'No'}
    </div>""", unsafe_allow_html=True)

# ============================================================
# Predicción
# ============================================================
st.markdown("---")
st.markdown('<div class="hero-title" style="font-size:1.5rem;">🤖 Predicción del Modelo</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#64748b;font-size:0.88rem;margin-top:4px;'>Modelo de Regresión Logística entrenado específicamente para este deporte.</p>", unsafe_allow_html=True)

caso = crear_caso(sport, action_type, competition, match_time, score_difference,
                  is_home_player, player, opponent, dominant_side, fatigue_level,
                  competition_importance, pressure_index, pressure_level)

try:
    probabilidades = predecir_probabilidades(modelos, caso)
except Exception as e:
    st.error(f"Error en la predicción: {e}")
    st.stop()

decision_top = probabilidades.iloc[0]["decision_zone"]
prob_top = probabilidades.iloc[0]["probability"]

col_pred, col_chart = st.columns([1, 1.2])

with col_pred:
    st.markdown(f"""<div class="result-box">
        <div style="font-size:0.72rem;color:#6366f1;text-transform:uppercase;letter-spacing:2px;">
            Decisión más probable</div>
        <div class="result-decision">{decision_top.upper()}</div>
        <div class="result-prob">Confianza del modelo: <strong style='color:#a78bfa'>{prob_top:.1%}</strong></div>
        <div style="font-size:0.78rem;color:#4a4a6a;margin-top:8px;">
            {player} vs {opponent} · {SPORT_ICONS.get(sport, '')} {sport.capitalize()}
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    prob_show = probabilidades.copy()
    prob_show["probability"] = (prob_show["probability"] * 100).round(2)
    prob_show = prob_show.rename(columns={
        "decision_zone": "Decisión", "probability": "Probabilidad (%)"})
    st.dataframe(prob_show, width=460, hide_index=True)

with col_chart:
    st.plotly_chart(crear_chart_probabilidades(probabilidades, sport),
                    use_container_width=True, key="probs_chart")

# ============================================================
# DINÁMICA CON EL PÚBLICO — Sección mejorada
# ============================================================
st.markdown("---")
st.markdown('<div class="hero-title" style="font-size:1.5rem;">🗳️ Dinámica con el Público</div>', unsafe_allow_html=True)

opciones_voto = probabilidades["decision_zone"].tolist()

tab_voto, tab_resultados, tab_reveal = st.tabs([
    "📊 Votar", "📈 Resultados en vivo", "🎯 Revelar predicción"
])

with tab_voto:
    st.markdown(f"""<div class="vote-section">
        <div class="vote-title">¿Qué decidirá {player}?</div>
        <div class="vote-subtitle">
            {SPORT_ICONS.get(sport, '')} {sport.capitalize()} · {action_type} · {competition}<br>
            Minuto {match_time} · Presión <strong>{pressure_level.upper()}</strong> ({pressure_index:.2f})
        </div>
    </div>""", unsafe_allow_html=True)

    voto_elegido = st.radio(
        "Selecciona tu predicción:",
        opciones_voto,
        format_func=lambda x: f"🎯 {x.upper().replace('_', ' ')}",
        horizontal=True,
        key="radio_voto"
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("✅ Enviar mi voto", use_container_width=True, type="primary"):
            if voto_elegido not in st.session_state.votos:
                st.session_state.votos[voto_elegido] = 0
            st.session_state.votos[voto_elegido] += 1
            st.session_state.voto_enviado = True
            st.success(f"¡Voto registrado! Elegiste **{voto_elegido.upper().replace('_', ' ')}**")

    if st.session_state.votos:
        total_votos = sum(st.session_state.votos.values())
        st.markdown(f"<p style='text-align:center;color:#64748b;margin-top:12px;'>Total de votos: <strong style='color:#a78bfa'>{total_votos}</strong></p>", unsafe_allow_html=True)


with tab_resultados:
    if not st.session_state.votos:
        st.markdown("""<div class="info-box">
            Aún no hay votos. Pide al público que vote en la pestaña anterior.
        </div>""", unsafe_allow_html=True)
    else:
        total_votos = sum(st.session_state.votos.values())
        st.markdown(f'<div class="section-title">Votos del público ({total_votos} total)</div>', unsafe_allow_html=True)

        # Ordenar por votos
        votos_ordenados = sorted(st.session_state.votos.items(), key=lambda x: x[1], reverse=True)
        
        # Mostrar barras de progreso personalizadas
        for opcion, n in votos_ordenados:
            pct = n / total_votos if total_votos > 0 else 0
            bar_width = int(pct * 100)
            st.markdown(f"""<div class="vote-bar-container">
                <div class="vote-bar-label">
                    <span>{opcion.upper().replace('_', ' ')}</span>
                    <span>{n} votos ({pct:.0%})</span>
                </div>
                <div class="vote-bar-bg">
                    <div class="vote-bar-fill" style="width:{bar_width}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Gráfico de barras
        st.markdown("<br>", unsafe_allow_html=True)
        votos_df = pd.DataFrame(votos_ordenados, columns=["Decisión", "Votos"])
        votos_df["Porcentaje"] = (votos_df["Votos"] / total_votos * 100).round(1)
        
        fig_votos = go.Figure(go.Bar(
            x=votos_df["Decisión"].str.upper().str.replace("_", " "),
            y=votos_df["Porcentaje"],
            marker_color=["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd", "#ddd6fe"][:len(votos_df)],
            text=[f"{p:.1f}%" for p in votos_df["Porcentaje"]],
            textposition="outside",
            textfont=dict(color="#e2e8f0", family="DM Mono"),
        ))
        fig_votos.update_layout(
            height=280, margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickfont=dict(color="#e2e8f0", size=12, family="DM Mono")),
            yaxis=dict(tickfont=dict(color="#64748b"), gridcolor="rgba(42,42,58,0.8)",
                       title="% votos"),
            font=dict(family="Space Grotesk", color="#e2e8f0"),
        )
        st.plotly_chart(fig_votos, use_container_width=True, key="chart_votos")

        col_r1, col_r2 = st.columns(2)
        voto_mayoritario = max(st.session_state.votos, key=st.session_state.votos.get)
        pct_mayoritario = st.session_state.votos[voto_mayoritario] / total_votos
        
        col_r1.metric("🏆 Voto mayoritario", voto_mayoritario.upper().replace("_", " "),
                      f"{pct_mayoritario:.0%} del público")
        col_r2.metric("🤖 Predicción modelo", decision_top.upper().replace("_", " "),
                      f"Confianza: {prob_top:.0%}")


with tab_reveal:
    st.markdown('<div class="section-title">Momento del reveal</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
        Usa esta sección al final: <strong>primero</strong> recoge los votos del público, 
        <strong>luego</strong> revela la predicción del modelo y compara.
    </div>""", unsafe_allow_html=True)

    col_reveal1, col_reveal2 = st.columns(2)
    
    with col_reveal1:
        if st.button("🎭 Revelar predicción del modelo", use_container_width=True, type="primary"):
            st.session_state.mostrar_resultado = True

    with col_reveal2:
        if st.button("🔄 Resetear todos los votos", use_container_width=True):
            st.session_state.votos = {}
            st.session_state.voto_enviado = False
            st.session_state.mostrar_resultado = False
            st.rerun()

    if st.session_state.mostrar_resultado:
        st.markdown(f"""<div class="reveal-box">
            <div style="font-size:0.75rem;color:#6366f1;text-transform:uppercase;letter-spacing:2px;">
                El modelo predice</div>
            <div class="result-decision" style="font-size:2.8rem;margin:10px 0;">
                {decision_top.upper().replace('_', ' ')}</div>
            <div style="color:#64748b;font-size:0.9rem;">
                Probabilidad asignada: <strong style="color:#a78bfa">{prob_top:.1%}</strong>
            </div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.votos:
            total_votos = sum(st.session_state.votos.values())
            voto_mayoritario = max(st.session_state.votos, key=st.session_state.votos.get)
            pct_mayoritario = st.session_state.votos[voto_mayoritario] / total_votos

            st.markdown("<br>", unsafe_allow_html=True)
            
            if voto_mayoritario == decision_top:
                st.success(f"✅ ¡El público y el modelo coinciden! Ambos eligieron **{decision_top.upper().replace('_', ' ')}** ({pct_mayoritario:.0%} del público)")
            else:
                st.warning(f"""⚡ El público eligió **{voto_mayoritario.upper().replace('_', ' ')}** ({pct_mayoritario:.0%} del voto),
                    pero el modelo predice **{decision_top.upper().replace('_', ' ')}** con {prob_top:.1%} de probabilidad.
                    ¿Quién tiene razón?""")

            # Comparativa visual
            datos_comp = []
            for opcion in opciones_voto:
                votos_opc = st.session_state.votos.get(opcion, 0)
                pct_votos = votos_opc / total_votos if total_votos > 0 else 0
                prob_modelo = probabilidades[probabilidades["decision_zone"] == opcion]["probability"].values
                pct_modelo = float(prob_modelo[0]) if len(prob_modelo) > 0 else 0
                datos_comp.append({"Decisión": opcion.upper().replace("_", " "),
                                   "% Público": pct_votos * 100,
                                   "% Modelo": pct_modelo * 100})
            
            df_comp = pd.DataFrame(datos_comp).sort_values("% Modelo", ascending=False)
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name="Público", x=df_comp["Decisión"], y=df_comp["% Público"],
                marker_color="#f59e0b", opacity=0.85))
            fig_comp.add_trace(go.Bar(
                name="Modelo", x=df_comp["Decisión"], y=df_comp["% Modelo"],
                marker_color="#6366f1", opacity=0.85))
            fig_comp.update_layout(
                barmode="group", height=300,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="#e2e8f0")),
                xaxis=dict(tickfont=dict(color="#e2e8f0", size=11, family="DM Mono")),
                yaxis=dict(tickfont=dict(color="#64748b"), gridcolor="rgba(42,42,58,0.5)",
                           title="Porcentaje (%)"),
                font=dict(family="Space Grotesk", color="#e2e8f0"),
                title=dict(text="Público vs. Modelo", font=dict(color="#a78bfa"))
            )
            st.plotly_chart(fig_comp, use_container_width=True, key="comp_chart")

# ============================================================
# Análisis del dataset
# ============================================================
st.markdown("---")
st.markdown('<div class="hero-title" style="font-size:1.5rem;">📊 Análisis del Dataset</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#64748b;font-size:0.88rem;margin-bottom:16px;'>Exploración del conjunto de datos sintético multideporte.</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset", "🔥 Presión", "📊 Decisiones", "🤖 Modelo"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total eventos", df_preparado.shape[0])
    col2.metric("Deportes", df_preparado["sport"].nunique())
    col3.metric("Decisiones distintas", df_preparado["decision_zone"].nunique())
    col4.metric("Accuracy media", f"{resultados['accuracy']:.2%}")
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df_preparado.head(50), use_container_width=True, hide_index=True)

with tab2:
    tabla_pm = presion_media_por_deporte(df_preparado)
    fig_p = go.Figure(go.Bar(
        x=tabla_pm.index,
        y=tabla_pm.values,
        marker=dict(color=[SPORT_COLORS.get(s, "#6366f1") for s in tabla_pm.index],
                    line=dict(width=0)),
        text=[f"{v:.3f}" for v in tabla_pm.values],
        textposition="outside",
        textfont=dict(color="#e2e8f0", family="DM Mono"),
    ))
    fig_p.update_layout(
        title="Presión media por deporte",
        height=360, margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="#e2e8f0", size=12)),
        yaxis=dict(tickfont=dict(color="#64748b"), gridcolor="rgba(42,42,58,0.8)"),
        font=dict(family="Space Grotesk", color="#e2e8f0"),
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
    st.markdown('<div class="section-title">Rendimiento por deporte</div>', unsafe_allow_html=True)
    acc_data = [(sport, modelos[sport]["accuracy"]) for sport in modelos]
    acc_df = pd.DataFrame(acc_data, columns=["Deporte", "Accuracy"]).sort_values("Accuracy", ascending=False)
    
    for _, row in acc_df.iterrows():
        col_s, col_a = st.columns([2, 1])
        with col_s:
            st.write(f"{SPORT_ICONS.get(row['Deporte'], '')} {row['Deporte'].capitalize()}")
        with col_a:
            st.write(f"`{row['Accuracy']:.4f}`")
    
    st.markdown("---")
    sport_detail = st.selectbox("Ver detalle de deporte:", list(modelos.keys()),
        format_func=lambda s: f"{SPORT_ICONS.get(s,'')} {s.capitalize()}")
    
    if sport_detail in modelos:
        info = modelos[sport_detail]
        st.metric("Accuracy", f"{info['accuracy']:.4f}")
        st.write("Clases del modelo:", info["classes"])
        st.write("**Reporte de clasificación:**")
        st.text(info["classification_report"])

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("""<div style="text-align:center;color:#4a4a5a;font-size:0.8rem;padding:20px 0;font-family:'DM Mono',monospace;">
    Los datos usados son <strong style="color:#6366f1">sintéticos</strong>. 
    Aplicación demostrativa del pipeline de ML en deportes.<br>
    Predicción de Trayectoria en Deportes · 2025
</div>""", unsafe_allow_html=True)
