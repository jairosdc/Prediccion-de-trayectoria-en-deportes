"""
app.py

Interfaz web con Streamlit para el proyecto:

Predicción de trayectoria en deportes

La app permite:
1. Cargar el dataset sintético multideporte.
2. Entrenar el modelo automáticamente.
3. Elegir un deporte y una situación concreta.
4. Calcular la presión del momento.
5. Predecir las probabilidades de decisión del jugador.

Para ejecutar:

    streamlit run app.py
"""

import os
import pandas as pd
import streamlit as st

from src.preprocessing import (
    cargar_datos,
    preparar_dataframe,
    COLUMNAS_MODELO
)

from src.train_model import (
    entrenar_y_evaluar,
    predecir_probabilidades
)

from src.pressure import (
    calcular_pressure_index,
    clasificar_presion
)

from src.analysis import (
    decisiones_por_deporte,
    decisiones_por_presion,
    decisiones_por_lado_dominante,
    presion_media_por_deporte
)


# ============================================================
# 1. Configuración general
# ============================================================

RUTA_DATASET = "data/synthetic_multisport_decisions.csv"


st.set_page_config(
    page_title="Predicción de trayectoria en deportes",
    page_icon=None,
    layout="wide"
)


# ============================================================
# 2. Estilo sencillo
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-top: 0px;
        margin-bottom: 25px;
    }

    .info-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #f5f5f5;
        border: 1px solid #dddddd;
        margin-bottom: 20px;
    }

    .metric-box {
        padding: 14px;
        border-radius: 10px;
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. Carga y entrenamiento cacheados
# ============================================================

@st.cache_data
def cargar_y_preparar_dataset():
    """
    Carga y prepara el dataset.
    """

    if not os.path.exists(RUTA_DATASET):
        return None

    df = cargar_datos(RUTA_DATASET)
    df_preparado = preparar_dataframe(df)

    return df, df_preparado


@st.cache_resource
def entrenar_modelo_cacheado():
    """
    Entrena el modelo una sola vez para que la app vaya más fluida.
    """

    df = cargar_datos(RUTA_DATASET)
    modelo, resultados, df_preparado = entrenar_y_evaluar(df)

    return modelo, resultados, df_preparado


# ============================================================
# 4. Funciones auxiliares de interfaz
# ============================================================

def obtener_opciones(df, columna):
    """
    Devuelve valores únicos ordenados de una columna.
    """

    if columna not in df.columns:
        return []

    valores = df[columna].dropna().unique().tolist()
    valores = sorted(valores)

    return valores


def filtrar_dataframe(df, sport=None, action_type=None, competition=None):
    """
    Filtra el dataframe según las opciones elegidas.
    """

    filtrado = df.copy()

    if sport is not None:
        filtrado = filtrado[filtrado["sport"] == sport]

    if action_type is not None:
        filtrado = filtrado[filtrado["action_type"] == action_type]

    if competition is not None:
        filtrado = filtrado[filtrado["competition"] == competition]

    return filtrado


def crear_caso_prediccion(
    sport,
    action_type,
    competition,
    match_time,
    score_difference,
    is_home_player,
    player,
    opponent,
    dominant_side,
    fatigue_level,
    competition_importance,
    pressure_index,
    pressure_level
):
    """
    Crea un dataframe con una sola fila para predecir.
    """

    caso = pd.DataFrame([{
        "sport": sport,
        "action_type": action_type,
        "competition": competition,
        "match_time": match_time,
        "score_difference": score_difference,
        "is_home_player": is_home_player,
        "player": player,
        "opponent": opponent,
        "dominant_side": dominant_side,
        "fatigue_level": fatigue_level,
        "competition_importance": competition_importance,
        "pressure_index": pressure_index,
        "pressure_level": pressure_level,
    }])

    caso = caso[COLUMNAS_MODELO]

    return caso


def texto_deporte(sport):
    """
    Devuelve una descripción corta según el deporte.
    """

    textos = {
        "football": "Penalti o acción de lanzamiento en fútbol.",
        "tennis": "Saque o punto importante en tenis.",
        "basketball": "Tiro o decisión ofensiva en baloncesto.",
        "handball": "Lanzamiento de 7 metros o acción directa en balonmano.",
        "hockey": "Shootout o mano a mano contra el portero.",
        "esports": "Duelo 1v1 con decisión táctica bajo presión."
    }

    return textos.get(sport, "Situación deportiva individual bajo presión.")


def nombre_variable_decision(sport):
    """
    Explica qué representa decision_zone según el deporte.
    """

    textos = {
        "football": "zona del lanzamiento",
        "tennis": "zona del saque",
        "basketball": "tipo o zona de tiro",
        "handball": "zona del lanzamiento",
        "hockey": "tipo de finalización",
        "esports": "decisión táctica",
    }

    return textos.get(sport, "decisión tomada")


# ============================================================
# 5. Cabecera
# ============================================================

st.markdown(
    '<div class="main-title">Predicción de trayectoria en deportes</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Modelo interactivo para estimar decisiones deportivas bajo presión en situaciones 1v1.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    Esta aplicación usa un dataset sintético multideporte. 
    El objetivo no es afirmar patrones reales definitivos, sino demostrar que el framework funciona:
    carga datos, calcula presión, entrena un modelo y devuelve probabilidades de decisión.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 6. Comprobar dataset
# ============================================================

if not os.path.exists(RUTA_DATASET):
    st.error(
        "No se ha encontrado el dataset. "
        "Coloca el archivo synthetic_multisport_decisions.csv dentro de la carpeta data/."
    )
    st.stop()


df_original, df_preparado = cargar_y_preparar_dataset()
modelo, resultados, df_modelo = entrenar_modelo_cacheado()


# ============================================================
# 7. Sidebar
# ============================================================

st.sidebar.title("Panel de simulación")

st.sidebar.markdown(
    """
    Elige un deporte y modifica la situación.
    El modelo calculará las probabilidades de cada decisión posible.
    """
)

sports = obtener_opciones(df_preparado, "sport")

sport = st.sidebar.selectbox(
    "Deporte",
    sports
)

df_sport = filtrar_dataframe(df_preparado, sport=sport)

action_types = obtener_opciones(df_sport, "action_type")

action_type = st.sidebar.selectbox(
    "Tipo de acción",
    action_types
)

df_action = filtrar_dataframe(
    df_preparado,
    sport=sport,
    action_type=action_type
)

competitions = obtener_opciones(df_action, "competition")

competition = st.sidebar.selectbox(
    "Competición",
    competitions
)

df_contexto = filtrar_dataframe(
    df_preparado,
    sport=sport,
    action_type=action_type,
    competition=competition
)


# ============================================================
# 8. Layout principal
# ============================================================

col_info, col_modelo = st.columns([1.1, 1])


with col_info:
    st.subheader("Situación elegida")

    st.write(texto_deporte(sport))

    st.write(
        f"El modelo va a estimar la **{nombre_variable_decision(sport)}** "
        "más probable para el contexto seleccionado."
    )

    st.markdown("#### Contexto del evento")

    col_a, col_b = st.columns(2)

    with col_a:
        match_time = st.slider(
            "Momento del partido / evento",
            min_value=0,
            max_value=120,
            value=80,
            help="Se interpreta como el avance del evento. Valores altos significan más cercanía al final."
        )

        score_difference = st.slider(
            "Diferencia de marcador",
            min_value=-5,
            max_value=5,
            value=0,
            help="Diferencia desde el punto de vista del jugador o equipo que realiza la acción."
        )

        is_home_player = st.selectbox(
            "¿Juega como local?",
            options=[1, 0],
            format_func=lambda x: "Sí" if x == 1 else "No"
        )

    with col_b:
        fatigue_level = st.slider(
            "Nivel de fatiga",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05
        )

        competition_importance = st.slider(
            "Importancia de la competición",
            min_value=0.0,
            max_value=1.0,
            value=float(round(df_contexto["competition_importance"].mean(), 2)),
            step=0.05
        )

        critical_moment = st.selectbox(
            "¿Momento crítico?",
            options=[0, 1],
            format_func=lambda x: "Sí" if x == 1 else "No"
        )


with col_modelo:
    st.subheader("Jugador y rival")

    players = obtener_opciones(df_contexto, "player")
    opponents = obtener_opciones(df_contexto, "opponent")
    dominant_sides = obtener_opciones(df_contexto, "dominant_side")

    if len(players) == 0 or len(opponents) == 0:
        st.error("No hay suficientes datos para este filtro.")
        st.stop()

    player = st.selectbox(
        "Jugador",
        players
    )

    opponent = st.selectbox(
        "Rival / portero / oponente",
        opponents
    )

    dominant_side = st.selectbox(
        "Lado dominante",
        dominant_sides
    )

    pressure_index = calcular_pressure_index(
        match_time=match_time,
        score_difference=score_difference,
        competition_importance=competition_importance,
        fatigue_level=fatigue_level,
        critical_moment=critical_moment
    )

    pressure_level = clasificar_presion(pressure_index)

    st.markdown("#### Presión calculada")

    metric_1, metric_2 = st.columns(2)

    with metric_1:
        st.metric("Índice de presión", pressure_index)

    with metric_2:
        st.metric("Nivel", pressure_level)


# ============================================================
# 9. Predicción
# ============================================================

st.divider()

st.header("Predicción del modelo")

caso = crear_caso_prediccion(
    sport=sport,
    action_type=action_type,
    competition=competition,
    match_time=match_time,
    score_difference=score_difference,
    is_home_player=is_home_player,
    player=player,
    opponent=opponent,
    dominant_side=dominant_side,
    fatigue_level=fatigue_level,
    competition_importance=competition_importance,
    pressure_index=pressure_index,
    pressure_level=pressure_level
)

probabilidades = predecir_probabilidades(modelo, caso)

col_pred, col_chart = st.columns([1, 1.2])

with col_pred:
    st.subheader("Probabilidades estimadas")

    probabilidades_mostrar = probabilidades.copy()
    probabilidades_mostrar["probability"] = (
        probabilidades_mostrar["probability"] * 100
    ).round(2)

    probabilidades_mostrar = probabilidades_mostrar.rename(
        columns={
            "decision_zone": "decisión",
            "probability": "probabilidad (%)"
        }
    )

    st.dataframe(
        probabilidades_mostrar,
        use_container_width=True,
        hide_index=True
    )

    decision_mas_probable = probabilidades.iloc[0]["decision_zone"]
    probabilidad_mas_alta = probabilidades.iloc[0]["probability"]

    st.success(
        f"La decisión más probable es **{decision_mas_probable}**, "
        f"con una probabilidad aproximada de **{probabilidad_mas_alta:.2%}**."
    )

with col_chart:
    st.subheader("Gráfico de probabilidades")

    grafico = probabilidades.copy()
    grafico = grafico.set_index("decision_zone")

    st.bar_chart(grafico)


# ============================================================
# 10. Dinámica con el público
# ============================================================

st.divider()

st.header("Dinámica para la presentación")

st.write(
    "Antes de enseñar la predicción, se puede pedir al público que vote "
    "qué decisión cree que tomará el jugador."
)

opciones_decision = probabilidades["decision_zone"].tolist()

voto_publico = st.radio(
    "Voto del público",
    opciones_decision,
    horizontal=True
)

if st.button("Comparar voto del público con el modelo"):
    if voto_publico == decision_mas_probable:
        st.success("El público coincide con el modelo.")
    else:
        st.warning(
            f"El público eligió **{voto_publico}**, "
            f"pero el modelo considera más probable **{decision_mas_probable}**."
        )

    st.write("Tabla completa de probabilidades:")
    st.dataframe(
        probabilidades_mostrar,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 11. Análisis del dataset
# ============================================================

st.divider()

st.header("Análisis rápido del dataset")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Dataset",
        "Presión",
        "Decisiones",
        "Modelo"
    ]
)

with tab1:
    st.subheader("Vista general")

    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        st.metric("Eventos", df_preparado.shape[0])

    with col_d2:
        st.metric("Deportes", df_preparado["sport"].nunique())

    with col_d3:
        st.metric("Decisiones distintas", df_preparado["decision_zone"].nunique())

    st.dataframe(
        df_preparado.head(50),
        use_container_width=True
    )

with tab2:
    st.subheader("Presión media por deporte")

    tabla_presion_media = presion_media_por_deporte(df_preparado)

    st.dataframe(tabla_presion_media, use_container_width=True)
    st.bar_chart(tabla_presion_media)

    st.subheader("Distribución de niveles de presión")

    conteo_presion = df_preparado["pressure_level"].value_counts(normalize=True)
    st.dataframe(conteo_presion, use_container_width=True)
    st.bar_chart(conteo_presion)

with tab3:
    st.subheader("Decisiones por deporte")

    tabla_deporte = decisiones_por_deporte(df_preparado)
    st.dataframe(tabla_deporte, use_container_width=True)

    st.subheader("Decisiones por presión")

    tabla_presion = decisiones_por_presion(df_preparado)
    st.dataframe(tabla_presion, use_container_width=True)

    st.subheader("Decisiones por lado dominante")

    tabla_lado = decisiones_por_lado_dominante(df_preparado)
    st.dataframe(tabla_lado, use_container_width=True)

with tab4:
    st.subheader("Evaluación básica del modelo")

    st.metric("Accuracy", round(resultados["accuracy"], 4))

    st.write("Clases del modelo:")
    st.write(list(resultados["classes"]))

    st.write("Matriz de confusión:")
    matriz = pd.DataFrame(
        resultados["confusion_matrix"],
        index=resultados["classes"],
        columns=resultados["classes"]
    )

    st.dataframe(matriz, use_container_width=True)

    st.write("Reporte de clasificación:")
    st.text(resultados["classification_report"])


# ============================================================
# 12. Aviso final
# ============================================================

st.divider()

st.caption(
    "Aviso: los datos usados son sintéticos. La aplicación demuestra el funcionamiento "
    "del pipeline, pero no permite extraer conclusiones empíricas reales sobre deportistas concretos."
)