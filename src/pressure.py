"""
pressure.py

Funciones para calcular la presión de una acción deportiva.

La presión se calcula usando:
- momento del partido
- diferencia de marcador
- importancia de la competición
- fatiga
- si es un momento crítico

La fórmula principal es:

P = 0.30*T + 0.25*S + 0.20*C + 0.15*F + 0.10*M

donde:
T = presión temporal
S = presión por marcador
C = importancia competitiva
F = fatiga
M = momento crítico
"""

import pandas as pd


def calcular_presion_temporal(match_time):
    """
    Calcula presión temporal entre 0 y 1.

    match_time se interpreta como porcentaje/momento del partido.
    Por ejemplo:
    - 0   -> inicio
    - 50  -> mitad
    - 100 -> final

    Cuanto más avanzado el evento, mayor presión.
    """

    if pd.isna(match_time):
        return 0.5

    match_time = float(match_time)

    if match_time < 0:
        return 0

    if match_time > 100:
        return 1

    return match_time / 100


def calcular_presion_marcador(score_difference):
    """
    Calcula presión según el marcador.

    score_difference es la diferencia de puntos/goles desde el punto de vista
    del jugador/equipo que realiza la acción.

    Si el marcador está igualado, hay más presión.

    Ejemplos:
    score_difference = 0  -> presión máxima
    score_difference = 1  -> presión alta
    score_difference = 3  -> presión menor
    """

    if pd.isna(score_difference):
        return 0.5

    score_difference = abs(float(score_difference))

    return 1 / (1 + score_difference)


def clasificar_presion(pressure_index):
    """
    Clasifica la presión en baja, media o alta.
    """

    if pressure_index < 0.35:
        return "baja"
    elif pressure_index < 0.70:
        return "media"
    else:
        return "alta"


def calcular_pressure_index(
    match_time,
    score_difference,
    competition_importance,
    fatigue_level,
    critical_moment=0
):
    """
    Calcula el índice de presión total.

    Parámetros:
    - match_time: momento del partido entre 0 y 100
    - score_difference: diferencia de marcador
    - competition_importance: importancia de la competición entre 0 y 1
    - fatigue_level: fatiga del jugador entre 0 y 1
    - critical_moment: 1 si es un momento crítico, 0 si no
    """

    T = calcular_presion_temporal(match_time)
    S = calcular_presion_marcador(score_difference)

    if pd.isna(competition_importance):
        C = 0.5
    else:
        C = float(competition_importance)

    if pd.isna(fatigue_level):
        F = 0.5
    else:
        F = float(fatigue_level)

    if pd.isna(critical_moment):
        M = 0
    else:
        M = float(critical_moment)

    # Nos aseguramos de que cada componente esté entre 0 y 1
    C = max(0, min(1, C))
    F = max(0, min(1, F))
    M = max(0, min(1, M))

    pressure_index = (
        0.30 * T +
        0.25 * S +
        0.20 * C +
        0.15 * F +
        0.10 * M
    )

    pressure_index = max(0, min(1, pressure_index))

    return round(pressure_index, 4)


def añadir_presion(df):
    """
    Añade dos columnas al dataframe:

    - pressure_index
    - pressure_level

    Si no existe critical_moment, se crea con valor 0.
    """

    df = df.copy()

    if "critical_moment" not in df.columns:
        df["critical_moment"] = 0

    df["pressure_index"] = df.apply(
        lambda row: calcular_pressure_index(
            row["match_time"],
            row["score_difference"],
            row["competition_importance"],
            row["fatigue_level"],
            row["critical_moment"]
        ),
        axis=1
    )

    df["pressure_level"] = df["pressure_index"].apply(clasificar_presion)

    return df


anadir_presion = añadir_presion