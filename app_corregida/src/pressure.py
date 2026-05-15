"""
pressure.py

Funciones para calcular la presión de una acción deportiva.
CORREGIDO: calcular_presion_temporal ahora acepta match_time hasta 120.
"""

import pandas as pd


def calcular_presion_temporal(match_time):
    """
    CORREGIDO: acepta match_time de 0 a 120 (partidos de fútbol, etc.)
    """
    if pd.isna(match_time):
        return 0.5
    match_time = float(match_time)
    if match_time <= 0:
        return 0.0
    if match_time >= 120:
        return 1.0
    return match_time / 120


def calcular_presion_marcador(score_difference):
    if pd.isna(score_difference):
        return 0.5
    score_difference = abs(float(score_difference))
    return 1 / (1 + score_difference)


def clasificar_presion(pressure_index):
    if pressure_index < 0.35:
        return "baja"
    elif pressure_index < 0.70:
        return "media"
    else:
        return "alta"


def calcular_pressure_index(
    match_time, score_difference, competition_importance,
    fatigue_level, critical_moment=0
):
    T = calcular_presion_temporal(match_time)
    S = calcular_presion_marcador(score_difference)
    C = float(competition_importance) if not pd.isna(competition_importance) else 0.5
    F = float(fatigue_level) if not pd.isna(fatigue_level) else 0.5
    M = float(critical_moment) if not pd.isna(critical_moment) else 0

    C = max(0, min(1, C))
    F = max(0, min(1, F))
    M = max(0, min(1, M))

    pressure_index = 0.30 * T + 0.25 * S + 0.20 * C + 0.15 * F + 0.10 * M
    return round(max(0, min(1, pressure_index)), 4)


def añadir_presion(df):
    df = df.copy()
    if "critical_moment" not in df.columns:
        df["critical_moment"] = 0

    df["pressure_index"] = df.apply(
        lambda row: calcular_pressure_index(
            row["match_time"], row["score_difference"],
            row["competition_importance"], row["fatigue_level"],
            row["critical_moment"]
        ), axis=1
    )
    df["pressure_level"] = df["pressure_index"].apply(clasificar_presion)
    return df


anadir_presion = añadir_presion
