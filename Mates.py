"""
mathematical_core.py

Funciones sencillas para calcular la presión de un lanzamiento y preparar
un primer modelo de predicción.

La idea del proyecto es predecir hacia dónde puede tirar un jugador en un
momento de presión, usando variables como:

- minuto del partido
- diferencia de goles
- importancia de la competición
- pierna buena del jugador
- jugador
- portero

Este archivo no pretende ser perfecto. Es una primera versión simple y entendible.
"""

import pandas as pd
import numpy as np


# ============================================================
# 1. Funciones para calcular presión
# ============================================================


def presion_por_minuto(minuto):
    """
    Devuelve una presión entre 0 y 1 según el minuto del partido.

    Cuanto más tarde ocurre el penalti, más presión.
    """
    if pd.isna(minuto):
        return 0

    minuto = float(minuto)

    if minuto < 0:
        return 0

    if minuto > 90:
        return 1

    return minuto / 90



def presion_por_marcador(diferencia_goles):
    """
    Devuelve una presión entre 0 y 1 según el marcador.

    diferencia_goles se mide desde el punto de vista del equipo que lanza.

    Ejemplos:
    - Si van empatados: diferencia = 0 -> mucha presión
    - Si gana o pierde de muchos goles: menos presión
    """
    if pd.isna(diferencia_goles):
        return 0.5

    diferencia_goles = abs(float(diferencia_goles))

    return 1 / (1 + diferencia_goles)



def presion_por_competicion(competicion):
    """
    Asigna una presión aproximada según la importancia de la competición.
    """
    if pd.isna(competicion):
        return 0.5

    competicion = str(competicion).lower()

    if "champions" in competicion:
        return 0.9
    elif "mundial" in competicion or "world cup" in competicion:
        return 1.0
    elif "euro" in competicion:
        return 0.9
    elif "liga" in competicion or "premier" in competicion:
        return 0.6
    elif "copa" in competicion:
        return 0.7
    elif "amistoso" in competicion or "friendly" in competicion:
        return 0.2
    else:
        return 0.5



def calcular_presion_total(minuto, diferencia_goles, competicion):
    """
    Calcula una presión total entre 0 y 1.

    Fórmula:
        presión = 0.4 * presión_minuto
                + 0.4 * presión_marcador
                + 0.2 * presión_competición
    """
    p_minuto = presion_por_minuto(minuto)
    p_marcador = presion_por_marcador(diferencia_goles)
    p_competicion = presion_por_competicion(competicion)

    presion = 0.4 * p_minuto + 0.4 * p_marcador + 0.2 * p_competicion

    return round(presion, 4)



def añadir_presion_dataframe(df):
    """
    Añade una columna llamada pressure_index al dataframe.

    El dataframe necesita tener estas columnas:
    - minute
    - score_difference
    - competition
    """
    df = df.copy()

    df["pressure_index"] = df.apply(
        lambda fila: calcular_presion_total(
            fila["minute"],
            fila["score_difference"],
            fila["competition"]
        ),
        axis=1
    )

    return df


# ============================================================
# 2. Funciones para clasificar la presión
# ============================================================


def clasificar_presion(presion):
    """
    Convierte la presión numérica en una categoría.
    """
    if presion < 0.35:
        return "baja"
    elif presion < 0.70:
        return "media"
    else:
        return "alta"



def añadir_tipo_presion(df):
    """
    Añade una columna pressure_level con los valores:
    - baja
    - media
    - alta
    """
    df = df.copy()

    df["pressure_level"] = df["pressure_index"].apply(clasificar_presion)

    return df


# ============================================================
# 3. Funciones para estudiar patrones
# ============================================================


def tabla_tiros_por_presion(df):
    """
    Calcula la proporción de tiros a izquierda, centro y derecha
    según el nivel de presión.

    El dataframe debe tener:
    - pressure_level
    - shot_zone
    """
    tabla = pd.crosstab(
        df["pressure_level"],
        df["shot_zone"],
        normalize="index"
    )

    return tabla



def tabla_tiros_por_pierna(df):
    """
    Calcula la proporción de tiros según la pierna buena del jugador.

    El dataframe debe tener:
    - strong_foot
    - shot_zone
    """
    tabla = pd.crosstab(
        df["strong_foot"],
        df["shot_zone"],
        normalize="index"
    )

    return tabla



def tabla_tiros_por_pierna_y_presion(df):
    """
    Calcula la distribución de tiros según pierna buena y presión.
    """
    tabla = pd.crosstab(
        [df["strong_foot"], df["pressure_level"]],
        df["shot_zone"],
        normalize="index"
    )

    return tabla


# ============================================================
# 4. Función sencilla para preparar el dataframe
# ============================================================


def preparar_dataframe(df):
    """
    Prepara el dataframe para el modelo.

    Hace tres cosas:
    1. Añade pressure_index
    2. Añade pressure_level
    3. Elimina filas sin zona de tiro
    """
    df = df.copy()

    df = añadir_presion_dataframe(df)
    df = añadir_tipo_presion(df)

    df = df.dropna(subset=["shot_zone"])

    return df


# ============================================================
# 5. Ejemplo de uso
# ============================================================

if __name__ == "__main__":

    datos = {
        "minute": [10, 45, 80, 90, 30],
        "score_difference": [2, 0, -1, 0, 3],
        "competition": ["La Liga", "Champions League", "Premier League", "Champions League", "Friendly"],
        "player": ["Jugador A", "Jugador B", "Jugador C", "Jugador D", "Jugador E"],
        "goalkeeper": ["Portero A", "Portero B", "Portero C", "Portero D", "Portero E"],
        "strong_foot": ["right", "left", "right", "right", "left"],
        "shot_zone": ["left", "center", "right", "center", "left"]
    }

    df = pd.DataFrame(datos)

    df = preparar_dataframe(df)

    print(df)
    print("\nTiros por presión:")
    print(tabla_tiros_por_presion(df))

    print("\nTiros por pierna:")
    print(tabla_tiros_por_pierna(df))

    print("\nTiros por pierna y presión:")
    print(tabla_tiros_por_pierna_y_presion(df))
