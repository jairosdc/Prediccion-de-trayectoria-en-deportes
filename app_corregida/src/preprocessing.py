"""
preprocessing.py

Funciones sencillas para cargar, validar y preparar el dataframe antes de
entrenar el modelo.

La idea es que cualquier dataset sintético o real tenga una estructura común.
El modelo intentará predecir:

    decision_zone

usando variables del contexto del evento deportivo.
"""

import pandas as pd


try:
    from src.pressure import añadir_presion, clasificar_presion
except ModuleNotFoundError:
    from pressure import añadir_presion, clasificar_presion


# ============================================================
# 1. Columnas del proyecto
# ============================================================

COLUMNAS_OBLIGATORIAS = [
    "sport",
    "action_type",
    "competition",
    "match_time",
    "score_difference",
    "is_home_player",
    "player",
    "opponent",
    "dominant_side",
    "fatigue_level",
    "competition_importance",
    "decision_zone",
]


COLUMNAS_MODELO = [
    "sport",
    "action_type",
    "competition",
    "match_time",
    "score_difference",
    "is_home_player",
    "player",
    "opponent",
    "dominant_side",
    "fatigue_level",
    "competition_importance",
    "pressure_index",
    "pressure_level",
]


COLUMNA_OBJETIVO = "decision_zone"


# ============================================================
# 2. Cargar datos
# ============================================================

def cargar_datos(ruta):
    """
    Carga un archivo CSV y devuelve un dataframe.
    """

    df = pd.read_csv(ruta)

    return df


# ============================================================
# 3. Validar columnas
# ============================================================

def validar_columnas(df):
    """
    Comprueba que el dataframe tenga las columnas mínimas necesarias.
    """

    columnas_faltantes = []

    for columna in COLUMNAS_OBLIGATORIAS:
        if columna not in df.columns:
            columnas_faltantes.append(columna)

    if len(columnas_faltantes) > 0:
        raise ValueError(f"Faltan columnas obligatorias: {columnas_faltantes}")

    return True


# ============================================================
# 4. Limpieza básica
# ============================================================

def limpiar_datos(df):
    """
    Limpia el dataframe de forma sencilla.

    Hace:
    - copia del dataframe
    - elimina duplicados
    - normaliza textos
    - convierte columnas numéricas
    - elimina filas sin variable objetivo
    """

    df = df.copy()

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Quitar filas sin decisión final
    df = df.dropna(subset=[COLUMNA_OBJETIVO])

    # Columnas numéricas
    columnas_numericas = [
        "match_time",
        "score_difference",
        "is_home_player",
        "fatigue_level",
        "competition_importance",
    ]

    for columna in columnas_numericas:
        if columna in df.columns:
            df[columna] = pd.to_numeric(df[columna], errors="coerce")

    # Rellenar numéricas si hay algún nulo
    if "match_time" in df.columns:
        df["match_time"] = df["match_time"].fillna(df["match_time"].median())

    if "score_difference" in df.columns:
        df["score_difference"] = df["score_difference"].fillna(0)

    if "is_home_player" in df.columns:
        df["is_home_player"] = df["is_home_player"].fillna(0)

    if "fatigue_level" in df.columns:
        df["fatigue_level"] = df["fatigue_level"].fillna(0.5)

    if "competition_importance" in df.columns:
        df["competition_importance"] = df["competition_importance"].fillna(0.5)

    # Columnas categóricas
    columnas_categoricas = [
        "sport",
        "action_type",
        "competition",
        "player",
        "opponent",
        "dominant_side",
        "decision_zone",
        "action_result",
        "pressure_level",
    ]

    for columna in columnas_categoricas:
        if columna in df.columns:
            df[columna] = (
                df[columna]
                .astype(str)
                .str.strip()
                .str.lower()
            )

    return df


# ============================================================
# 5. Preparar presión
# ============================================================

def preparar_presion(df, recalcular_presion=False):
    """
    Añade o recalcula las columnas de presión.

    Si el dataframe ya tiene pressure_index y pressure_level, las respeta.
    Si faltan, las calcula.
    """

    df = df.copy()

    if recalcular_presion:
        df = añadir_presion(df)
        return df

    if "pressure_index" not in df.columns:
        df = añadir_presion(df)
        return df

    df["pressure_index"] = pd.to_numeric(
        df["pressure_index"],
        errors="coerce"
    ).fillna(0.5)

    if "pressure_level" not in df.columns:
        df["pressure_level"] = df["pressure_index"].apply(clasificar_presion)
    else:
        df["pressure_level"] = (
            df["pressure_level"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return df


# ============================================================
# 6. Preparar dataframe completo
# ============================================================

def preparar_dataframe(df, recalcular_presion=False):
    """
    Función principal de preprocessing.

    Hace:
    1. Valida columnas mínimas.
    2. Limpia datos.
    3. Añade o recalcula presión.
    4. Elimina filas incompletas para el modelo.
    """

    validar_columnas(df)

    df = limpiar_datos(df)

    df = preparar_presion(df, recalcular_presion=recalcular_presion)

    # Asegurar que las columnas del modelo existen
    for columna in COLUMNAS_MODELO:
        if columna not in df.columns:
            raise ValueError(f"Falta la columna necesaria para el modelo: {columna}")

    # Eliminar filas que sigan teniendo nulos en variables importantes
    df = df.dropna(subset=COLUMNAS_MODELO + [COLUMNA_OBJETIVO])

    return df


# ============================================================
# 7. Separar variables para el modelo
# ============================================================

def separar_X_y(df):
    """
    Separa el dataframe en variables predictoras X y variable objetivo y.
    """

    X = df[COLUMNAS_MODELO]
    y = df[COLUMNA_OBJETIVO]

    return X, y


# ============================================================
# 8. Resumen rápido
# ============================================================

def resumen_dataset(df):
    """
    Imprime un resumen sencillo del dataset.
    """

    print("Resumen del dataset")
    print("-------------------")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")

    print("\nColumnas:")
    print(list(df.columns))

    if "sport" in df.columns:
        print("\nEventos por deporte:")
        print(df["sport"].value_counts())

    if "pressure_level" in df.columns:
        print("\nDistribución de presión:")
        print(df["pressure_level"].value_counts(normalize=True))

    if "decision_zone" in df.columns:
        print("\nDistribución de decisiones:")
        print(df["decision_zone"].value_counts(normalize=True))


# ============================================================
# 9. Prueba rápida
# ============================================================

if __name__ == "__main__":

    ruta = "data/synthetic_multisport_decisions.csv"

    df = cargar_datos(ruta)

    df = preparar_dataframe(df)

    resumen_dataset(df)

    X, y = separar_X_y(df)

    print("\nVariables predictoras:")
    print(X.head())

    print("\nVariable objetivo:")
    print(y.head())