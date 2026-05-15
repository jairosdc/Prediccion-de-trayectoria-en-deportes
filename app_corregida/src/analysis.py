"""
analysis.py

Funciones sencillas para analizar el dataset.

Sirve para sacar tablas útiles para la presentación:
- decisiones por deporte
- decisiones por presión
- decisiones por lado dominante
- presión media por deporte
"""

import pandas as pd


# ============================================================
# 1. Tabla de decisiones por deporte
# ============================================================

def decisiones_por_deporte(df):
    """
    Calcula la distribución de decision_zone para cada deporte.
    """

    tabla = pd.crosstab(
        df["sport"],
        df["decision_zone"],
        normalize="index"
    )

    return tabla


# ============================================================
# 2. Tabla de decisiones por presión
# ============================================================

def decisiones_por_presion(df):
    """
    Calcula la distribución de decisiones según presión baja/media/alta.
    """

    tabla = pd.crosstab(
        df["pressure_level"],
        df["decision_zone"],
        normalize="index"
    )

    return tabla


# ============================================================
# 3. Tabla de decisiones por deporte y presión
# ============================================================

def decisiones_por_deporte_y_presion(df):
    """
    Calcula la distribución de decisiones según deporte y nivel de presión.
    """

    tabla = pd.crosstab(
        [df["sport"], df["pressure_level"]],
        df["decision_zone"],
        normalize="index"
    )

    return tabla


# ============================================================
# 4. Tabla de decisiones por lado dominante
# ============================================================

def decisiones_por_lado_dominante(df):
    """
    Calcula la distribución de decisiones según el lado dominante del jugador.
    """

    tabla = pd.crosstab(
        df["dominant_side"],
        df["decision_zone"],
        normalize="index"
    )

    return tabla


# ============================================================
# 5. Presión media por deporte
# ============================================================

def presion_media_por_deporte(df):
    """
    Calcula la presión media de cada deporte.
    """

    tabla = (
        df.groupby("sport")["pressure_index"]
        .mean()
        .sort_values(ascending=False)
    )

    return tabla


# ============================================================
# 6. Resumen general
# ============================================================

def resumen_analisis(df):
    """
    Imprime varias tablas de análisis.
    """

    print("Decisiones por deporte")
    print("----------------------")
    print(decisiones_por_deporte(df))

    print("\nDecisiones por presión")
    print("----------------------")
    print(decisiones_por_presion(df))

    print("\nDecisiones por lado dominante")
    print("-----------------------------")
    print(decisiones_por_lado_dominante(df))

    print("\nPresión media por deporte")
    print("-------------------------")
    print(presion_media_por_deporte(df))


# ============================================================
# 7. Guardar tablas
# ============================================================

def guardar_tabla(tabla, ruta):
    """
    Guarda una tabla en formato CSV.
    """

    tabla.to_csv(ruta)

    print(f"Tabla guardada en: {ruta}")


# ============================================================
# 8. Prueba rápida
# ============================================================

if __name__ == "__main__":

    from preprocessing import cargar_datos, preparar_dataframe

    ruta = "data/synthetic_multisport_decisions.csv"

    df = cargar_datos(ruta)
    df = preparar_dataframe(df)

    resumen_analisis(df)