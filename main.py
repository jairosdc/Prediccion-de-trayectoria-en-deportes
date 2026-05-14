"""
main.py

Archivo principal del proyecto.

Este script hace el flujo completo:

1. Carga el dataset sintético.
2. Prepara los datos.
3. Calcula o revisa la presión.
4. Muestra análisis básicos.
5. Entrena un modelo de predicción.
6. Evalúa el modelo.
7. Hace una predicción de ejemplo.

Para ejecutarlo:

    python main.py
"""

import os
import pandas as pd

from src.preprocessing import (
    cargar_datos,
    preparar_dataframe,
    resumen_dataset,
    COLUMNAS_MODELO
)

from src.analysis import (
    decisiones_por_deporte,
    decisiones_por_presion,
    decisiones_por_lado_dominante,
    presion_media_por_deporte
)

from src.train_model import (
    entrenar_y_evaluar,
    predecir_probabilidades
)


# ============================================================
# 1. Configuración inicial
# ============================================================

RUTA_DATASET = "data/synthetic_multisport_decisions.csv"


# ============================================================
# 2. Comprobar que existe el dataset
# ============================================================

def comprobar_dataset(ruta):
    """
    Comprueba si existe el archivo CSV.
    """

    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se ha encontrado el dataset en: {ruta}\n\n"
            "Asegúrate de que el archivo esté en la carpeta data/ "
            "y se llame synthetic_multisport_decisions.csv"
        )


# ============================================================
# 3. Mostrar análisis básico
# ============================================================

def mostrar_analisis(df):
    """
    Muestra tablas sencillas para entender el dataset.
    """

    print("\n" + "=" * 60)
    print("ANÁLISIS BÁSICO DEL DATASET")
    print("=" * 60)

    print("\n1. Decisiones por deporte")
    print("-------------------------")
    print(decisiones_por_deporte(df))

    print("\n2. Decisiones por nivel de presión")
    print("----------------------------------")
    print(decisiones_por_presion(df))

    print("\n3. Decisiones por lado dominante")
    print("--------------------------------")
    print(decisiones_por_lado_dominante(df))

    print("\n4. Presión media por deporte")
    print("----------------------------")
    print(presion_media_por_deporte(df))


# ============================================================
# 4. Mostrar resultados del modelo
# ============================================================

def mostrar_resultados_modelo(resultados):
    """
    Imprime los resultados principales del modelo.
    """

    print("\n" + "=" * 60)
    print("RESULTADOS DEL MODELO")
    print("=" * 60)

    print("\nAccuracy:")
    print(round(resultados["accuracy"], 4))

    print("\nClases que predice el modelo:")
    print(resultados["classes"])

    print("\nMatriz de confusión:")
    print(resultados["confusion_matrix"])

    print("\nReporte de clasificación:")
    print(resultados["classification_report"])


# ============================================================
# 5. Crear un caso de ejemplo para predecir
# ============================================================

def crear_caso_ejemplo(df):
    """
    Crea un caso de ejemplo tomando una fila del dataframe.

    Esto sirve para comprobar que el modelo devuelve probabilidades.
    """

    caso = df[COLUMNAS_MODELO].sample(1, random_state=7)

    return caso


# ============================================================
# 6. Mostrar predicción de ejemplo
# ============================================================

def mostrar_prediccion_ejemplo(modelo, df):
    """
    Muestra una predicción de ejemplo.
    """

    print("\n" + "=" * 60)
    print("PREDICCIÓN DE EJEMPLO")
    print("=" * 60)

    caso = crear_caso_ejemplo(df)

    print("\nCaso usado para predecir:")
    print(caso)

    probabilidades = predecir_probabilidades(modelo, caso)

    print("\nProbabilidades estimadas:")
    print(probabilidades)

    decision_mas_probable = probabilidades.iloc[0]["decision_zone"]
    probabilidad = probabilidades.iloc[0]["probability"]

    print("\nDecisión más probable:")
    print(f"{decision_mas_probable} con probabilidad aproximada de {probabilidad:.2%}")


# ============================================================
# 7. Función principal
# ============================================================

def main():
    """
    Ejecuta todo el flujo del proyecto.
    """

    print("=" * 60)
    print("PREDICCIÓN DE TRAYECTORIA EN DEPORTES")
    print("=" * 60)

    print("\nCargando dataset...")

    comprobar_dataset(RUTA_DATASET)

    df = cargar_datos(RUTA_DATASET)

    print("Dataset cargado correctamente.")

    print("\nPreparando dataframe...")

    df_preparado = preparar_dataframe(df)

    print("Dataframe preparado correctamente.")

    print("\nResumen del dataset preparado:")
    resumen_dataset(df_preparado)

    mostrar_analisis(df_preparado)

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO DEL MODELO")
    print("=" * 60)

    print("\nEntrenando modelo...")

    modelo, resultados, df_modelo = entrenar_y_evaluar(df)

    print("Modelo entrenado correctamente.")

    mostrar_resultados_modelo(resultados)

    mostrar_prediccion_ejemplo(modelo, df_modelo)

    print("\n" + "=" * 60)
    print("EJECUCIÓN FINALIZADA")
    print("=" * 60)

    print(
        "\nEl proyecto ha cargado los datos, calculado la presión, "
        "entrenado el modelo y generado una predicción de ejemplo."
    )


# ============================================================
# 8. Ejecutar
# ============================================================

if __name__ == "__main__":
    main()