"""
train_model.py

Archivo para entrenar un modelo sencillo de predicción.

El modelo intenta predecir:

    decision_zone

a partir de variables del contexto deportivo.

Usamos Regresión Logística porque:
- es sencilla
- es rápida
- sirve para clasificación multiclase
- devuelve probabilidades
- es fácil de defender en una presentación
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


try:
    from src.preprocessing import preparar_dataframe, separar_X_y
except ModuleNotFoundError:
    from preprocessing import preparar_dataframe, separar_X_y


# ============================================================
# 1. Crear pipeline del modelo
# ============================================================

def crear_modelo():
    """
    Crea el pipeline completo:

    1. Codifica variables categóricas con OneHotEncoder.
    2. Escala variables numéricas con StandardScaler.
    3. Entrena una regresión logística multiclase.
    """

    columnas_categoricas = [
        "sport",
        "action_type",
        "competition",
        "player",
        "opponent",
        "dominant_side",
        "pressure_level",
    ]

    columnas_numericas = [
        "match_time",
        "score_difference",
        "is_home_player",
        "fatigue_level",
        "competition_importance",
        "pressure_index",
    ]

    preprocesado = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
            ("num", StandardScaler(), columnas_numericas),
        ]
    )

    modelo = Pipeline(
        steps=[
            ("preprocesado", preprocesado),
            ("modelo", LogisticRegression(max_iter=5000))
        ]
    )

    return modelo



# ============================================================
# 2. Dividir datos
# ============================================================

def dividir_datos(X, y, test_size=0.25, random_state=42):
    """
    Divide los datos en entrenamiento y prueba.

    Si alguna clase tiene muy pocos ejemplos, no usa stratify para evitar errores.
    """

    conteos = y.value_counts()

    if conteos.min() >= 2:
        stratify = y
    else:
        stratify = None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )


# ============================================================
# 3. Entrenar modelo
# ============================================================

def entrenar_modelo(df, recalcular_presion=False):
    """
    Prepara el dataframe, separa X e y, divide los datos y entrena el modelo.
    """

    df = preparar_dataframe(df, recalcular_presion=recalcular_presion)

    X, y = separar_X_y(df)

    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    modelo = crear_modelo()

    modelo.fit(X_train, y_train)

    return modelo, X_train, X_test, y_train, y_test, df


# ============================================================
# 4. Evaluar modelo
# ============================================================

def evaluar_modelo(modelo, X_test, y_test):
    """
    Evalúa el modelo con métricas básicas.
    """

    predicciones = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, predicciones)

    matriz = confusion_matrix(
        y_test,
        predicciones,
        labels=modelo.classes_
    )

    reporte = classification_report(
        y_test,
        predicciones,
        zero_division=0
    )

    resultados = {
        "accuracy": accuracy,
        "confusion_matrix": matriz,
        "classes": modelo.classes_,
        "classification_report": reporte,
    }

    return resultados


# ============================================================
# 5. Entrenar y evaluar directamente
# ============================================================

def entrenar_y_evaluar(df, recalcular_presion=False):
    """
    Función cómoda para hacer todo de golpe:
    - preparar datos
    - entrenar modelo
    - evaluar modelo
    """

    modelo, X_train, X_test, y_train, y_test, df_preparado = entrenar_modelo(
        df,
        recalcular_presion=recalcular_presion
    )

    resultados = evaluar_modelo(modelo, X_test, y_test)

    return modelo, resultados, df_preparado


# ============================================================
# 6. Predecir un nuevo caso
# ============================================================

def predecir_probabilidades(modelo, nuevo_caso):
    """
    Devuelve las probabilidades de cada decisión posible.

    nuevo_caso debe ser un dataframe con una sola fila y las mismas columnas
    que X.
    """

    probabilidades = modelo.predict_proba(nuevo_caso)[0]
    clases = modelo.classes_

    tabla = pd.DataFrame({
        "decision_zone": clases,
        "probability": probabilidades
    })

    tabla = tabla.sort_values("probability", ascending=False)

    return tabla


# ============================================================
# 7. Prueba rápida
# ============================================================

if __name__ == "__main__":

    ruta = "data/synthetic_multisport_decisions.csv"

    df = pd.read_csv(ruta)

    modelo, resultados, df_preparado = entrenar_y_evaluar(df)

    print("Accuracy:")
    print(resultados["accuracy"])

    print("\nClases:")
    print(resultados["classes"])

    print("\nMatriz de confusión:")
    print(resultados["confusion_matrix"])

    print("\nReporte:")
    print(resultados["classification_report"])