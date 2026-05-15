"""
train_model.py

Entrena un modelo POR DEPORTE para que las predicciones sean coherentes.
Corrige el error original donde el modelo mezcla clases de todos los deportes.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    from src.preprocessing import preparar_dataframe, separar_X_y, COLUMNAS_MODELO
except ModuleNotFoundError:
    from preprocessing import preparar_dataframe, separar_X_y, COLUMNAS_MODELO


COLUMNAS_CATEGORICAS = [
    "action_type", "competition", "player", "opponent",
    "dominant_side", "pressure_level",
]

COLUMNAS_NUMERICAS = [
    "match_time", "score_difference", "is_home_player",
    "fatigue_level", "competition_importance", "pressure_index",
]

# Columnas del modelo sin 'sport' (ya que entrenamos por deporte)
COLUMNAS_MODELO_SPORT = [c for c in COLUMNAS_MODELO if c != "sport"]


def crear_modelo():
    preprocesado = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), COLUMNAS_CATEGORICAS),
            ("num", StandardScaler(), COLUMNAS_NUMERICAS),
        ]
    )
    modelo = Pipeline(steps=[
        ("preprocesado", preprocesado),
        ("modelo", LogisticRegression(max_iter=5000))
    ])
    return modelo


def dividir_datos(X, y, test_size=0.25, random_state=42):
    conteos = y.value_counts()
    stratify = y if conteos.min() >= 2 else None
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=stratify)


def entrenar_y_evaluar(df, recalcular_presion=False):
    """
    CORREGIDO: Entrena un modelo independiente por cada deporte.
    Devuelve un dict de modelos y el df preparado.
    """
    df_prep = preparar_dataframe(df, recalcular_presion=recalcular_presion)
    modelos = {}
    resultados_globales = {"accuracy": 0, "classes": [], "confusion_matrix": [], "classification_report": ""}
    accs = []

    for sport in df_prep["sport"].unique():
        df_sport = df_prep[df_prep["sport"] == sport].copy()
        X = df_sport[COLUMNAS_MODELO_SPORT]
        y = df_sport["decision_zone"]

        if len(y.unique()) < 2 or len(y) < 10:
            continue

        X_train, X_test, y_train, y_test = dividir_datos(X, y)
        m = crear_modelo()
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accs.append(acc)

        modelos[sport] = {
            "model": m,
            "accuracy": acc,
            "classes": list(m.classes_),
            "confusion_matrix": confusion_matrix(y_test, preds, labels=m.classes_).tolist(),
            "classification_report": classification_report(y_test, preds, zero_division=0),
        }

    resultados_globales["accuracy"] = sum(accs) / len(accs) if accs else 0
    resultados_globales["classes"] = list(df_prep["decision_zone"].unique())
    resultados_globales["confusion_matrix"] = []
    resultados_globales["classification_report"] = f"Accuracy media entre deportes: {resultados_globales['accuracy']:.4f}"
    resultados_globales["por_deporte"] = {s: {"accuracy": modelos[s]["accuracy"]} for s in modelos}

    return modelos, resultados_globales, df_prep


def predecir_probabilidades(modelos, nuevo_caso):
    """
    CORREGIDO: Usa el modelo del deporte correcto.
    nuevo_caso debe incluir la columna 'sport'.
    """
    sport = nuevo_caso["sport"].iloc[0]
    
    if sport not in modelos:
        raise ValueError(f"No hay modelo entrenado para el deporte: {sport}")
    
    m = modelos[sport]["model"]
    X = nuevo_caso[COLUMNAS_MODELO_SPORT]
    
    probabilidades = m.predict_proba(X)[0]
    clases = m.classes_

    tabla = pd.DataFrame({
        "decision_zone": clases,
        "probability": probabilidades
    }).sort_values("probability", ascending=False)

    return tabla
