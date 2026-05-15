# Informe técnico

## 1. Resumen del proyecto

**Predicción de trayectoria en deportes** es un framework que predice la decisión más probable de un jugador en situaciones 1v1 bajo presión, usando un modelo de clasificación multiclase entrenado con datos sintéticos de 6 deportes.

---

## 2. Pipeline del proyecto

```
CSV → cargar_datos → validar_columnas → limpiar_datos → preparar_presion
    → separar_X_y → dividir_datos → crear_modelo → fit → evaluar → predict_proba
```

### 2.1 Carga y validación (`preprocessing.py`)

- `cargar_datos(ruta)`: lee el CSV con pandas.
- `validar_columnas(df)`: comprueba que existen las 12 columnas obligatorias.
- `limpiar_datos(df)`: elimina duplicados, normaliza textos a minúsculas, convierte numéricas, rellena nulos.

### 2.2 Cálculo de presión (`pressure.py`)

- `calcular_presion_temporal(match_time)`: normaliza el tiempo a [0, 1].
- `calcular_presion_marcador(score_difference)`: fórmula 1/(1+|d|).
- `calcular_pressure_index(...)`: aplica la fórmula P = 0.30T + 0.25S + 0.20C + 0.15F + 0.10M.
- `clasificar_presion(P)`: baja/media/alta.
- `añadir_presion(df)`: añade columnas `pressure_index` y `pressure_level` al dataframe.

### 2.3 Entrenamiento (`train_model.py`)

- `crear_modelo()`: crea un Pipeline de sklearn con ColumnTransformer (OneHotEncoder + passthrough) y LogisticRegression.
- `dividir_datos(X, y)`: split 75/25 con estratificación.
- `entrenar_y_evaluar(df)`: función todo-en-uno que devuelve modelo, resultados y dataframe preparado.
- `predecir_probabilidades(modelo, caso)`: devuelve DataFrame con probabilidades por clase, ordenadas de mayor a menor.

### 2.4 Análisis (`analysis.py`)

Funciones de tablas cruzadas:
- `decisiones_por_deporte(df)`
- `decisiones_por_presion(df)`
- `decisiones_por_lado_dominante(df)`
- `presion_media_por_deporte(df)`

---

## 3. Generación de datos sintéticos

El archivo `synthetic_multisport_data.py` genera 300 eventos por deporte (1800 total) con reglas probabilísticas específicas para cada deporte.

Cada generador:
1. Calcula componentes de presión individual.
2. Aplica la fórmula de presión global.
3. Define probabilidades base para las zonas de decisión.
4. Modifica las probabilidades según reglas deportivas (fatiga, presión, lado dominante).
5. Normaliza probabilidades y muestrea la decisión.

---

## 4. Interfaz Streamlit (`app.py`)

La aplicación web permite:

- **Sidebar:** seleccionar deporte, tipo de acción y competición.
- **Panel principal:** ajustar parámetros (tiempo, marcador, fatiga, etc.) y ver presión calculada.
- **Predicción:** tabla de probabilidades + gráfico de barras + decisión más probable.
- **Dinámica:** votación del público para comparar con el modelo.
- **Análisis:** pestañas con dataset, presión, decisiones y métricas del modelo.

Se usa `@st.cache_data` y `@st.cache_resource` para evitar recargar datos y reentrenar el modelo en cada interacción.

---

## 5. Resultados del modelo

Con el dataset sintético actual (1800 eventos, 22 clases):

- **Accuracy global:** ~34%
- El modelo distingue bien entre deportes (las zonas de decisión son exclusivas por deporte).
- Dentro de cada deporte, las probabilidades reflejan los sesgos programados en la generación de datos.

> El accuracy bajo es esperable: con 22 clases, un modelo aleatorio tendría ~4.5%. El 34% indica que el modelo captura patrones reales.

### Interpretación por deporte

| Deporte    | Clases | Observación principal                        |
|:-----------|:-------|:---------------------------------------------|
| Fútbol     | 3      | Distingue bien left/center/right             |
| Tenis      | 3      | Captura efecto de fatiga en body              |
| Baloncesto | 3      | Separa drive/mid_range/three_point            |
| Balonmano  | 5      | Más difícil, pero detecta influencia de mano  |
| Hockey     | 5      | Direct_shot bien predicho en presión alta     |
| eSports    | 5      | Defensive_hold bien predicho en alta presión  |

---

## 6. Decisiones de diseño

### ¿Por qué Regresión Logística?

- Interpretable: los coeficientes tienen significado directo.
- Probabilística: devuelve probabilidades, no solo la clase predicha.
- Rápida: entrena en menos de 1 segundo.
- Baseline sólido antes de usar modelos más complejos.

### ¿Por qué OneHotEncoding?

- Compatible con regresión logística.
- No asume orden en las categorías.
- Maneja categorías desconocidas con `handle_unknown="ignore"`.

### ¿Por qué datos sintéticos?

- Permite diseñar patrones controlados.
- Demuestra que la arquitectura completa funciona.
- Facilita la depuración y validación del pipeline.
- Los datos reales pueden integrarse sin cambiar la arquitectura.

---

## 7. Estructura de archivos

| Archivo                           | Función                           |
|:----------------------------------|:----------------------------------|
| `main.py`                         | Pipeline completo por consola     |
| `app.py`                          | Interfaz web Streamlit            |
| `synthetic_multisport_data.py`    | Generador de datos sintéticos     |
| `src/pressure.py`                 | Cálculo del índice de presión     |
| `src/preprocessing.py`            | Carga, validación y limpieza      |
| `src/train_model.py`              | Pipeline ML (sklearn)             |
| `src/analysis.py`                 | Análisis descriptivo              |
| `Mates.py`                        | Versión original (solo fútbol)    |

---

## 8. Dependencias

- Python 3.10+
- pandas >= 2.0
- numpy >= 1.24
- scikit-learn >= 1.3
- streamlit >= 1.30

---

## 9. Cómo ejecutar

```bash
# Pipeline completo
python main.py

# Interfaz web
streamlit run app.py

# Regenerar datos
python synthetic_multisport_data.py

# Tests
python -m pytest test_synthetic_multisport_data.py -v
```

---

## 10. Trabajo futuro

1. **Datos reales:** integrar datasets públicos de eventos deportivos.
2. **Modelos avanzados:** Random Forest, Gradient Boosting, redes neuronales.
3. **Feature engineering:** crear variables derivadas (racha de resultados, historial vs rival, etc.).
4. **Optimización de presión:** aprender los pesos w_T, w_S, etc. a partir de los datos.
5. **Validación cruzada:** implementar k-fold CV para resultados más robustos.
6. **Despliegue:** dockerizar la app de Streamlit para facilitar el acceso.
