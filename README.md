# Predicción de trayectoria en deportes

Framework matemático-computacional para predecir la decisión más probable de un jugador en situaciones deportivas de alta presión (duelos 1v1).

---

## Descripción

El proyecto construye un pipeline completo de machine learning que:

1. Genera datos sintéticos multideporte.
2. Calcula un **índice de presión** basado en múltiples factores contextuales.
3. Entrena un modelo de clasificación multiclase (Regresión Logística).
4. Predice la **zona de decisión** (`decision_zone`) más probable del jugador.
5. Ofrece una interfaz interactiva con Streamlit para simular escenarios.

> **Aviso:** los datos son sintéticos. El objetivo es demostrar que la arquitectura funciona y puede recibir datos reales en el futuro. No se pretenden extraer conclusiones empíricas sobre deportistas concretos.

---

## Deportes incluidos

| Deporte      | Acción modelada          | Zonas de decisión                                          |
|:-------------|:-------------------------|:-----------------------------------------------------------|
| Fútbol       | Penalti                  | `left`, `center`, `right`                                  |
| Tenis        | Saque decisivo           | `wide`, `body`, `center`                                   |
| Baloncesto   | Tiro decisivo            | `drive`, `mid_range`, `three_point`                        |
| Balonmano    | Lanzamiento de 7 metros  | `low_left`, `low_right`, `high_left`, `high_right`, `center` |
| Hockey       | Shootout                 | `left_deke`, `right_deke`, `direct_shot`, `high_shot`, `low_shot` |
| eSports      | Duelo 1v1                | `aggressive_push`, `defensive_hold`, `flank`, `bait`, `direct_duel` |

---

## Variable objetivo

```
decision_zone
```

Representa la decisión tomada por el jugador. Su significado varía según el deporte (zona de lanzamiento, tipo de tiro, decisión táctica, etc.).

---

## Índice de presión

La presión se calcula con una combinación lineal ponderada:

```
P = 0.30·T + 0.25·S + 0.20·C + 0.15·F + 0.10·M
```

| Componente | Descripción              | Peso |
|:-----------|:-------------------------|:-----|
| T          | Presión temporal         | 0.30 |
| S          | Presión por marcador     | 0.25 |
| C          | Importancia competición  | 0.20 |
| F          | Fatiga                   | 0.15 |
| M          | Momento crítico          | 0.10 |

**Clasificación:**

| Rango              | Nivel   |
|:--------------------|:--------|
| P < 0.35            | Baja    |
| 0.35 ≤ P < 0.70    | Media   |
| P ≥ 0.70            | Alta    |

---

## Variables del dataset

El dataset principal es `data/synthetic_multisport_decisions.csv` con las siguientes columnas:

| Columna                  | Tipo        | Descripción                                   |
|:-------------------------|:------------|:----------------------------------------------|
| `event_id`               | Texto       | Identificador único del evento                |
| `sport`                  | Categórica  | Deporte                                       |
| `action_type`            | Categórica  | Tipo de acción                                |
| `competition`            | Categórica  | Nombre de la competición                      |
| `match_time`             | Numérica    | Momento del partido/evento                    |
| `score_difference`       | Numérica    | Diferencia de marcador                        |
| `is_home_player`         | Binaria     | 1 si juega como local, 0 si visitante         |
| `player`                 | Categórica  | Identificador del jugador                     |
| `opponent`               | Categórica  | Identificador del rival                       |
| `dominant_side`          | Categórica  | Lado dominante (right/left)                   |
| `fatigue_level`          | Numérica    | Nivel de fatiga (0–1)                         |
| `competition_importance` | Numérica    | Importancia de la competición (0–1)           |
| `pressure_index`         | Numérica    | Índice de presión calculado (0–1)             |
| `pressure_level`         | Categórica  | Nivel de presión (baja/media/alta)            |
| `decision_zone`          | Categórica  | **Variable objetivo** — decisión del jugador  |
| `action_result`          | Categórica  | Resultado de la acción                        |

---

## Estructura del proyecto

```
Prediccion-de-trayectoria-en-deportes/
│
├── README.md                              ← Este archivo
├── requirements.txt                       ← Dependencias del proyecto
├── main.py                                ← Ejecuta el pipeline completo por consola
├── app.py                                 ← Interfaz web con Streamlit
│
├── Mates.py                               ← Funciones matemáticas originales (solo fútbol)
├── synthetic_multisport_data.py           ← Generador del dataset sintético
├── test_synthetic_multisport_data.py      ← Tests del generador
│
├── data/
│   ├── synthetic_multisport_decisions.csv ← Dataset multideporte principal
│   ├── synthetic_football_penalties.csv   ← Subconjunto: fútbol
│   ├── synthetic_tennis_serves.csv        ← Subconjunto: tenis
│   ├── synthetic_basketball_shots.csv     ← Subconjunto: baloncesto
│   ├── synthetic_handball_throws.csv      ← Subconjunto: balonmano
│   ├── synthetic_hockey_shootouts.csv     ← Subconjunto: hockey
│   └── synthetic_esports_duels.csv        ← Subconjunto: eSports
│
├── src/
│   ├── __init__.py
│   ├── pressure.py                        ← Cálculo del índice de presión
│   ├── preprocessing.py                   ← Carga, validación y limpieza de datos
│   ├── train_model.py                     ← Pipeline de entrenamiento y evaluación
│   └── analysis.py                        ← Funciones de análisis descriptivo
│
└── docs/
    ├── mathematical_framework.md          ← Marco matemático del proyecto
    └── technical_report.md                ← Informe técnico del proyecto
```

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/jairosdc/Prediccion-de-trayectoria-en-deportes.git
cd Prediccion-de-trayectoria-en-deportes

# Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

### Pipeline completo por consola

```bash
python main.py
```

Ejecuta el flujo completo: carga de datos → preprocesamiento → análisis descriptivo → entrenamiento → evaluación → predicción de ejemplo.

### Interfaz web interactiva

```bash
streamlit run app.py
```

Abre una aplicación web donde se puede:

- Elegir un deporte y tipo de acción.
- Ajustar los parámetros de la situación (fatiga, marcador, minuto, etc.).
- Ver el índice de presión calculado.
- Obtener las probabilidades de cada decisión posible.
- Comparar la predicción del modelo con un voto del público.

### Regenerar datos sintéticos

```bash
python synthetic_multisport_data.py
```

Regenera todos los CSV de la carpeta `data/`.

### Ejecutar tests

```bash
python -m pytest test_synthetic_multisport_data.py -v
```

---

## Modelo utilizado

- **Algoritmo:** Regresión Logística multiclase (solver `lbfgs`, `max_iter=1000`).
- **Preprocesado:** OneHotEncoder para variables categóricas + passthrough para numéricas.
- **División:** 75% entrenamiento / 25% test con estratificación cuando es posible.
- **Métricas:** accuracy, matriz de confusión, classification report (precision, recall, f1).

La elección de Regresión Logística se justifica porque:
- Es interpretable y fácil de explicar en una presentación académica.
- Devuelve probabilidades directamente.
- Es rápida de entrenar con datos de este tamaño.
- Sirve como baseline sólido antes de probar modelos más complejos.

---

## Documentación adicional

- [Marco matemático](docs/mathematical_framework.md) — Formalización matemática completa del índice de presión y el modelo de predicción.
- [Informe técnico](docs/technical_report.md) — Descripción técnica detallada del pipeline, decisiones de diseño y resultados.

---

## Licencia

Proyecto académico.

---

## Autor

**jairosdc** — [GitHub](https://github.com/jairosdc)