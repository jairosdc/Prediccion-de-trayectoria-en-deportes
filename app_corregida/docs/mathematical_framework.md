# Marco matemático

## 1. Introducción

Este documento formaliza las bases matemáticas del proyecto **Predicción de trayectoria en deportes**.

El objetivo es estimar:

```
P(decision_zone = z_k | x)
```

donde z_k es una zona de decisión posible y x es el vector de contexto.

---

## 2. Variables del modelo

### 2.1 Vector de características

| Variable                 | Tipo        | Dominio          |
|:-------------------------|:------------|:-----------------|
| sport                    | Categórica  | 6 valores        |
| action_type              | Categórica  | 6 valores        |
| competition              | Categórica  | Variable         |
| match_time               | Numérica    | [0, 120]         |
| score_difference         | Numérica    | [-5, 5]          |
| is_home_player           | Binaria     | {0, 1}           |
| player                   | Categórica  | Variable         |
| opponent                 | Categórica  | Variable         |
| dominant_side            | Categórica  | {right, left}    |
| fatigue_level            | Numérica    | [0, 1]           |
| competition_importance   | Numérica    | [0, 1]           |
| pressure_index           | Numérica    | [0, 1]           |
| pressure_level           | Categórica  | {baja, media, alta} |

### 2.2 Variable objetivo

```
y ∈ {z_1, z_2, ..., z_K}
```

K = 22 zonas de decisión en la versión actual (combinación de todos los deportes).

---

## 3. Índice de presión

### 3.1 Definición

```
P = 0.30·T + 0.25·S + 0.20·C + 0.15·F + 0.10·M
```

| Componente | Peso | Descripción              |
|:-----------|:-----|:-------------------------|
| T          | 0.30 | Presión temporal         |
| S          | 0.25 | Presión por marcador     |
| C          | 0.20 | Importancia competición  |
| F          | 0.15 | Fatiga                   |
| M          | 0.10 | Momento crítico          |

Resultado acotado a [0, 1].

### 3.2 Componentes

**Presión temporal:**
```
T(t) = t / t_max
```
Saturada a [0, 1].

**Presión por marcador:**
```
S(d) = 1 / (1 + |d|)
```
Máxima cuando d = 0 (empate).

**Importancia competitiva (C):** valor directo en [0, 1].

**Fatiga (F):** valor directo en [0, 1].

**Momento crítico (M):** valor en [0, 1].

### 3.3 Clasificación

```
pressure_level =
  baja   si P < 0.35
  media  si 0.35 ≤ P < 0.70
  alta   si P ≥ 0.70
```

---

## 4. Modelo de predicción

### 4.1 Regresión logística multinomial

Para un vector codificado x' (tras OneHotEncoding):

```
P(y = z_k | x') = exp(w_k · x' + b_k) / Σ_j exp(w_j · x' + b_j)
```

### 4.2 Función de pérdida

Cross-entropy multiclase:

```
L = -(1/N) Σ_i Σ_k 1[y_i = z_k] · log P(y_i = z_k | x_i')
```

### 4.3 Codificación

- Categóricas → One-Hot Encoding
- Numéricas → passthrough (sin transformación)

### 4.4 Decisión final

```
ŷ = argmax_k P(y = z_k | x')
```

---

## 5. Métricas de evaluación

- **Accuracy:** predicciones correctas / total
- **Precision_k:** TP_k / (TP_k + FP_k)
- **Recall_k:** TP_k / (TP_k + FN_k)
- **F1_k:** 2 · Precision · Recall / (Precision + Recall)
- **Matriz de confusión:** C[i][j] = muestras de clase i clasificadas como j

---

## 6. Reglas de datos sintéticos

### Fútbol
- Diestros cruzan más a la izquierda; zurdos a la derecha
- Presión alta → más tiros al centro

### Tenis
- Fatiga alta → más saques al cuerpo
- Presión alta → más saques al cuerpo

### Baloncesto
- Fatiga alta → menos triples, más media distancia
- Presión alta → más penetración, menos triples

### Balonmano
- Mano dominante influye en el lado del lanzamiento
- Presión alta → más tiros centrales/bajos

### Hockey
- Fatiga alta → menos dekes, más tiro directo
- Presión alta → más tiro directo

### eSports
- Fatiga alta → más pasividad
- Presión alta → más juego defensivo y bait

---

## 7. Justificación de pesos

1. **Tiempo (0.30):** factor más determinante en la literatura deportiva
2. **Marcador (0.25):** empates generan más presión
3. **Importancia (0.20):** finales > amistosos
4. **Fatiga (0.15):** afecta la toma de decisiones
5. **Momento crítico (0.10):** complemento para situaciones especiales

> Pesos fijos como primera aproximación. Optimizables con datos reales.

---

## 8. Limitaciones

- Datos sintéticos (patrones programados, no reales)
- Regresión logística asume separabilidad lineal
- Sin variables de secuencia temporal
- Pesos de presión fijos, no aprendidos

---

## 9. Extensiones futuras

- Datos reales de eventos deportivos
- Modelos más complejos (Random Forest, XGBoost, redes neuronales)
- Optimización de pesos de presión
- Historial del jugador
- Análisis de sensibilidad
