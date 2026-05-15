# PRD: Mejora del Framework de Predicción de Trayectorias Deportivas

## 1. Visión del Producto
Transformar el dashboard actual en una experiencia inmersiva y de alto rendimiento que no solo prediga resultados, sino que los visualice dinámicamente y explique el "por qué" detrás de cada decisión.

## 2. Objetivos
- **WOW Factor**: Implementar animaciones que simulen la ejecución de la acción (específicamente fútbol).
- **Rigor Técnico**: Evolucionar el modelo de Regresión Logística a modelos ensemble.
- **Transparencia**: Añadir interpretabilidad al modelo (Explainable AI).
- **Premium UX**: Refinar la interfaz hacia un estilo "Glassmorphism" con feedback en tiempo real.

## 3. Características Propuestas

### A. Simulación Visual (Animación de Penalti)
- **Descripción**: Al generar una predicción de fútbol, mostrar una animación 2D/3D del jugador chutando el balón hacia la portería.
- **Tecnología**: CSS Keyframes o Lottie.
- **Interacción**: La trayectoria del balón debe coincidir con la `decision_zone` predicha.

### B. Comparador de Modelos
- **Descripción**: Selector para elegir entre Logistic Regression (Baseline) y Random Forest / XGBoost.
- **Métricas**: Mostrar la diferencia de Accuracy y confianza entre modelos en tiempo real.

### C. Análisis SHAP (Feature Importance)
- **Descripción**: Gráfico dinámico que muestra qué variable (Fatiga, Presión, Lado Dominante) tuvo más peso en la predicción actual.
- **Visualización**: Waterfall plot o Bar plot simplificado.

### D. Mapa de Calor Táctico
- **Descripción**: Sustituir el selector de zona de texto por un mapa interactivo de la portería/cancha.
- **Interacción**: Al pasar el ratón, mostrar la probabilidad histórica en esa zona.

## 4. Viabilidad de la Animación
Es **totalmente viable**. Podemos usar un componente de HTML/CSS inyectado en Streamlit que reaccione al estado de la predicción. 
- **Opción A (Rápida)**: Animación CSS con el balón moviéndose a coordenadas específicas.
- **Opción B (Premium)**: Integración de un canvas de JavaScript para una simulación más fluida.

## 5. Roadmap de Ejecución (GSD)
1. **Fase 1**: Implementación del Prototipo de Animación (Fútbol).
2. **Fase 2**: Integración de Modelos Avanzados (Random Forest).
3. **Fase 3**: Implementación de SHAP y Feedback Visual.
