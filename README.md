# Clasificación de tumores con scikit-learn (y por qué la exactitud engaña)

Proyecto de Machine Learning supervisado sobre el conjunto público **Breast Cancer Wisconsin (Diagnostic)**, que viene incluido en scikit-learn (569 casos, 30 medidas numéricas por caso, sin valores faltantes). El objetivo no es «lograr 99 %», sino mostrar **cómo se evalúa bien un modelo**: separar datos, evitar fugas, elegir el modelo y el umbral sin mirar la prueba, y reportar la métrica que importa para el problema.

**Costo: cero. Corre en segundos en una laptop, sin GPU.**

> Es un ejercicio de aprendizaje. No es un dispositivo médico ni sirve para diagnosticar nada.

## Cómo correrlo

```bash
uv run --with scikit-learn --with matplotlib --with pandas python entrenar.py
# o:  pip install -r requirements.txt && python entrenar.py
```

Genera `resultados.json` y tres figuras en `figuras/`. Es determinista (semilla 42).

## El problema y la métrica que importa

Se predice si un tumor es **maligno** (clase positiva) o benigno. El error grave es un **falso negativo**: decir «benigno» cuando era maligno. Por eso la métrica principal es el **recall** (de los malignos reales, cuántos se detectan), vigilando la precisión para no marcar de más.

El conjunto tiene 212 malignos (37,3 %) y 357 benignos: no está muy desbalanceado, pero alcanza para mostrar el problema de la exactitud.

## Método

1. **Separar antes de explorar**: 80 % entrenamiento (455) / 20 % prueba (114), estratificado. La prueba no se toca hasta el final.
2. **Pipeline con el escalado dentro**: así el escalador se ajusta solo con los datos de cada pliegue y no hay **fuga de datos** en la validación cruzada.
3. **Tres modelos** comparados con validación cruzada de 5 pliegues, solo sobre entrenamiento: una línea base que siempre dice «benigno», regresión logística y random forest.
4. **El modelo se elige con la validación cruzada** (recall, luego F1), no mirando la prueba.
5. **El umbral de decisión también se elige sin la prueba**: con predicciones de validación cruzada, el umbral más alto que logra recall ≥ 0,97.
6. **Evaluación final una sola vez** en la prueba, y comparación entrenamiento vs prueba para revisar sobreajuste.

## Resultados (salida real de `entrenar.py`)

**Validación cruzada (entrenamiento, media de 5 pliegues):**

| Modelo | Exactitud | Precisión | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Base «siempre benigno» | 0,626 | 0,000 | **0,000** | 0,000 | 0,500 |
| Regresión logística | 0,974 | 0,977 | 0,953 | 0,964 | 0,996 |
| Random forest | 0,960 | 0,958 | 0,935 | 0,946 | 0,988 |

Elegido: **regresión logística** (además de ganar en todas las métricas, es más simple y se interpreta mejor).

**Prueba (114 casos nunca vistos), regresión logística:**

| Umbral | Exactitud | Precisión | Recall | F1 | VP | FP | VN | FN |
|---|---|---|---|---|---|---|---|---|
| 0,50 (por defecto) | 0,965 | 0,975 | 0,929 | 0,951 | 39 | 1 | 71 | 3 |
| **0,315 (elegido en validación)** | **0,974** | 0,976 | **0,952** | 0,964 | 40 | 1 | 71 | **2** |

AUC en prueba: 0,996. Sobreajuste: exactitud 0,987 en entrenamiento vs 0,965 en prueba (brecha de 2 puntos, aceptable).

Variables más influyentes (coeficientes de la regresión logística, escala estandarizada): `worst texture`, `radius error`, `worst symmetry`, `mean concave points`, `worst concavity`.

Figuras: `figuras/matriz_confusion.png`, `figuras/precision_recall.png`, `figuras/distribucion_probabilidades.png`.

## Lo que muestra el proyecto

- **La exactitud engaña**: el modelo que siempre dice «benigno» tiene 63 % de exactitud y **recall 0**: no detecta ni un solo maligno.
- **Bajar el umbral cambia el tipo de error**: de 0,5 a 0,315 se evita un falso negativo a costa de nada en esta muestra (1 falso positivo en ambos casos).
- **Elegir sin mirar la prueba**: modelo y umbral salieron de la validación cruzada; la prueba solo reporta.

## Limitaciones (importante)

- **La prueba es pequeña**: solo 42 casos malignos. Un falso negativo más o menos mueve el recall unos 2,4 puntos, así que la diferencia entre 0,929 y 0,952 **no es estadísticamente sólida**. No se calcularon intervalos de confianza.
- **El umbral no alcanzó en la prueba el recall de 0,97 buscado en validación** (fue 0,952): es la variación normal entre muestras, y es una razón para no sobreinterpretar un solo corte.
- Un solo conjunto de datos, limpio y público; no se probó en otro hospital ni con otro instrumento.
- No se hizo ajuste de hiperparámetros ni selección de variables.

## Siguientes pasos

1. Intervalos de confianza con *bootstrap* sobre la prueba.
2. Repetir con validación cruzada anidada para elegir hiperparámetros sin sesgo.
3. Calibrar las probabilidades y revisar la curva de calibración.
4. Probar con otro dataset tabular (por ejemplo uno con clases muy desbalanceadas) para ver el efecto del umbral.

## Experimentos

Se corren con `uv run --with scikit-learn --with matplotlib --with pandas python experimentos.py`. Salida real.

### 1. Cambiar la semilla (misma regla: umbral con recall ≥ 0,97 en validación)

| Semilla | Umbral elegido | Recall en prueba | Precisión en prueba | FN | FP | Malignos en prueba |
|---|---|---|---|---|---|---|
| 0 | 0,418 | 0,952 | 0,909 | 2 | 4 | 42 |
| 7 | 0,376 | 0,952 | 0,976 | 2 | 1 | 42 |
| 42 | 0,315 | 0,952 | 0,976 | 2 | 1 | 42 |
| 123 | 0,288 | 0,976 | 0,837 | 1 | 8 | 42 |
| 2024 | 0,444 | 0,929 | 0,975 | 3 | 1 | 42 |

Recall en prueba: mínimo 0,929, máximo 0,976, media 0,952. **El umbral elegido varía mucho** (de 0,288 a 0,444) y la precisión también (0,837 a 0,976). Con 42 malignos en la prueba, un caso cambia el recall 2,4 puntos: el resultado de una sola semilla no es una cifra fija. Por eso conviene hablar de «alrededor de 95 %» y no de una cifra exacta.

### 2. Objetivo de recall ≥ 0,99 en validación (semilla 42)

El umbral baja a 0,094: recall 0,976 (1 falso negativo) pero precisión 0,891 (5 falsos positivos). Querer más recall se paga con más falsas alarmas, y el 0,99 de validación tampoco se alcanza en la prueba.

### 3. Sin escalado de variables (semilla 42, solo entrenamiento)

| | Recall | F1 | Avisos |
|---|---|---|---|
| Sin escalado | 0,924 | 0,937 | `ConvergenceWarning` |
| Con escalado | 0,953 | 0,964 | ninguno |

Sin escalar, la regresión logística converge peor y rinde menos.

### 4. Fuga de datos: escalar todo antes de separar

| | Recall | F1 |
|---|---|---|
| Con fuga | 0,953 | 0,964 |
| Sin fuga (pipeline) | 0,953 | 0,964 |

**En este conjunto no hay diferencia visible**: la fuga por escalado es sutil porque la media y la desviación de todo el conjunto se parecen mucho a las del entrenamiento. Eso **no** significa que la regla sobre la fuga sea inútil: con imputación de datos faltantes, selección de variables o codificaciones aprendidas de los datos, la fuga sí puede inflar los resultados. Aquí la regla se sigue por prevención, no porque este experimento la haya demostrado.
