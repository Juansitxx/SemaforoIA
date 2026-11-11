# 🚦 SemaforoIA · clasificador de vehículos

App de Streamlit que clasifica una imagen como **carro**, **moto** o **bicicleta**
usando transfer learning sobre **MobileNetV2** (PyTorch), y muestra un mapa
**Grad-CAM** con las regiones que más pesaron en la decisión.

Esta rama (`despliegue-streamlit`) contiene únicamente lo necesario para desplegar:
es una rama huérfana, independiente del historial de `main`.

En vivo: https://semaforoiadatatech.streamlit.app

## Resultados en el conjunto de prueba

247 imágenes que el modelo nunca vio durante el entrenamiento.

| Métrica | Valor |
|---|---|
| Accuracy | 92.7 % |
| F1 macro | 92.5 % |
| Errores | 18 / 247 |

| Clase | Precisión | Recall | F1 | AUC | Imágenes |
|---|---|---|---|---|---|
| bicicleta | 0.931 | 0.900 | 0.915 | 0.974 | 90 |
| carro | 0.977 | 0.944 | 0.960 | 0.993 | 90 |
| moto | 0.863 | 0.940 | 0.900 | 0.977 | 67 |

El conjunto está equilibrado (90 / 90 / 67), así que el accuracy y el F1 macro
dicen prácticamente lo mismo. Las tres clases quedan por encima de 0.90 de F1.

## Historia de este modelo

La primera versión se entrenó con un dataset muy desbalanceado: **711 carros,
141 motos y solo 33 bicicletas**. Daba 98.5 % de accuracy, pero era un espejismo:
el conjunto de prueba tenía 107 carros y 5 bicicletas, así que acertar siempre
"carro" ya puntuaba altísimo. El F1 macro real era 90.7 % y el recall de
bicicleta, **0.60**.

La segunda versión se entrenó sobre **1645 imágenes equilibradas** de Open
Images V7, seleccionadas por área de la caja delimitadora para que el vehículo
domine el encuadre. Midiendo **ambos modelos sobre el mismo conjunto de prueba**:

| Métrica | Modelo v1 | Modelo v2 | Δ |
|---|---|---|---|
| Accuracy | 80.6 % | 92.7 % | +12.1 pp |
| F1 macro | 79.7 % | 92.5 % | +12.8 pp |
| Errores | 48 | 18 | −30 |
| **Recall bicicleta** | 60.0 % | **90.0 %** | **+30.0 pp** |
| F1 moto | 73.8 % | 90.0 % | +16.2 pp |

El 98.5 % de accuracy que anunciaba la v1 no sobrevive al cambio de conjunto de
prueba: sobre datos equilibrados caía al 80.6 %. Es el ejemplo de libro de por
qué el accuracy engaña con clases desbalanceadas.

## El modelo

- MobileNetV2 preentrenada en ImageNet, con cabeza nueva
  `Dropout → Linear(1280, 256) → ReLU → Dropout → Linear(256, 3)`
- La v2 se inicializó desde los pesos de la v1 en vez de desde ImageNet puro:
  así conserva lo aprendido del dataset original, que ya no estaba disponible.
- Entrenamiento en dos fases: extracción de características con el extractor
  congelado (las características se precalculan una sola vez) y luego
  fine-tuning de los últimos 5 bloques con una tasa de aprendizaje 10× menor.
- 46 épocas, mejor época la 43 · entrada 224 × 224 px · 2 552 579 parámetros
- Entrenado en CPU · latencia de inferencia 34 ms

Nota honesta sobre el entrenamiento: en esta segunda vuelta **la fase de
fine-tuning no aportó nada**. La fase 1 ya alcanzaba F1 0.952 en validación y la
fase 2 se quedó en 0.935, con la pérdida de entrenamiento bajando mientras la
validación empeoraba, es decir sobreajuste. Tiene sentido, porque el extractor
ya venía ajustado a esta tarea desde la v1. Se conserva el mejor estado medido
sobre validación.

## Ejecutar en local

```bash
python -m venv .venv
.venv/Scripts/activate      # en Linux o macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

```
app.py              la aplicación completa
requirements.txt    dependencias (PyTorch en versión CPU)
modelo/             pesos entrenados (.pt)
artefactos/         métricas en JSON y gráficas de la evaluación
ejemplos/           imágenes de prueba para la demo
```

Las 5 imágenes de `ejemplos/` vienen del dataset original, que no forma parte
del entrenamiento de la v2: son datos completamente fuera de su distribución.
El modelo las clasifica correctamente las 5, con confianza de 89.8 % a 100 %.

## Herramientas de datos

Los scripts que recolectan y limpian las imágenes están en la rama
`datos-recoleccion` de este mismo repositorio.

## Límites conocidos

- Solo conoce tres clases: ante cualquier otra imagen responderá con una de ellas.
- Clasifica la imagen completa; no localiza ni cuenta vehículos.
- La confusión que queda es sobre todo bicicleta ↔ moto: 7 bicicletas clasificadas
  como moto y 4 motos como bicicleta. Son las dos clases visualmente más próximas.

## Licencias

Las imágenes de entrenamiento de la v2 provienen de Open Images V7, con licencia
CC BY.
