# 🚦 SemaforoIA · clasificador de vehículos

App de Streamlit que clasifica una imagen como **carro**, **moto** o **bicicleta**
usando transfer learning sobre **MobileNetV2** (PyTorch), y muestra un mapa
**Grad-CAM** con las regiones que más pesaron en la decisión.

Esta rama (`despliegue-streamlit`) contiene únicamente lo necesario para desplegar:
es una rama huérfana, independiente del historial de `main`.

## Resultados en el conjunto de prueba

133 imágenes que el modelo nunca vio durante el entrenamiento.

| Métrica | Valor |
|---|---|
| Accuracy | 98.5 % |
| F1 macro | 90.7 % |
| Errores | 2 / 133 |

| Clase | Precisión | Recall | F1 | Imágenes |
|---|---|---|---|---|
| bicicleta | 1.000 | 0.600 | 0.750 | 5 |
| carro | 0.991 | 1.000 | 0.995 | 107 |
| moto | 0.955 | 1.000 | 0.977 | 21 |

El conjunto está desbalanceado (107 carros frente a 5 bicicletas), así que el
**F1 macro es la métrica honesta**: la clase bicicleta es la débil, con solo 33
imágenes en todo el dataset. El desbalance se compensó con pesos por clase en la
función de pérdida.

## El modelo

- MobileNetV2 preentrenada en ImageNet, con cabeza nueva
  `Dropout → Linear(1280, 256) → ReLU → Dropout → Linear(256, 3)`
- Entrenamiento en dos fases: extracción de características con el extractor
  congelado, y luego fine-tuning de los últimos 5 bloques con una tasa de
  aprendizaje 10× menor
- 15 épocas, mejor época la 10 · entrada 224 × 224 px · 2 552 579 parámetros
- Latencia de inferencia: 23.8 ms en CPU

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

## Límites conocidos

- Solo conoce tres clases: ante cualquier otra imagen responderá con una de ellas.
- Clasifica la imagen completa; no localiza ni cuenta vehículos.
