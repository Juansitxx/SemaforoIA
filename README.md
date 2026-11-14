# 🚦 SemaforoIA · Clasificador de vehículos con Knowledge Distillation

App de Streamlit que clasifica una imagen como **carro**, **moto** o **bicicleta**
usando **Transfer Learning** y **Knowledge Distillation (Destilación de Conocimiento)** en PyTorch,
mostrando además un mapa **Grad-CAM** con las regiones que más pesaron en la decisión.

Esta rama (`despliegue-streamlit`) contiene lo necesario para desplegar en producción e incluye
ambos modelos para poder seleccionarlos y compararlos en vivo desde la barra lateral:

1. **Modelo Principal de Uso (`MobileNetV2` · Estudiante Destilado):** Seleccionado por defecto.
   Pesa **10.0 MB**, tiene **2.552.579 parámetros** y ejecuta en **~35–45 ms en CPU**, entrenado
   mediante *Knowledge Distillation* a partir del Maestro.
2. **Modelo Maestro / Padre (`ResNet50` · Referencia):** Arquitectura profunda de **24.558.659 parámetros**
   (**94.0 MB** en FP32, almacenada en FP16 en `modelo/resnet50_maestro.pt` con **47.0 MB**). Se entrenó
   para enseñar a `MobileNetV2` y está disponible en el selector de la app para comparar predicciones y mapas Grad-CAM.

En vivo: https://semaforoiadatatech.streamlit.app

---

## Comparativa: Maestro (`ResNet50`) vs. Estudiante Principal (`MobileNetV2`)

Medidos sobre el **mismo conjunto de prueba de 247 imágenes** que ninguno de los modelos vio durante el entrenamiento:

| Métrica | `MobileNetV2` (Estudiante · **Principal**) | `ResNet50` (Maestro / Padre) | Ventaja del Estudiante en producción |
|---|---|---|---|
| **Rol en la aplicación** | **Modelo principal de producción** | Arquitectura Padre (referencia) | — |
| **Parámetros** | **2.552.579** | 24.558.659 | **9.6× menos parámetros** (−89.6 %) |
| **Peso en memoria (FP32)** | **10.0 MB** | 94.0 MB *(47.0 MB en disco FP16)* | **9.4× más liviano** |
| **Latencia en CPU** | **~35–45 ms** | ~170 ms | **~3.8× más rápido** |
| **Accuracy en test** | **92.7 %** | **95.1 %** | Retiene **97.5 %** del desempeño del Maestro |
| **F1 macro en test** | **92.5 %** | **94.9 %** | Retiene **97.5 %** del F1 del Maestro |
| **Errores (sobre 247)** | 18 / 247 | 12 / 247 | Solo 6 imágenes de diferencia en 247 pruebas |

### Desglose por clase en el conjunto de prueba (247 imágenes)

**Modelo Principal (`MobileNetV2` · Estudiante Destilado):**

| Clase | Precisión | Recall | F1 | AUC | Imágenes |
|---|---|---|---|---|---|
| bicicleta | 0.931 | 0.900 | 0.915 | 0.974 | 90 |
| carro | 0.977 | 0.944 | 0.960 | 0.993 | 90 |
| moto | 0.863 | 0.940 | 0.900 | 0.977 | 67 |

**Modelo Maestro (`ResNet50` · Padre):**

| Clase | Precisión | Recall | F1 | AUC | Imágenes |
|---|---|---|---|---|---|
| bicicleta | 0.965 | 0.922 | 0.943 | 0.990 | 90 |
| carro | 1.000 | 0.978 | 0.989 | 0.996 | 90 |
| moto | 0.877 | 0.955 | 0.914 | 0.987 | 67 |

---

## ¿Cómo se aplicó Knowledge Distillation?

1. **Entrenamiento del Maestro (`ResNet50`):**
   Se partió de `ResNet50` preentrenada en ImageNet V2 (`2048` características) y se le añadió una cabeza
   `Dropout(0.3) → Linear(2048, 512) → ReLU → Dropout(0.3) → Linear(512, 3)`, entrenándola sobre las
   1.151 imágenes de entrenamiento equilibradas con pesos por clase.
2. **Transferencia de conocimiento (`ResNet50` ➔ `MobileNetV2`):**
   Con el Maestro congelado, se calcularon sus *logits* $z_T$ y se entrenó el Estudiante `MobileNetV2`
   minimizando una combinación de **Divergencia KL** sobre probabilidades suavizadas con **Temperatura $T = 4.0$**
   (*soft targets*, con peso $\alpha = 0.7$) y **Entropía Cruzada** contra las etiquetas reales ($1 - \alpha = 0.3$):
   $$\mathcal{L}_{\text{KD}} = \alpha \cdot T^2 \cdot \text{KLDiv}\!\left(\sigma\!\left(\frac{z_S}{T}\right), \sigma\!\left(\frac{z_T}{T}\right)\right) + (1 - \alpha) \cdot \text{CrossEntropy}(z_S, y)$$

---

## Historia del dataset (v1 vs v2)

La primera versión se entrenó con un dataset muy desbalanceado: **711 carros,
141 motos y solo 33 bicicletas**. Daba 98.5 % de accuracy sobre un conjunto de prueba también
desbalanceado, pero al medirla sobre las **247 imágenes equilibradas** de la v2 caía al **80.6 %**:

| Métrica | Modelo v1 | Modelo v2 (`MobileNetV2` Principal) | Maestro (`ResNet50`) |
|---|---|---|---|
| Accuracy | 80.6 % | **92.7 %** (+12.1 pp) | **95.1 %** (+14.5 pp) |
| F1 macro | 79.7 % | **92.5 %** (+12.8 pp) | **94.9 %** (+15.2 pp) |
| Errores | 48 / 247 | **18 / 247** (−30) | **12 / 247** (−36) |
| **Recall bicicleta** | 60.0 % | **90.0 %** (+30.0 pp) | **92.2 %** (+32.2 pp) |

---

## Ejecutar en local

```bash
python -m venv .venv
.venv/Scripts/activate      # en Linux o macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

```
app.py                         aplicación Streamlit con selector de modelo (MobileNetV2 / ResNet50)
requirements.txt               dependencias (PyTorch en versión CPU)
modelo/
  mobilenetv2_semaforo.pt      pesos del Estudiante Destilado (Modelo Principal, 10.0 MB)
  resnet50_maestro.pt          pesos del Modelo Maestro ResNet50 (47.0 MB en FP16)
artefactos/                    métricas en JSON y gráficas de evaluación y destilación
ejemplos/                      imágenes de prueba para la demo
```

## Herramientas de datos y destilación

Los scripts que recolectan, limpian, entrenan al Maestro `ResNet50` y aplican Knowledge Distillation
(`destilar.py` y `refinar_destilacion.py`) están en la rama `datos-recoleccion` de este mismo repositorio.

## Licencias

Las imágenes de entrenamiento provienen de Open Images V7, con licencia CC BY.
