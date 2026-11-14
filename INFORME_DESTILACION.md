# Informe Técnico: Knowledge Distillation en SemaforoIA (`ResNet50` ➔ `MobileNetV2`)

## 1. Planteamiento del problema y objetivo
En un sistema de control semafórico inteligente (**SemaforoIA**), el clasificador de vehículos (`bicicleta`, `carro`, `moto`) debe operar con **baja latencia y bajo consumo de memoria** sobre hardware de borde o instancias en la nube basadas en CPU, sin sacrificar capacidad de discriminación entre clases visualmente cercanas (especialmente `bicicleta` vs. `moto`).

Para lograrlo se implementó un esquema de **Knowledge Distillation (Destilación de Conocimiento)**:
1. **Modelo Maestro / Padre (`ResNet50`):** Una red profunda de alta capacidad (**24,56 millones de parámetros**, **94,0 MB** en precisión completa FP32) encargada de aprender fronteras de decisión ricas sobre el dataset equilibrado.
2. **Modelo Estudiante / Hijo (`MobileNetV2`):** Una arquitectura ligera (**2,55 millones de parámetros**, **10,0 MB**) que aprende simultáneamente de las etiquetas reales y de las distribuciones de probabilidad suavizadas (*soft targets*) generadas por el Maestro.

---

## 2. Metodología del proceso de destilación

### 2.1 Dataset y partición experimental
Ambos modelos fueron entrenados y evaluados exactamente sobre las mismas particiones (`1.645` imágenes verificadas de Open Images V7, filtradas por dominancia de caja delimitadora):
* **Entrenamiento (`train` — 70 %):** `1.151` imágenes (`420` bicicletas, `420` carros, `311` motos). Se aplicaron pesos inversamente proporcionales a la frecuencia de clase (`[0.913, 0.913, 1.234]`).
* **Validación (`val` — 15 %):** `247` imágenes (`90` bicicletas, `90` carros, `67` motos).
* **Prueba (`test` — 15 %):** `247` imágenes (`90` bicicletas, `90` carros, `67` motos), reservadas exclusivamente para la medición final.

### 2.2 Entrenamiento del Maestro (`ResNet50`)
* Se partió de los pesos preentrenados `IMAGENET1K_V2` y se reemplazó la capa final por una cabeza clasificadora `Dropout(0.3) → Linear(2048, 512) → ReLU → Dropout(0.3) → Linear(512, 3)`.
* Se precalcularon los vectores de características de `2048` dimensiones sobre el extractor congelado y se entrenó la cabeza con `AdamW` ($\text{lr}=10^{-3}$) y planificador cosenoidal, alcanzando un **F1 macro en validación de 97,84 %** y un **Accuracy en prueba de 95,14 %**.

### 2.3 Función de pérdida de Knowledge Distillation
Durante el entrenamiento del Estudiante (`MobileNetV2`), en lugar de usar solo etiquetas rígidas (*hard labels* `[0, 1, 0]`), el Maestro transfiere su conocimiento mediante una **Temperatura $T = 4,0$** y un factor de balance **$\alpha = 0,70$**:

$$\mathcal{L}_{\text{total}} = \alpha \cdot T^2 \cdot \text{KLDiv}\!\left(\log\sigma\!\left(\frac{z_S}{T}\right), \sigma\!\left(\frac{z_T}{T}\right)\right) + (1 - \alpha) \cdot \text{CrossEntropy}(z_S, y) + \lambda \|W_S - W_{\text{base}}\|_2^2$$

* **$\alpha = 0,70$ (`KLDivLoss`):** Obliga al Estudiante a imitar las relaciones inter-clase descubiertas por `ResNet50`.
* **$1 - \alpha = 0,30$ (`CrossEntropyLoss`):** Mantiene al Estudiante alineado con la etiqueta verdadera ponderada por clase.
* **Anclaje proximal ($\lambda = 0,05$):** Evita el olvido catastrófico sobre los pesos base ya convergidos del Estudiante.

---

## 3. Tablas comparativas de resultados (Conjunto de Prueba: 247 imágenes)

### 3.1 Comparativa global: Eficiencia vs. Rendimiento

| Criterio | `MobileNetV2` v1 *(Dataset anterior)* | `MobileNetV2` Destilado *(Estudiante · **Principal**)* | `ResNet50` *(Maestro / Padre)* | Relación Estudiante vs. Maestro |
| :--- | :---: | :---: | :---: | :--- |
| **Rol en el sistema** | Histórico (descartado) | **Principal en producción** | Referencia / Maestro KD | — |
| **Parámetros totales** | 2.552.579 | **2.552.579** | 24.558.659 | **9,6× menos parámetros** (−89,6 %) |
| **Peso en memoria (FP32)** | 10,0 MB | **10,0 MB** | 94,0 MB *(47,0 MB en FP16)* | **9,4× más liviano** (−89,4 %) |
| **Latencia media en CPU** | 34,2 ms | **34,2 – 44,5 ms** | 169,7 ms | **~3,8× a 4,9× más rápido** |
| **Accuracy en Test (`n=247`)** | 80,57 % | **92,71 %** | **95,14 %** | Retiene el **97,45 %** del Accuracy del Maestro |
| **F1 Macro en Test (`n=247`)** | 79,74 % | **92,52 %** | **94,87 %** | Retiene el **97,52 %** del F1 del Maestro |
| **Errores totales (de 247)** | 48 errores | **18 errores** | **12 errores** | Solo 6 imágenes de diferencia en 247 pruebas |
| **Confianza media en ejemplos** | 84,4 % | **98,0 %** | **99,9 %** | Alta certidumbre en ambas redes |

---

### 3.2 Desglose por clase en el conjunto de prueba (`n = 247`)

| Clase | Soporte (`test`) | Métrica | `MobileNetV2` (Estudiante · **Principal**) | `ResNet50` (Maestro / Padre) | Diferencia |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **🚲 Bicicleta** | 90 | Precisión / Recall / **F1** / AUC | 0,931 / 0,900 / **0,915** / 0,974 | 0,965 / 0,922 / **0,943** / 0,990 | −0,028 F1 |
| **🚗 Carro** | 90 | Precisión / Recall / **F1** / AUC | 0,977 / 0,944 / **0,960** / 0,993 | 1,000 / 0,978 / **0,989** / 0,996 | −0,029 F1 |
| **🏍️ Moto** | 67 | Precisión / Recall / **F1** / AUC | 0,863 / 0,940 / **0,900** / 0,977 | 0,877 / 0,955 / **0,914** / 0,987 | −0,014 F1 |

---

## 4. Análisis y veredicto: ¿Cuál modelo es mejor?

* **En exactitud absoluta de laboratorio:** El mejor es **`ResNet50` (Maestro)** con **95,14 % de Accuracy y 94,87 % de F1 macro** (solo 12 errores en 247 imágenes y 100 % de precisión en la clase `carro`).
* **En ingeniería y despliegue en producción (Veredicto final):** El ganador y **modelo principal del sistema es `MobileNetV2` (Estudiante Destilado)**, porque comprime el tamaño del modelo en casi **10 veces** (`2,55M` vs. `24,56M` parámetros; `10,0 MB` vs. `94,0 MB`) y multiplica por **~4× la velocidad en CPU** (`~35-45 ms` vs. `169,7 ms`), reteniendo el **97,5 % del F1 macro** del Maestro (una diferencia de apenas 6 imágenes sobre 247).
