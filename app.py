"""SemaforoIA · clasificador de vehículos (carro, moto, bicicleta) con MobileNetV2 + Grad-CAM.

Ejecutar en local:  streamlit run app/app.py
"""
from pathlib import Path
import json

import matplotlib
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms as T
from torchvision.models import mobilenet_v2

AQUI = Path(__file__).resolve().parent
RUTA_MODELO = AQUI / "modelo" / "mobilenetv2_semaforo.pt"
DIR_ARTEFACTOS = AQUI / "artefactos"
DIR_EJEMPLOS = AQUI / "ejemplos"
UMBRAL_CONFIANZA = 0.60
EMOJI = {"carro": "🚗", "moto": "🏍️", "bicicleta": "🚲"}

st.set_page_config(page_title="SemaforoIA · Clasificador de vehículos", page_icon=" ", layout="wide")


@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_modelo():
    paquete = torch.load(RUTA_MODELO, map_location="cpu", weights_only=True)
    modelo = mobilenet_v2(weights=None)
    modelo.classifier = nn.Sequential(
        nn.Dropout(0.2), nn.Linear(modelo.last_channel, 256), nn.ReLU(inplace=True),
        nn.Dropout(0.3), nn.Linear(256, len(paquete["clases"])),
    )
    modelo.load_state_dict(paquete["state_dict"])
    modelo.eval()
    tam = paquete["tam_imagen"]
    recortar = T.Compose([T.Resize(256), T.CenterCrop(tam)])
    normalizar = T.Compose([T.ToTensor(), T.Normalize(paquete["media"], paquete["desviacion"])])
    return modelo, paquete["clases"], recortar, normalizar


def leer_json(nombre: str) -> dict:
    ruta = DIR_ARTEFACTOS / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def preparar(archivo) -> Image.Image:
    """Abre la imagen, corrige la orientación y resuelve la transparencia sobre blanco."""
    imagen = ImageOps.exif_transpose(Image.open(archivo))
    if "A" in imagen.getbands() or "transparency" in imagen.info:
        rgba = imagen.convert("RGBA")
        fondo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(fondo, rgba).convert("RGB")
    return imagen.convert("RGB")


def predecir_con_gradcam(modelo, tensor, clase=None):
    """Probabilidades y mapa Grad-CAM (0-1) para la clase indicada (por defecto, la predicha)."""
    x = tensor.unsqueeze(0).requires_grad_(True)
    mapas = modelo.features(x)                              # última capa convolucional: [1, 1280, 7, 7]
    logits = modelo.classifier(mapas.mean(dim=(2, 3)))
    probabilidades = F.softmax(logits, dim=1)[0].detach().numpy()
    clase = int(logits.argmax(1)) if clase is None else clase
    gradiente = torch.autograd.grad(logits[0, clase], mapas)[0]
    pesos = gradiente.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((pesos * mapas).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return probabilidades, cam.detach().numpy()


def superponer(imagen: Image.Image, cam: np.ndarray, alfa: float) -> Image.Image:
    base = np.asarray(imagen, dtype=np.float32) / 255
    calor = matplotlib.colormaps["jet"](cam)[..., :3]
    mezcla = (1 - alfa) * base + alfa * calor
    return Image.fromarray((mezcla * 255).astype(np.uint8))


def grande(imagen: Image.Image, lado: int = 420) -> Image.Image:
    return imagen.resize((lado, lado), Image.Resampling.LANCZOS)


# ------------------------------------------------------------------ cabecera
st.title("🚦 SemaforoIA")
st.caption("Clasificador de vehículos con transfer learning · MobileNetV2 · carro, moto o bicicleta")

if not RUTA_MODELO.exists():
    st.error("No se encontró `app/modelo/mobilenetv2_semaforo.pt`. Ejecuta primero el notebook "
             "`02_entrenamiento.ipynb` para generar el modelo.")
    st.stop()

modelo, CLASES, recortar, normalizar = cargar_modelo()
metricas = leer_json("metricas_test.json")
entrenamiento = leer_json("resumen_entrenamiento.json")
comparacion = leer_json("comparacion_modelos.json")

PESO_MB = RUTA_MODELO.stat().st_size / 1048576
# Separador de miles con punto, como se escribe en español.
PARAMETROS = f"{entrenamiento.get('parametros_totales', 0):,}".replace(",", ".")

with st.sidebar:
    st.header("El modelo")
    st.markdown(
        "- **Arquitectura:** MobileNetV2 preentrenada en ImageNet\n"
        "- **Cabeza nueva:** Dropout → Linear → ReLU → Dropout → Linear(3)\n"
        "- **Entrada:** 224 × 224 px\n"
        f"- **Clases:** {', '.join(CLASES)}\n"
        f"- **Tamaño del archivo:** {PESO_MB:.1f} MB\n"
        f"- **Parámetros:** {PARAMETROS}"
    )
    if metricas:
        st.divider()
        st.metric("Accuracy en test", f"{metricas['accuracy']:.1%}")
        st.metric("F1 macro en test", f"{metricas['f1_macro']:.1%}")
        st.caption(f"Medido sobre {metricas['n_test']} imágenes que el modelo nunca vio.")
    st.divider()
    alfa = st.slider("Intensidad del mapa de calor", 0.2, 0.8, 0.45, 0.05)

pestana_clasificar, pestana_metricas, pestana_como = st.tabs(["Clasificar una imagen", "Métricas del modelo", "Cómo funciona"])

# ------------------------------------------------------------------ clasificar
with pestana_clasificar:
    ejemplos = sorted(DIR_EJEMPLOS.glob("*.jpg")) if DIR_EJEMPLOS.exists() else []
    columna_subir, columna_ejemplo = st.columns([3, 2])
    with columna_subir:
        subida = st.file_uploader("Sube una foto de un carro, una moto o una bicicleta",
                                  type=["jpg", "jpeg", "png", "webp"])
    with columna_ejemplo:
        opciones = ["—"] + [p.stem.replace("_", " ") for p in ejemplos]
        elegido = st.selectbox("…o prueba con una imagen de ejemplo (del conjunto de prueba)", opciones,
                               disabled=not ejemplos)

    origen = subida if subida is not None else (ejemplos[opciones.index(elegido) - 1] if elegido != "—" else None)

    if origen is None:
        st.info("Sube una imagen o elige un ejemplo para ver la predicción y el Grad-CAM.")
    else:
        try:
            imagen = preparar(origen)
        except Exception:
            st.error("No se pudo leer el archivo como imagen. Prueba con un JPG o PNG.")
            st.stop()

        vista = recortar(imagen)                       # exactamente lo que recibe la red
        probabilidades, cam = predecir_con_gradcam(modelo, normalizar(vista))
        ganadora = int(probabilidades.argmax())
        confianza = float(probabilidades[ganadora])

        columna_imagen, columna_cam, columna_resultado = st.columns([2, 2, 2])
        with columna_imagen:
            st.subheader("Imagen")
            st.image(grande(vista), caption="Recorte central 224 × 224 que recibe la red")
        with columna_cam:
            st.subheader("Grad-CAM")
            st.image(grande(superponer(vista, cam, alfa)),
                     caption=f"Rojo = zonas que más pesaron para decir «{CLASES[ganadora]}»")
        with columna_resultado:
            st.subheader("Predicción")
            st.markdown(f"## {EMOJI.get(CLASES[ganadora], '')} {CLASES[ganadora].capitalize()}")
            st.metric("Confianza", f"{confianza:.1%}")
            if confianza < UMBRAL_CONFIANZA:
                st.warning("Confianza baja. El modelo solo conoce carros, motos y bicicletas: si la imagen "
                           "muestra otra cosa, igual elegirá una de las tres.")
            st.markdown("**Probabilidad por clase**")
            for i in np.argsort(probabilidades)[::-1]:
                st.progress(float(probabilidades[i]), text=f"{EMOJI.get(CLASES[i], '')} {CLASES[i]} · {probabilidades[i]:.1%}")

# ------------------------------------------------------------------ métricas
with pestana_metricas:
    if not metricas:
        st.info("Aún no hay métricas. Ejecuta `03_evaluacion_gradcam.ipynb` para generarlas.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metricas['accuracy']:.1%}")
        c2.metric("F1 macro", f"{metricas['f1_macro']:.1%}")
        c3.metric("Imágenes de prueba", metricas["n_test"])
        c4.metric("Errores", metricas["errores"])
        st.caption("Todas las métricas se calcularon sobre el conjunto de prueba (15 % de los datos), "
                   "que no se usó ni para entrenar ni para elegir el mejor modelo.")
        st.caption(
            f"El archivo del modelo pesa **{PESO_MB:.1f} MB** y tiene **{PARAMETROS} parámetros**. "
            "MobileNetV2 está diseñada para ser ligera: una ResNet50 rondaría los 100 MB y 25 "
            f"millones de parámetros. Por eso la inferencia tarda "
            f"{metricas.get('latencia', {}).get('cpu_ms', '?')} ms en CPU."
        )

        if comparacion:
            anterior, actual = comparacion["anterior"], comparacion["actual"]
            with st.container(border=True):
                st.subheader("Modelo con más confianza")
                st.caption(
                    f"La versión anterior se entrenó con un conjunto desbalanceado "
                    f"(711 carros frente a 33 bicicletas). Esta se entrenó con uno equilibrado. "
                    f"Ambas medidas sobre las mismas {comparacion['n_test']} imágenes de prueba."
                )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("F1 macro", f"{actual['f1_macro']:.1%}",
                          f"{(actual['f1_macro']-anterior['f1_macro'])*100:+.1f} pp")
                m2.metric("Accuracy", f"{actual['accuracy']:.1%}",
                          f"{(actual['accuracy']-anterior['accuracy'])*100:+.1f} pp")
                m3.metric("Errores", actual["errores"],
                          actual["errores"] - anterior["errores"], delta_color="inverse")
                m4.metric("Confianza media en los ejemplos",
                          f"{actual['confianza_media_ejemplos']:.1%}",
                          f"{(actual['confianza_media_ejemplos']-anterior['confianza_media_ejemplos'])*100:+.1f} pp")

                st.markdown("**Recall por clase: cuántas de cada clase reconoce**")
                st.table({
                    "clase": list(actual["por_clase"]),
                    "antes": [f"{v['recall']:.1%}" for v in anterior["por_clase"].values()],
                    "ahora": [f"{v['recall']:.1%}" for v in actual["por_clase"].values()],
                    "cambio": [f"{(actual['por_clase'][c]['recall']-anterior['por_clase'][c]['recall'])*100:+.1f} pp"
                               for c in actual["por_clase"]],
                })

                st.markdown("**Confianza en cada imagen de ejemplo**")
                st.table({
                    "ejemplo": [e["ejemplo"] for e in actual["ejemplos"]],
                    "antes": [f"{e['confianza']:.1%}" for e in anterior["ejemplos"]],
                    "ahora": [f"{e['confianza']:.1%}" for e in actual["ejemplos"]],
                })
                st.caption(
                    "Las dos versiones aciertan los 5 ejemplos, pero la actual lo hace con "
                    "mucha menos duda. Solo `moto_1` baja, de 95.1 % a 89.8 %. "
                    "Antes `carro_2` se quedaba en 56.9 %, por debajo del umbral de aviso."
                )

        st.subheader("Por clase")
        st.table({
            "clase": list(metricas["por_clase"]),
            "precisión": [f"{v['precision']:.3f}" for v in metricas["por_clase"].values()],
            "recall": [f"{v['recall']:.3f}" for v in metricas["por_clase"].values()],
            "F1": [f"{v['f1']:.3f}" for v in metricas["por_clase"].values()],
            "AUC": [f"{v['auc']:.3f}" for v in metricas["por_clase"].values()],
            "imágenes": [v["soporte"] for v in metricas["por_clase"].values()],
        })

        figuras = [
            ("matriz_confusion.png", "Matriz de confusión", "Filas: clase real. Columnas: predicción. La diagonal son los aciertos."),
            ("curvas_entrenamiento.png", "Curvas de entrenamiento", "Fase 1: solo la cabeza. Fase 2: fine-tuning de los últimos bloques. La estrella marca la época guardada."),
            ("curvas_roc.png", "Curvas ROC", "Una clase contra el resto. AUC = 1 es separación perfecta."),
            ("gradcam_aciertos.png", "Grad-CAM en aciertos", "Dónde miró la red cuando acertó."),
            ("gradcam_errores.png", "Grad-CAM en errores", "Los errores cometidos con mayor confianza."),
            ("distribucion_clases.png", "Datos por clase y partición", "El desbalance se compensó con pesos por clase en la pérdida."),
        ]
        for archivo, titulo, pie in figuras:
            if (DIR_ARTEFACTOS / archivo).exists():
                st.subheader(titulo)
                st.image(str(DIR_ARTEFACTOS / archivo), caption=pie)

# ------------------------------------------------------------------ cómo funciona
with pestana_como:
    st.markdown(
        """
### Transfer learning
MobileNetV2 ya aprendió a reconocer formas, texturas y objetos con 1,28 millones de imágenes de
ImageNet. En vez de entrenar desde cero con ~2 000 fotos, **reutilizamos ese extractor** y
reemplazamos su última capa (1000 clases) por una cabeza propia de 3 clases.

1. **Fase 1 – extracción de características:** el extractor queda congelado y solo se entrena la cabeza.
2. **Fase 2 – fine-tuning:** se descongelan los últimos bloques y se ajustan con una tasa de aprendizaje 10× menor.

### Funciones de activación
- **ReLU6** dentro de MobileNetV2: `min(max(0, x), 6)`.
- **ReLU** en la cabeza: `max(0, x)`; aporta la no linealidad sin desvanecer el gradiente.
- **Softmax** a la salida: convierte los 3 puntajes en probabilidades que suman 1.

### Grad-CAM
Toma los 1280 mapas 7 × 7 de la última capa convolucional, pondera cada uno por el gradiente del
puntaje de la clase y los suma. El resultado indica **qué regiones de la imagen empujaron la
decisión**: sirve para comprobar que la red mira el vehículo y no el fondo.

### Límites conocidos
- Solo conoce 3 clases: ante cualquier otra imagen responderá con una de ellas.
- Clasifica la imagen completa; no cuenta ni localiza vehículos.
"""
    )
    if entrenamiento:
        st.caption(f"Entrenado {entrenamiento.get('epocas_ejecutadas', '?')} épocas · mejor época "
                   f"{entrenamiento.get('mejor_epoca', '?')} · PyTorch {entrenamiento.get('pytorch', '')}")
