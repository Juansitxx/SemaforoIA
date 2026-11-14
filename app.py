"""SemaforoIA Â· clasificador de vehÃ­culos (carro, moto, bicicleta) con Knowledge Distillation:
ResNet50 (Modelo Maestro / Padre) -> MobileNetV2 (Modelo Estudiante / Principal en producciÃ³n) + Grad-CAM.

Ejecutar en local:  streamlit run app.py
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
from torchvision.models import mobilenet_v2, resnet50

AQUI = Path(__file__).resolve().parent
RUTA_ESTUDIANTE = AQUI / "modelo" / "mobilenetv2_semaforo.pt"
RUTA_MAESTRO = AQUI / "modelo" / "resnet50_maestro.pt"
DIR_ARTEFACTOS = AQUI / "artefactos"
DIR_EJEMPLOS = AQUI / "ejemplos"
UMBRAL_CONFIANZA = 0.60
EMOJI = {"carro": "ðŸš—", "moto": "ðŸï¸", "bicicleta": "ðŸš²"}

MODELOS_INFO = {
    "mobilenetv2": {
        "etiqueta": "MobileNetV2 Â· Estudiante Destilado (Principal / ProducciÃ³n)",
        "nombre_corto": "MobileNetV2 (Estudiante Â· Principal)",
        "ruta": RUTA_ESTUDIANTE,
        "metricas_json": "metricas_test.json",
        "arquitectura": "MobileNetV2 preentrenada en ImageNet + Knowledge Distillation desde ResNet50",
        "cabeza": "Dropout(0.2) â†’ Linear(1280, 256) â†’ ReLU â†’ Dropout(0.3) â†’ Linear(256, 3)",
        "rol": "Estudiante (Hijo) Â· Modelo principal de uso en producciÃ³n",
    },
    "resnet50": {
        "etiqueta": "ResNet50 Â· Modelo Maestro / Padre (Referencia pesada)",
        "nombre_corto": "ResNet50 (Maestro / Padre)",
        "ruta": RUTA_MAESTRO,
        "metricas_json": "metricas_resnet50.json",
        "arquitectura": "ResNet50 preentrenada en ImageNet V2 (Arquitectura Padre para destilaciÃ³n)",
        "cabeza": "Dropout(0.3) â†’ Linear(2048, 512) â†’ ReLU â†’ Dropout(0.3) â†’ Linear(512, 3)",
        "rol": "Maestro (Padre) Â· Usado para enseÃ±ar a MobileNetV2 mediante Knowledge Distillation",
    },
}

st.set_page_config(page_title="SemaforoIA Â· Clasificador de vehÃ­culos", page_icon="ðŸš¦", layout="wide")


@st.cache_resource(show_spinner="Cargando el modeloâ€¦")
def cargar_modelo(clave: str):
    ruta = MODELOS_INFO[clave]["ruta"]
    paquete = torch.load(ruta, map_location="cpu", weights_only=True)
    clases = paquete["clases"]
    if clave == "mobilenetv2":
        modelo = mobilenet_v2(weights=None)
        modelo.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(modelo.last_channel, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, len(clases)),
        )
    else:
        modelo = resnet50(weights=None)
        modelo.fc = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(2048, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, len(clases)),
        )
    estado = {
        k: (v.float() if v.is_floating_point() else v)
        for k, v in paquete["state_dict"].items()
    }
    modelo.load_state_dict(estado)
    modelo.eval()
    tam = paquete.get("tam_imagen", 224)
    recortar = T.Compose([T.Resize(256), T.CenterCrop(tam)])
    normalizar = T.Compose([T.ToTensor(), T.Normalize(paquete["media"], paquete["desviacion"])])
    parametros = sum(p.numel() for p in modelo.parameters())
    peso_fp32_mb = paquete.get("peso_fp32_mb", round(ruta.stat().st_size / 1048576, 1))
    peso_archivo_mb = round(ruta.stat().st_size / 1048576, 1)
    return modelo, clases, recortar, normalizar, parametros, peso_fp32_mb, peso_archivo_mb


def leer_json(nombre: str) -> dict:
    ruta = DIR_ARTEFACTOS / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def preparar(archivo) -> Image.Image:
    """Abre la imagen, corrige la orientaciÃ³n y resuelve la transparencia sobre blanco."""
    imagen = ImageOps.exif_transpose(Image.open(archivo))
    if "A" in imagen.getbands() or "transparency" in imagen.info:
        rgba = imagen.convert("RGBA")
        fondo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(fondo, rgba).convert("RGB")
    return imagen.convert("RGB")


def predecir_con_gradcam(modelo, clave_modelo: str, tensor, clase=None):
    """Probabilidades y mapa Grad-CAM (0-1) para MobileNetV2 o ResNet50."""
    x = tensor.unsqueeze(0).requires_grad_(True)
    if clave_modelo == "mobilenetv2":
        mapas = modelo.features(x)                          # [1, 1280, 7, 7]
        logits = modelo.classifier(mapas.mean(dim=(2, 3)))
    else:
        x0 = modelo.maxpool(modelo.relu(modelo.bn1(modelo.conv1(x))))
        mapas = modelo.layer4(modelo.layer3(modelo.layer2(modelo.layer1(x0))))  # [1, 2048, 7, 7]
        logits = modelo.fc(modelo.avgpool(mapas).flatten(1))

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
st.title("ðŸš¦ SemaforoIA")
st.caption(
    "Clasificador de vehÃ­culos con Transfer Learning y Knowledge Distillation Â· "
    "**MobileNetV2 (Modelo Principal)** destilado desde **ResNet50 (Modelo Maestro)** Â· carro, moto o bicicleta"
)

if not RUTA_ESTUDIANTE.exists():
    st.error("No se encontrÃ³ `modelo/mobilenetv2_semaforo.pt`.")
    st.stop()

entrenamiento = leer_json("resumen_entrenamiento.json")
comparacion = leer_json("comparacion_modelos.json")
metricas_mbv2 = leer_json("metricas_test.json")
metricas_resnet = leer_json("metricas_resnet50.json")

modelos_disponibles = ["mobilenetv2"] + (["resnet50"] if RUTA_MAESTRO.exists() else [])

with st.sidebar:
    st.header("SelecciÃ³n de modelo")
    clave_activa = st.selectbox(
        "Arquitectura para inferencia",
        options=modelos_disponibles,
        index=0,
        format_func=lambda k: MODELOS_INFO[k]["etiqueta"],
        help="MobileNetV2 es el modelo principal para despliegue por su bajo peso (10 MB) y baja latencia. "
             "ResNet50 es el modelo Maestro (Padre) con el que se aplicÃ³ Knowledge Distillation.",
    )
    if clave_activa == "mobilenetv2":
        st.success("**Modelo principal activo (MobileNetV2):** optimizado mediante *Knowledge Distillation* para operar ligero y rÃ¡pido en CPU/borde.")
    else:
        st.warning("**Modelo Maestro activo (ResNet50):** arquitectura padre pesada (~94 MB en FP32). Se incluye como referencia comparativa; el modelo principal de producciÃ³n es **MobileNetV2**.")

modelo, CLASES, recortar, normalizar, num_params, peso_fp32_mb, peso_archivo_mb = cargar_modelo(clave_activa)
metricas = leer_json(MODELOS_INFO[clave_activa]["metricas_json"]) or metricas_mbv2
PARAMETROS = f"{num_params:,}".replace(",", ".")
info_activa = MODELOS_INFO[clave_activa]

with st.sidebar:
    st.subheader("Ficha tÃ©cnica")
    texto_peso = (
        f"{peso_fp32_mb:.1f} MB (comprimido a {peso_archivo_mb:.1f} MB en FP16)"
        if peso_archivo_mb != peso_fp32_mb else f"{peso_fp32_mb:.1f} MB"
    )
    st.markdown(
        f"- **Rol:** {info_activa['rol']}\n"
        f"- **Arquitectura:** {info_activa['arquitectura']}\n"
        f"- **Cabeza:** `{info_activa['cabeza']}`\n"
        f"- **Entrada:** 224 Ã— 224 px\n"
        f"- **Clases:** {', '.join(CLASES)}\n"
        f"- **TamaÃ±o del modelo:** {texto_peso}\n"
        f"- **ParÃ¡metros:** {PARAMETROS}"
    )
    if metricas:
        st.divider()
        st.metric("Accuracy en test", f"{metricas['accuracy']:.1%}")
        st.metric("F1 macro en test", f"{metricas['f1_macro']:.1%}")
        st.caption(f"Medido sobre {metricas['n_test']} imÃ¡genes que el modelo nunca vio.")
    st.divider()
    alfa = st.slider("Intensidad del mapa de calor", 0.2, 0.8, 0.45, 0.05)

pestana_clasificar, pestana_metricas, pestana_como = st.tabs([
    "Clasificar una imagen",
    "MÃ©tricas y DestilaciÃ³n",
    "CÃ³mo funciona",
])

# ------------------------------------------------------------------ clasificar
with pestana_clasificar:
    st.info(
        f"**Modelo en uso:** {info_activa['nombre_corto']}. "
        "**Nota:** El modelo principal del proyecto es **MobileNetV2 (Estudiante)** por ser **9.4Ã— mÃ¡s liviano** "
        "y **~3.8Ã— mÃ¡s rÃ¡pido** en CPU; puedes alternar en la barra lateral con **ResNet50 (Maestro)** para comparar sus predicciones y mapas Grad-CAM."
    )
    ejemplos = sorted(DIR_EJEMPLOS.glob("*.jpg")) if DIR_EJEMPLOS.exists() else []
    columna_subir, columna_ejemplo = st.columns([3, 2])
    with columna_subir:
        subida = st.file_uploader("Sube una foto de un carro, una moto o una bicicleta",
                                  type=["jpg", "jpeg", "png", "webp"])
    with columna_ejemplo:
        opciones = ["â€”"] + [p.stem.replace("_", " ") for p in ejemplos]
        elegido = st.selectbox("â€¦o prueba con una imagen de ejemplo (del conjunto de prueba)", opciones,
                               disabled=not ejemplos)

    origen = subida if subida is not None else (ejemplos[opciones.index(elegido) - 1] if elegido != "â€”" else None)

    if origen is None:
        st.caption("Sube una imagen o elige un ejemplo para ver la predicciÃ³n y el Grad-CAM.")
    else:
        try:
            imagen = preparar(origen)
        except Exception:
            st.error("No se pudo leer el archivo como imagen. Prueba con un JPG o PNG.")
            st.stop()

        vista = recortar(imagen)                       # exactamente lo que recibe la red
        tensor_entrada = normalizar(vista)
        probabilidades, cam = predecir_con_gradcam(modelo, clave_activa, tensor_entrada)
        ganadora = int(probabilidades.argmax())
        confianza = float(probabilidades[ganadora])

        columna_imagen, columna_cam, columna_resultado = st.columns([2, 2, 2])
        with columna_imagen:
            st.subheader("Imagen")
            st.image(grande(vista), caption="Recorte central 224 Ã— 224 que recibe la red")
        with columna_cam:
            st.subheader(f"Grad-CAM Â· {info_activa['nombre_corto']}")
            st.image(grande(superponer(vista, cam, alfa)),
                     caption=f"Rojo = zonas que mÃ¡s pesaron para decir Â«{CLASES[ganadora]}Â»")
        with columna_resultado:
            st.subheader("PredicciÃ³n")
            st.markdown(f"## {EMOJI.get(CLASES[ganadora], '')} {CLASES[ganadora].capitalize()}")
            st.metric("Confianza", f"{confianza:.1%}")
            if confianza < UMBRAL_CONFIANZA:
                st.warning("Confianza baja. El modelo solo conoce carros, motos y bicicletas: si la imagen "
                           "muestra otra cosa, igual elegirÃ¡ una de las tres.")
            st.markdown("**Probabilidad por clase**")
            for i in np.argsort(probabilidades)[::-1]:
                st.progress(float(probabilidades[i]), text=f"{EMOJI.get(CLASES[i], '')} {CLASES[i]} Â· {probabilidades[i]:.1%}")

# ------------------------------------------------------------------ mÃ©tricas
with pestana_metricas:
    if not metricas:
        st.info("AÃºn no hay mÃ©tricas disponibles.")
    else:
        st.subheader(f"Rendimiento en test Â· {info_activa['nombre_corto']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metricas['accuracy']:.1%}")
        c2.metric("F1 macro", f"{metricas['f1_macro']:.1%}")
        c3.metric("ImÃ¡genes de prueba", metricas["n_test"])
        c4.metric("Errores", metricas["errores"])
        st.caption("Todas las mÃ©tricas se calcularon sobre el conjunto de prueba (15 % de los datos, 247 imÃ¡genes), "
                   "que no se usÃ³ ni para entrenar ni para elegir el mejor modelo.")
        st.caption(
            f"El modelo seleccionado pesa **{peso_fp32_mb:.1f} MB** en precisiÃ³n completa, tiene **{PARAMETROS} parÃ¡metros** "
            f"y tarda **{metricas.get('latencia', {}).get('cpu_ms', '?')} ms** por imagen en CPU."
        )

        if metricas_resnet and metricas_mbv2:
            with st.container(border=True):
                st.subheader("Knowledge Distillation: Maestro (ResNet50) âž” Estudiante Principal (MobileNetV2)")
                kd_info = entrenamiento.get("destilacion", {})
                st.caption(
                    f"Se entrenÃ³ primero un modelo padre pesado (**ResNet50**, 24.6M de parÃ¡metros y 94.0 MB) y se "
                    f"transfiriÃ³ su conocimiento (probabilidades suavizadas con Temperatura $T = {kd_info.get('temperatura', 4.0)}$ "
                    f"y peso $\\alpha = {kd_info.get('alfa_kd', 0.7)}$) hacia **MobileNetV2**, que es el **modelo principal de producciÃ³n**."
                )
                k1, k2, k3, k4 = st.columns(4)
                k1.metric(
                    "ReducciÃ³n de parÃ¡metros",
                    "2,55 M vs 24,56 M",
                    f"-{(1 - 2552579 / 24558659) * 100:.1f}% (9.6Ã— menor)",
                )
                k2.metric(
                    "ReducciÃ³n de peso (FP32)",
                    "10,0 MB vs 94,0 MB",
                    "-89.4% (9.4Ã— mÃ¡s ligero)",
                )
                lat_m = metricas_resnet.get("latencia", {}).get("cpu_ms", 169.7)
                lat_e = metricas_mbv2.get("latencia", {}).get("cpu_ms", 44.5)
                k3.metric(
                    "Latencia en CPU",
                    f"{lat_e:.1f} ms vs {lat_m:.1f} ms",
                    f"{lat_m / max(lat_e, 0.1):.1f}Ã— mÃ¡s rÃ¡pido",
                )
                k4.metric(
                    "RetenciÃ³n de F1 macro",
                    f"{metricas_mbv2['f1_macro']:.1%} vs {metricas_resnet['f1_macro']:.1%}",
                    f"{(metricas_mbv2['f1_macro'] / metricas_resnet['f1_macro']) * 100:.1f}% del Maestro",
                )

                st.markdown("**Comparativa directa sobre las 247 imÃ¡genes de prueba**")
                st.table({
                    "Modelo": [
                        "MobileNetV2 Â· Estudiante Destilado (Principal)",
                        "ResNet50 Â· Maestro / Padre (Referencia)",
                    ],
                    "ParÃ¡metros": ["2.552.579", "24.558.659"],
                    "Peso (FP32)": ["10.0 MB", "94.0 MB (47.0 MB en FP16)"],
                    "Latencia CPU": [f"{lat_e:.1f} ms", f"{lat_m:.1f} ms"],
                    "Accuracy Test": [f"{metricas_mbv2['accuracy']:.2%}", f"{metricas_resnet['accuracy']:.2%}"],
                    "F1 Macro Test": [f"{metricas_mbv2['f1_macro']:.2%}", f"{metricas_resnet['f1_macro']:.2%}"],
                    "Errores (de 247)": [metricas_mbv2["errores"], metricas_resnet["errores"]],
                })

        if comparacion:
            anterior, actual = comparacion["anterior"], comparacion["actual"]
            with st.container(border=True):
                st.subheader("EvoluciÃ³n del dataset: v1 (desbalanceado) vs v2 (equilibrado)")
                st.caption(
                    f"La primera versiÃ³n se entrenÃ³ con un conjunto desbalanceado "
                    f"(711 carros frente a 33 bicicletas). La versiÃ³n actual se entrenÃ³ sobre 1.645 imÃ¡genes equilibradas. "
                    f"Ambas medidas sobre las mismas {comparacion['n_test']} imÃ¡genes de prueba."
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

                st.markdown("**Recall por clase: cuÃ¡ntas de cada clase reconoce**")
                st.table({
                    "clase": list(actual["por_clase"]),
                    "antes": [f"{v['recall']:.1%}" for v in anterior["por_clase"].values()],
                    "ahora": [f"{v['recall']:.1%}" for v in actual["por_clase"].values()],
                    "cambio": [f"{(actual['por_clase'][c]['recall']-anterior['por_clase'][c]['recall'])*100:+.1f} pp"
                               for c in actual["por_clase"]],
                })

        st.subheader(f"Desglose por clase Â· {info_activa['nombre_corto']}")
        st.table({
            "clase": list(metricas["por_clase"]),
            "precisiÃ³n": [f"{v['precision']:.3f}" for v in metricas["por_clase"].values()],
            "recall": [f"{v['recall']:.3f}" for v in metricas["por_clase"].values()],
            "F1": [f"{v['f1']:.3f}" for v in metricas["por_clase"].values()],
            "AUC": [f"{v.get('auc', 0.0):.3f}" for v in metricas["por_clase"].values()],
            "imÃ¡genes": [v["soporte"] for v in metricas["por_clase"].values()],
        })

        figuras = [
            ("matriz_confusion.png", "Matriz de confusiÃ³n (Modelo Principal MobileNetV2)", "Filas: clase real. Columnas: predicciÃ³n. La diagonal son los aciertos."),
            ("curvas_entrenamiento.png", "Curvas de Knowledge Distillation (Maestro ResNet50 âž” Estudiante MobileNetV2)", "Izquierda: pÃ©rdida KL-Divergence (T=4) + Cross-Entropy. Derecha: validaciÃ³n del Estudiante frente al techo del Maestro ResNet50."),
            ("curvas_roc.png", "Curvas ROC", "Una clase contra el resto. AUC = 1 es separaciÃ³n perfecta."),
            ("gradcam_aciertos.png", "Grad-CAM en aciertos", "DÃ³nde mirÃ³ la red cuando acertÃ³."),
            ("gradcam_errores.png", "Grad-CAM en errores", "Los errores cometidos con mayor confianza."),
            ("distribucion_clases.png", "Datos por clase y particiÃ³n", "El desbalance se compensÃ³ con pesos por clase en la pÃ©rdida."),
        ]
        for archivo, titulo, pie in figuras:
            if (DIR_ARTEFACTOS / archivo).exists():
                st.subheader(titulo)
                st.image(str(DIR_ARTEFACTOS / archivo), caption=pie)

# ------------------------------------------------------------------ cÃ³mo funciona
with pestana_como:
    st.markdown(
        """
### 1. Â¿Por quÃ© hay dos modelos y cuÃ¡l es el principal?
- **Modelo Principal de ProducciÃ³n (`MobileNetV2` Â· Estudiante):** Es el modelo predeterminado de la aplicaciÃ³n. Pesa apenas **10.0 MB**, tiene **2,55 millones de parÃ¡metros** y clasifica en **~35â€“45 ms en CPU**, ideal para ejecutarse en dispositivos de borde o servidores ligeros sin GPU.
- **Modelo Maestro (`ResNet50` Â· Padre):** Es una arquitectura profunda de **24,56 millones de parÃ¡metros (~94 MB en FP32)** que alcanza **95.1 % de accuracy** en prueba, pero requiere casi 10 veces mÃ¡s memoria y casi 4 veces mÃ¡s tiempo de cÃ³mputo por imagen.

### 2. Knowledge Distillation (DestilaciÃ³n de Conocimiento)
En lugar de entrenar `MobileNetV2` Ãºnicamente con etiquetas rÃ­gidas (`[0, 1, 0]`), entrenamos primero el **Maestro (`ResNet50`)** sobre el mismo dataset y usamos sus predicciones suavizadas con una **Temperatura $T = 4.0$** (*soft targets*) para guiar a `MobileNetV2`:

$$\\mathcal{L}_{\\text{total}} = \\alpha \\cdot T^2 \\cdot \\text{KLDiv}\\!\\left(\\sigma\\!\\left(\\frac{z_S}{T}\\right), \\sigma\\!\\left(\\frac{z_T}{T}\\right)\\right) + (1 - \\alpha) \\cdot \\text{CrossEntropy}(z_S, y)$$

De esta forma, `MobileNetV2` aprende no solo quÃ© clase es cada foto, sino las relaciones visuales entre clases que descubriÃ³ `ResNet50` (por ejemplo, cuÃ¡ndo el Ã¡ngulo de una moto comparte rasgos con una bicicleta).

### 3. Transfer Learning y Funciones de ActivaciÃ³n
- **ReLU6** dentro de MobileNetV2: `min(max(0, x), 6)`.
- **ReLU** en las cabezas clasificadoras: `max(0, x)`.
- **Softmax** a la salida: convierte los 3 puntajes (*logits*) en probabilidades que suman 1.

### 4. Grad-CAM en ambas arquitecturas
Extrae los mapas de activaciÃ³n `7 Ã— 7` de la Ãºltima etapa convolucional (`features` en MobileNetV2 y `layer4` en ResNet50), los pondera por el gradiente de la clase predicha y genera el mapa de calor que muestra **en quÃ© parte de la imagen se fijÃ³ cada modelo**.
"""
    )
    if entrenamiento:
        st.caption(
            f"DestilaciÃ³n ejecutada durante {entrenamiento.get('epocas_ejecutadas', '?')} Ã©pocas Â· "
            f"Temperatura T={entrenamiento.get('temperatura_kd', 4.0)} Â· Î±={entrenamiento.get('alfa_kd', 0.7)} Â· "
            f"PyTorch {entrenamiento.get('pytorch', '')}"
        )
