"""Evalua el modelo sobre el conjunto de prueba y genera todos los artefactos.

Produce exactamente los nombres de archivo que la app busca:
  metricas_test.json, matriz_confusion.png, curvas_roc.png,
  gradcam_aciertos.png, gradcam_errores.png

AUC y ROC se calculan a mano (formula de rangos de Mann-Whitney) para no meter
scipy ni scikit-learn en el entorno que corre la app.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.models import mobilenet_v2

CACHE = Path("cache")
RUTA_MODELO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("modelo") / "mobilenetv2_semaforo.pt"
SALIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artefactos")

paquete = torch.load(RUTA_MODELO, map_location="cpu", weights_only=True)
CLASES = paquete["clases"]
TAM = paquete["tam_imagen"]

modelo = mobilenet_v2(weights=None)
modelo.classifier = nn.Sequential(
    nn.Dropout(0.2), nn.Linear(1280, 256), nn.ReLU(inplace=True),
    nn.Dropout(0.3), nn.Linear(256, len(CLASES)),
)
modelo.load_state_dict(paquete["state_dict"])
modelo.eval()

recortar = T.CenterCrop(TAM)
normalizar = T.Compose([T.ToTensor(), T.Normalize(paquete["media"], paquete["desviacion"])])

rutas, reales = [], []
for i, clase in enumerate(CLASES):
    for p in sorted((CACHE / "test" / clase).glob("*.jpg")):
        rutas.append(p)
        reales.append(i)
reales = np.array(reales)
print("imagenes de prueba:", len(rutas), flush=True)

probabilidades = []
with torch.no_grad():
    for i in range(0, len(rutas), 32):
        lote = []
        for p in rutas[i:i + 32]:
            with Image.open(p) as im:
                lote.append(normalizar(recortar(im.convert("RGB"))))
        probabilidades.append(F.softmax(modelo(torch.stack(lote)), dim=1))
probabilidades = torch.cat(probabilidades).numpy()
predichas = probabilidades.argmax(1)

aciertos = int((predichas == reales).sum())
accuracy = aciertos / len(reales)

matriz = np.zeros((len(CLASES), len(CLASES)), dtype=int)
for r, p in zip(reales, predichas):
    matriz[r, p] += 1


def auc_una_contra_resto(puntajes, positivos):
    """AUC exacta por rangos; empata promediando, como hace scikit-learn."""
    n_pos = int(positivos.sum())
    n_neg = len(positivos) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    orden = np.argsort(puntajes, kind="mergesort")
    rangos = np.empty(len(puntajes), dtype=float)
    i = 0
    while i < len(orden):
        j = i
        while j + 1 < len(orden) and puntajes[orden[j + 1]] == puntajes[orden[i]]:
            j += 1
        rangos[orden[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    suma = rangos[positivos.astype(bool)].sum()
    return (suma - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def curva_roc(puntajes, positivos):
    orden = np.argsort(-puntajes, kind="mergesort")
    pos = positivos[orden]
    tpr = np.concatenate([[0], np.cumsum(pos) / max(pos.sum(), 1)])
    fpr = np.concatenate([[0], np.cumsum(1 - pos) / max((1 - pos).sum(), 1)])
    return fpr, tpr


por_clase = {}
for i, clase in enumerate(CLASES):
    vp = int(((predichas == i) & (reales == i)).sum())
    fp = int(((predichas == i) & (reales != i)).sum())
    fn = int(((predichas != i) & (reales == i)).sum())
    prec = vp / (vp + fp) if vp + fp else 0.0
    rec = vp / (vp + fn) if vp + fn else 0.0
    por_clase[clase] = {
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "soporte": int((reales == i).sum()),
        "auc": float(auc_una_contra_resto(probabilidades[:, i], (reales == i).astype(int))),
    }

f1_macro = sum(v["f1"] for v in por_clase.values()) / len(CLASES)

muestra = normalizar(recortar(Image.open(rutas[0]).convert("RGB"))).unsqueeze(0)
with torch.no_grad():
    for _ in range(3):
        modelo(muestra)
    t0 = time.perf_counter()
    for _ in range(20):
        modelo(muestra)
    cpu_ms = (time.perf_counter() - t0) / 20 * 1000

metricas = {
    "accuracy": accuracy,
    "f1_macro": f1_macro,
    "n_test": len(reales),
    "errores": len(reales) - aciertos,
    "por_clase": por_clase,
    "matriz_confusion": matriz.tolist(),
    "clases": CLASES,
    "latencia": {"cpu_ms": round(cpu_ms, 1)},
}
SALIDA.mkdir(exist_ok=True)
with open(SALIDA / "metricas_test.json", "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)

print("accuracy {:.4f}  f1_macro {:.4f}  errores {}".format(
    accuracy, f1_macro, metricas["errores"]), flush=True)
for c, v in por_clase.items():
    print("  {:10s} P {:.3f}  R {:.3f}  F1 {:.3f}  AUC {:.3f}  n={}".format(
        c, v["precision"], v["recall"], v["f1"], v["auc"], v["soporte"]), flush=True)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.imshow(matriz, cmap="Blues")
ax.set_xticks(range(len(CLASES)), CLASES)
ax.set_yticks(range(len(CLASES)), CLASES)
ax.set_xlabel("prediccion")
ax.set_ylabel("clase real")
ax.set_title("Matriz de confusion")
for r in range(len(CLASES)):
    for c in range(len(CLASES)):
        ax.text(c, r, matriz[r, c], ha="center", va="center",
                color="white" if matriz[r, c] > matriz.max() / 2 else "black")
fig.tight_layout()
fig.savefig(SALIDA / "matriz_confusion.png", dpi=120)

fig, ax = plt.subplots(figsize=(5.5, 5))
for i, clase in enumerate(CLASES):
    fpr, tpr = curva_roc(probabilidades[:, i], (reales == i).astype(int))
    ax.plot(fpr, tpr, label="{} (AUC {:.3f})".format(clase, por_clase[clase]["auc"]))
ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=0.8)
ax.set_xlabel("tasa de falsos positivos")
ax.set_ylabel("tasa de verdaderos positivos")
ax.set_title("Curvas ROC, una clase contra el resto")
ax.legend()
fig.tight_layout()
fig.savefig(SALIDA / "curvas_roc.png", dpi=120)


def gradcam(tensor, clase):
    x = tensor.unsqueeze(0).requires_grad_(True)
    mapas = modelo.features(x)
    logits = modelo.classifier(mapas.mean(dim=(2, 3)))
    gradiente = torch.autograd.grad(logits[0, clase], mapas)[0]
    cam = F.relu((gradiente.mean(dim=(2, 3), keepdim=True) * mapas).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().numpy()


def rejilla(indices, titulo, archivo):
    if not len(indices):
        return
    indices = list(indices)[:6]
    cols = min(3, len(indices))
    filas = (len(indices) + cols - 1) // cols
    fig, ejes = plt.subplots(filas, cols, figsize=(4 * cols, 4.2 * filas), squeeze=False)
    for eje in ejes.flat:
        eje.axis("off")
    for eje, k in zip(ejes.flat, indices):
        with Image.open(rutas[k]) as im:
            vista = recortar(im.convert("RGB"))
        cam = gradcam(normalizar(vista), int(predichas[k]))
        base = np.asarray(vista, dtype=np.float32) / 255
        calor = matplotlib.colormaps["jet"](cam)[..., :3]
        eje.imshow(0.55 * base + 0.45 * calor)
        eje.set_title("real {} / dijo {}\nconfianza {:.1%}".format(
            CLASES[reales[k]], CLASES[predichas[k]], probabilidades[k].max()), fontsize=9)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(SALIDA / archivo, dpi=110)


correctos = np.where(predichas == reales)[0]
fallos = np.where(predichas != reales)[0]
rejilla(correctos[np.argsort(-probabilidades[correctos].max(1))],
        "Grad-CAM en aciertos: donde miro la red", "gradcam_aciertos.png")
rejilla(fallos[np.argsort(-probabilidades[fallos].max(1))],
        "Grad-CAM en errores cometidos con mas confianza", "gradcam_errores.png")

print("artefactos generados en", SALIDA, flush=True)
