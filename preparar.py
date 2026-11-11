"""Limpia, reparte en train/val/test y deja una caché redimensionada a 256 px.

La caché evita decodificar JPEG grandes en cada época, que en 2 hilos de CPU es
la mitad del coste del entrenamiento.
"""
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import limpiar

CRUDOS = Path("datos_crudos")
CACHE = Path("cache")
SEMILLA = 42
LADO_CORTO = 256

print("limpieza:", limpiar.limpiar(CRUDOS), flush=True)

random.seed(SEMILLA)
reparto = {}
for carpeta in sorted(p for p in CRUDOS.iterdir() if p.is_dir()):
    archivos = sorted(carpeta.glob("*"))
    random.shuffle(archivos)
    n = len(archivos)
    corte_tr, corte_va = int(n * 0.70), int(n * 0.85)
    reparto[carpeta.name] = {
        "train": archivos[:corte_tr],
        "val": archivos[corte_tr:corte_va],
        "test": archivos[corte_va:],
    }


def a_lado_corto(imagen: Image.Image) -> Image.Image:
    ancho, alto = imagen.size
    escala = LADO_CORTO / min(ancho, alto)
    if escala >= 1:
        return imagen
    return imagen.resize((round(ancho * escala), round(alto * escala)), Image.Resampling.LANCZOS)


conteo = Counter()
for clase, particiones in reparto.items():
    for particion, archivos in particiones.items():
        destino = CACHE / particion / clase
        destino.mkdir(parents=True, exist_ok=True)
        for i, origen in enumerate(archivos):
            try:
                with Image.open(origen) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    a_lado_corto(im).save(destino / f"{i:05d}.jpg", quality=92)
                conteo[(clase, particion)] += 1
            except Exception as e:
                print(f"  omitida {origen.name}: {e}", flush=True)

resumen = {f"{c}/{p}": n for (c, p), n in sorted(conteo.items())}
print("caché:", resumen, flush=True)
Path("reparto.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")

# Figura de distribución que la app muestra en la pestaña de métricas.
clases = sorted({c for c, _ in conteo})
particiones = ["train", "val", "test"]
fig, ax = plt.subplots(figsize=(7, 4))
ancho = 0.25
for i, particion in enumerate(particiones):
    ax.bar([x + i * ancho for x in range(len(clases))],
           [conteo[(c, particion)] for c in clases], ancho, label=particion)
ax.set_xticks([x + ancho for x in range(len(clases))])
ax.set_xticklabels(clases)
ax.set_ylabel("imágenes")
ax.set_title("Imágenes por clase y partición")
ax.legend()
fig.tight_layout()
Path("artefactos").mkdir(exist_ok=True)
fig.savefig("artefactos/distribucion_clases.png", dpi=120)
print("distribucion_clases.png generada", flush=True)
