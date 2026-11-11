"""Reentrena MobileNetV2 en dos fases respetando el contrato que espera la app.

Fase 1: extractor congelado. Las caracteristicas se calculan UNA vez y la cabeza
        se entrena sobre ellas, lo que convierte media hora en segundos.
Fase 2: se descongelan los ultimos 5 bloques y se ajusta todo con aumentacion.

    python entrenar.py [epocas_fase2]
"""
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import mobilenet_v2

CACHE = Path("cache")
SALIDA = Path("artefactos")
MODELO = Path("modelo")
# Partimos del modelo que ya esta en produccion en vez de ImageNet puro: ya viene
# de ImageNet y ademas aprendio del dataset original, que no esta en esta maquina.
BASE = Path("modelo_base.pt")
TAM = 224
MEDIA = [0.485, 0.456, 0.406]
DESVIACION = [0.229, 0.224, 0.225]
LOTE = 32
EPOCAS_CABEZA = 40
EPOCAS_FINETUNE = int(sys.argv[1]) if len(sys.argv) > 1 else 6
BLOQUES_DESCONGELADOS = 5

CLASES = sorted(p.name for p in (CACHE / "train").iterdir() if p.is_dir())
print("clases:", CLASES, flush=True)

determinista = T.Compose([T.CenterCrop(TAM), T.ToTensor(), T.Normalize(MEDIA, DESVIACION)])
aumentada = T.Compose([
    T.RandomResizedCrop(TAM, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(), T.Normalize(MEDIA, DESVIACION),
])


class Carpeta(Dataset):
    def __init__(self, particion, transformacion):
        self.items = [(p, i) for i, c in enumerate(CLASES)
                      for p in sorted((CACHE / particion / c).glob("*.jpg"))]
        self.transformacion = transformacion

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ruta, etiqueta = self.items[i]
        with Image.open(ruta) as im:
            return self.transformacion(im.convert("RGB")), etiqueta


def f1_macro(reales, predichas):
    total = 0.0
    for c in range(len(CLASES)):
        vp = sum(1 for r, p in zip(reales, predichas) if r == c and p == c)
        fp = sum(1 for r, p in zip(reales, predichas) if r != c and p == c)
        fn = sum(1 for r, p in zip(reales, predichas) if r == c and p != c)
        prec = vp / (vp + fp) if vp + fp else 0.0
        rec = vp / (vp + fn) if vp + fn else 0.0
        total += 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return total / len(CLASES)


def nueva_cabeza(salidas):
    return nn.Sequential(
        nn.Dropout(0.2), nn.Linear(1280, 256), nn.ReLU(inplace=True),
        nn.Dropout(0.3), nn.Linear(256, salidas),
    )


def evaluar(modelo, cargador):
    modelo.eval()
    reales, predichas = [], []
    with torch.no_grad():
        for x, y in cargador:
            predichas += modelo(x).argmax(1).tolist()
            reales += y.tolist()
    aciertos = sum(1 for r, p in zip(reales, predichas) if r == p)
    return aciertos / len(reales), f1_macro(reales, predichas)


cuenta = [len(list((CACHE / "train" / c).glob("*.jpg"))) for c in CLASES]
pesos = torch.tensor([sum(cuenta) / (len(CLASES) * n) for n in cuenta], dtype=torch.float)
print("imagenes de entrenamiento por clase:", dict(zip(CLASES, cuenta)), flush=True)
print("pesos de la perdida:", [round(p, 3) for p in pesos.tolist()], flush=True)
criterio = nn.CrossEntropyLoss(weight=pesos)

modelo = mobilenet_v2(weights=None)
modelo.classifier = nueva_cabeza(len(CLASES))
paquete_base = torch.load(BASE, map_location="cpu", weights_only=True)
if paquete_base["clases"] != CLASES:
    raise SystemExit("las clases del modelo base {} no coinciden con {}".format(
        paquete_base["clases"], CLASES))
modelo.load_state_dict(paquete_base["state_dict"])
print("inicializado desde el modelo en produccion:", BASE, flush=True)

print("\n== fase 1: caracteristicas precalculadas ==", flush=True)
inicio = time.perf_counter()


def extraer(particion):
    conjunto = Carpeta(particion, determinista)
    cargador = DataLoader(conjunto, batch_size=LOTE, shuffle=False, num_workers=0)
    modelo.eval()
    trozos, etiquetas = [], []
    with torch.no_grad():
        for x, y in cargador:
            trozos.append(modelo.features(x).mean(dim=(2, 3)))
            etiquetas.append(y)
    return torch.cat(trozos), torch.cat(etiquetas)


rasgos_tr, etiq_tr = extraer("train")
rasgos_va, etiq_va = extraer("val")
print("extraidas en {:.0f} s ({} train, {} val)".format(
    time.perf_counter() - inicio, len(rasgos_tr), len(rasgos_va)), flush=True)

optimizador = torch.optim.Adam(modelo.classifier.parameters(), lr=1e-3)
historial = []
# El mejor estado se sigue en LAS DOS fases. Partiendo de un modelo ya ajustado,
# la cabeza converge en un par de epocas y el fine-tuning posterior puede
# empeorar: quedarse solo con el mejor de la fase 2 tira ese optimo.
mejor_f1, mejor_estado, mejor_epoca, mejor_fase = -1.0, None, 0, 0
for epoca in range(1, EPOCAS_CABEZA + 1):
    modelo.classifier.train()
    orden = torch.randperm(len(rasgos_tr))
    perdida_total = 0.0
    for i in range(0, len(orden), LOTE):
        idx = orden[i:i + LOTE]
        optimizador.zero_grad()
        perdida = criterio(modelo.classifier(rasgos_tr[idx]), etiq_tr[idx])
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item() * len(idx)
    modelo.classifier.eval()
    with torch.no_grad():
        pred_va = modelo.classifier(rasgos_va).argmax(1).tolist()
    f1_va = f1_macro(etiq_va.tolist(), pred_va)
    acc_va = sum(1 for r, p in zip(etiq_va.tolist(), pred_va) if r == p) / len(pred_va)
    historial.append({"fase": 1, "epoca": epoca, "perdida": perdida_total / len(orden),
                      "val_accuracy": acc_va, "val_f1_macro": f1_va})
    if f1_va > mejor_f1:
        mejor_f1, mejor_epoca, mejor_fase = f1_va, epoca, 1
        mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
    if epoca % 10 == 0:
        print("  epoca {:2d}  perdida {:.4f}  val F1 {:.4f}".format(
            epoca, historial[-1]["perdida"], f1_va), flush=True)

print("\n== fase 2: fine-tuning de los ultimos bloques ==", flush=True)
for parametro in modelo.parameters():
    parametro.requires_grad = False
for bloque in modelo.features[-BLOQUES_DESCONGELADOS:]:
    for parametro in bloque.parameters():
        parametro.requires_grad = True
for parametro in modelo.classifier.parameters():
    parametro.requires_grad = True

cargador_tr = DataLoader(Carpeta("train", aumentada), batch_size=LOTE, shuffle=True, num_workers=0)
cargador_va = DataLoader(Carpeta("val", determinista), batch_size=LOTE, shuffle=False, num_workers=0)
optimizador = torch.optim.Adam([p for p in modelo.parameters() if p.requires_grad], lr=1e-4)

for epoca in range(1, EPOCAS_FINETUNE + 1):
    t0 = time.perf_counter()
    modelo.train()
    perdida_total = 0.0
    for x, y in cargador_tr:
        optimizador.zero_grad()
        perdida = criterio(modelo(x), y)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item() * len(y)
    acc_va, f1_va = evaluar(modelo, cargador_va)
    historial.append({"fase": 2, "epoca": EPOCAS_CABEZA + epoca,
                      "perdida": perdida_total / len(cargador_tr.dataset),
                      "val_accuracy": acc_va, "val_f1_macro": f1_va})
    marca = ""
    if f1_va > mejor_f1:
        mejor_f1, mejor_epoca, mejor_fase = f1_va, EPOCAS_CABEZA + epoca, 2
        mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        marca = "  <- mejor"
    print("  epoca {}/{}  perdida {:.4f}  val acc {:.4f}  val F1 {:.4f}  ({:.0f}s){}".format(
        epoca, EPOCAS_FINETUNE, historial[-1]["perdida"], acc_va, f1_va,
        time.perf_counter() - t0, marca), flush=True)

modelo.load_state_dict(mejor_estado)

MODELO.mkdir(exist_ok=True)
SALIDA.mkdir(exist_ok=True)
torch.save({"state_dict": modelo.state_dict(), "clases": CLASES, "tam_imagen": TAM,
            "media": MEDIA, "desviacion": DESVIACION},
           MODELO / "mobilenetv2_semaforo.pt")

with open(SALIDA / "historial_entrenamiento.csv", "w", newline="", encoding="utf-8") as f:
    escritor = csv.DictWriter(f, fieldnames=list(historial[0]))
    escritor.writeheader()
    escritor.writerows(historial)

resumen = {
    "arquitectura": "MobileNetV2 (ImageNet) + cabeza Dropout-Linear(1280,256)-ReLU-Dropout-Linear(256,3)",
    "preentrenado_imagenet": True,
    "epocas_ejecutadas": len(historial),
    "mejor_epoca": mejor_epoca,
    "mejor_fase": mejor_fase,
    "val_f1_macro": mejor_f1,
    "val_accuracy": next(h["val_accuracy"] for h in historial if h["epoca"] == mejor_epoca),
    "parametros_totales": sum(p.numel() for p in modelo.parameters()),
    "lote": LOTE, "lr_cabeza": 1e-3, "lr_fine_tuning": 1e-4,
    "bloques_descongelados": BLOQUES_DESCONGELADOS,
    "pytorch": torch.__version__, "dispositivo": "cpu",
    "imagenes_entrenamiento": dict(zip(CLASES, cuenta)),
}
with open(SALIDA / "resumen_entrenamiento.json", "w", encoding="utf-8") as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)

fig, (iz, de) = plt.subplots(1, 2, figsize=(11, 4))
iz.plot([h["epoca"] for h in historial], [h["perdida"] for h in historial])
iz.axvline(EPOCAS_CABEZA + 0.5, ls="--", c="gray")
iz.set_title("Perdida de entrenamiento")
iz.set_xlabel("epoca")
de.plot([h["epoca"] for h in historial], [h["val_f1_macro"] for h in historial], label="val F1 macro")
de.plot([h["epoca"] for h in historial], [h["val_accuracy"] for h in historial], label="val accuracy")
de.axvline(EPOCAS_CABEZA + 0.5, ls="--", c="gray")
de.scatter([mejor_epoca], [mejor_f1], marker="*", s=200, c="crimson", zorder=5)
de.set_title("Validacion")
de.set_xlabel("epoca")
de.legend()
fig.suptitle("Fase 1: solo la cabeza  |  Fase 2: fine-tuning")
fig.tight_layout()
fig.savefig(SALIDA / "curvas_entrenamiento.png", dpi=120)

print("\nlisto. mejor epoca {}, val F1 macro {:.4f}".format(mejor_epoca, mejor_f1), flush=True)
