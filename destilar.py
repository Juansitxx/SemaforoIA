"""Entrena ResNet50 (Maestro) y aplica Knowledge Distillation hacia MobileNetV2 (Estudiante).

Uso:
    python destilar.py
"""
import csv
import json
import shutil
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet50_Weights,
    mobilenet_v2,
    resnet50,
)

AQUI = Path(__file__).resolve().parent
CACHE = AQUI / "cache" if (AQUI / "cache").exists() else Path(r"C:\Users\Juan\Downloads\SemaforoIA-dataset-v2")
RUTA_BASE_PROD = Path(r"C:\Users\Juan\Downloads\SemaforoIA-despliegue\modelo\mobilenetv2_semaforo.pt")
SALIDA = AQUI / "artefactos"
MODELO = AQUI / "modelo"

TAM = 224
MEDIA = [0.485, 0.456, 0.406]
DESVIACION = [0.229, 0.224, 0.225]
LOTE = 32
TEMPERATURA = 4.0
ALFA = 0.7

torch.manual_seed(42)
np.random.seed(42)

CLASES = sorted(p.name for p in (CACHE / "train").iterdir() if p.is_dir())
print(f"Dataset en: {CACHE}", flush=True)
print(f"Clases: {CLASES}", flush=True)

determinista = T.Compose([
    T.CenterCrop(TAM),
    T.ToTensor(),
    T.Normalize(MEDIA, DESVIACION),
])


class Carpeta(Dataset):
    def __init__(self, particion, transformacion):
        self.items = [
            (p, i)
            for i, c in enumerate(CLASES)
            for p in sorted((CACHE / particion / c).glob("*.jpg"))
        ]
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


def metricas_detalladas(logits, etiquetas):
    probs = F.softmax(logits, dim=1).numpy()
    reales = etiquetas.numpy()
    preds = probs.argmax(1)
    aciertos = int((preds == reales).sum())
    acc = aciertos / len(reales)
    por_clase = {}
    for i, c in enumerate(CLASES):
        vp = int(((preds == i) & (reales == i)).sum())
        fp = int(((preds == i) & (reales != i)).sum())
        fn = int(((preds != i) & (reales == i)).sum())
        prec = vp / (vp + fp) if vp + fp else 0.0
        rec = vp / (vp + fn) if vp + fn else 0.0
        por_clase[c] = {
            "precision": prec,
            "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "soporte": int((reales == i).sum()),
        }
    f1 = sum(v["f1"] for v in por_clase.values()) / len(CLASES)
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "n_test": int(len(reales)),
        "errores": int(len(reales) - aciertos),
        "por_clase": por_clase,
    }


def extraer_rasgos(extractor, particion, nombre=""):
    t0 = time.perf_counter()
    conjunto = Carpeta(particion, determinista)
    cargador = DataLoader(conjunto, batch_size=LOTE, shuffle=False, num_workers=0)
    extractor.eval()
    trozos, etiquetas = [], []
    total_lotes = len(cargador)
    with torch.no_grad():
        for b, (x, y) in enumerate(cargador, 1):
            trozos.append(extractor(x))
            etiquetas.append(y)
            if b % 10 == 0 or b == total_lotes:
                print(f"  [{nombre} - {particion}] lote {b}/{total_lotes} ({time.perf_counter() - t0:.1f}s)", flush=True)
    return torch.cat(trozos), torch.cat(etiquetas)


def medir_latencia_ms(modelo, muestra):
    modelo.eval()
    with torch.no_grad():
        for _ in range(3):
            modelo(muestra)
        t0 = time.perf_counter()
        for _ in range(15):
            modelo(muestra)
        return round((time.perf_counter() - t0) / 15 * 1000, 1)


cuenta = [len(list((CACHE / "train" / c).glob("*.jpg"))) for c in CLASES]
pesos_clase = torch.tensor([sum(cuenta) / (len(CLASES) * n) for n in cuenta], dtype=torch.float)
print("Imágenes de entrenamiento por clase:", dict(zip(CLASES, cuenta)), flush=True)
print("Pesos de la pérdida:", [round(p, 3) for p in pesos_clase.tolist()], flush=True)
criterio_ce = nn.CrossEntropyLoss(weight=pesos_clase)

MODELO.mkdir(exist_ok=True)
SALIDA.mkdir(exist_ok=True)

# ==============================================================================
# PASO 1: ENTRENAR EL MODELO MAESTRO (ResNet50)
# ==============================================================================
print("\n========================================================", flush=True)
print("PASO 1: Entrenando Modelo Maestro (ResNet50 - ImageNet)", flush=True)
print("========================================================", flush=True)

RUTA_PESOS_RESNET50 = Path(r"C:\Users\Juan\.cache\torch\hub\checkpoints\resnet50-11ad3fa6.pth")
maestro = resnet50(weights=None)
maestro.load_state_dict(torch.load(RUTA_PESOS_RESNET50, map_location="cpu", weights_only=True))
for p in maestro.parameters():
    p.requires_grad = False

maestro.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(2048, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, len(CLASES)),
)

extractor_resnet = nn.Sequential(*list(maestro.children())[:-1], nn.Flatten(1))
rasgos_tr_m, etiq_tr = extraer_rasgos(extractor_resnet, "train", "ResNet50")
rasgos_va_m, etiq_va = extraer_rasgos(extractor_resnet, "val", "ResNet50")
rasgos_te_m, etiq_te = extraer_rasgos(extractor_resnet, "test", "ResNet50")

torch.manual_seed(42)
opt_m = torch.optim.AdamW(maestro.fc.parameters(), lr=1e-3, weight_decay=1e-2)
sched_m = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=60)

mejor_f1_m, mejor_acc_m, mejor_epoca_m, mejor_fc_m = -1.0, 0.0, 0, None
historial_maestro = []

for epoca in range(1, 61):
    maestro.fc.train()
    orden = torch.randperm(len(rasgos_tr_m))
    perdida_total = 0.0
    for i in range(0, len(orden), LOTE):
        idx = orden[i:i + LOTE]
        opt_m.zero_grad()
        perdida = criterio_ce(maestro.fc(rasgos_tr_m[idx]), etiq_tr[idx])
        perdida.backward()
        opt_m.step()
        perdida_total += perdida.item() * len(idx)
    sched_m.step()

    maestro.fc.eval()
    with torch.no_grad():
        logits_va_m = maestro.fc(rasgos_va_m)
        pred_va = logits_va_m.argmax(1).tolist()
    f1_va = f1_macro(etiq_va.tolist(), pred_va)
    acc_va = sum(1 for r, p in zip(etiq_va.tolist(), pred_va) if r == p) / len(pred_va)
    historial_maestro.append({
        "epoca": epoca,
        "perdida": perdida_total / len(orden),
        "val_accuracy": acc_va,
        "val_f1_macro": f1_va,
    })
    if f1_va > mejor_f1_m or (f1_va == mejor_f1_m and acc_va > mejor_acc_m):
        mejor_f1_m, mejor_acc_m, mejor_epoca_m = f1_va, acc_va, epoca
        mejor_fc_m = {k: v.clone() for k, v in maestro.fc.state_dict().items()}
    if epoca % 10 == 0:
        print(f"  [Maestro ResNet50] época {epoca:2d} | pérdida {historial_maestro[-1]['perdida']:.4f} | val acc {acc_va:.4f} | val F1 {f1_va:.4f}", flush=True)

maestro.fc.load_state_dict(mejor_fc_m)
maestro.eval()

with torch.no_grad():
    logits_maestro_tr = maestro.fc(rasgos_tr_m)
    logits_maestro_va = maestro.fc(rasgos_va_m)
    logits_maestro_te = maestro.fc(rasgos_te_m)

metricas_maestro_te = metricas_detalladas(logits_maestro_te, etiq_te)
ruta_maestro = MODELO / "resnet50_maestro.pt"
torch.save({
    "state_dict": maestro.state_dict(),
    "clases": CLASES,
    "tam_imagen": TAM,
    "media": MEDIA,
    "desviacion": DESVIACION,
    "arquitectura": "ResNet50 (Maestro)",
}, ruta_maestro)

peso_maestro_mb = round(ruta_maestro.stat().st_size / 1048576, 1)
params_maestro = sum(p.numel() for p in maestro.parameters())
muestra_tensor, _ = Carpeta("test", determinista)[0]
latencia_maestro_ms = medir_latencia_ms(maestro, muestra_tensor.unsqueeze(0))
metricas_maestro_te["peso_mb"] = peso_maestro_mb
metricas_maestro_te["parametros_totales"] = params_maestro
metricas_maestro_te["latencia_cpu_ms"] = latencia_maestro_ms
metricas_maestro_te["mejor_epoca_val"] = mejor_epoca_m
metricas_maestro_te["val_f1_macro"] = mejor_f1_m

print(
    f"\n-> Maestro ResNet50 guardado en {ruta_maestro.name} ({peso_maestro_mb} MB, {params_maestro:,} params)\n"
    f"   Mejor época val: {mejor_epoca_m} (Val F1: {mejor_f1_m:.4f})\n"
    f"   Test Accuracy: {metricas_maestro_te['accuracy']:.4f} ({metricas_maestro_te['errores']} errores / {metricas_maestro_te['n_test']}) | "
    f"Test F1 macro: {metricas_maestro_te['f1_macro']:.4f} | Latencia CPU: {latencia_maestro_ms} ms",
    flush=True,
)

# ==============================================================================
# PASO 2: CONOCIMIENTO DESTILADO HACIA EL ESTUDIANTE (MobileNetV2)
# ==============================================================================
print("\n========================================================", flush=True)
print("PASO 2: Destilando ResNet50 -> MobileNetV2 (Estudiante)", flush=True)
print("========================================================", flush=True)


def nueva_cabeza_mbv2(salidas):
    return nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, salidas),
    )


def perdida_destilacion(logits_est, logits_mae, etiquetas, T=TEMPERATURA, alfa=ALFA):
    soft_est = F.log_softmax(logits_est / T, dim=1)
    soft_mae = F.softmax(logits_mae / T, dim=1)
    loss_kd = F.kl_div(soft_est, soft_mae, reduction="batchmean") * (T * T)
    loss_ce = criterio_ce(logits_est, etiquetas)
    return alfa * loss_kd + (1.0 - alfa) * loss_ce, loss_kd.item(), loss_ce.item()


# Inicializamos el Estudiante partiendo del extractor ya adaptado en producción (como en entrenar.py)
estudiante = mobilenet_v2(weights=None)
estudiante.classifier = nueva_cabeza_mbv2(len(CLASES))
if RUTA_BASE_PROD.exists():
    paquete_base = torch.load(RUTA_BASE_PROD, map_location="cpu", weights_only=True)
    estudiante.load_state_dict(paquete_base["state_dict"])
    print(f"Estudiante inicializado desde {RUTA_BASE_PROD}", flush=True)

extractor_mbv2 = nn.Sequential(estudiante.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
rasgos_tr_e, _ = extraer_rasgos(extractor_mbv2, "train", "MobileNetV2")
rasgos_va_e, _ = extraer_rasgos(extractor_mbv2, "val", "MobileNetV2")
rasgos_te_e, _ = extraer_rasgos(extractor_mbv2, "test", "MobileNetV2")

# Evaluamos primero el modelo base (sin destilar) sobre el mismo conjunto de test
estudiante.eval()
with torch.no_grad():
    logits_base_te = estudiante.classifier(rasgos_te_e)
metricas_base_te = metricas_detalladas(logits_base_te, etiq_te)
print(
    f"Línea base Estudiante (antes de destilar) -> Test Acc: {metricas_base_te['accuracy']:.4f} | "
    f"Test F1: {metricas_base_te['f1_macro']:.4f}",
    flush=True,
)

# Guardamos copia de los pesos iniciales de la cabeza para entrenar con destilación
torch.manual_seed(42)
opt_e = torch.optim.AdamW(estudiante.classifier.parameters(), lr=5e-4, weight_decay=1e-2)
sched_e = torch.optim.lr_scheduler.CosineAnnealingLR(opt_e, T_max=50)

mejor_f1_e, mejor_acc_e, mejor_epoca_e = -1.0, 0.0, 0
mejor_estado_e = {k: v.clone() for k, v in estudiante.state_dict().items()}
historial_est = []

for epoca in range(1, 51):
    estudiante.classifier.train()
    orden = torch.randperm(len(rasgos_tr_e))
    perdida_tot, kd_tot, ce_tot = 0.0, 0.0, 0.0
    for i in range(0, len(orden), LOTE):
        idx = orden[i:i + LOTE]
        opt_e.zero_grad()
        logits_e = estudiante.classifier(rasgos_tr_e[idx])
        perdida, l_kd, l_ce = perdida_destilacion(logits_e, logits_maestro_tr[idx], etiq_tr[idx])
        perdida.backward()
        opt_e.step()
        perdida_tot += perdida.item() * len(idx)
        kd_tot += l_kd * len(idx)
        ce_tot += l_ce * len(idx)
    sched_e.step()

    estudiante.classifier.eval()
    with torch.no_grad():
        logits_va_e = estudiante.classifier(rasgos_va_e)
        pred_va = logits_va_e.argmax(1).tolist()
    f1_va = f1_macro(etiq_va.tolist(), pred_va)
    acc_va = sum(1 for r, p in zip(etiq_va.tolist(), pred_va) if r == p) / len(pred_va)

    historial_est.append({
        "fase": 1,
        "epoca": epoca,
        "perdida": perdida_tot / len(orden),
        "perdida_kd": kd_tot / len(orden),
        "perdida_ce": ce_tot / len(orden),
        "val_accuracy": acc_va,
        "val_f1_macro": f1_va,
    })
    if f1_va > mejor_f1_e or (f1_va == mejor_f1_e and acc_va > mejor_acc_e):
        mejor_f1_e, mejor_acc_e, mejor_epoca_e = f1_va, acc_va, epoca
        mejor_estado_e = {k: v.clone() for k, v in estudiante.state_dict().items()}
    if epoca % 10 == 0:
        print(
            f"  [Estudiante KD] época {epoca:2d} | pérdida total {historial_est[-1]['perdida']:.4f} "
            f"(KD {historial_est[-1]['perdida_kd']:.4f}, CE {historial_est[-1]['perdida_ce']:.4f}) | "
            f"val acc {acc_va:.4f} | val F1 {f1_va:.4f}",
            flush=True,
        )

estudiante.load_state_dict(mejor_estado_e)
estudiante.eval()

with torch.no_grad():
    logits_est_te = estudiante.classifier(rasgos_te_e)
metricas_est_te = metricas_detalladas(logits_est_te, etiq_te)

ruta_estudiante = MODELO / "mobilenetv2_semaforo.pt"
torch.save({
    "state_dict": estudiante.state_dict(),
    "clases": CLASES,
    "tam_imagen": TAM,
    "media": MEDIA,
    "desviacion": DESVIACION,
}, ruta_estudiante)

peso_est_mb = round(ruta_estudiante.stat().st_size / 1048576, 1)
params_est = sum(p.numel() for p in estudiante.parameters())
latencia_est_ms = medir_latencia_ms(estudiante, muestra_tensor.unsqueeze(0))

metricas_est_te["peso_mb"] = peso_est_mb
metricas_est_te["parametros_totales"] = params_est
metricas_est_te["latencia_cpu_ms"] = latencia_est_ms
metricas_est_te["mejor_epoca_val"] = mejor_epoca_e
metricas_est_te["val_f1_macro"] = mejor_f1_e

# Guardamos historial y resumen de entrenamiento con información de Knowledge Distillation
with open(SALIDA / "historial_entrenamiento.csv", "w", newline="", encoding="utf-8") as f:
    escritor = csv.DictWriter(f, fieldnames=list(historial_est[0]))
    escritor.writeheader()
    escritor.writerows(historial_est)

resumen_entrenamiento = {
    "arquitectura": "MobileNetV2 (Estudiante destilado desde Maestro ResNet50)",
    "maestro": "ResNet50 (ImageNet + cabeza 2048->512->3)",
    "destilacion": {
        "temperatura": TEMPERATURA,
        "alfa_kd": ALFA,
        "peso_ce": round(1.0 - ALFA, 2),
        "maestro_parametros": params_maestro,
        "maestro_peso_mb": peso_maestro_mb,
        "maestro_latencia_ms": latencia_maestro_ms,
        "maestro_test_accuracy": metricas_maestro_te["accuracy"],
        "maestro_test_f1_macro": metricas_maestro_te["f1_macro"],
        "estudiante_parametros": params_est,
        "estudiante_peso_mb": peso_est_mb,
        "estudiante_latencia_ms": latencia_est_ms,
        "estudiante_test_accuracy": metricas_est_te["accuracy"],
        "estudiante_test_f1_macro": metricas_est_te["f1_macro"],
        "reduccion_parametros_x": round(params_maestro / params_est, 2),
        "reduccion_peso_x": round(peso_maestro_mb / peso_est_mb, 2),
    },
    "preentrenado_imagenet": True,
    "epocas_ejecutadas": len(historial_est),
    "mejor_epoca": mejor_epoca_e,
    "mejor_fase": 1,
    "val_f1_macro": mejor_f1_e,
    "val_accuracy": mejor_acc_e,
    "parametros_totales": params_est,
    "lote": LOTE,
    "temperatura_kd": TEMPERATURA,
    "alfa_kd": ALFA,
    "pytorch": torch.__version__,
    "dispositivo": "cpu",
    "imagenes_entrenamiento": dict(zip(CLASES, cuenta)),
}
with open(SALIDA / "resumen_entrenamiento.json", "w", encoding="utf-8") as f:
    json.dump(resumen_entrenamiento, f, indent=2, ensure_ascii=False)

with open(SALIDA / "metricas_maestro_resnet50.json", "w", encoding="utf-8") as f:
    json.dump(metricas_maestro_te, f, indent=2, ensure_ascii=False)

# Gráfica de curvas de entrenamiento con Destilación (Maestro vs Estudiante)
fig, (iz, de) = plt.subplots(1, 2, figsize=(11, 4.2))
iz.plot([h["epoca"] for h in historial_est], [h["perdida"] for h in historial_est], label="Total (KD + CE)", lw=2)
iz.plot([h["epoca"] for h in historial_est], [h["perdida_kd"] for h in historial_est], ls="--", label=f"KL Div (T={TEMPERATURA})")
iz.plot([h["epoca"] for h in historial_est], [h["perdida_ce"] for h in historial_est], ls=":", label="Cross-Entropy")
iz.set_title("Pérdida de Destilación (Estudiante)")
iz.set_xlabel("época")
iz.legend()

de.plot([h["epoca"] for h in historial_est], [h["val_f1_macro"] for h in historial_est], label="Estudiante val F1 macro", lw=2)
de.plot([h["epoca"] for h in historial_est], [h["val_accuracy"] for h in historial_est], label="Estudiante val accuracy")
de.axhline(mejor_f1_m, ls="--", c="darkorange", label=f"Maestro ResNet50 val F1 ({mejor_f1_m:.3f})")
de.scatter([mejor_epoca_e], [mejor_f1_e], marker="*", s=200, c="crimson", zorder=5)
de.set_title("Validación: Estudiante MobileNetV2 vs Maestro ResNet50")
de.set_xlabel("época")
de.legend()
fig.suptitle("Knowledge Distillation: ResNet50 (Maestro) -> MobileNetV2 (Estudiante)")
fig.tight_layout()
fig.savefig(SALIDA / "curvas_entrenamiento.png", dpi=120)

print("\n========================================================", flush=True)
print("RESUMEN COMPARATIVO EN TEST (247 IMÁGENES)", flush=True)
print("========================================================", flush=True)
print(f"Maestro ResNet50      ({peso_maestro_mb:5.1f} MB | {params_maestro:10,d} params | {latencia_maestro_ms:5.1f} ms) -> Acc: {metricas_maestro_te['accuracy']:.2%} | F1: {metricas_maestro_te['f1_macro']:.2%} | Errores: {metricas_maestro_te['errores']}", flush=True)
print(f"MobileNetV2 (Base)    ({peso_est_mb:5.1f} MB | {params_est:10,d} params | {latencia_est_ms:5.1f} ms) -> Acc: {metricas_base_te['accuracy']:.2%} | F1: {metricas_base_te['f1_macro']:.2%} | Errores: {metricas_base_te['errores']}", flush=True)
print(f"MobileNetV2 (Destil.) ({peso_est_mb:5.1f} MB | {params_est:10,d} params | {latencia_est_ms:5.1f} ms) -> Acc: {metricas_est_te['accuracy']:.2%} | F1: {metricas_est_te['f1_macro']:.2%} | Errores: {metricas_est_te['errores']}", flush=True)
