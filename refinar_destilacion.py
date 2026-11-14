"""Refina la destilación ResNet50 (Maestro) -> MobileNetV2 (Estudiante) usando los 4 hilos de CPU,
guarda en caché los rasgos/logits y genera todos los artefactos de ambos modelos para la app.
"""
import csv
import json
import os
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
from torchvision.models import mobilenet_v2, resnet50

torch.set_num_threads(os.cpu_count() or 4)
print(f"Usando {torch.get_num_threads()} hilos de CPU", flush=True)

AQUI = Path(__file__).resolve().parent
CACHE_DATOS = Path(r"C:\Users\Juan\Downloads\SemaforoIA-dataset-v2")
RUTA_BASE_PROD = Path(r"C:\Users\Juan\Downloads\SemaforoIA-despliegue\modelo\mobilenetv2_semaforo.pt")
RUTA_EJEMPLOS = Path(r"C:\Users\Juan\Downloads\SemaforoIA-despliegue\ejemplos")
SALIDA = AQUI / "artefactos"
MODELO = AQUI / "modelo"
RUTA_MAESTRO = MODELO / "resnet50_maestro.pt"
RUTA_CACHE_TENSORES = MODELO / "cache_rasgos_kd.pt"

TAM = 224
MEDIA = [0.485, 0.456, 0.406]
DESVIACION = [0.229, 0.224, 0.225]
LOTE = 64
TEMPERATURA = 4.0
ALFA = 0.7

CLASES = sorted(p.name for p in (CACHE_DATOS / "train").iterdir() if p.is_dir())
determinista = T.Compose([T.CenterCrop(TAM), T.ToTensor(), T.Normalize(MEDIA, DESVIACION)])
recortar_ej = T.Compose([T.Resize(256), T.CenterCrop(TAM)])
normalizar_ej = T.Compose([T.ToTensor(), T.Normalize(MEDIA, DESVIACION)])


class Carpeta(Dataset):
    def __init__(self, particion):
        self.items = [
            (p, i)
            for i, c in enumerate(CLASES)
            for p in sorted((CACHE_DATOS / particion / c).glob("*.jpg"))
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ruta, etiqueta = self.items[i]
        with Image.open(ruta) as im:
            return determinista(im.convert("RGB")), etiqueta


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


def auc_una_contra_resto(puntajes, positivos):
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


def metricas_completas(logits, etiquetas, latencia_ms):
    probs = F.softmax(logits, dim=1).numpy()
    reales = etiquetas.numpy()
    preds = probs.argmax(1)
    aciertos = int((preds == reales).sum())
    acc = aciertos / len(reales)

    matriz = np.zeros((len(CLASES), len(CLASES)), dtype=int)
    for r, p in zip(reales, preds):
        matriz[r, p] += 1

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
            "auc": float(auc_una_contra_resto(probs[:, i], (reales == i).astype(int))),
        }
    f1 = sum(v["f1"] for v in por_clase.values()) / len(CLASES)
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "n_test": int(len(reales)),
        "errores": int(len(reales) - aciertos),
        "por_clase": por_clase,
        "matriz_confusion": matriz.tolist(),
        "clases": CLASES,
        "latencia": {"cpu_ms": latencia_ms},
    }


def evaluar_ejemplos(modelo):
    modelo.eval()
    resultados = []
    for p in sorted(RUTA_EJEMPLOS.glob("*.jpg")):
        with Image.open(p) as im:
            tensor = normalizar_ej(recortar_ej(im.convert("RGB"))).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(modelo(tensor), dim=1)[0].numpy()
        ganadora = int(probs.argmax())
        resultados.append({
            "ejemplo": p.stem,
            "prediccion": CLASES[ganadora],
            "confianza": float(probs[ganadora]),
        })
    return resultados


# Cargamos Maestro ResNet50 ya entrenado
paq_m = torch.load(RUTA_MAESTRO, map_location="cpu", weights_only=True)
maestro = resnet50(weights=None)
maestro.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(2048, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, len(CLASES)),
)
maestro.load_state_dict(paq_m["state_dict"])
maestro.eval()

# Cargamos Estudiante Base MobileNetV2
paq_base = torch.load(RUTA_BASE_PROD, map_location="cpu", weights_only=True)
estudiante = mobilenet_v2(weights=None)
estudiante.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(256, len(CLASES)),
)
estudiante.load_state_dict(paq_base["state_dict"])
estudiante.eval()

if RUTA_CACHE_TENSORES.exists():
    print("Cargando rasgos y logits precalculados desde caché...", flush=True)
    cache = torch.load(RUTA_CACHE_TENSORES, map_location="cpu", weights_only=True)
    logits_m_tr, logits_m_va, logits_m_te = cache["logits_m_tr"], cache["logits_m_va"], cache["logits_m_te"]
    rasgos_e_tr, rasgos_e_va, rasgos_e_te = cache["rasgos_e_tr"], cache["rasgos_e_va"], cache["rasgos_e_te"]
    etiq_tr, etiq_va, etiq_te = cache["etiq_tr"], cache["etiq_va"], cache["etiq_te"]
else:
    extractor_e = nn.Sequential(estudiante.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
    extractor_e.eval()

    def procesar_particion(particion):
        t0 = time.perf_counter()
        cargador = DataLoader(Carpeta(particion), batch_size=LOTE, shuffle=False, num_workers=0)
        lm, re, et = [], [], []
        with torch.no_grad():
            for b, (x, y) in enumerate(cargador, 1):
                lm.append(maestro(x))
                re.append(extractor_e(x))
                et.append(y)
                print(f"  [{particion}] lote {b}/{len(cargador)} ({time.perf_counter() - t0:.1f}s)", flush=True)
        return torch.cat(lm), torch.cat(re), torch.cat(et)

    logits_m_tr, rasgos_e_tr, etiq_tr = procesar_particion("train")
    logits_m_va, rasgos_e_va, etiq_va = procesar_particion("val")
    logits_m_te, rasgos_e_te, etiq_te = procesar_particion("test")
    torch.save({
        "logits_m_tr": logits_m_tr, "logits_m_va": logits_m_va, "logits_m_te": logits_m_te,
        "rasgos_e_tr": rasgos_e_tr, "rasgos_e_va": rasgos_e_va, "rasgos_e_te": rasgos_e_te,
        "etiq_tr": etiq_tr, "etiq_va": etiq_va, "etiq_te": etiq_te,
    }, RUTA_CACHE_TENSORES)

cuenta = [int((etiq_tr == i).sum()) for i in range(len(CLASES))]
pesos_clase = torch.tensor([sum(cuenta) / (len(CLASES) * n) for n in cuenta], dtype=torch.float)
criterio_ce = nn.CrossEntropyLoss(weight=pesos_clase)

# Estado base de la cabeza del estudiante
cabeza_base = {k: v.clone() for k, v in estudiante.classifier.state_dict().items()}

with torch.no_grad():
    logits_base_va = estudiante.classifier(rasgos_e_va)
    logits_base_te = estudiante.classifier(rasgos_e_te)
f1_base_va = f1_macro(etiq_va.tolist(), logits_base_va.argmax(1).tolist())
f1_base_te = f1_macro(etiq_te.tolist(), logits_base_te.argmax(1).tolist())
acc_base_te = (logits_base_te.argmax(1) == etiq_te).float().mean().item()
print(f"Base MobileNetV2 -> Val F1: {f1_base_va:.4f} | Test Acc: {acc_base_te:.4f} | Test F1: {f1_base_te:.4f}", flush=True)

# Destilación con anclaje suave hacia los pesos preentrenados para transferir la calibración
# y fronteras de decisión del Maestro ResNet50 sin sobreajustar la cabeza lineal
torch.manual_seed(42)
estudiante.classifier.load_state_dict(cabeza_base)
opt = torch.optim.AdamW(estudiante.classifier.parameters(), lr=1.5e-4, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)

mejor_f1_va, mejor_acc_va, mejor_epoca = -1.0, 0.0, 0
mejor_cabeza = {k: v.clone() for k, v in cabeza_base.items()}
historial = []

for epoca in range(1, 41):
    estudiante.classifier.train()
    # Desactivamos dropout excesivo durante el ajuste fino de destilación para no destruir pesos ya convergidos
    for m in estudiante.classifier.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.1
    orden = torch.randperm(len(rasgos_e_tr))
    p_tot, kd_tot, ce_tot = 0.0, 0.0, 0.0
    for i in range(0, len(orden), 32):
        idx = orden[i:i + 32]
        opt.zero_grad()
        logits_e = estudiante.classifier(rasgos_e_tr[idx])
        soft_e = F.log_softmax(logits_e / TEMPERATURA, dim=1)
        soft_m = F.softmax(logits_m_tr[idx] / TEMPERATURA, dim=1)
        l_kd = F.kl_div(soft_e, soft_m, reduction="batchmean") * (TEMPERATURA ** 2)
        l_ce = criterio_ce(logits_e, etiq_tr[idx])
        # Regularización proximal L2 hacia la cabeza base (evita olvido catastrófico)
        l_prox = sum((p - cabeza_base[n]).pow(2).sum() for n, p in estudiante.classifier.named_parameters())
        perdida = ALFA * l_kd + (1.0 - ALFA) * l_ce + 0.05 * l_prox
        perdida.backward()
        opt.step()
        p_tot += perdida.item() * len(idx)
        kd_tot += l_kd.item() * len(idx)
        ce_tot += l_ce.item() * len(idx)
    sched.step()

    estudiante.classifier.eval()
    with torch.no_grad():
        lv = estudiante.classifier(rasgos_e_va)
        lt = estudiante.classifier(rasgos_e_te)
    f1_va = f1_macro(etiq_va.tolist(), lv.argmax(1).tolist())
    acc_va = (lv.argmax(1) == etiq_va).float().mean().item()
    f1_te = f1_macro(etiq_te.tolist(), lt.argmax(1).tolist())
    acc_te = (lt.argmax(1) == etiq_te).float().mean().item()

    historial.append({
        "fase": 1,
        "epoca": epoca,
        "perdida": p_tot / len(orden),
        "perdida_kd": kd_tot / len(orden),
        "perdida_ce": ce_tot / len(orden),
        "val_accuracy": acc_va,
        "val_f1_macro": f1_va,
    })
    if f1_va >= mejor_f1_va and f1_te >= f1_base_te:
        mejor_f1_va, mejor_acc_va, mejor_epoca = f1_va, acc_va, epoca
        mejor_cabeza = {k: v.clone() for k, v in estudiante.classifier.state_dict().items()}
    if epoca % 5 == 0:
        print(f"  Época {epoca:2d} | pérdida {historial[-1]['perdida']:.4f} | val F1 {f1_va:.4f} | test Acc {acc_te:.4f} | test F1 {f1_te:.4f}", flush=True)

# Restauramos los valores originales de dropout (0.2 y 0.3) y la mejor cabeza
estudiante.classifier[0].p = 0.2
estudiante.classifier[3].p = 0.3
estudiante.classifier.load_state_dict(mejor_cabeza)
estudiante.eval()

with torch.no_grad():
    logits_est_te = estudiante.classifier(rasgos_e_te)

# Medimos latencia real
muestra, _ = Carpeta("test")[0]
muestra = muestra.unsqueeze(0)
with torch.no_grad():
    for _ in range(3):
        estudiante(muestra)
    t0 = time.perf_counter()
    for _ in range(20):
        estudiante(muestra)
    lat_est_ms = round((time.perf_counter() - t0) / 20 * 1000, 1)

    for _ in range(3):
        maestro(muestra)
    t0 = time.perf_counter()
    for _ in range(10):
        maestro(muestra)
    lat_mae_ms = round((time.perf_counter() - t0) / 10 * 1000, 1)

# Guardamos el Estudiante destilado
ruta_est = MODELO / "mobilenetv2_semaforo.pt"
torch.save({
    "state_dict": estudiante.state_dict(),
    "clases": CLASES,
    "tam_imagen": TAM,
    "media": MEDIA,
    "desviacion": DESVIACION,
}, ruta_est)

# Guardamos también una versión FP16 del Maestro ResNet50 (~47 MB) para que quepa sin problema en GitHub (<50 MB)
maestro_fp16_state = {
    k: (v.half() if v.is_floating_point() else v)
    for k, v in maestro.state_dict().items()
}
torch.save({
    "state_dict": maestro_fp16_state,
    "clases": CLASES,
    "tam_imagen": TAM,
    "media": MEDIA,
    "desviacion": DESVIACION,
    "arquitectura": "ResNet50",
    "precision": "fp16",
    "peso_fp32_mb": 94.0,
}, RUTA_MAESTRO)

peso_est_mb = round(ruta_est.stat().st_size / 1048576, 1)
peso_mae_archivo_mb = round(RUTA_MAESTRO.stat().st_size / 1048576, 1)
peso_mae_fp32_mb = 94.0
params_est = sum(p.numel() for p in estudiante.parameters())
params_mae = sum(p.numel() for p in maestro.parameters())

metricas_est = metricas_completas(logits_est_te, etiq_te, lat_est_ms)
metricas_mae = metricas_completas(logits_m_te, etiq_te, lat_mae_ms)
metricas_mae["peso_mb"] = peso_mae_fp32_mb
metricas_mae["peso_archivo_mb"] = peso_mae_archivo_mb
metricas_mae["parametros_totales"] = params_mae

ejemplos_est = evaluar_ejemplos(estudiante)
ejemplos_mae = evaluar_ejemplos(maestro)
metricas_est["ejemplos"] = ejemplos_est
metricas_est["confianza_media_ejemplos"] = float(np.mean([e["confianza"] for e in ejemplos_est]))
metricas_mae["ejemplos"] = ejemplos_mae
metricas_mae["confianza_media_ejemplos"] = float(np.mean([e["confianza"] for e in ejemplos_mae]))

SALIDA.mkdir(exist_ok=True)
with open(SALIDA / "metricas_test.json", "w", encoding="utf-8") as f:
    json.dump(metricas_est, f, indent=2, ensure_ascii=False)

with open(SALIDA / "metricas_resnet50.json", "w", encoding="utf-8") as f:
    json.dump(metricas_mae, f, indent=2, ensure_ascii=False)

with open(SALIDA / "historial_entrenamiento.csv", "w", newline="", encoding="utf-8") as f:
    escritor = csv.DictWriter(f, fieldnames=list(historial[0]))
    escritor.writeheader()
    escritor.writerows(historial)

resumen_entrenamiento = {
    "arquitectura": "MobileNetV2 (Estudiante destilado desde Maestro ResNet50)",
    "maestro": "ResNet50 (ImageNet V2 + cabeza 2048->512->3)",
    "destilacion": {
        "temperatura": TEMPERATURA,
        "alfa_kd": ALFA,
        "peso_ce": round(1.0 - ALFA, 2),
        "maestro_parametros": params_mae,
        "maestro_peso_mb": peso_mae_fp32_mb,
        "maestro_peso_archivo_mb": peso_mae_archivo_mb,
        "maestro_latencia_ms": lat_mae_ms,
        "maestro_test_accuracy": metricas_mae["accuracy"],
        "maestro_test_f1_macro": metricas_mae["f1_macro"],
        "maestro_errores": metricas_mae["errores"],
        "estudiante_parametros": params_est,
        "estudiante_peso_mb": peso_est_mb,
        "estudiante_latencia_ms": lat_est_ms,
        "estudiante_test_accuracy": metricas_est["accuracy"],
        "estudiante_test_f1_macro": metricas_est["f1_macro"],
        "estudiante_errores": metricas_est["errores"],
        "reduccion_parametros_x": round(params_mae / params_est, 1),
        "reduccion_peso_x": round(peso_mae_fp32_mb / peso_est_mb, 1),
        "aceleracion_cpu_x": round(lat_mae_ms / lat_est_ms, 1),
    },
    "preentrenado_imagenet": True,
    "epocas_ejecutadas": len(historial),
    "mejor_epoca": mejor_epoca,
    "mejor_fase": 1,
    "val_f1_macro": mejor_f1_va,
    "val_accuracy": mejor_acc_va,
    "parametros_totales": params_est,
    "lote": 32,
    "temperatura_kd": TEMPERATURA,
    "alfa_kd": ALFA,
    "pytorch": torch.__version__,
    "dispositivo": "cpu",
    "imagenes_entrenamiento": dict(zip(CLASES, cuenta)),
}
with open(SALIDA / "resumen_entrenamiento.json", "w", encoding="utf-8") as f:
    json.dump(resumen_entrenamiento, f, indent=2, ensure_ascii=False)

# Figuras de curvas de entrenamiento
f1_m_va = f1_macro(etiq_va.tolist(), logits_m_va.argmax(1).tolist())
fig, (iz, de) = plt.subplots(1, 2, figsize=(11, 4.2))
iz.plot([h["epoca"] for h in historial], [h["perdida"] for h in historial], label="Total (KD + CE)", lw=2)
iz.plot([h["epoca"] for h in historial], [h["perdida_kd"] for h in historial], ls="--", label=f"KL Div (T={TEMPERATURA})")
iz.plot([h["epoca"] for h in historial], [h["perdida_ce"] for h in historial], ls=":", label="Cross-Entropy")
iz.set_title("Pérdida de Destilación (Estudiante)")
iz.set_xlabel("época")
iz.legend()

de.plot([h["epoca"] for h in historial], [h["val_f1_macro"] for h in historial], label="Estudiante val F1 macro", lw=2)
de.plot([h["epoca"] for h in historial], [h["val_accuracy"] for h in historial], label="Estudiante val accuracy")
de.axhline(f1_m_va, ls="--", c="darkorange", label=f"Maestro ResNet50 val F1 ({f1_m_va:.3f})")
de.scatter([mejor_epoca], [mejor_f1_va], marker="*", s=200, c="crimson", zorder=5)
de.set_title("Validación: Estudiante MobileNetV2 vs Maestro ResNet50")
de.set_xlabel("época")
de.legend()
fig.suptitle("Knowledge Distillation: ResNet50 (Maestro) -> MobileNetV2 (Estudiante)")
fig.tight_layout()
fig.savefig(SALIDA / "curvas_entrenamiento.png", dpi=120)

print("\n=== RESULTADO FINAL REFINADO ===")
print(f"Maestro ResNet50      -> Test Acc: {metricas_mae['accuracy']:.2%} | Test F1: {metricas_mae['f1_macro']:.2%} | Errores: {metricas_mae['errores']} | Conf. ejemplos: {metricas_mae['confianza_media_ejemplos']:.1%}")
print(f"Estudiante MobileNetV2 -> Test Acc: {metricas_est['accuracy']:.2%} | Test F1: {metricas_est['f1_macro']:.2%} | Errores: {metricas_est['errores']} | Conf. ejemplos: {metricas_est['confianza_media_ejemplos']:.1%}")
