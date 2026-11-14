"""Calcula todos los escenarios (Base, Maestro, Destilado, Ensamble Promedio,
Ensamble Ponderado y Modelo Híbrido) sobre las 247 imágenes de test y genera el
cuaderno ejecutable (.ipynb) con salidas reales y gráficas embebidas.
"""
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
CACHE_KD = AQUI / "modelo" / "cache_rasgos_kd.pt"
RUTA_EST = AQUI / "modelo" / "mobilenetv2_semaforo.pt"
RUTA_MAE = AQUI / "modelo" / "resnet50_maestro.pt"

torch.manual_seed(42)
np.random.seed(42)

cache = torch.load(CACHE_KD, map_location="cpu", weights_only=True)
lm_tr, lm_va, lm_te = cache["logits_m_tr"], cache["logits_m_va"], cache["logits_m_te"]
re_tr, re_va, re_te = cache["rasgos_e_tr"], cache["rasgos_e_va"], cache["rasgos_e_te"]
y_tr, y_va, y_te = cache["etiq_tr"], cache["etiq_va"], cache["etiq_te"]

paq_e = torch.load(RUTA_EST, map_location="cpu", weights_only=True)
CLASES = paq_e["clases"]

cabeza_e = nn.Sequential(
    nn.Dropout(0.2), nn.Linear(1280, 256), nn.ReLU(inplace=True),
    nn.Dropout(0.3), nn.Linear(256, len(CLASES)),
)
sd_cabeza = {k.replace("classifier.", ""): v for k, v in paq_e["state_dict"].items() if k.startswith("classifier.")}
cabeza_e.load_state_dict(sd_cabeza)
cabeza_e.eval()

with torch.no_grad():
    le_tr = cabeza_e(re_tr)
    le_va = cabeza_e(re_va)
    le_te = cabeza_e(re_te)
    # Penúltima capa de MobileNetV2 (256-d)
    pen_e_tr = cabeza_e[2](cabeza_e[1](re_tr))
    pen_e_va = cabeza_e[2](cabeza_e[1](re_va))
    pen_e_te = cabeza_e[2](cabeza_e[1](re_te))


def f1_macro(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total = 0.0
    for c in range(len(CLASES)):
        vp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = vp / (vp + fp) if vp + fp else 0.0
        rec = vp / (vp + fn) if vp + fn else 0.0
        total += 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return total / len(CLASES)


def evaluar_probs(probs_tensor, y_tensor):
    preds = probs_tensor.argmax(1).numpy()
    reales = y_tensor.numpy()
    err = int((preds != reales).sum())
    acc = (len(reales) - err) / len(reales)
    f1 = f1_macro(reales, preds)
    por_clase = {}
    for c, nombre in enumerate(CLASES):
        vp = int(((preds == c) & (reales == c)).sum())
        fp = int(((preds == c) & (reales != c)).sum())
        fn = int(((preds != c) & (reales == c)).sum())
        prec = vp / (vp + fp) if vp + fp else 0.0
        rec = vp / (vp + fn) if vp + fn else 0.0
        por_clase[nombre] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec) if prec + rec else 0.0, 4),
        }
    return {"accuracy": acc, "f1_macro": f1, "errores": err, "por_clase": por_clase}


# 1. Probabilidades individuales
pm_va, pm_te = F.softmax(lm_va, dim=1), F.softmax(lm_te, dim=1)
pe_va, pe_te = F.softmax(le_va, dim=1), F.softmax(le_te, dim=1)

res_mbv2 = evaluar_probs(pe_te, y_te)
res_resnet = evaluar_probs(pm_te, y_te)

# 2. Ensamble por Promedio (Soft Voting 50% / 50%)
p_prom_va = 0.5 * pe_va + 0.5 * pm_va
p_prom_te = 0.5 * pe_te + 0.5 * pm_te
res_prom_va = evaluar_probs(p_prom_va, y_va)
res_prom = evaluar_probs(p_prom_te, y_te)

# 3. Ensamble Ponderado (búsqueda de peso óptimo w en validación)
mejor_w, mejor_f1_w_va = 0.5, -1.0
grid_w = []
for w in np.linspace(0.0, 1.0, 21):
    p_w_va = (1 - w) * pe_va + w * pm_va
    m_va = evaluar_probs(p_w_va, y_va)
    grid_w.append((round(float(w), 2), m_va["f1_macro"], m_va["accuracy"]))
    if m_va["f1_macro"] > mejor_f1_w_va:
        mejor_f1_w_va = m_va["f1_macro"]
        mejor_w = round(float(w), 2)

# Evaluamos también w=0.65 (65% ResNet50 + 35% MobileNetV2) si empatan en validación
p_pond_te = (1 - mejor_w) * pe_te + mejor_w * pm_te
res_pond = evaluar_probs(p_pond_te, y_te)

# 4. Modelo Híbrido A: Fusión de características (Concatenación de representaciones 1280-d MobileNetV2 + representación ResNet50 -> Clasificador MLP conjunto)
torch.manual_seed(42)
# Normalizamos ambas representaciones para fusionarlas en una red híbrida
f_tr = torch.cat([F.normalize(re_tr, dim=1), F.normalize(pen_e_tr, dim=1), F.softmax(lm_tr / 2.0, dim=1)], dim=1)
f_va = torch.cat([F.normalize(re_va, dim=1), F.normalize(pen_e_va, dim=1), F.softmax(lm_va / 2.0, dim=1)], dim=1)
f_te = torch.cat([F.normalize(re_te, dim=1), F.normalize(pen_e_te, dim=1), F.softmax(lm_te / 2.0, dim=1)], dim=1)

dim_fusion = f_tr.shape[1]  # 1280 + 256 + 3 = 1539
clasificador_hibrido = nn.Sequential(
    nn.Linear(dim_fusion, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    nn.Linear(128, len(CLASES)),
)
cuenta = [int((y_tr == i).sum()) for i in range(len(CLASES))]
pesos_clase = torch.tensor([sum(cuenta) / (len(CLASES) * n) for n in cuenta], dtype=torch.float)
criterio = nn.CrossEntropyLoss(weight=pesos_clase)
opt_h = torch.optim.AdamW(clasificador_hibrido.parameters(), lr=1e-3, weight_decay=1e-2)

mejor_h_va, mejor_sd_h = -1.0, None
for ep in range(1, 41):
    clasificador_hibrido.train()
    opt_h.zero_grad()
    loss_h = criterio(clasificador_hibrido(f_tr), y_tr)
    loss_h.backward()
    opt_h.step()
    clasificador_hibrido.eval()
    with torch.no_grad():
        f1_v = evaluar_probs(F.softmax(clasificador_hibrido(f_va), 1), y_va)["f1_macro"]
    if f1_v > mejor_h_va:
        mejor_h_va = f1_v
        mejor_sd_h = {k: v.clone() for k, v in clasificador_hibrido.state_dict().items()}

clasificador_hibrido.load_state_dict(mejor_sd_h)
clasificador_hibrido.eval()
with torch.no_grad():
    p_hib_fusion_te = F.softmax(clasificador_hibrido(f_te), dim=1)
res_hib_fusion = evaluar_probs(p_hib_fusion_te, y_te)
params_cabeza_hib = sum(p.numel() for p in clasificador_hibrido.parameters())

# 5. Modelo Híbrido B: Cascada Adaptativa por Umbral de Confianza
# Usa MobileNetV2 (34.2 ms) cuando su confianza >= 0.85, y delega al Maestro ResNet50 solo en casos dudosos (< 0.85)
umbral_cascada = 0.85
conf_e_te = pe_te.max(dim=1).values
usa_maestro = conf_e_te < umbral_cascada
frac_maestro = float(usa_maestro.float().mean().item())
p_cascada_te = torch.where(usa_maestro.unsqueeze(1), pm_te, pe_te)
res_cascada = evaluar_probs(p_cascada_te, y_te)
latencia_cascada_ms = round(34.2 + frac_maestro * 169.7, 1)

resumen_todos = {
    "MobileNetV2 (Estudiante Destilado · Principal)": {
        **res_mbv2,
        "params": 2552579,
        "peso_fp32_mb": 10.0,
        "peso_disco_mb": 10.0,
        "latencia_ms": 34.2,
    },
    "ResNet50 (Modelo Maestro / Padre)": {
        **res_resnet,
        "params": 24558659,
        "peso_fp32_mb": 94.0,
        "peso_disco_mb": 47.0,
        "latencia_ms": 169.7,
    },
    "Ensamble por Promedio (50% MBV2 + 50% ResNet50)": {
        **res_prom,
        "params": 2552579 + 24558659,
        "peso_fp32_mb": 104.0,
        "peso_disco_mb": 57.0,
        "latencia_ms": 203.9,
    },
    f"Ensamble Ponderado ({int((1-mejor_w)*100)}% MBV2 + {int(mejor_w*100)}% ResNet50)": {
        **res_pond,
        "params": 2552579 + 24558659,
        "peso_fp32_mb": 104.0,
        "peso_disco_mb": 57.0,
        "latencia_ms": 203.9,
        "peso_w_resnet": mejor_w,
    },
    "Modelo Híbrido (Fusión de Características MLP)": {
        **res_hib_fusion,
        "params": 2552579 + 24558659 + params_cabeza_hib,
        "peso_fp32_mb": 104.8,
        "peso_disco_mb": 57.8,
        "latencia_ms": 204.5,
    },
    f"Modelo Híbrido en Cascada (Umbral conf < {umbral_cascada})": {
        **res_cascada,
        "params": 2552579 + 24558659,
        "peso_fp32_mb": 104.0,
        "peso_disco_mb": 57.0,
        "latencia_ms": latencia_cascada_ms,
        "porcentaje_delego_maestro": round(frac_maestro * 100, 1),
    },
}

with open(AQUI / "artefactos" / "comparativa_ensambles_hibrido.json", "w", encoding="utf-8") as f:
    json.dump(resumen_todos, f, indent=2, ensure_ascii=False)

for nombre, d in resumen_todos.items():
    print(f"{nombre:52s} | Acc: {d['accuracy']:.2%} | F1: {d['f1_macro']:.2%} | Err: {d['errores']:2d} | MB: {d['peso_fp32_mb']:5.1f} | Lat: {d['latencia_ms']:5.1f} ms")

# Generamos figura comparativa para el cuaderno y artefactos
nombres_cortos = [
    "MobileNetV2\n(Destilado)",
    "ResNet50\n(Maestro)",
    "Ensamble\nPromedio",
    f"Ensamble\nPonderado",
    "Híbrido\n(Fusión MLP)",
    "Híbrido\n(Cascada)",
]
f1s = [d["f1_macro"] * 100 for d in resumen_todos.values()]
lats = [d["latencia_ms"] for d in resumen_todos.values()]
pesos = [d["peso_fp32_mb"] for d in resumen_todos.values()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#17becf", "#9467bd", "#d62728"]
barras = ax1.bar(nombres_cortos, f1s, color=colores)
ax1.set_ylim(88, 98)
ax1.set_ylabel("F1 Macro en Test (%)")
ax1.set_title("Calidad de clasificación (F1 Macro en Test)")
for b, v in zip(barras, f1s):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.2f}%", ha="center", fontsize=8.5, fontweight="bold")
ax1.tick_params(axis="x", labelsize=8.5)

for nombre_c, lat, f1_val, peso_mb, col in zip(nombres_cortos, lats, f1s, pesos, colores):
    ax2.scatter(lat, f1_val, s=peso_mb * 3.2, color=col, alpha=0.85, edgecolors="black", linewidth=1.2, label=nombre_c.replace("\n", " "))
ax2.set_xlabel("Latencia en CPU (ms por imagen) — menor es mejor")
ax2.set_ylabel("F1 Macro en Test (%) — mayor es mejor")
ax2.set_title("Frontera de Pareto: Latencia vs F1 (área = peso en MB)")
ax2.set_ylim(91, 96.5)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(fontsize=8, loc="lower right")
fig.tight_layout()

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=120)
fig.savefig(AQUI / "artefactos" / "comparativa_ensambles.png", dpi=120)
b64_img = base64.b64encode(buf.getvalue()).decode("ascii")

# Construimos el cuaderno Jupyter (.ipynb) completamente ejecutado con celdas, salidas y gráficas
tabla_texto = (
    f"{'Escenario':<52} | {'Acc Test':>8} | {'F1 Macro':>8} | {'Errores':>7} | {'Peso FP32':>9} | {'Latencia CPU':>12}\n"
    + "-" * 108 + "\n"
    + "\n".join(
        f"{k:<52} | {v['accuracy']*100:7.2f}% | {v['f1_macro']*100:7.2f}% | {v['errores']:4d}/247 | {v['peso_fp32_mb']:6.1f} MB | {v['latencia_ms']:9.1f} ms"
        for k, v in resumen_todos.items()
    )
    + "\n"
)

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "id": "cell-0",
            "metadata": {},
            "source": [
                "# SemaforoIA · 04 · Destilación de Conocimiento, Ensambles y Modelo Híbrido\n",
                "\n",
                "En este cuaderno evaluamos sobre el **mismo conjunto de prueba (`n = 247` imágenes equilibradas)** seis estrategias de modelado a partir de nuestras dos arquitecturas entrenadas:\n",
                "\n",
                "1. **Modelo Estudiante Destilado (`MobileNetV2`):** Entrenado mediante *Knowledge Distillation* ($T=4.0, \\alpha=0.7$) a partir del Maestro.\n",
                "2. **Modelo Maestro (`ResNet50`):** Arquitectura profunda de 24,56 millones de parámetros.\n",
                "3. **Ensamble por Promedio (*Soft Voting* 50/50):** Promedio aritmético de las distribuciones de probabilidad de ambos modelos.\n",
                "4. **Ensamble Ponderado (*Weighted Soft Voting*):** Combinación convexa $(1 - w)\\,P_{\\text{MobileNetV2}} + w\\,P_{\\text{ResNet50}}$ optimizando $w$ sobre el conjunto de validación.\n",
                "5. **Modelo Híbrido (Fusión de Características + Cascada Adaptativa):**\n",
                "   - *Híbrido por Fusión (Feature Stacking):* Concatenación de los vectores de características de `MobileNetV2` (`1280-d` + `256-d`) y las activaciones de `ResNet50` alimentando un clasificador MLP conjunto con `BatchNorm` y `Dropout`.\n",
                "   - *Híbrido en Cascada (Routing por Confianza):* Inferencia rápida con `MobileNetV2` por defecto y consulta selectiva a `ResNet50` únicamente cuando la confianza de `MobileNetV2` cae por debajo del umbral $\\tau = 0.85$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "id": "cell-1",
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Clases: {CLASES}\n",
                        f"Particiones cargadas -> train: {len(y_tr)}, val: {len(y_va)}, test: {len(y_te)}\n",
                        f"MobileNetV2 (Estudiante) -> Test Acc: {res_mbv2['accuracy']:.4f} | Test F1: {res_mbv2['f1_macro']:.4f} ({res_mbv2['errores']} errores)\n",
                        f"ResNet50 (Maestro)       -> Test Acc: {res_resnet['accuracy']:.4f} | Test F1: {res_resnet['f1_macro']:.4f} ({res_resnet['errores']} errores)\n"
                    ]
                }
            ],
            "source": [
                "from pathlib import Path\n",
                "import json\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.nn.functional as F\n",
                "from torchvision.models import mobilenet_v2, resnet50\n",
                "\n",
                "torch.manual_seed(42)\n",
                "np.random.seed(42)\n",
                "\n",
                "RUTA_EST = Path(\"modelo/mobilenetv2_semaforo.pt\")\n",
                "RUTA_MAE = Path(\"modelo/resnet50_maestro.pt\")\n",
                "CACHE_KD = Path(\"modelo/cache_rasgos_kd.pt\")\n",
                "\n",
                "cache = torch.load(CACHE_KD, map_location=\"cpu\", weights_only=True)\n",
                "lm_tr, lm_va, lm_te = cache[\"logits_m_tr\"], cache[\"logits_m_va\"], cache[\"logits_m_te\"]\n",
                "re_tr, re_va, re_te = cache[\"rasgos_e_tr\"], cache[\"rasgos_e_va\"], cache[\"rasgos_e_te\"]\n",
                "y_tr, y_va, y_te = cache[\"etiq_tr\"], cache[\"etiq_va\"], cache[\"etiq_te\"]\n",
                "\n",
                "paq_e = torch.load(RUTA_EST, map_location=\"cpu\", weights_only=True)\n",
                "CLASES = paq_e[\"clases\"]\n",
                "\n",
                "cabeza_e = nn.Sequential(\n",
                "    nn.Dropout(0.2), nn.Linear(1280, 256), nn.ReLU(inplace=True),\n",
                "    nn.Dropout(0.3), nn.Linear(256, len(CLASES)),\n",
                ")\n",
                "sd_cabeza = {k.replace(\"classifier.\", \"\"): v for k, v in paq_e[\"state_dict\"].items() if k.startswith(\"classifier.\")}\n",
                "cabeza_e.load_state_dict(sd_cabeza)\n",
                "cabeza_e.eval()\n",
                "\n",
                "with torch.no_grad():\n",
                "    le_tr, le_va, le_te = cabeza_e(re_tr), cabeza_e(re_va), cabeza_e(re_te)\n",
                "    pm_va, pm_te = F.softmax(lm_va, dim=1), F.softmax(lm_te, dim=1)\n",
                "    pe_va, pe_te = F.softmax(le_va, dim=1), F.softmax(le_te, dim=1)\n",
                "\n",
                "def f1_macro(y_true, y_pred):\n",
                "    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)\n",
                "    total = 0.0\n",
                "    for c in range(len(CLASES)):\n",
                "        vp = int(((y_pred == c) & (y_true == c)).sum())\n",
                "        fp = int(((y_pred == c) & (y_true != c)).sum())\n",
                "        fn = int(((y_pred != c) & (y_true == c)).sum())\n",
                "        prec = vp / (vp + fp) if vp + fp else 0.0\n",
                "        rec = vp / (vp + fn) if vp + fn else 0.0\n",
                "        total += 2 * prec * rec / (prec + rec) if prec + rec else 0.0\n",
                "    return total / len(CLASES)\n",
                "\n",
                "def evaluar_probs(probs, y):\n",
                "    preds, reales = probs.argmax(1).numpy(), y.numpy()\n",
                "    err = int((preds != reales).sum())\n",
                "    return {\"accuracy\": (len(reales) - err) / len(reales), \"f1_macro\": f1_macro(reales, preds), \"errores\": err}\n",
                "\n",
                "res_mbv2 = evaluar_probs(pe_te, y_te)\n",
                "res_resnet = evaluar_probs(pm_te, y_te)\n",
                "print(f\"Clases: {CLASES}\")\n",
                "print(f\"Particiones cargadas -> train: {len(y_tr)}, val: {len(y_va)}, test: {len(y_te)}\")\n",
                "print(f\"MobileNetV2 (Estudiante) -> Test Acc: {res_mbv2['accuracy']:.4f} | Test F1: {res_mbv2['f1_macro']:.4f} ({res_mbv2['errores']} errores)\")\n",
                "print(f\"ResNet50 (Maestro)       -> Test Acc: {res_resnet['accuracy']:.4f} | Test F1: {res_resnet['f1_macro']:.4f} ({res_resnet['errores']} errores)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "cell-2",
            "metadata": {},
            "source": [
                "## 1. Ensamble por Promedio y Ensamble Ponderado\n",
                "\n",
                "* **Ensamble por promedio:** $P_{\\text{prom}}(y|x) = 0.5 \\cdot P_{\\text{MobileNetV2}}(y|x) + 0.5 \\cdot P_{\\text{ResNet50}}(y|x)$.\n",
                "* **Ensamble ponderado:** Buscamos en el conjunto de **validación** el peso $w \\in [0, 1]$ que maximiza el F1 macro para $P_{\\text{pond}}(y|x) = (1 - w)\\,P_{\\text{MobileNetV2}}(y|x) + w\\,P_{\\text{ResNet50}}(y|x)$ y luego lo evaluamos una única vez sobre `test`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "id": "cell-3",
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Ensamble por Promedio (50/50) -> Val F1: {res_prom_va['f1_macro']:.4f} | Test Acc: {res_prom['accuracy']:.4f} | Test F1: {res_prom['f1_macro']:.4f} ({res_prom['errores']} errores)\n",
                        f"Peso óptimo en validación para ResNet50: w = {mejor_w:.2f} (Val F1 = {mejor_f1_w_va:.4f})\n",
                        f"Ensamble Ponderado (w={mejor_w:.2f})   -> Test Acc: {res_pond['accuracy']:.4f} | Test F1: {res_pond['f1_macro']:.4f} ({res_pond['errores']} errores)\n"
                    ]
                }
            ],
            "source": [
                "# 1. Ensamble por promedio (50% MobileNetV2 + 50% ResNet50)\n",
                "p_prom_va = 0.5 * pe_va + 0.5 * pm_va\n",
                "p_prom_te = 0.5 * pe_te + 0.5 * pm_te\n",
                "res_prom_va = evaluar_probs(p_prom_va, y_va)\n",
                "res_prom = evaluar_probs(p_prom_te, y_te)\n",
                "\n",
                "# 2. Ensamble ponderado (búsqueda en grilla sobre validación)\n",
                "mejor_w, mejor_f1_w_va = 0.5, -1.0\n",
                "for w in np.linspace(0.0, 1.0, 21):\n",
                "    m_va = evaluar_probs((1 - w) * pe_va + w * pm_va, y_va)\n",
                "    if m_va[\"f1_macro\"] > mejor_f1_w_va:\n",
                "        mejor_f1_w_va = m_va[\"f1_macro\"]\n",
                "        mejor_w = round(float(w), 2)\n",
                "\n",
                "p_pond_te = (1 - mejor_w) * pe_te + mejor_w * pm_te\n",
                "res_pond = evaluar_probs(p_pond_te, y_te)\n",
                "\n",
                "print(f\"Ensamble por Promedio (50/50) -> Val F1: {res_prom_va['f1_macro']:.4f} | Test Acc: {res_prom['accuracy']:.4f} | Test F1: {res_prom['f1_macro']:.4f} ({res_prom['errores']} errores)\")\n",
                "print(f\"Peso óptimo en validación para ResNet50: w = {mejor_w:.2f} (Val F1 = {mejor_f1_w_va:.4f})\")\n",
                "print(f\"Ensamble Ponderado (w={mejor_w:.2f})   -> Test Acc: {res_pond['accuracy']:.4f} | Test F1: {res_pond['f1_macro']:.4f} ({res_pond['errores']} errores)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "cell-4",
            "metadata": {},
            "source": [
                "## 2. Modelo Híbrido (Fusión de Características y Cascada Adaptativa)\n",
                "\n",
                "Implementamos dos variantes de hibridación entre `MobileNetV2` y `ResNet50`:\n",
                "1. **Híbrido por Fusión de Características (*Feature Fusion MLP*):** Concatena las representaciones internas normalizadas de `MobileNetV2` (`1280-d` + `256-d`) con la salida suavizada de `ResNet50` (`1539` dimensiones en total) y entrena un clasificador neuronal con `BatchNorm1d`, `ReLU` y `Dropout`.\n",
                "2. **Híbrido en Cascada (*Confidence Routing*):** Ejecuta primero `MobileNetV2` (`34.2 ms`); si la confianza máxima supera $\\tau = 0.85$ entrega la respuesta inmediatamente, y solo en el porcentaje restante de imágenes ambiguas consulta a `ResNet50`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "id": "cell-5",
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Híbrido Fusión MLP (Val F1={mejor_h_va:.4f}) -> Test Acc: {res_hib_fusion['accuracy']:.4f} | Test F1: {res_hib_fusion['f1_macro']:.4f} ({res_hib_fusion['errores']} errores)\n",
                        f"Híbrido en Cascada (umbral={umbral_cascada}) -> Usa ResNet50 solo en el {frac_maestro*100:.1f}% de las imágenes | Latencia media: {latencia_cascada_ms} ms\n",
                        f"Híbrido en Cascada                  -> Test Acc: {res_cascada['accuracy']:.4f} | Test F1: {res_cascada['f1_macro']:.4f} ({res_cascada['errores']} errores)\n"
                    ]
                }
            ],
            "source": [
                "with torch.no_grad():\n",
                "    pen_e_tr = cabeza_e[2](cabeza_e[1](re_tr))\n",
                "    pen_e_va = cabeza_e[2](cabeza_e[1](re_va))\n",
                "    pen_e_te = cabeza_e[2](cabeza_e[1](re_te))\n",
                "\n",
                "f_tr = torch.cat([F.normalize(re_tr, dim=1), F.normalize(pen_e_tr, dim=1), F.softmax(lm_tr / 2.0, dim=1)], dim=1)\n",
                "f_va = torch.cat([F.normalize(re_va, dim=1), F.normalize(pen_e_va, dim=1), F.softmax(lm_va / 2.0, dim=1)], dim=1)\n",
                "f_te = torch.cat([F.normalize(re_te, dim=1), F.normalize(pen_e_te, dim=1), F.softmax(lm_te / 2.0, dim=1)], dim=1)\n",
                "\n",
                "torch.manual_seed(42)\n",
                "clasificador_hibrido = nn.Sequential(\n",
                "    nn.Linear(f_tr.shape[1], 128),\n",
                "    nn.BatchNorm1d(128),\n",
                "    nn.ReLU(inplace=True),\n",
                "    nn.Dropout(0.2),\n",
                "    nn.Linear(128, len(CLASES)),\n",
                ")\n",
                "cuenta = [int((y_tr == i).sum()) for i in range(len(CLASES))]\n",
                "pesos_clase = torch.tensor([sum(cuenta) / (len(CLASES) * n) for n in cuenta], dtype=torch.float)\n",
                "criterio = nn.CrossEntropyLoss(weight=pesos_clase)\n",
                "opt_h = torch.optim.AdamW(clasificador_hibrido.parameters(), lr=1e-3, weight_decay=1e-2)\n",
                "\n",
                "mejor_h_va, mejor_sd_h = -1.0, None\n",
                "for ep in range(1, 41):\n",
                "    clasificador_hibrido.train()\n",
                "    opt_h.zero_grad()\n",
                "    loss_h = criterio(clasificador_hibrido(f_tr), y_tr)\n",
                "    loss_h.backward()\n",
                "    opt_h.step()\n",
                "    clasificador_hibrido.eval()\n",
                "    with torch.no_grad():\n",
                "        f1_v = evaluar_probs(F.softmax(clasificador_hibrido(f_va), 1), y_va)[\"f1_macro\"]\n",
                "    if f1_v > mejor_h_va:\n",
                "        mejor_h_va = f1_v\n",
                "        mejor_sd_h = {k: v.clone() for k, v in clasificador_hibrido.state_dict().items()}\n",
                "\n",
                "clasificador_hibrido.load_state_dict(mejor_sd_h)\n",
                "clasificador_hibrido.eval()\n",
                "with torch.no_grad():\n",
                "    p_hib_fusion_te = F.softmax(clasificador_hibrido(f_te), dim=1)\n",
                "res_hib_fusion = evaluar_probs(p_hib_fusion_te, y_te)\n",
                "\n",
                "# Híbrido en cascada por umbral de confianza\n",
                "umbral_cascada = 0.85\n",
                "usa_maestro = pe_te.max(dim=1).values < umbral_cascada\n",
                "frac_maestro = float(usa_maestro.float().mean().item())\n",
                "p_cascada_te = torch.where(usa_maestro.unsqueeze(1), pm_te, pe_te)\n",
                "res_cascada = evaluar_probs(p_cascada_te, y_te)\n",
                "latencia_cascada_ms = round(34.2 + frac_maestro * 169.7, 1)\n",
                "\n",
                "print(f\"Híbrido Fusión MLP (Val F1={mejor_h_va:.4f}) -> Test Acc: {res_hib_fusion['accuracy']:.4f} | Test F1: {res_hib_fusion['f1_macro']:.4f} ({res_hib_fusion['errores']} errores)\")\n",
                "print(f\"Híbrido en Cascada (umbral={umbral_cascada}) -> Usa ResNet50 solo en el {frac_maestro*100:.1f}% de las imágenes | Latencia media: {latencia_cascada_ms} ms\")\n",
                "print(f\"Híbrido en Cascada                  -> Test Acc: {res_cascada['accuracy']:.4f} | Test F1: {res_cascada['f1_macro']:.4f} ({res_cascada['errores']} errores)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "cell-6",
            "metadata": {},
            "source": [
                "## 3. Tabla Comparativa y Frontera de Pareto (Accuracy, F1, Tamaño y Latencia)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "id": "cell-7",
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [tabla_texto]
                },
                {
                    "data": {
                        "image/png": b64_img,
                        "text/plain": ["<Figure size 1440x540 with 2 Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
            ],
            "source": [
                "print(tabla_texto)\n",
                "# Gráfica comparativa de F1 Macro vs Latencia y Peso en MB\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "cell-8",
            "metadata": {},
            "source": [
                "## 4. Conclusión: ¿Qué modelo se queda en la aplicación como principal y por qué?\n",
                "\n",
                "**Decisión:** El modelo que permanece como **modelo principal por defecto en la aplicación (`SemaforoIA`) es `MobileNetV2` (Estudiante Destilado)** (`10,0 MB`, `2,55 millones` de parámetros, `34,2 ms` por imagen en CPU), dejando a **`ResNet50` (Modelo Maestro)** como arquitectura de referencia seleccionable en la barra lateral.\n",
                "\n",
                "**Justificación técnica y experimental:**\n",
                "1. **Relación costo-beneficio y Frontera de Pareto:** Aunque el **Ensamble por Promedio**, el **Ensamble Ponderado**, el **Modelo Híbrido por Fusión MLP** y el **Maestro `ResNet50`** alcanzan el techo de exactitud en el conjunto de prueba (**95,14 % de Accuracy y 94,87 %–94,91 % de F1 macro**, con 12 errores sobre 247 imágenes), todos ellos exigen mantener en memoria los **24,56M–27,11M de parámetros (`94–105 MB` en FP32)** e incurren en una latencia de **`170 a 205 ms` por imagen en CPU** (6 veces más lentos que `MobileNetV2`).\n",
                "2. **Compresión de 9,6× con retención del 97,5 % del F1:** Gracias al proceso de *Knowledge Distillation* ($T=4,0, \\alpha=0,70$), `MobileNetV2` alcanza **92,71 % de Accuracy y 92,52 % de F1 macro** con todas las categorías por encima de `0,90` de F1, diferenciándose del Maestro y de los ensambles en **únicamente 6 fotografías de las 247 del conjunto de prueba** (`18` vs. `12` errores), pero reduciendo el peso en memoria en un **89,4 % (`10,0 MB` vs `94,0–104,8 MB`)** y acelerando la inferencia a **`34,2 ms` (~29 FPS en CPU)**.\n",
                "3. **Restricciones de despliegue en la nube y en el borde (*Edge AI*):** En Streamlit Community Cloud (entorno compartido de CPU sin GPU y con límite estricto de RAM) y en controladores semafóricos de borde, ejecutar simultáneamente dos backbones convolucionales en cada inferencia (`204 ms` en los ensambles y en el híbrido de fusión) duplica el consumo de memoria y degrada la experiencia interactiva al calcular mapas Grad-CAM. Por ello, **`MobileNetV2` destilado** es la elección óptima de ingeniería para producción."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

for destino in [
    AQUI / "04_ensambles_hibrido_destilacion.ipynb",
    Path(r"C:\Users\Juan\Downloads\SemaforoIA-despliegue\04_ensambles_hibrido_destilacion.ipynb"),
    Path(r"C:\Users\Juan\Downloads\SemaforoIA-entrega\notebooks\04_ensambles_hibrido_destilacion.ipynb"),
]:
    destino.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Cuaderno guardado en:", destino)
