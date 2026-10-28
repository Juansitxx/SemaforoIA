"""MobileNetV3-Small para el experimento de clasificación de recortes.

No localiza vehículos en escenas, no cuenta carros y no controla semáforos.
--probar comprueba forward/backward con datos sintéticos, sin entrenar ni guardar pesos.
El constructor reutilizable usa pesos ImageNet por defecto para futuro transfer learning.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

from imagenes_base import AjustarCuadrado

CLASES = ("fondo", "vehiculo")


def crear_modelo(preentrenado: bool = True, ajuste_completo: bool = False):
    from torch import nn
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    pesos = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if preentrenado else None
    modelo = mobilenet_v3_small(weights=pesos)
    if not ajuste_completo:
        for parametro in modelo.parameters():
            parametro.requires_grad = False
    entradas = modelo.classifier[-1].in_features
    modelo.classifier[-1] = nn.Linear(entradas, len(CLASES))
    return modelo


def configurar_modo_entrenamiento(modelo, ajuste_completo: bool = False):
    """En modo base, preserva BatchNorm y dropout de la parte congelada."""
    if ajuste_completo:
        modelo.train()
    else:
        modelo.eval()
        modelo.classifier[-1].train()


def crear_transformaciones(entrenamiento: bool = False):
    from torchvision import transforms as T
    from torchvision.models import MobileNet_V3_Small_Weights

    # Leer la especificación no descarga los pesos.
    referencia = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    pasos = [AjustarCuadrado(224)]
    if entrenamiento:
        pasos += [
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        ]
    pasos += [T.ToTensor(), T.Normalize(mean=referencia.mean, std=referencia.std)]
    return T.Compose(pasos)


def probar(proyecto: Path, dispositivo: str, preentrenado: bool):
    import torch
    import torchvision
    from PIL import Image

    torch.manual_seed(42)
    if dispositivo == "auto":
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    if dispositivo == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA no está disponible en este intérprete Python")
    equipo = torch.device(dispositivo)

    modelo = crear_modelo(preentrenado=preentrenado).to(equipo)
    configurar_modo_entrenamiento(modelo)
    transformar = crear_transformaciones(entrenamiento=False)
    # Dos imágenes planas: comprueban dimensiones y gradientes, no reconocimiento.
    imagenes = [Image.new("RGB", (320, 180), (80, 80, 80)),
                Image.new("RGB", (180, 320), (180, 180, 180))]
    lote = torch.stack([transformar(imagen) for imagen in imagenes]).to(equipo)
    logits = modelo(lote)
    if tuple(logits.shape) != (2, 2) or not torch.isfinite(logits).all().item():
        raise RuntimeError("La red no devolvió logits finitos con forma [2, 2]")
    logits.sum().backward()
    gradiente = modelo.classifier[-1].weight.grad
    if gradiente is None or not torch.isfinite(gradiente).all().item():
        raise RuntimeError("No se obtuvieron gradientes válidos en la capa de salida")
    if any(p.grad is not None for p in modelo.features.parameters()):
        raise RuntimeError("La parte congelada recibió gradientes inesperados")
    if equipo.type == "cuda":
        torch.cuda.synchronize()

    resumen = {
        "modelo": "MobileNetV3-Small",
        "tarea": "clasificacion_de_recortes",
        "clases": list(CLASES),
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda_compilacion": torch.version.cuda,
        "dispositivo": str(logits.device),
        "gpu": torch.cuda.get_device_name(0) if equipo.type == "cuda" else None,
        "pesos_backbone": "ImageNet IMAGENET1K_V1" if preentrenado else "aleatorios (solo prueba)",
        "capa_salida": "nueva, inicializada aleatoriamente, sin entrenar",
        "entrada": list(lote.shape),
        "salida": list(logits.shape),
        "parametros_totales": sum(p.numel() for p in modelo.parameters()),
        "parametros_entrenables": sum(p.numel() for p in modelo.parameters() if p.requires_grad),
        "prueba_forward_backward": "correcta",
        "pasos_optimizador": 0,
        "entrenado_para_semaforoia": False,
        "pesos_guardados": False,
        "nota": "No son métricas de precisión ni una validación sobre vehículos reales.",
    }
    sello = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    salida = proyecto / "outputs" / "prueba_modelo" / sello
    salida.mkdir(parents=True, exist_ok=False)
    (salida / "resultado.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    print(f"\nPrueba técnica correcta. Resultado: {salida}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probar", action="store_true", required=True)
    parser.add_argument("--dispositivo", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--preentrenado", action="store_true",
                        help="Descarga/reutiliza pesos oficiales ImageNet; la salida de 2 clases sigue sin entrenar")
    parser.add_argument("--proyecto", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        probar(args.proyecto.resolve(), args.dispositivo, args.preentrenado)
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
