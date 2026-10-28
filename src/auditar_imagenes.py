"""Inventario de imágenes para SemaforoIA, sin mover, borrar o etiquetar originales.

Uso desde la raíz del proyecto:
    python src/auditar_imagenes.py

El manifiesto es una plantilla pendiente de revisión, NO un dataset entrenable.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import warnings

from PIL import Image
from imagenes_base import cargar_rgb


EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CAMPOS = [
    "archivo", "estado", "ancho", "alto", "sha256_pixeles",
    "duplicado_de", "clase", "grupo_escena", "particion", "observacion",
]


def auditar(proyecto: Path, entrada: Path) -> Path:
    proyecto = proyecto.resolve()
    entrada = entrada.resolve()
    if not entrada.is_dir():
        raise ValueError(f"No existe la carpeta de imágenes: {entrada}")
    if not entrada.is_relative_to(proyecto):
        raise ValueError("La carpeta de imágenes debe estar dentro del proyecto")

    archivos = sorted(
        (p for p in entrada.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONES),
        key=lambda p: p.as_posix().casefold(),
    )
    if not archivos:
        raise ValueError(f"No se encontraron imágenes compatibles en: {entrada}")

    sello = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    salida = proyecto / "outputs" / "auditoria" / sello
    salida.mkdir(parents=True, exist_ok=False)
    primeras = {}
    filas = []
    formatos = Counter()
    pequenas = 0
    for numero, ruta in enumerate(archivos, 1):
        relativa = ruta.relative_to(proyecto).as_posix()
        fila = dict.fromkeys(CAMPOS, "")
        fila["archivo"] = relativa
        formatos[ruta.suffix.lower()] += 1
        try:
            # También detecta archivos truncados que algunos lectores toleran.
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(ruta) as comprobacion:
                    comprobacion.verify()
                imagen = cargar_rgb(ruta)
            ancho, alto = imagen.size
            digest = hashlib.sha256()
            digest.update(f"RGB:{ancho}:{alto}:".encode("ascii"))
            digest.update(imagen.tobytes())
            huella = digest.hexdigest()
            fila.update(estado="ok", ancho=ancho, alto=alto, sha256_pixeles=huella)
            if huella in primeras:
                fila["duplicado_de"] = primeras[huella]
            else:
                primeras[huella] = relativa
            if min(ancho, alto) < 96:
                pequenas += 1
                fila["observacion"] = "Lado menor de 96 px: revisar calidad; no descartada"
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as error:
            fila["estado"] = "error"
            fila["observacion"] = f"{type(error).__name__}: {error}"
        filas.append(fila)
        if numero % 100 == 0:
            print(f"Revisadas {numero}/{len(archivos)} imágenes", flush=True)

    with (salida / "inventario.csv").open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS)
        escritor.writeheader()
        escritor.writerows(filas)

    resumen = {
        "entrada": str(entrada),
        "total_archivos": len(filas),
        "imagenes_legibles": sum(f["estado"] == "ok" for f in filas),
        "imagenes_con_error": sum(f["estado"] == "error" for f in filas),
        "imagenes_unicas_por_pixeles": len(primeras),
        "copias_identicas_adicionales": sum(bool(f["duplicado_de"]) for f in filas),
        "imagenes_con_lado_menor_a_96_px": pequenas,
        "formatos": dict(sorted(formatos.items())),
        "estado_dataset": "PENDIENTE_DE_ETIQUETADO_Y_REVISION",
        "notas": [
            "No se han modificado, movido ni eliminado las imágenes originales.",
            "La auditoría no decide si una imagen contiene un vehículo, un dibujo o un collage.",
            "Solo se detectan duplicados exactos tras orientar y convertir a RGB.",
            "Imágenes parecidas, recodificadas o de una misma sesión requieren revisión manual.",
            "clase: fondo o vehiculo solo para ejemplos adecuados de clasificación de recortes.",
            "grupo_escena: mismo identificador para imágenes relacionadas de un mismo origen o sesión.",
            "particion: train, val o test; todos los archivos de un grupo deben ir a la misma partición.",
            "Separar grupos antes de aplicar aumento de datos; no asignar etiquetas por el nombre del archivo.",
        ],
    }
    (salida / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    print(f"\nResultados: {salida}")
    return salida


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proyecto", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--entrada", type=Path, default=Path("imagenes"))
    args = parser.parse_args()
    entrada = args.entrada if args.entrada.is_absolute() else args.proyecto / args.entrada
    try:
        auditar(args.proyecto, entrada)
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
