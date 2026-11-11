"""Quita duplicados, imágenes pequeñas y archivos corruptos de lo recolectado.

    python limpiar.py recolectado          # limpia esa carpeta
    python limpiar.py --autoprueba         # comprueba que la lógica funciona

Nada se borra: lo descartado se mueve a  descartadas/<motivo>/  para que puedas
revisarlo. Si el umbral quedó agresivo, lo devuelves y ya.

Usa dHash (hash perceptual por gradiente) para detectar casi duplicados: la foto
reescalada, recomprimida o con otra marca de agua da el mismo hash o muy cercano.
Solo necesita Pillow.
"""
import sys
from pathlib import Path
from PIL import Image

LADO_MINIMO = 96          # el pipeline de entrenamiento descarta por debajo
DISTANCIA_MAXIMA = 5      # bits distintos; 0 = idénticas, <=5 = casi iguales
EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def hash_diferencial(imagen: Image.Image, lado: int = 8) -> int:
    """Compara cada píxel con su vecino de la derecha: 64 bits de estructura."""
    gris = imagen.convert("L").resize((lado + 1, lado), Image.Resampling.LANCZOS)
    pixeles = gris.tobytes()          # modo L: un byte por píxel, fila por fila
    bits = 0
    for y in range(lado):
        fila = pixeles[y * (lado + 1):(y + 1) * (lado + 1)]
        for x in range(lado):
            bits = bits << 1 | int(fila[x] < fila[x + 1])
    return bits


def distancia(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def limpiar(carpeta: Path) -> dict:
    descartes = Path("descartadas")
    conteo = {"conservadas": 0, "duplicada": 0, "pequena": 0, "corrupta": 0}
    vistos: dict[int, Path] = {}

    def apartar(ruta: Path, motivo: str) -> None:
        destino = descartes / motivo / ruta.name
        destino.parent.mkdir(parents=True, exist_ok=True)
        # Si ya existe uno con ese nombre, le ponemos sufijo en vez de pisarlo.
        n = 1
        while destino.exists():
            destino = destino.with_name(f"{ruta.stem}_{n}{ruta.suffix}")
            n += 1
        ruta.rename(destino)
        conteo[motivo] += 1

    for ruta in sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in EXTENSIONES):
        try:
            with Image.open(ruta) as imagen:
                imagen.load()
                if min(imagen.size) < LADO_MINIMO:
                    apartar(ruta, "pequena")
                    continue
                huella = hash_diferencial(imagen)
        except Exception:
            apartar(ruta, "corrupta")
            continue

        # ponytail: comparación O(n²) contra los hashes ya vistos. Con unos pocos
        # miles de imágenes sobra; si llegas a decenas de miles, indexa por
        # prefijo de bits (BK-tree) en vez de recorrer todo.
        if any(distancia(huella, visto) <= DISTANCIA_MAXIMA for visto in vistos):
            apartar(ruta, "duplicada")
            continue

        vistos[huella] = ruta
        conteo["conservadas"] += 1

    return conteo


def _imagen(semilla: int, lado: int = 300) -> Image.Image:
    """Degradado suave y determinista; la orientación cambia con la semilla.

    Suave a propósito: así el hash sobrevive al reescalado, que es justo lo que
    la prueba necesita comprobar.
    """
    if semilla == 1:
        datos = bytes((x * 255 // lado) for _ in range(lado) for x in range(lado))
    else:
        datos = bytes((y * 255 // lado) for y in range(lado) for _ in range(lado))
    return Image.frombytes("L", (lado, lado), datos).convert("RGB")


def autoprueba() -> None:
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        anterior = Path.cwd()
        try:
            os.chdir(raiz)
            origen = raiz / "fotos"
            origen.mkdir()
            _imagen(1).save(origen / "a.jpg")
            _imagen(1).save(origen / "a_copia.jpg")                      # duplicada exacta
            _imagen(1).resize((180, 180)).save(origen / "a_chica.jpg")   # casi duplicada
            _imagen(2).save(origen / "b.jpg")                            # distinta
            _imagen(1, lado=50).save(origen / "mini.jpg")                # muy pequeña
            (origen / "rota.jpg").write_bytes(b"esto no es una imagen")  # corrupta

            r = limpiar(origen)
            assert r["conservadas"] == 2, f"esperaba conservar 2, conservó {r['conservadas']}"
            assert r["duplicada"] == 2, f"esperaba 2 duplicadas, hubo {r['duplicada']}"
            assert r["pequena"] == 1, f"esperaba 1 pequeña, hubo {r['pequena']}"
            assert r["corrupta"] == 1, f"esperaba 1 corrupta, hubo {r['corrupta']}"
            assert (raiz / "descartadas" / "duplicada").is_dir(), "no apartó las duplicadas"
            print("autoprueba OK:", r)
        finally:
            os.chdir(anterior)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--autoprueba":
        autoprueba()
    elif len(sys.argv) > 1:
        print(limpiar(Path(sys.argv[1])))
    else:
        sys.exit(__doc__)
