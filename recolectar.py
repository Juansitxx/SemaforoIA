"""Recolecta imágenes para las clases bicicleta, carro y moto.

Dos modos:

    python recolectar.py openimages      # recomendado: etiquetas verificadas
    python recolectar.py buscar          # complemento: scraping de buscadores

El modo `openimages` necesita `pip install fiftyone` (pesado, pensado para Colab).
El modo `buscar` necesita `pip install icrawler`.

Las imágenes quedan en  recolectado/<clase>/  listas para pasar por limpiar.py
"""
import shutil
import sys
from pathlib import Path

# Nuestros nombres -> los de Open Images.
CLASES = {"bicicleta": "Bicycle", "moto": "Motorcycle", "carro": "Car"}
POR_CLASE = 700
AREA_MINIMA = 0.15      # el vehículo debe ocupar al menos este % del encuadre
DESTINO = Path("recolectado")


def _area(deteccion) -> float:
    """Área relativa de la caja: bounding_box es [x, y, ancho, alto] en 0-1."""
    return deteccion.bounding_box[2] * deteccion.bounding_box[3]


def _cajas(muestra) -> list:
    """Las detecciones de la muestra. fiftyone las ha llamado de las dos formas."""
    for campo in ("ground_truth", "detections"):
        if muestra.has_field(campo):
            valor = muestra.get_field(campo)
            if valor is not None:
                return valor.detections
    return []


def _es_dominante(muestra, objetivo: str, otras: list[str], area_minima: float) -> bool:
    """La clase objetivo manda en la imagen y ningún otro vehículo la supera.

    Usamos cajas y no las etiquetas de clasificación a propósito. Las etiquetas
    de clasificación de Open Images marcan *presencia*, no *dominancia*: una foto
    de patinadores con una bicicleta de fondo viene marcada como Bicycle, y para
    un clasificador de imagen completa eso es una etiqueta equivocada.
    """
    detecciones = _cajas(muestra)
    propias = [_area(d) for d in detecciones if d.label == objetivo]
    if not propias or max(propias) < area_minima:
        return False
    ajenas = [_area(d) for d in detecciones if d.label in otras]
    return not ajenas or max(ajenas) < max(propias)


def desde_openimages(por_clase: int = POR_CLASE, split: str = "train",
                     area_minima: float = AREA_MINIMA) -> dict:
    import fiftyone.zoo as foz

    resultado = {}
    for nuestra, suya in CLASES.items():
        otras = [v for k, v in CLASES.items() if k != nuestra]
        conjunto = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=[suya],
            max_samples=por_clase * 3,     # pedimos de más: el filtro recorta fuerte
            only_matching=True,
            dataset_name=f"oi-{nuestra}-{split}",
            overwrite=True,
        )
        salida = DESTINO / nuestra
        salida.mkdir(parents=True, exist_ok=True)

        guardadas = 0
        for muestra in conjunto:
            if guardadas >= por_clase:
                break
            if not _es_dominante(muestra, suya, otras, area_minima):
                continue
            origen = Path(muestra.filepath)
            shutil.copy2(origen, salida / f"{nuestra}_{origen.name}")
            guardadas += 1

        resultado[nuestra] = guardadas
        print(f"{nuestra}: {guardadas} de {len(conjunto)} descargadas -> {salida}")
    return resultado


def desde_buscadores(por_clase: int = 300) -> None:
    """Complemento. Espera mucho ruido: revisa a mano antes de entrenar."""
    from icrawler.builtin import BingImageCrawler

    consultas = {
        "bicicleta": ["bicicleta en la calle", "bicycle side view photo",
                      "cyclist riding bicycle street"],
        "moto": ["motocicleta en la calle", "motorcycle side view photo",
                 "motorbike parked street"],
        "carro": ["carro en la calle", "car side view photo"],
    }
    for clase, frases in consultas.items():
        salida = DESTINO / clase
        salida.mkdir(parents=True, exist_ok=True)
        for frase in frases:
            BingImageCrawler(storage={"root_dir": str(salida)}).crawl(
                keyword=frase,
                max_num=por_clase // len(frases),
                min_size=(200, 200),
                file_idx_offset="auto",
            )
        print(f"{clase}: {len(list(salida.glob('*')))} archivos en {salida}")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "openimages"
    if modo == "openimages":
        desde_openimages()
    elif modo == "buscar":
        desde_buscadores()
    else:
        sys.exit(f"Modo desconocido: {modo}. Usa 'openimages' o 'buscar'.")
