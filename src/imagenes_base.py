"""Lectura común de imágenes. No modifica el archivo original."""
from pathlib import Path

from PIL import Image, ImageOps


def cargar_rgb(ruta: str | Path) -> Image.Image:
    """Aplica orientación EXIF, resuelve transparencia y devuelve una copia RGB."""
    with Image.open(ruta) as original:
        if getattr(original, "n_frames", 1) != 1:
            raise ValueError("Imagen con varios fotogramas: requiere revisión manual")
        imagen = ImageOps.exif_transpose(original)
        if "A" in imagen.getbands() or "transparency" in imagen.info:
            rgba = imagen.convert("RGBA")
            fondo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(fondo, rgba).convert("RGB")
        return imagen.convert("RGB").copy()


class AjustarCuadrado:
    """Conserva la imagen completa y su proporción; añade márgenes grises."""

    def __init__(self, lado: int = 224):
        if lado < 1:
            raise ValueError("El lado debe ser positivo")
        self.lado = lado

    def __call__(self, imagen: Image.Image) -> Image.Image:
        imagen = imagen.convert("RGB")
        escala = self.lado / max(imagen.size)
        ancho = max(1, min(self.lado, round(imagen.width * escala)))
        alto = max(1, min(self.lado, round(imagen.height * escala)))
        ajustada = imagen.resize((ancho, alto), Image.Resampling.BILINEAR)
        salida = Image.new("RGB", (self.lado, self.lado), (127, 127, 127))
        salida.paste(ajustada, ((self.lado - ancho) // 2, (self.lado - alto) // 2))
        return salida
