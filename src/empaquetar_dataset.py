"""
Empaqueta las imágenes y las anotaciones XML generadas en dataset_semaforo.zip
listo para subir a Google Colab.
"""
from pathlib import Path
import zipfile
import sys

ROOT = Path(r"c:\Users\juanj\Desktop\SemaforoIA")
IMAGES_DIR = ROOT / "imagenes"
XML_DIR = ROOT / "annotations_xml"
ZIP_OUT = ROOT / "dataset_semaforo.zip"

print(f"Empaquetando dataset en {ZIP_OUT}...", flush=True)

xml_files = list(XML_DIR.glob("*.xml"))
print(f"Anotaciones XML encontradas: {len(xml_files)}", flush=True)

if not xml_files:
    print("Aún no hay archivos XML generados. Espera a que termine el auto-etiquetado.", flush=True)
    sys.exit(1)

with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # 1. Agregar XMLs
    for xml_file in xml_files:
        zipf.write(xml_file, arcname=f"annotations_xml/{xml_file.name}")
        
    # 2. Agregar imágenes que corresponden a los XMLs
    added_imgs = 0
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for img_file in IMAGES_DIR.rglob("*"):
        if img_file.is_file() and img_file.suffix.lower() in extensions:
            # Mantener ruta relativa dentro de imagenes/
            rel_path = img_file.relative_to(ROOT)
            zipf.write(img_file, arcname=str(rel_path).replace("\\", "/"))
            added_imgs += 1

print(f"[OK] Archivo creado exitosamente: {ZIP_OUT}")
print(f"   - XMLs incluidos: {len(xml_files)}")
print(f"   - Imágenes incluidas: {added_imgs}")
print(f"   - Tamaño del zip: {round(ZIP_OUT.stat().st_size / (1024*1024), 2)} MB")
