"""
Script de auto-etiquetado optimizado (Pascal VOC XML) para SemaforoIA.
Usa SSD MobileNet V2 con entrada fija (320, 320) para máxima velocidad de inferencia en CPU.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import tensorflow as tf
from PIL import Image
import sys

ROOT = Path(r"c:\Users\juanj\Desktop\SemaforoIA")
IMAGES_DIR = ROOT / "imagenes"
XML_DIR = ROOT / "annotations_xml"
XML_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = ROOT / "pretrained" / "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8" / "saved_model"
print(f"Cargando modelo detector desde {MODEL_DIR}...", flush=True)
detect_fn = tf.saved_model.load(str(MODEL_DIR))

# Warmup inicial de TensorFlow con forma fija (1, 320, 320, 3)
warmup_tensor = tf.zeros((1, 320, 320, 3), dtype=tf.uint8)
detect_fn(warmup_tensor)
print("Detector compilado y listo en memoria.", flush=True)

# Clases de vehículos en COCO: 3 (car), 4 (motorcycle), 6 (bus), 8 (truck)
VEHICLE_CLASSES = {3: "car", 4: "car", 6: "car", 8: "car"}
CONFIDENCE_THRESHOLD = 0.28

extensions = {".jpg", ".jpeg", ".png", ".bmp"}
image_files = sorted([p for p in IMAGES_DIR.rglob("*") if p.is_file() and p.suffix.lower() in extensions])
print(f"Total de imágenes encontradas: {len(image_files)}", flush=True)

def create_voc_xml(image_path, image_shape, boxes, scores, classes):
    height, width, depth = image_shape
    
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = image_path.parent.name
    ET.SubElement(annotation, "filename").text = image_path.name
    ET.SubElement(annotation, "path").text = str(image_path)
    
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "SemaforoIA_AutoLabeled"
    
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    
    ET.SubElement(annotation, "segmented").text = "0"
    
    valid_objects = 0
    for box, score, cls in zip(boxes, scores, classes):
        cls_int = int(cls)
        if score >= CONFIDENCE_THRESHOLD and cls_int in VEHICLE_CLASSES:
            ymin, xmin, ymax, xmax = box
            
            # Mapear coordenadas normalizadas (0 a 1) a dimensiones originales de la imagen
            abs_xmin = max(1, int(round(xmin * width)))
            abs_ymin = max(1, int(round(ymin * height)))
            abs_xmax = min(width, int(round(xmax * width)))
            abs_ymax = min(height, int(round(ymax * height)))
            
            if abs_xmax <= abs_xmin or abs_ymax <= abs_ymin:
                continue
                
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = "car"
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(abs_xmin)
            ET.SubElement(bndbox, "ymin").text = str(abs_ymin)
            ET.SubElement(bndbox, "xmax").text = str(abs_xmax)
            ET.SubElement(bndbox, "ymax").text = str(abs_ymax)
            
            valid_objects += 1
            
    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
    return xml_str, valid_objects

total_boxes = 0
labeled_images = 0

print("Iniciando auto-etiquetado acelerado...", flush=True)
for i, img_path in enumerate(image_files):
    xml_filename = f"{img_path.stem}.xml"
    xml_out_path = XML_DIR / xml_filename
    
    # Si ya existe, podemos contar y continuar para acelerar
    if xml_out_path.exists() and xml_out_path.stat().st_size > 100:
        labeled_images += 1
        continue

    try:
        with Image.open(img_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            orig_w, orig_h = pil_img.size
            
            # Redimensión fija a 320x320 para aprovechar al 100% la optimización del modelo
            infer_img = pil_img.resize((320, 320), Image.Resampling.BILINEAR)
            
            inp = tf.convert_to_tensor(np.expand_dims(np.array(infer_img), 0), dtype=tf.uint8)
            res = detect_fn(inp)
            
            boxes = res['detection_boxes'].numpy()[0]
            scores = res['detection_scores'].numpy()[0]
            classes = res['detection_classes'].numpy()[0]
            
            xml_content, num_objs = create_voc_xml(img_path, (orig_h, orig_w, 3), boxes, scores, classes)
            
            with open(xml_out_path, "w", encoding="utf-8") as f:
                f.write(xml_content)
                
            total_boxes += num_objs
            labeled_images += 1
            
        if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
            print(f"Progreso: {i + 1}/{len(image_files)} imágenes procesadas", flush=True)
            
    except Exception as e:
        print(f"Error procesando {img_path.name}: {e}", flush=True)

print("=====================================================", flush=True)
print(f"Auto-etiquetado completado exitosamente!", flush=True)
print(f"Total imágenes listas con XML: {labeled_images}", flush=True)
print(f"Archivos XML guardados en: {XML_DIR}", flush=True)
print("=====================================================", flush=True)
