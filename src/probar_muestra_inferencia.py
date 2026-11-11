"""
Script de inferencia vehicular balanceada (60 imágenes) usando el detector preentrenado SSD MobileNet V2 COCO.
Detecta con alta precisión:
- Carros (clase 3) + Buses/Camiones (clases 6 y 8)
- Motocicletas (clase 4)
- Bicicletas (clase 2)
Genera:
1. outputs/reporte_inferencia_muestra.csv
2. outputs/mosaico_detecciones.png (Imágenes con cajas y etiquetas dibujadas)
3. outputs/resumen_metricas_inferencia.png (Distribución y gráficos)
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parent.parent
IMGS_DIR = ROOT / 'imagenes'
OUT_DIR = ROOT / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = ROOT / 'pretrained' / 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8' / 'saved_model'
print(f"✅ Cargando detector preentrenado desde: {MODEL_DIR}", flush=True)

detect_fn = tf.saved_model.load(str(MODEL_DIR))
print("Detector compilado y listo en memoria.", flush=True)

# Mapeo oficial COCO para vehículos
# 2: bicycle, 3: car, 4: motorcycle, 6: bus, 8: truck
COCO_VEHICLES = {
    3: {'name': 'Carro', 'color': '#e74c3c'},
    6: {'name': 'Carro (Bus)', 'color': '#c0392b'},
    8: {'name': 'Carro (Camión)', 'color': '#962d22'},
    4: {'name': 'Moto', 'color': '#f39c12'},
    2: {'name': 'Bicicleta', 'color': '#2ecc71'}
}
MIN_SCORE = 0.28  # Umbral estándar de detección

# Selección balanceada de 60 imágenes (20 carros, 20 motos, 20 bicis)
carros_imgs = (
    list((IMGS_DIR / 'Frente de Carros').glob('*.jpg')) +
    list((IMGS_DIR / 'Frente de Carros').glob('*.png')) +
    list((IMGS_DIR / 'Lateral de Carros').glob('*.jpg')) +
    list((IMGS_DIR / 'Parte trasera de Carros').glob('*.jpg'))
)[:20]

motos_imgs = (
    list((IMGS_DIR / 'Motos').glob('*.jpg')) +
    list((IMGS_DIR / 'Motos').glob('*.png'))
)[:20]

bicis_imgs = (
    list((IMGS_DIR / 'Bicicletas').glob('*.jpg')) +
    list((IMGS_DIR / 'Bicicletas').glob('*.png'))
)[:20]

selected_images = [
    (p, 'Carro') for p in carros_imgs
] + [
    (p, 'Moto') for p in motos_imgs
] + [
    (p, 'Bicicleta') for p in bicis_imgs
]

print(f"Total imágenes a evaluar: {len(selected_images)} (20 Carros, 20 Motos, 20 Bicicletas)\n", flush=True)

results_list = []
sample_visualizations = []

t_start = time.time()
for idx, (img_path, expected_cat) in enumerate(selected_images, 1):
    t_img = time.time()
    try:
        image = Image.open(img_path).convert('RGB')
        im_w, im_h = image.size
        
        # Inferencia con tensor de la imagen
        inp = tf.convert_to_tensor(np.expand_dims(np.array(image), 0), dtype=tf.uint8)
        detections = detect_fn(inp)
        
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(int)
        
        n_carros, n_motos, n_bicis = 0, 0, 0
        valid_scores = []
        valid_boxes_for_vis = []
        
        for box, score, cls_id in zip(boxes, scores, classes):
            if score >= MIN_SCORE and cls_id in COCO_VEHICLES:
                valid_scores.append(score)
                info = COCO_VEHICLES[cls_id]
                valid_boxes_for_vis.append((box, score, info['name'], info['color']))
                
                if cls_id in [3, 6, 8]:
                    n_carros += 1
                elif cls_id == 4:
                    n_motos += 1
                elif cls_id == 2:
                    n_bicis += 1
                    
        total_vehiculos = n_carros + n_motos + n_bicis
        mean_conf = float(np.mean(valid_scores)) if valid_scores else 0.0
        max_conf = float(np.max(valid_scores)) if valid_scores else 0.0
        
        results_list.append({
            'indice': idx,
            'archivo': img_path.name,
            'carpeta_origen': expected_cat,
            'carros_detectados': n_carros,
            'motos_detectadas': n_motos,
            'bicis_detectadas': n_bicis,
            'total_detectados': total_vehiculos,
            'confianza_media': round(mean_conf, 3),
            'confianza_maxima': round(max_conf, 3),
            'tiempo_seg': round(time.time() - t_img, 2)
        })
        
        # Guardar 2 muestras de cada categoría con detecciones para el mosaico
        if valid_boxes_for_vis:
            if (expected_cat == 'Carro' and len([s for s in sample_visualizations if s[0] == 'Carro']) < 2) or \
               (expected_cat == 'Moto' and len([s for s in sample_visualizations if s[0] == 'Moto']) < 2) or \
               (expected_cat == 'Bicicleta' and len([s for s in sample_visualizations if s[0] == 'Bicicleta']) < 2):
                sample_visualizations.append((expected_cat, img_path.name, np.array(image), valid_boxes_for_vis))
                
        print(f"[{idx:02d}/60] {expected_cat:9s} | {img_path.name:15s} -> Detectados: Carros={n_carros}, Motos={n_motos}, Bicis={n_bicis} ({time.time() - t_img:.2f}s)", flush=True)
        
    except Exception as e:
        print(f"[{idx:02d}/60] Error procesando {img_path.name}: {e}", flush=True)

df_results = pd.DataFrame(results_list)
csv_out = OUT_DIR / 'reporte_inferencia_muestra.csv'
df_results.to_csv(csv_out, index=False, encoding='utf-8')
print(f"\n💾 Reporte CSV guardado en: {csv_out}", flush=True)

# Generar Mosaico Visual de Detecciones
if sample_visualizations:
    num_samples = len(sample_visualizations)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < num_samples:
            cat, filename, img_arr, boxes_vis = sample_visualizations[i]
            ax.imshow(img_arr)
            h, w, _ = img_arr.shape
            for box, score, label_name, color in boxes_vis:
                ymin, xmin, ymax, xmax = box
                aymin, axmin, aymax, axmax = ymin*h, xmin*w, ymax*h, xmax*w
                bw, bh = axmax - axmin, aymax - aymin
                rect = patches.Rectangle((axmin, aymin), bw, bh, linewidth=2.5, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(axmin, max(0, aymin - 4), f"{label_name} {score*100:.0f}%",
                        color='white', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.85, edgecolor='none'))
            ax.set_title(f"Carpeta: {cat}\n{filename}", fontsize=11)
        ax.axis('off')
    plt.suptitle("Muestra Visual de Detecciones (SSD MobileNet V2)", fontsize=15, y=0.98)
    plt.tight_layout()
    mosaico_path = OUT_DIR / 'mosaico_detecciones.png'
    plt.savefig(mosaico_path, dpi=300)
    plt.close()
    print(f"🖼️ Mosaico visual guardado en: {mosaico_path}", flush=True)

# Generar Gráficas de Resumen
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

totales_por_categoria = [
    df_results['carros_detectados'].sum(),
    df_results['motos_detectadas'].sum(),
    df_results['bicis_detectadas'].sum()
]
cats = ['Carros', 'Motos', 'Bicicletas']
colores = ['#e74c3c', '#f39c12', '#2ecc71']

bars = ax1.bar(cats, totales_por_categoria, color=colores, edgecolor='black', alpha=0.85)
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, int(yval), ha='center', va='bottom', fontweight='bold', fontsize=11)
ax1.set_ylabel('Total Vehículos Detectados')
ax1.set_title('Conteo Total por Categoría (60 Imágenes)')
ax1.grid(axis='y', alpha=0.3)

ax2.plot(df_results.index, df_results['carros_detectados'], label='Carros', color='#e74c3c', marker='o', alpha=0.7)
ax2.plot(df_results.index, df_results['motos_detectadas'], label='Motos', color='#f39c12', marker='^', alpha=0.7)
ax2.plot(df_results.index, df_results['bicis_detectadas'], label='Bicis', color='#2ecc71', marker='s', alpha=0.7)
ax2.set_xlabel('Índice de Imagen (1-60)')
ax2.set_ylabel('Vehículos Detectados')
ax2.set_title('Conteo por Imagen Evaluada')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
resumen_chart_path = OUT_DIR / 'resumen_metricas_inferencia.png'
plt.savefig(resumen_chart_path, dpi=300)
plt.close()
print(f"📊 Gráfico de resumen guardado en: {resumen_chart_path}", flush=True)

t_total = time.time() - t_start
print("\n" + "="*55)
print(f"🏁 PRUEBA COMPLETADA EN {t_total:.1f} SEGUNDOS ({t_total/60:.1f} min)")
print("="*55)
print(f"Total imágenes analizadas:  {len(df_results)}")
print(f"🚗 Carros detectados:       {df_results['carros_detectados'].sum()}")
print(f"🏍️ Motos detectadas:        {df_results['motos_detectadas'].sum()}")
print(f"🚲 Bicicletas detectadas:   {df_results['bicis_detectadas'].sum()}")
print(f"🚦 Total vehículos:         {df_results['total_detectados'].sum()}")
val_conf = df_results[df_results['total_detectados']>0]['confianza_media']
avg_c = val_conf.mean()*100 if len(val_conf)>0 else 0
print(f"📈 Confianza promedio:      {avg_c:.1f}%")
print("="*55)
