# Recolección de datos · SemaforoIA

Herramientas para ampliar el dataset del clasificador de vehículos
(carro / moto / bicicleta) antes de reentrenar.

## Por qué existe esto

El dataset actual está muy desbalanceado: **carro 711, moto 141, bicicleta 33**
imágenes. Con solo 33 bicicletas, al conjunto de prueba le tocaron 5 y el modelo
acierta 3 de esas 5 (recall 0.60). El accuracy del 98.5 % está inflado por la
clase mayoritaria; el número honesto es el F1 macro, 90.7 %.

No es un problema de arquitectura ni de hiperparámetros. Es falta de datos.

## Por qué Open Images y no scraping

En la primera ronda se recolectaron 2149 imágenes de buscadores y solo
sobrevivieron 885: se descartaron 1192 por duplicadas y 72 por pequeñas. Entre
las que pasaron se coló un logotipo etiquetado como bicicleta.

Open Images V7 trae las tres clases ya etiquetadas y verificadas por humanos,
con licencia CC BY. El scraping queda como complemento, no como vía principal.

## Por qué cajas y no etiquetas de clasificación

Las etiquetas de clasificación de Open Images marcan **presencia**, no
**dominancia**: dicen "aquí hay una bicicleta en alguna parte". Probándolo sobre
una muestra real de 12 imágenes etiquetadas como `Bicycle` aparecieron un
biplano, un skater con una bicicleta diminuta en una esquina y unos patinadores
con una bicicleta de fondo. Solo 7-8 de las 12 servían.

Para un clasificador que decide sobre la imagen completa eso es ruido de
etiqueta. Por eso `recolectar.py` usa las **cajas delimitadoras** y se queda
solo con las imágenes donde la clase objetivo ocupa al menos el 15 % del
encuadre y ningún otro vehículo ocupa más. Repitiendo la misma prueba con el
filtro activo, las 12 de 12 tenían la bicicleta dominante.

El umbral está en `AREA_MINIMA` dentro de `recolectar.py`. Súbelo si quieres
imágenes más limpias y menos cantidad, bájalo si te quedas corto de volumen.

## Uso

El pipeline completo, en el orden en que se ejecuta:

```bash
python recolectar.py openimages     # vía principal: etiquetas verificadas
python recolectar.py buscar         # complemento: scraping de Bing (ruidoso)
python limpiar.py recolectado       # deduplica, filtra pequeñas y corruptas
python preparar.py                  # reparte 70/15/15 y cachea a 256 px
python entrenar.py 6                # dos fases; el argumento son épocas de fase 2
python evaluar.py modelo/mobilenetv2_semaforo.pt artefactos
python comparar.py artefactos_viejo/metricas_test.json artefactos/metricas_test.json
```

`entrenar.py` espera un `modelo_base.pt` del que partir: el checkpoint que esté
en producción. Inicializar desde ahí en vez de desde ImageNet puro conserva lo
que el modelo ya aprendió del dataset original.

`evaluar.py` acepta ruta de modelo y carpeta de salida, así que se puede correr
sobre dos modelos distintos y comparar con `comparar.py` sobre el **mismo**
conjunto de prueba. Comparar métricas calculadas sobre conjuntos diferentes no
significa nada, y es justo el error que hacía parecer que la v1 tenía 98.5 %.

## Resultado de la primera ejecución (11-11-2026)

Con 1645 imágenes equilibradas, medido contra el modelo que estaba en producción
sobre el mismo conjunto de prueba de 247 imágenes:

| Métrica | v1 | v2 | Δ |
|---|---|---|---|
| Accuracy | 80.6 % | 92.7 % | +12.1 pp |
| F1 macro | 79.7 % | 92.5 % | +12.8 pp |
| Recall bicicleta | 60.0 % | 90.0 % | +30.0 pp |

### Dos hallazgos para la siguiente vuelta

**1. La fase de fine-tuning no aportó.** La fase 1 llegó a F1 0.961 en validación
y la fase 2 no superó 0.921 en ninguna de sus 6 épocas, con la pérdida de
entrenamiento bajando mientras la validación empeoraba. Partiendo de un modelo ya
ajustado a la tarea, la fase 2 sobra o necesita una tasa de aprendizaje mucho
menor.

**2. Mejor validación no significa mejor test.** El seguimiento del mejor estado
cubría solo la fase 2, así que se descartaba el óptimo de la fase 1. Al
corregirlo, el modelo seleccionado subió de 0.935 a 0.961 en validación... y
**bajó** de 92.5 % a 91.0 % de F1 macro en test.

Con 247 imágenes de validación, elegir la mejor época sobre ese conjunto se
sobreajusta a él. La diferencia en test son 4 imágenes de 247, dentro del ruido,
así que los dos modelos son estadísticamente indistinguibles. Se conservó en
producción el primero, por estar ya verificado.

La correccion del seguimiento sigue estando en `entrenar.py` porque es correcta:
no mirar la fase 1 era un fallo real. Pero la leccion es que la decision de que
modelo desplegar se toma sobre el conjunto de prueba, nunca sobre validacion.

Las imágenes quedan en `recolectado/<clase>/`. Lo que `limpiar.py` descarta no se
borra: se mueve a `descartadas/<motivo>/` por si hay que rescatar algo.

Después de limpiar, revisa las carpetas a ojo antes de entrenar. El filtro por
área reduce muchísimo el ruido, pero no lo elimina: siguen pasando primeros
planos de una rueda o vehículos atípicos, como un triciclo de mano.

## Qué está probado

| Componente | Estado |
|---|---|
| `limpiar.py`, autoprueba | verificado: `python limpiar.py --autoprueba` |
| `limpiar.py` sobre imágenes reales | verificado: detecta recomprimidas y reescaladas, sin falsos positivos |
| `recolectar.py openimages` | verificado contra Open Images V7 con fiftyone 1.22.1 |
| `recolectar.py buscar` | **sin probar** |

La prueba del modo `openimages` se hizo con el split `validation` y muestras
pequeñas. Con `split="train"` la descarga de las anotaciones es bastante más
pesada: hazla en Colab, no en un portátil.

Aviso: fiftyone guarda lo que descarga en `~/fiftyone/`, no en esta carpeta. Si
te quedas sin disco, mira ahí.

## Dependencias

- `recolectar.py openimages` → `pip install fiftyone` (pesado; pensado para Colab)
- `recolectar.py buscar` → `pip install icrawler`
- `limpiar.py` → solo Pillow

## Meta

Llegar a 500-700 imágenes por clase, o al menos 400. Prioridad: bicicleta
primero, moto después; carro ya tiene suficientes.

El número a vencer es F1 macro 90.7 %, y sobre todo el recall de bicicleta, hoy
en 0.60. Al crecer el dataset cambia la partición de prueba, así que las cifras
no serán comparables una a una con las actuales.

## Licencias

Las imágenes de Open Images son CC BY: cita la fuente en el trabajo. Lo que baje
el scraper es material con copyright de terceros; sirve para entrenar en un
proyecto de aula, pero no publiques el dataset.
