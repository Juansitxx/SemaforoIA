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

```bash
python recolectar.py openimages     # vía principal: etiquetas verificadas
python recolectar.py buscar         # complemento: scraping de Bing (ruidoso)
python limpiar.py recolectado       # deduplica, filtra pequeñas y corruptas
```

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
