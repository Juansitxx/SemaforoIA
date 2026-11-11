"""Compara dos evaluaciones sobre el MISMO conjunto de prueba y dicta el veredicto.

    python comparar.py artefactos_viejo/metricas_test.json artefactos/metricas_test.json
"""
import json
import sys

viejo = json.load(open(sys.argv[1], encoding="utf-8"))
nuevo = json.load(open(sys.argv[2], encoding="utf-8"))

assert viejo["n_test"] == nuevo["n_test"], "los conjuntos de prueba no coinciden"
print("conjunto de prueba comun: {} imagenes\n".format(nuevo["n_test"]))

fila = "{:<22} {:>10} {:>10} {:>10}"
print(fila.format("metrica", "produccion", "nuevo", "delta"))
print("-" * 54)


def linea(nombre, a, b, pct=True):
    d = b - a
    fmt = "{:.1%}" if pct else "{:+.0f}"
    print(fila.format(nombre, fmt.format(a) if pct else "{:.0f}".format(a),
                      fmt.format(b) if pct else "{:.0f}".format(b),
                      "{:+.1f} pp".format(d * 100) if pct else "{:+.0f}".format(d)))


linea("accuracy", viejo["accuracy"], nuevo["accuracy"])
linea("F1 macro", viejo["f1_macro"], nuevo["f1_macro"])
linea("errores", viejo["errores"], nuevo["errores"], pct=False)
print()
for clase in nuevo["clases"]:
    linea("recall " + clase, viejo["por_clase"][clase]["recall"], nuevo["por_clase"][clase]["recall"])
    linea("F1 " + clase, viejo["por_clase"][clase]["f1"], nuevo["por_clase"][clase]["f1"])

delta_f1 = nuevo["f1_macro"] - viejo["f1_macro"]
print("\nVEREDICTO: delta F1 macro = {:+.1f} pp -> {}".format(
    delta_f1 * 100, "EL NUEVO GANA" if delta_f1 > 0 else "EL NUEVO NO MEJORA"))
