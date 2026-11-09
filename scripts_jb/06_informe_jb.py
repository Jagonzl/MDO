"""Resume resultados_jb/ en una tabla: baseline, hparams elegidos (03) y F1/accuracy en test (04).

Uso: python scripts_jb/06_informe_jb.py   -> resultados_jb/resumen_jb.csv y resumen_jb.md
"""
import json

import pandas as pd

from common_jb import MODELOS, RESULTADOS, TAREAS


def leer(ruta):
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def main():
    filas = []
    for tarea in TAREAS:
        base = leer(RESULTADOS / tarea / "baseline" / "metricas.json") or {}
        for nombre, r in base.items():
            filas.append({"tarea": tarea, "modelo": f"[baseline] {nombre}",
                          "test_f1_macro": r["macro avg"]["f1-score"], "test_accuracy": r["accuracy"]})
        for modelo in MODELOS:
            hp = leer(RESULTADOS / tarea / f"hparams_{modelo}.json")
            fin = leer(RESULTADOS / tarea / f"final_{modelo}" / "metricas.json")
            fila = {"tarea": tarea, "modelo": modelo}
            if hp:
                fila |= {"lr": hp["lr"], "batch": hp["batch"], "epochs": hp["epochs"],
                         "dev_f1_busqueda": hp["dev_f1_macro_small"], "min_busqueda": hp.get("min_busqueda"),
                         "gpu": hp.get("entorno", {}).get("gpu")}
            if fin:
                fila |= {"dev_f1_final": fin["dev_f1_macro"]["media"],
                         "test_f1_macro": fin["test_f1_macro"]["media"], "test_f1_desv": fin["test_f1_macro"]["desv"],
                         "test_accuracy": fin["test_accuracy"]["media"], "test_acc_desv": fin["test_accuracy"]["desv"],
                         "seed_guardada": fin["seed_guardada"]}
            filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(RESULTADOS / "resumen_jb.csv", index=False)
    vista = df.round(4)
    if "lr" in vista:
        vista["lr"] = df["lr"].map(lambda v: f"{v:.0e}" if pd.notna(v) else "")
    (RESULTADOS / "resumen_jb.md").write_text(vista.to_markdown(index=False), encoding="utf-8")
    print(vista.to_string(index=False))


if __name__ == "__main__":
    main()
