"""Clasifica MDO (con_MDO / sin_MDO) los objetos de unos xlsx (hoja 'objetos') con el mejor modelo final de MDO.

- Mejor modelo = mayor F1 macro medio en test (resultados_jb/mdo/final_*/metricas.json), versión best/.
- Entrada igual que en entrenamiento: (texto_oracion, "verbo: <verbo> ; objeto: <objeto>"); si supera
  MAX_LENGTH tokens se recorta con la misma ventana que 01_build_dataset.py.
- Filas únicas por (sent_id, verbo_id, objeto_id) de todos los xlsx de --entrada (no lee los .bak).
- Salida: <nombre>_mdo_jb.xlsx con 3 columnas: texto, mdo, seguridad_% (probabilidad softmax de la clase
  predicha, en %; no está calibrada). Aparte, <nombre>_mdo_jb_claves.csv con las claves para cruzar con el original.

Uso: python scripts_jb/07_clasificar_preseea_jb.py --entrada <dir xlsx> --salida <dir> --nombre PRESEEA
"""
import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common_jb import MAX_LENGTH, RESULTADOS, ROOT, TAREAS
from entrenamiento_jb import entorno, probabilidades

TAREA = "mdo"
CLAVE = ["sent_id", "verbo_id", "objeto_id"]

_spec = importlib.util.spec_from_file_location("build", ROOT / "scripts" / "01_build_dataset.py")
_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build)
recortar, normalizar = _build.recortar, _build.normalizar


def mejor_modelo() -> dict:
    finales = [json.loads(p.read_text(encoding="utf-8")) for p in (RESULTADOS / TAREA).glob("final_*/metricas.json")]
    m = max(finales, key=lambda m: m["test_f1_macro"]["media"])
    return {"modelo": m["modelo"], "ruta": m["ruta_modelo"], "seed": m["seed_guardada"],
            "test_f1_macro": m["test_f1_macro"]["media"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--nombre", required=True)
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)
    archivos = sorted(p for p in args.entrada.glob("*.xlsx") if not p.name.startswith("~$"))

    filas = pd.concat([pd.read_excel(p, sheet_name="objetos") for p in archivos], ignore_index=True)
    filas = filas.drop_duplicates(subset=CLAVE).reset_index(drop=True)
    texto = filas["texto_oracion"].map(normalizar)
    objeto = filas["objeto"].map(normalizar)
    df = pd.DataFrame({"texto": texto, "par": "verbo: " + filas["verbo"].map(normalizar) + " ; objeto: " + objeto})

    mm = mejor_modelo()
    tok = AutoTokenizer.from_pretrained(mm["ruta"])
    model = AutoModelForSequenceClassification.from_pretrained(mm["ruta"]).to("cuda" if torch.cuda.is_available() else "cpu")
    largo = pd.Series([len(x) for x in tok(df["texto"].tolist(), df["par"].tolist())["input_ids"]]) > MAX_LENGTH
    df.loc[largo, "texto"] = [recortar(t, o) for t, o in zip(df.loc[largo, "texto"], objeto[largo])]
    probs = probabilidades(model, tok, df)
    clases = TAREAS[TAREA]["clases"]

    salida = pd.DataFrame({"texto": texto, "mdo": [clases[i] for i in probs.argmax(-1)],
                           "seguridad_%": (probs.max(-1) * 100).round(1)})
    destino = args.salida / f"{args.nombre}_mdo_jb.xlsx"
    salida.to_excel(destino, index=False)
    pd.concat([filas[CLAVE + ["verbo", "objeto"]], salida[["mdo", "seguridad_%"]]], axis=1).to_csv(
        args.salida / f"{args.nombre}_mdo_jb_claves.csv", index=False)
    (args.salida / f"{args.nombre}_mdo_jb_modelo.json").write_text(json.dumps(
        {**mm, "recortadas": int(largo.sum()), "n": len(salida), "archivos": [p.name for p in archivos], "entorno": entorno()},
        indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(salida)} objetos de {len(archivos)} archivos | modelo {mm['modelo']} (seed {mm['seed']}, "
          f"test F1 {mm['test_f1_macro']:.3f}) | {int(largo.sum())} recortadas")
    print(salida["mdo"].value_counts().to_string())
    print("seguridad_% por clase (media / % filas < 90):")
    print(salida.groupby("mdo")["seguridad_%"].agg(media="mean", bajo_90=lambda s: round((s < 90).mean() * 100, 1)).to_string())
    print(f"-> {destino}")


if __name__ == "__main__":
    main()
