"""Copia de scripts/02_baseline.py que escribe en resultados_jb/ (mismas líneas base, sin GPU)."""
import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from common_jb import RESULTADOS, SEED, TAREAS, cargar_split

# Reutiliza regla_a e informe del original para no duplicarlos.
_spec = importlib.util.spec_from_file_location("baseline", Path(__file__).resolve().parent.parent / "scripts" / "02_baseline.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
regla_a, informe = _base.regla_a, _base.informe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TAREAS, required=True)
    args = ap.parse_args()
    clases = TAREAS[args.task]["clases"]
    train, test = cargar_split(args.task, "train"), cargar_split(args.task, "test")

    def entrada(d):
        return d["texto"] + " [SEP] " + d["par"]

    resultados = {}
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
    clf.fit(entrada(train), train["labels"])
    resultados["tfidf_logreg"] = informe(test["labels"], clf.predict(entrada(test)), clases)

    if args.task == "mdo":
        pred = [regla_a(t, o) for t, o in zip(test["Texto oración"], test["Objeto"])]
        resultados["regla_a"] = informe(test["labels"], pred, clases)

    salida = RESULTADOS / args.task / "baseline"
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "metricas.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    for nombre, r in resultados.items():
        print(f"\n== {args.task} / {nombre}")
        print(pd.DataFrame(r).T.round(3).to_string())


if __name__ == "__main__":
    main()
