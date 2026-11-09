"""Líneas base: TF-IDF + regresión logística y, en MDO, la regla 'a/al delante del objeto'."""
import argparse
import json
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

from common import RESULTADOS, SEED, TAREAS, cargar_split


def regla_a(texto: str, objeto: str) -> int:
    """1 (con MDO) si alguna aparición del objeto va precedida de 'a'/'al' en las 4 palabras anteriores."""
    palabras = texto.lower().split()
    obj = objeto.lower().split()[0]
    for i, p in enumerate(palabras):
        if re.sub(r"\W", "", p) == re.sub(r"\W", "", obj):
            previas = [re.sub(r"\W", "", w) for w in palabras[max(0, i - 4) : i]]
            if "a" in previas or "al" in previas:
                return 1
    return 0


def informe(y_true, y_pred, clases):
    return classification_report(y_true, y_pred, labels=list(range(len(clases))), target_names=clases,
                                 output_dict=True, zero_division=0)


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
