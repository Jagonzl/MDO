"""Búsqueda de learning rate, batch y épocas con un 40 % de train, evaluando en dev.

Uso:
  python scripts/03_small_test.py --task mdo --model mrbert_es --solo-tiempo   # mide velocidad y estima tiempos
  python scripts/03_small_test.py --task mdo --model mrbert_es                 # búsqueda completa
"""
import argparse
import itertools
import json
import tempfile

from sklearn.model_selection import train_test_split

from common import MODELOS, RESULTADOS, SEED, TAREAS, cargar_split
from entrenamiento import entrenar

FRACCION = 0.4
EPOCHS = 5
BATCHES = [8, 16]
LRS = {"mbert": [2e-5, 3e-5, 5e-5], "beto_cased": [2e-5, 3e-5, 5e-5], "beto_uncased": [2e-5, 3e-5, 5e-5],
       "mrbert_es": [3e-5, 5e-5, 8e-5], "mrbert": [3e-5, 5e-5, 8e-5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TAREAS, required=True)
    ap.add_argument("--model", choices=MODELOS, required=True)
    ap.add_argument("--solo-tiempo", action="store_true")
    args = ap.parse_args()

    train_full, dev = cargar_split(args.task, "train"), cargar_split(args.task, "dev")
    train, _ = train_test_split(train_full, train_size=FRACCION, stratify=train_full["labels"], random_state=SEED)
    salida = RESULTADOS / args.task
    combos = list(itertools.product(LRS[args.model], BATCHES))

    if args.solo_tiempo:
        with tempfile.TemporaryDirectory() as tmp:
            _, hist, spe = entrenar(args.task, args.model, train, dev, LRS[args.model][1], 16, 1, tmp, guardar=False)
        estimacion = {
            "modelo": args.model, "tarea": args.task, "seg_por_ejemplo": round(spe, 4),
            "min_busqueda_completa": round(spe * len(train) * EPOCHS * len(combos) / 60, 1),
            "min_entrenamiento_final_5_epochs": round(spe * len(train_full) * EPOCHS / 60, 1),
            "dev_tras_1_epoch": hist,
        }
        (salida / f"tiempo_{args.model}.json").write_text(json.dumps(estimacion, indent=2), encoding="utf-8")
        print(json.dumps(estimacion, indent=2))
        return

    resultados = []
    for lr, batch in combos:
        with tempfile.TemporaryDirectory() as tmp:
            _, hist, spe = entrenar(args.task, args.model, train, dev, lr, batch, EPOCHS, tmp, guardar=False)
        for h in hist:
            resultados.append({"lr": lr, "batch": batch, **h})
        print(f"lr={lr} batch={batch} -> " + ", ".join(f"e{h['epoch']}:{h['dev_f1_macro']:.3f}" for h in hist))

    mejor = max(resultados, key=lambda r: (round(r["dev_f1_macro"], 4), -r["epoch"]))
    hparams = {"modelo": args.model, "tarea": args.task, "lr": mejor["lr"], "batch": mejor["batch"],
               "epochs": mejor["epoch"], "dev_f1_macro_small": mejor["dev_f1_macro"], "busqueda": resultados}
    (salida / f"hparams_{args.model}.json").write_text(json.dumps(hparams, indent=2), encoding="utf-8")
    print(f"\nMejor: lr={mejor['lr']} batch={mejor['batch']} epochs={mejor['epoch']} f1_macro_dev={mejor['dev_f1_macro']:.3f}")


if __name__ == "__main__":
    main()
