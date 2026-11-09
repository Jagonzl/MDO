"""Entrenamiento final y evaluación en test (versión GPU, nuevo en *_jb).

Para cada semilla: entrena en el train completo con los hiperparámetros elegidos por 03_small_test_jb.py
(lr, batch, epochs), se queda con la mejor época según F1 macro en dev y evalúa en test.
Solo se conserva en disco el modelo de la semilla con mejor F1 en dev (el que se sube a HF).

Uso:
  python scripts_jb/04_final_jb.py --task mdo --model mrbert_es [--seeds 42 43 44]
  python scripts_jb/04_final_jb.py --par 3
"""
import argparse
import json
import shutil
import statistics

import pandas as pd
from sklearn.metrics import classification_report

from common_jb import MODELOS, MODELOS_DIR, PARES, RESULTADOS, TAREAS, cargar_split
from entrenamiento_jb import a_dataset, entorno, entrenar


def informe(y_true, y_pred, clases):
    return classification_report(y_true, y_pred, labels=list(range(len(clases))), target_names=clases,
                                 output_dict=True, zero_division=0)


def resumen(valores):
    return {"media": statistics.mean(valores), "desv": statistics.stdev(valores) if len(valores) > 1 else 0.0,
            "valores": valores}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TAREAS)
    ap.add_argument("--model", choices=MODELOS)
    ap.add_argument("--par", type=int, help="índice en PARES (sustituye a --task/--model)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()
    if args.par is not None:
        args.task, args.model = PARES[args.par]
    if not (args.task and args.model):
        ap.error("hace falta --task y --model, o --par")
    print(f"== {args.task} / {args.model} | {entorno()}", flush=True)

    clases = TAREAS[args.task]["clases"]
    hp = json.loads((RESULTADOS / args.task / f"hparams_{args.model}.json").read_text(encoding="utf-8"))
    train, dev, test = (cargar_split(args.task, s) for s in ("train", "dev", "test"))
    salida = RESULTADOS / args.task / f"final_{args.model}"
    salida.mkdir(parents=True, exist_ok=True)
    base_modelos = MODELOS_DIR / args.task / args.model

    por_semilla = {}
    for seed in args.seeds:
        dir_seed = base_modelos / f"seed{seed}"
        trainer, hist, spe = entrenar(args.task, args.model, train, dev, hp["lr"], hp["batch"], hp["epochs"],
                                      dir_seed, guardar=True, seed=seed)
        pred = trainer.predict(a_dataset(trainer.processing_class, test))
        y_pred = pred.predictions.argmax(-1)
        rep = informe(test["labels"], y_pred, clases)
        mejor_dev = max(hist, key=lambda h: h["dev_f1_macro"])
        por_semilla[seed] = {"dev_mejor_epoch": mejor_dev["epoch"], "dev_f1_macro": mejor_dev["dev_f1_macro"],
                             "test_accuracy": rep["accuracy"], "test_f1_macro": rep["macro avg"]["f1-score"],
                             "seg_por_ejemplo": round(spe, 4), "historial_dev": hist, "test": rep}
        pd.DataFrame({
            "Texto oración": test["Texto oración"], "Verbo": test["Verbo"], "Objeto": test["Objeto"],
            "real": [clases[i] for i in test["labels"]], "pred": [clases[i] for i in y_pred],
        }).to_csv(salida / f"predicciones_test_seed{seed}.csv", index=False)

        trainer.save_model(str(dir_seed / "final"))
        for ckpt in dir_seed.glob("checkpoint-*"):
            shutil.rmtree(ckpt)
        print(f"seed={seed}: dev_f1={mejor_dev['dev_f1_macro']:.3f} (e{mejor_dev['epoch']}) "
              f"test_f1={rep['macro avg']['f1-score']:.3f} test_acc={rep['accuracy']:.3f}", flush=True)

    mejor_seed = max(por_semilla, key=lambda s: por_semilla[s]["dev_f1_macro"])
    for seed in args.seeds:
        if seed != mejor_seed:
            shutil.rmtree(base_modelos / f"seed{seed}", ignore_errors=True)

    metricas = {
        "modelo": args.model, "tarea": args.task, "hparams": {k: hp[k] for k in ("lr", "batch", "epochs")},
        "seeds": args.seeds, "seed_guardada": mejor_seed,
        "ruta_modelo": str(base_modelos / f"seed{mejor_seed}" / "final"),
        "test_f1_macro": resumen([por_semilla[s]["test_f1_macro"] for s in args.seeds]),
        "test_accuracy": resumen([por_semilla[s]["test_accuracy"] for s in args.seeds]),
        "dev_f1_macro": resumen([por_semilla[s]["dev_f1_macro"] for s in args.seeds]),
        "entorno": entorno(), "por_semilla": por_semilla,
    }
    (salida / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    f1 = metricas["test_f1_macro"]
    print(f"\n{args.task}/{args.model}: test F1 macro = {f1['media']:.3f} ± {f1['desv']:.3f} (seed guardada {mejor_seed})")


if __name__ == "__main__":
    main()
