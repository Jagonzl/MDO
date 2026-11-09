"""Sube a Hugging Face (repos privados) los modelos finales guardados por 04_final_jb.py.

Se ejecuta en el nodo login (tiene internet). Token en la variable HF_TOKEN.
Uso: HF_TOKEN=$(cat .hf_token_write) python scripts_jb/05_subir_hf_jb.py [--usuario johnatanebonilla] [--par 0 1 ...]
"""
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

from common_jb import MODELOS, PARES, RESULTADOS, TAREAS

TARJETA = """---
language: es
library_name: transformers
pipeline_tag: text-classification
base_model: {base}
---

# {repo}

Clasificador `{tarea}` ({clases}) afinado desde [{base}](https://huggingface.co/{base})
sobre los corpus ESLORA y PRESEEA. Proyecto: https://github.com/Jagonzl/MDO (rama `jb`, versión GPU).

Entrada: par de textos `(oración, "verbo: <verbo> ; objeto: <objeto>")`, máximo 512 tokens.

| | valor |
|---|---|
| lr / batch / epochs | {lr} / {batch} / {epochs} |
| semilla guardada | {seed} (mejor F1 macro en dev) |
| F1 macro test (media ± desv., {n} semillas) | {f1:.3f} ± {f1d:.3f} |
| accuracy test (media ± desv.) | {acc:.3f} ± {accd:.3f} |
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usuario", default="johnatanebonilla")
    ap.add_argument("--par", type=int, nargs="*", help="índices en PARES; por defecto todos")
    args = ap.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])

    for i in args.par if args.par else range(len(PARES)):
        tarea, modelo = PARES[i]
        archivo = RESULTADOS / tarea / f"final_{modelo}" / "metricas.json"
        if not archivo.exists():
            print(f"[{i}] {tarea}/{modelo}: sin metricas.json, se omite")
            continue
        m = json.loads(archivo.read_text(encoding="utf-8"))
        ruta = Path(m["ruta_modelo"])
        repo = f"{args.usuario}/mdo-jb-{tarea}-{modelo.replace('_', '-')}"
        (ruta / "README.md").write_text(TARJETA.format(
            repo=repo, tarea=tarea, clases=", ".join(TAREAS[tarea]["clases"]), base=MODELOS[modelo],
            seed=m["seed_guardada"], n=len(m["seeds"]), **m["hparams"],
            f1=m["test_f1_macro"]["media"], f1d=m["test_f1_macro"]["desv"],
            acc=m["test_accuracy"]["media"], accd=m["test_accuracy"]["desv"]), encoding="utf-8")
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=str(ruta), commit_message="Modelo final (jb, GPU)",
                          ignore_patterns=["training_args.bin"])
        print(f"[{i}] {tarea}/{modelo} -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
