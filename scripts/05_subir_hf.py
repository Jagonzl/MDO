"""Sube a Hugging Face (repo privado) el modelo final guardado por 04_final.py, con su ficha.

Se ejecuta donde haya internet (en el clúster, el login). Token con permiso de escritura en la variable HF_TOKEN.
Uso: HF_TOKEN=... MDO_VERSION=segmentado python scripts/05_subir_hf.py --model beto_cased
     -> johnatanebonilla/mdo-beto-cased (el modelo final del proyecto)
"""
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

from common import MODELOS, RESULTADOS, cargar_split

TARJETA = """---
language: es
library_name: transformers
pipeline_tag: text-classification
base_model: {base}
---

# {nombre}

Clasificador del **marcado diferencial de objeto (MDO)** en español oral: decide si un objeto directo lleva marca
(`con_MDO`, p. ej. «conocí **a** mi mujer») o no (`sin_MDO`).

- **Modelo de partida**: [{base}](https://huggingface.co/{base}).
- **Datos**: muestra anotada a mano de ESLORA y PRESEEA ({n} objetos).
- **Proyecto**: código, datos y resultados en https://github.com/Jagonzl/MDO (rama `jb`).

## Entrada

Un par de textos: el trozo de oración y `"verbo: <verbo> ; objeto: <objeto>"`, con 512 tokens como máximo.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "{repo}"  # privado: hace falta un token con acceso
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo)
enc = tok("y conocí a mi mujer en la universidad", "verbo: conocí ; objeto: mujer", return_tensors="pt")
probs = model(**enc).logits.softmax(-1)[0]
print(model.config.id2label[int(probs.argmax())], float(probs.max()))
```

## Resultados

| | valor |
|---|---|
| Hiperparámetros (lr / batch / épocas) | {lr} / {batch} / {epochs} |
| F1 macro en test (media ± desv., {semillas} semillas) | {f1:.3f} ± {f1d:.3f} |
| Accuracy en test | {acc:.3f} ± {accd:.3f} |

Este es el modelo de la semilla {seed} (la mejor en dev), en su mejor época según dev.

## Limitaciones

- En la muestra, el MDO casi coincide con la animacidad: con_MDO es casi siempre humano y sin_MDO casi siempre
  inanimado. Hay pocos ejemplos de con_MDO inanimado, de sin_MDO humano y de objetos animados.
- Casi no hay clíticos (`lo`, `la`, `te`, `me`…) entre los objetos de entrenamiento; sobre ellos el modelo es menos
  fiable.
- La probabilidad que devuelve no está calibrada.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELOS, default="beto_cased")
    ap.add_argument("--usuario", default="johnatanebonilla")
    ap.add_argument("--repo", help="nombre completo del repo; por defecto <usuario>/mdo-<modelo>")
    args = ap.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])

    m = json.loads((RESULTADOS / "mdo" / f"final_{args.model}" / "metricas.json").read_text(encoding="utf-8"))
    ruta = Path(m["ruta_modelo"])
    repo = args.repo or f"{args.usuario}/mdo-{args.model.replace('_', '-')}"
    n = sum(len(cargar_split("mdo", s)) for s in ("train", "dev", "test"))
    (ruta / "README.md").write_text(TARJETA.format(
        nombre=repo.split("/")[-1], repo=repo, base=MODELOS[args.model], n=f"{n:,}".replace(",", " "),
        seed=m["seed_guardada"], semillas=len(m["seeds"]), **m["hparams"],
        f1=m["test_f1_macro"]["media"], f1d=m["test_f1_macro"]["desv"],
        acc=m["test_accuracy"]["media"], accd=m["test_accuracy"]["desv"]), encoding="utf-8")
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=str(ruta), commit_message="Modelo final",
                      ignore_patterns=["training_args.bin"])
    print(f"{args.model} -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
