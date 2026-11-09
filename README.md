# MDO

Clasificadores BERT en español sobre los corpus ESLORA y PRESEEA:

- **MDO**: objeto directo con marcado diferencial (`con_MDO`) frente a sin marcado (`sin_MDO`).
- **Definitud**: `Definido`, `Indefinido`, `Por determinar`.

Modelos comparados: `BSC-LT/MrBERT-es`, `BSC-LT/MrBERT`, `google-bert/bert-base-multilingual-cased`,
`dccuchile/bert-base-spanish-wwm-cased` y `dccuchile/bert-base-spanish-wwm-uncased`.
Los modelos entrenados se guardan en local (`modelos/`) y no se suben al repositorio.

## Estructura

- `dataset/raw/`: Excel originales.
- `dataset/processed/{mdo,definitud}/`: `train.csv`, `dev.csv`, `test.csv` (80/10/10) y `eliminadas.csv`.
- `scripts/`: construcción del dataset, líneas base, búsqueda de hiperparámetros, entrenamiento y evaluación.
- `resultados/{mdo,definitud}/`: estadísticas, hiperparámetros, tiempos y métricas en test.

## Uso

```
python scripts/01_build_dataset.py
python scripts/02_baseline.py --task mdo
python scripts/03_small_test.py --task mdo --model mrbert_es
```
