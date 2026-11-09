from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"
PROCESSED = ROOT / "dataset" / "processed"
MODELOS_DIR = ROOT / "modelos"
RESULTADOS = ROOT / "resultados"

SEED = 42
MAX_LENGTH = 512

MODELOS = {
    "mrbert_es": "BSC-LT/MrBERT-es",
    "mrbert": "BSC-LT/MrBERT",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "beto_cased": "dccuchile/bert-base-spanish-wwm-cased",
    "beto_uncased": "dccuchile/bert-base-spanish-wwm-uncased",
}

TAREAS = {
    "mdo": {"columna": "clase_mdo", "clases": ["sin_MDO", "con_MDO"]},
    "definitud": {"columna": "Definitud", "clases": ["Definido", "Indefinido", "Por determinar"]},
}


def cargar_split(tarea: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / tarea / f"{split}.csv")
    clases = TAREAS[tarea]["clases"]
    df["labels"] = df[TAREAS[tarea]["columna"]].map({c: i for i, c in enumerate(clases)})
    return df


def tokenizar(tokenizer, df: pd.DataFrame):
    return tokenizer(
        df["texto"].tolist(),
        df["par"].tolist(),
        truncation="only_first",
        max_length=MAX_LENGTH,
    )
