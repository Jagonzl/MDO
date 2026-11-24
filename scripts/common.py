"""Rutas, modelos, tareas y funciones comunes a todos los scripts.

Hay dos versiones del dataset de entrenamiento; se elige con la variable de entorno MDO_VERSION:
  original    (por defecto) muestra anotada sin las oraciones repetidas
  segmentado  muestra con las oraciones de varios objetos segmentadas
Cada versión lee de dataset/processed/<versión>, escribe en resultados/<versión> y guarda los modelos en
<MDO_MODELOS>/<versión> (por defecto modelos/, que no se sube al repositorio).
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"  # muestra/, preseea/, eslora/
SEG = ROOT / "dataset" / "segmentacion"  # muestra/, preseea/, eslora/
PROMPTS = ROOT / "prompts"

VERSIONES = ("original", "segmentado")
VERSION = os.environ.get("MDO_VERSION", "original")
if VERSION not in VERSIONES:
    raise ValueError(f"MDO_VERSION={VERSION!r}: debe ser una de {VERSIONES}")
PROCESSED = ROOT / "dataset" / "processed" / VERSION
RESULTADOS = ROOT / "resultados" / VERSION
CLASIFICACION = ROOT / "resultados" / "clasificacion"
MODELOS_DIR = Path(os.environ.get("MDO_MODELOS", ROOT / "modelos")) / VERSION

SEED = 42
MAX_LENGTH = 512
VENTANA_PALABRAS = 150

MODELOS = {
    "mrbert_es": "BSC-LT/MrBERT-es",
    "mrbert": "BSC-LT/MrBERT",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "beto_cased": "dccuchile/bert-base-spanish-wwm-cased",
    "beto_uncased": "dccuchile/bert-base-spanish-wwm-uncased",
}

TAREAS = {
    "mdo": {"columna": "clase_mdo", "clases": ["sin_MDO", "con_MDO"]},
}

# Orden fijo de los 5 pares (tarea, modelo) para los arrays de SLURM:
# 0 = mrbert_es, 1 = mrbert, 2 = mbert, 3 = beto_cased, 4 = beto_uncased.
PARES = [(t, m) for t in TAREAS for m in MODELOS]


def normalizar(s) -> str:
    return " ".join(str(s).split())


def recortar(texto: str, objeto: str) -> str:
    """Ventana de VENTANA_PALABRAS palabras centrada en la primera aparición del objeto (textos de más de 512 tokens)."""
    palabras = texto.split()
    objeto_l = objeto.lower().split()[0] if objeto else ""
    idx = next((i for i, p in enumerate(palabras) if objeto_l and objeto_l in p.lower()), len(palabras) // 2)
    inicio = max(0, idx - VENTANA_PALABRAS // 2)
    return " ".join(palabras[inicio : inicio + VENTANA_PALABRAS])


def cargar_split(tarea: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / tarea / f"{split}.csv")
    clases = TAREAS[tarea]["clases"]
    df["labels"] = df[TAREAS[tarea]["columna"]].map({c: i for i, c in enumerate(clases)})
    return df


def tokenizar(tokenizer, df: pd.DataFrame):
    """Entrada de los modelos: par de textos (oración, "verbo: X ; objeto: Y"), cortando solo la oración."""
    return tokenizer(df["texto"].tolist(), df["par"].tolist(), truncation="only_first", max_length=MAX_LENGTH)
