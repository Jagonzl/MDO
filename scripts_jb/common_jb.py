"""Igual que scripts/common.py (mismos datos, modelos y tareas), pero con salidas *_jb para la versión GPU.

Variables de entorno opcionales (para la versión con oraciones segmentadas):
  MDO_JB_PROCESSED   carpeta de los splits (por defecto dataset/processed, la de main)
  MDO_JB_RESULTADOS  carpeta de resultados (por defecto resultados_jb)
  MODELOS_DIR_JB     carpeta de modelos (por defecto modelos_jb)
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import common  # noqa: E402
from common import MAX_LENGTH, MODELOS, RAW, ROOT, SEED, TAREAS, tokenizar  # noqa: E402,F401

PROCESSED = Path(os.environ.get("MDO_JB_PROCESSED", common.PROCESSED))
RESULTADOS = Path(os.environ.get("MDO_JB_RESULTADOS", ROOT / "resultados_jb"))
MODELOS_DIR = Path(os.environ.get("MODELOS_DIR_JB", ROOT / "modelos_jb"))

# Orden fijo de los 10 pares (tarea, modelo) para los arrays de SLURM.
PARES = [(t, m) for t in TAREAS for m in MODELOS]


def cargar_split(tarea: str, split: str) -> pd.DataFrame:
    """Como common.cargar_split, pero lee de PROCESSED (que puede ser dataset/processed_jb)."""
    df = pd.read_csv(PROCESSED / tarea / f"{split}.csv")
    clases = TAREAS[tarea]["clases"]
    df["labels"] = df[TAREAS[tarea]["columna"]].map({c: i for i, c in enumerate(clases)})
    return df
