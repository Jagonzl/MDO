"""Igual que scripts/common.py (mismos datos, modelos y tareas), pero con salidas *_jb para la versión GPU."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from common import MAX_LENGTH, MODELOS, PROCESSED, RAW, ROOT, SEED, TAREAS, cargar_split, tokenizar  # noqa: E402,F401

RESULTADOS = ROOT / "resultados_jb"
MODELOS_DIR = Path(os.environ.get("MODELOS_DIR_JB", ROOT / "modelos_jb"))

# Orden fijo de los 10 pares (tarea, modelo) para los arrays de SLURM.
PARES = [(t, m) for t in TAREAS for m in MODELOS]
