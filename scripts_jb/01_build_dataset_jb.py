"""Construye el dataset MDO a partir del xlsx con las oraciones segmentadas (paso 2 del plan).

Igual que scripts/01_build_dataset.py (reutiliza normalizar, recortar, contar_tokens y estadisticas), con tres cambios:
  1. lee dataset/segmentacion_jb/completo_jb_<tag>.xlsx: las oraciones con varios OD ya están divididas, una por fila,
     así que ya no se eliminan por "oración repetida";
  2. solo se eliminan duplicados exactos (misma oración, verbo, objeto y clase), los conflictos (mismo
     oración+verbo+objeto con clases distintas) y las filas que el revisor marcó como "no es CD"
     (dataset/segmentacion_jb/inventario_revision_jb.csv);
  3. partición 80/10/10 agrupada por oración original (ID) y estratificada por clase (StratifiedGroupKFold), para que
     los segmentos de una misma oración no queden en splits distintos.
Salida: dataset/processed_jb/mdo/{train,dev,test,eliminadas}.csv y resultados_jb_seg/mdo/estadisticas_dataset.csv
Uso: python scripts_jb/01_build_dataset_jb.py [--tag gemma_v3]
"""
import argparse
import importlib.util

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer

from common_jb import MAX_LENGTH, MODELOS, ROOT, SEED

SEG = ROOT / "dataset" / "segmentacion_jb"
PROCESSED_JB = ROOT / "dataset" / "processed_jb"
RESULTADOS_SEG = ROOT / "resultados_jb_seg"
COLUMNAS = ["Texto oración", "Verbo", "Objeto", "Animacidad", "Determinante", "Definitud", "Corpus"]

_spec = importlib.util.spec_from_file_location("build", ROOT / "scripts" / "01_build_dataset.py")
_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build)


def cargar(tag: str) -> pd.DataFrame:
    df = pd.read_excel(SEG / f"completo_jb_{tag}.xlsx")
    for c in COLUMNAS:
        df[c] = df[c].map(_build.normalizar)
    df["par"] = "verbo: " + df["Verbo"] + " ; objeto: " + df["Objeto"]
    return df


def depurar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eliminadas = []
    inv = pd.read_csv(SEG / "inventario_revision_jb.csv")
    no_cd = set(zip(inv.loc[inv["categoria"] == "no es CD", "archivo"], inv.loc[inv["categoria"] == "no es CD", "fila_excel"]))
    m = [(a, f) in no_cd for a, f in zip(df["archivo"], df["fila_excel"])]
    eliminadas.append(df[m].assign(motivo="no es CD (revisión)"))
    df = df[[not x for x in m]]

    clave = ["Texto oración", "Verbo", "Objeto"]
    conflicto = df.groupby(clave)["clase_mdo"].transform("nunique") > 1
    eliminadas.append(df[conflicto].assign(motivo="misma oración, verbo y objeto con clases distintas"))
    df = df[~conflicto]
    dup = df.duplicated(subset=clave + ["clase_mdo"], keep="first")
    eliminadas.append(df[dup].assign(motivo="duplicado exacto (oración, verbo, objeto y clase)"))
    return df[~dup].reset_index(drop=True), pd.concat(eliminadas, ignore_index=True)


def partir(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(df, df["clase_mdo"], groups=df["ID"]))
    test_idx, dev_idx = folds[0][1], folds[1][1]
    resto = df.index.difference(test_idx).difference(dev_idx)
    return {"train": df.loc[resto], "dev": df.loc[dev_idx], "test": df.loc[test_idx]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gemma_v3")
    args = ap.parse_args()
    df = cargar(args.tag)
    print(f"Filas leídas: {len(df)} (segmentadas: {int((df['segmentada'] == 'sí').sum())})")

    tokenizers = {n: AutoTokenizer.from_pretrained(m) for n, m in MODELOS.items()}
    df["texto"] = df["Texto oración"]
    largos = _build.contar_tokens(df, tokenizers)
    exceden = largos.max(axis=1) > MAX_LENGTH
    df["recortada"] = exceden
    df.loc[exceden, "texto"] = [_build.recortar(t, o) for t, o in zip(df.loc[exceden, "Texto oración"], df.loc[exceden, "Objeto"])]
    print(f"Recortadas (> {MAX_LENGTH} tokens en algún modelo): {int(exceden.sum())}")

    datos, eliminadas = depurar(df)
    splits = partir(datos)
    (PROCESSED_JB / "mdo").mkdir(parents=True, exist_ok=True)
    (RESULTADOS_SEG / "mdo").mkdir(parents=True, exist_ok=True)
    eliminadas.to_csv(PROCESSED_JB / "mdo" / "eliminadas.csv", index=False)
    for nombre, d in splits.items():
        d.to_csv(PROCESSED_JB / "mdo" / f"{nombre}.csv", index=False)
    stats = _build.estadisticas(splits)
    stats.to_csv(RESULTADOS_SEG / "mdo" / "estadisticas_dataset.csv", index=False)

    ids = [set(d["ID"]) for d in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "fuga de oraciones entre particiones"
    print(f"\n== mdo: {len(datos)} filas, {len(eliminadas)} eliminadas "
          f"({eliminadas['motivo'].value_counts().to_dict()}), " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    print("segmentadas por split:", {k: int((v["segmentada"] == "sí").sum()) for k, v in splits.items()})
    print(stats[stats["columna"] == "clase_mdo"].to_string(index=False))


if __name__ == "__main__":
    main()
