"""Construye las particiones 80/10/10 (train/dev/test) de la muestra anotada.

--version original    a partir de los dos Excel de dataset/raw/muestra. Las oraciones
                      repetidas (oraciones largas con varios objetos) se eliminan, porque tienen el mismo texto.
--version segmentado  a partir de dataset/segmentacion/muestra/completo_<tag>.xlsx (salida de 00c), donde esas
                      oraciones ya están segmentadas, un trozo por objeto. Se eliminan solo los
                      duplicados exactos, los conflictos (misma oración, verbo y objeto con clases distintas) y las
                      filas que el revisor marcó como «no es CD» (inventario_revision.csv). La partición se agrupa
                      por oración original (ID), para que los trozos de una misma oración no queden en splits
                      distintos.
Salida: dataset/processed/<versión>/mdo/{train,dev,test,eliminadas}.csv y
        resultados/<versión>/mdo/estadisticas_dataset.csv
Uso: python scripts/01_build_dataset.py --version original
     python scripts/01_build_dataset.py --version segmentado [--tag gemma_v3]
"""
import argparse

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from transformers import AutoTokenizer

from common import MAX_LENGTH, MODELOS, RAW, ROOT, SEED, SEG, normalizar, recortar

COLUMNAS = ["Texto oración", "Verbo", "Objeto", "Animacidad", "Determinante", "Definitud", "Corpus"]


# --- versión original -------------------------------------------------------------------------------------------

def cargar_raw() -> pd.DataFrame:
    partes = []
    for patron, clase in [("*_MDO_*.xlsx", "con_MDO"), ("*sin_MDO*.xlsx", "sin_MDO")]:
        (archivo,) = (RAW / "muestra").glob(patron)
        df = pd.read_excel(archivo)
        df["clase_mdo"] = clase
        df["archivo"] = archivo.name
        df["fila_excel"] = df.index + 2
        partes.append(df)
    df = pd.concat(partes, ignore_index=True)
    for c in COLUMNAS:
        df[c] = df[c].map(normalizar)
    df["par"] = "verbo: " + df["Verbo"] + " ; objeto: " + df["Objeto"]
    return df


def dataset_mdo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    textos_por_clase = df.groupby("Texto oración")["clase_mdo"].nunique()
    en_ambos = set(textos_por_clase[textos_por_clase > 1].index)

    eliminadas = []
    m_ambos = df["Texto oración"].isin(en_ambos)
    eliminadas.append(df[m_ambos].assign(motivo="oración presente en ambos archivos"))
    resto = df[~m_ambos]
    m_rep = resto.duplicated(subset=["clase_mdo", "Texto oración"], keep="first")
    eliminadas.append(resto[m_rep].assign(motivo="oración repetida en el mismo archivo"))
    return resto[~m_rep].reset_index(drop=True), pd.concat(eliminadas, ignore_index=True)


def split_mdo(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train, temp = train_test_split(df, test_size=0.2, stratify=df["clase_mdo"], random_state=SEED)
    dev, test = train_test_split(temp, test_size=0.5, stratify=temp["clase_mdo"], random_state=SEED)
    return {"train": train, "dev": dev, "test": test}


# --- versión segmentada -----------------------------------------------------------------------------------------

def cargar_segmentado(tag: str) -> pd.DataFrame:
    df = pd.read_excel(SEG / "muestra" / f"completo_{tag}.xlsx")
    for c in COLUMNAS:
        df[c] = df[c].map(normalizar)
    df["par"] = "verbo: " + df["Verbo"] + " ; objeto: " + df["Objeto"]
    return df


def depurar_segmentado(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eliminadas = []
    inv = pd.read_csv(SEG / "muestra" / "inventario_revision.csv")
    no_cd = inv[inv["categoria"] == "no es CD"]
    no_cd = set(zip(no_cd["archivo"], no_cd["fila_excel"]))
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


def split_segmentado(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(df, df["clase_mdo"], groups=df["ID"]))
    test_idx, dev_idx = folds[0][1], folds[1][1]
    resto = df.index.difference(test_idx).difference(dev_idx)
    return {"train": df.loc[resto], "dev": df.loc[dev_idx], "test": df.loc[test_idx]}


# --- común ------------------------------------------------------------------------------------------------------

def contar_tokens(df: pd.DataFrame, tokenizers: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {nombre: [len(x) for x in tok(df["texto"].tolist(), df["par"].tolist())["input_ids"]] for nombre, tok in tokenizers.items()},
        index=df.index,
    )


def estadisticas(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    filas = []
    for split, d in splits.items():
        for col in ["clase_mdo", "Definitud", "Animacidad", "Corpus"]:
            for valor, n in d[col].value_counts().items():
                filas.append({"split": split, "columna": col, "clase": valor, "n": n})
    tabla = pd.DataFrame(filas).pivot_table(index=["columna", "clase"], columns="split", values="n", fill_value=0)
    tabla = tabla[["train", "dev", "test"]]
    tabla["total"] = tabla.sum(axis=1)
    return tabla.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["original", "segmentado"], required=True)
    ap.add_argument("--tag", default="gemma_v3", help="solo --version segmentado")
    args = ap.parse_args()
    processed, resultados = ROOT / "dataset" / "processed" / args.version, ROOT / "resultados" / args.version

    df = cargar_raw() if args.version == "original" else cargar_segmentado(args.tag)
    print(f"Filas leídas: {len(df)}")

    tokenizers = {n: AutoTokenizer.from_pretrained(m) for n, m in MODELOS.items()}
    df["texto"] = df["Texto oración"]
    largos = contar_tokens(df, tokenizers)
    exceden = largos.max(axis=1) > MAX_LENGTH
    df["recortada"] = exceden
    df.loc[exceden, "texto"] = [recortar(t, o) for t, o in zip(df.loc[exceden, "Texto oración"], df.loc[exceden, "Objeto"])]
    print(f"Oraciones que exceden {MAX_LENGTH} tokens en algún modelo (recortadas): {int(exceden.sum())}")

    if args.version == "original":
        resumen = []
        for n in tokenizers:
            s = largos[n]
            resumen.append({"modelo": n, "media": round(s.mean(), 1), "p50": s.quantile(0.5), "p95": s.quantile(0.95),
                            "p99": s.quantile(0.99), "max": s.max(), f"n_mayor_{MAX_LENGTH}": int((s > MAX_LENGTH).sum())})
        resumen = pd.DataFrame(resumen)
        resultados.mkdir(parents=True, exist_ok=True)
        resumen.to_csv(resultados / "longitudes_tokens.csv", index=False)
        print(resumen.to_string(index=False))
        tareas = [("mdo", dataset_mdo, split_mdo, "Texto oración")]
    else:
        tareas = [("mdo", depurar_segmentado, split_segmentado, "ID")]

    for tarea, construir, partir, grupo in tareas:
        datos, eliminadas = construir(df)
        splits = partir(datos)
        (processed / tarea).mkdir(parents=True, exist_ok=True)
        (resultados / tarea).mkdir(parents=True, exist_ok=True)
        eliminadas.to_csv(processed / tarea / "eliminadas.csv", index=False)
        for nombre, d in splits.items():
            d.to_csv(processed / tarea / f"{nombre}.csv", index=False)
        stats = estadisticas(splits)
        stats.to_csv(resultados / tarea / "estadisticas_dataset.csv", index=False)
        grupos = [set(d[grupo]) for d in splits.values()]
        assert not (grupos[0] & grupos[1] or grupos[0] & grupos[2] or grupos[1] & grupos[2]), "fuga entre particiones"
        print(f"\n== {tarea}: {len(datos)} filas, {len(eliminadas)} eliminadas "
              f"({eliminadas['motivo'].value_counts().to_dict()}), " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
        print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
