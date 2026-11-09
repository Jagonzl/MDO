"""Construye los datasets de las tareas MDO y Definitud con particiones 80/10/10."""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from transformers import AutoTokenizer

from common import MAX_LENGTH, MODELOS, PROCESSED, RAW, RESULTADOS, SEED

VENTANA_PALABRAS = 150


def normalizar(s) -> str:
    return " ".join(str(s).split())


def cargar_raw() -> pd.DataFrame:
    partes = []
    for patron, clase in [("*_MDO_*.xlsx", "con_MDO"), ("*sin_MDO*.xlsx", "sin_MDO")]:
        (archivo,) = RAW.glob(patron)
        df = pd.read_excel(archivo)
        df["clase_mdo"] = clase
        df["archivo"] = archivo.name
        df["fila_excel"] = df.index + 2
        partes.append(df)
    df = pd.concat(partes, ignore_index=True)
    for c in ["Texto oración", "Verbo", "Objeto", "Animacidad", "Determinante", "Definitud", "Corpus"]:
        df[c] = df[c].map(normalizar)
    df["par"] = "verbo: " + df["Verbo"] + " ; objeto: " + df["Objeto"]
    return df


def recortar(texto: str, objeto: str) -> str:
    """Ventana de palabras centrada en la primera aparición del objeto."""
    palabras = texto.split()
    objeto_l = objeto.lower().split()[0] if objeto else ""
    idx = next((i for i, p in enumerate(palabras) if objeto_l and objeto_l in p.lower()), len(palabras) // 2)
    inicio = max(0, idx - VENTANA_PALABRAS // 2)
    return " ".join(palabras[inicio : inicio + VENTANA_PALABRAS])


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


def dataset_definitud(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m_dup = df.duplicated(subset=["Texto oración", "Verbo", "Objeto"], keep="first")
    return df[~m_dup].reset_index(drop=True), df[m_dup].assign(motivo="duplicado exacto de oración, verbo y objeto")


def split_definitud(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(df, df["Definitud"], groups=df["Texto oración"]))
    test_idx, dev_idx = folds[0][1], folds[1][1]
    resto = df.index.difference(test_idx).difference(dev_idx)
    return {"train": df.loc[resto], "dev": df.loc[dev_idx], "test": df.loc[test_idx]}


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
    df = cargar_raw()
    print(f"Filas leídas: {len(df)}")

    tokenizers = {n: AutoTokenizer.from_pretrained(m) for n, m in MODELOS.items()}

    df["texto"] = df["Texto oración"]
    largos = contar_tokens(df, tokenizers)
    exceden = largos.max(axis=1) > MAX_LENGTH
    df["recortada"] = exceden
    df.loc[exceden, "texto"] = [recortar(t, o) for t, o in zip(df.loc[exceden, "Texto oración"], df.loc[exceden, "Objeto"])]
    print(f"Oraciones que exceden {MAX_LENGTH} tokens en algún modelo (recortadas): {int(exceden.sum())}")

    resumen = []
    for n in tokenizers:
        s = largos[n]
        resumen.append({"modelo": n, "media": round(s.mean(), 1), "p50": s.quantile(0.5), "p95": s.quantile(0.95),
                        "p99": s.quantile(0.99), "max": s.max(), f"n_mayor_{MAX_LENGTH}": int((s > MAX_LENGTH).sum())})
    resumen = pd.DataFrame(resumen)
    resumen.to_csv(RESULTADOS / "longitudes_tokens.csv", index=False)
    print(resumen.to_string(index=False))

    for tarea, construir, partir in [("mdo", dataset_mdo, split_mdo), ("definitud", dataset_definitud, split_definitud)]:
        datos, eliminadas = construir(df)
        splits = partir(datos)
        (PROCESSED / tarea).mkdir(parents=True, exist_ok=True)
        (RESULTADOS / tarea).mkdir(parents=True, exist_ok=True)
        eliminadas.to_csv(PROCESSED / tarea / "eliminadas.csv", index=False)
        for nombre, d in splits.items():
            d.to_csv(PROCESSED / tarea / f"{nombre}.csv", index=False)
        stats = estadisticas(splits)
        stats.to_csv(RESULTADOS / tarea / "estadisticas_dataset.csv", index=False)
        grupos = [set(d["Texto oración"]) for d in splits.values()]
        assert not (grupos[0] & grupos[1] or grupos[0] & grupos[2] or grupos[1] & grupos[2]), "fuga entre particiones"
        print(f"\n== {tarea}: {len(datos)} filas, {len(eliminadas)} eliminadas, "
              + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
        print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
