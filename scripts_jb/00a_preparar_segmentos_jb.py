"""Paso 1 de la segmentación: encuentra las oraciones con más de un objeto y fija la posición de cada verbo/objeto.

Una oración "repetida" (misma oración en varias filas, en el mismo archivo o en los dos) es una oración larga con varios
objetos directos. Para cada fila se busca la posición (índice de palabra, separando por espacios) de su verbo y de su
objeto:
  - si varias filas comparten el verbo, van en orden de aparición en la oración (orden de fila dentro de cada archivo);
  - entre las asignaciones posibles se elige la que deja cada objeto más cerca de su verbo (preferentemente detrás).

Salida: dataset/segmentacion_jb/tareas.jsonl (una línea por oración) y filas.csv (todas las filas, con su posición).
Uso: python scripts_jb/00a_preparar_segmentos_jb.py
"""
import importlib.util
import itertools
import json
import re

import pandas as pd

from common_jb import RAW, ROOT

SALIDA = ROOT / "dataset" / "segmentacion_jb"
DISTANCIA_MAX = 15  # palabras entre verbo y objeto a partir de las que se avisa para revisar

_spec = importlib.util.spec_from_file_location("build", ROOT / "scripts" / "01_build_dataset.py")
_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build)
normalizar = _build.normalizar


def norm(w: str) -> str:
    return re.sub(r"\W", "", str(w).lower())


def cargar_raw() -> pd.DataFrame:
    """Igual que cargar_raw() de 01_build_dataset.py, pero sin construir 'par' y conservando todas las columnas."""
    partes = []
    for patron, clase in [("*_MDO_*.xlsx", "con_MDO"), ("*sin_MDO*.xlsx", "sin_MDO")]:
        (archivo,) = RAW.glob(patron)
        df = pd.read_excel(archivo)
        df["clase_mdo"] = clase
        df["archivo"] = archivo.name
        df["fila_excel"] = df.index + 2
        partes.append(df)
    df = pd.concat(partes, ignore_index=True)
    df["texto_norm"] = df["Texto oración"].map(normalizar)
    return df


def posiciones(palabras: list[str], forma: str) -> list[int]:
    f = norm(forma)
    exactas = [i for i, p in enumerate(palabras) if norm(p) == f]
    return exactas or [i for i, p in enumerate(palabras) if f and norm(p).startswith(f)]


def coste(v: int, objs: list[int]) -> tuple[float, int]:
    """Distancia verbo→objeto (penaliza el objeto delante del verbo); devuelve (coste, posición del objeto)."""
    if not objs:
        return 1e6, -1
    return min(((o - v) if o >= v else 3 * (v - o), o) for o in objs)


def asignar(grupo: pd.DataFrame, palabras: list[str]) -> dict:
    """Devuelve {índice de fila: (pos_verbo, pos_objeto, aviso)}."""
    res = {}
    for _, filas in grupo.groupby(grupo["Verbo"].map(norm), sort=False):
        filas = filas.sort_values(["archivo", "fila_excel"])
        cand_v = posiciones(palabras, filas["Verbo"].iloc[0])
        objs = {i: posiciones(palabras, o) for i, o in zip(filas.index, filas["Objeto"])}
        if not cand_v:
            for i in filas.index:
                res[i] = (-1, -1, "verbo no encontrado")
            continue
        n = len(filas)
        repetir = len(cand_v) < n  # menos apariciones del verbo que filas: varias filas comparten verbo (p. ej. OD coordinados)
        combos = itertools.product(cand_v, repeat=n) if repetir else itertools.permutations(cand_v, n)
        mejor, mejor_coste, mejor_orden = None, None, False
        for combo in combos:
            c = sum(coste(v, objs[i])[0] for v, i in zip(combo, filas.index))
            # orden de aparición: dentro de cada archivo, filas posteriores -> verbo igual o posterior
            orden = all(a <= b for arch in filas["archivo"].unique()
                        for a, b in itertools.pairwise([v for v, ar in zip(combo, filas["archivo"]) if ar == arch]))
            clave = (not orden, c)
            if mejor is None or clave < (not mejor_orden, mejor_coste):
                mejor, mejor_coste, mejor_orden = combo, c, orden
        for v, i in zip(mejor, filas.index):
            _, o = coste(v, objs[i])
            aviso = "" if o >= 0 else "objeto no encontrado"
            if o >= 0 and abs(o - v) > DISTANCIA_MAX:
                aviso = f"verbo y objeto a {abs(o - v)} palabras (revisar anotación)"
            if not mejor_orden:
                aviso = (aviso + "; " if aviso else "") + "no respeta el orden de filas"
            res[i] = (v, o, aviso)
    return res


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = cargar_raw()
    n_por_texto = df.groupby("texto_norm")["texto_norm"].transform("size")
    df["multi"] = n_por_texto > 1
    df["pos_verbo"], df["pos_objeto"], df["aviso"] = -1, -1, ""

    tareas = []
    for k, (texto, grupo) in enumerate(df[df["multi"]].groupby("texto_norm", sort=False)):
        palabras = texto.split()
        asign = asignar(grupo, palabras)
        objetivos = []
        for t, i in enumerate(grupo.index, start=1):
            v, o, aviso = asign[i]
            df.loc[i, ["pos_verbo", "pos_objeto", "aviso"]] = [v, o, aviso]
            df.loc[i, "oracion_id"], df.loc[i, "t"] = k, t
            objetivos.append({"t": t, "fila": int(i), "verbo": str(grupo.at[i, "Verbo"]), "pos_verbo": v,
                              "objeto": str(grupo.at[i, "Objeto"]), "pos_objeto": o, "aviso": aviso})
        tareas.append({"oracion_id": k, "palabras": palabras, "objetivos": objetivos})

    with open(SALIDA / "tareas.jsonl", "w", encoding="utf-8") as fh:
        for t in tareas:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    df.to_csv(SALIDA / "filas.csv", index=True, index_label="fila")

    m = df[df["multi"]]
    print(f"filas: {len(df)} | filas en oraciones con varios objetos: {len(m)} | oraciones a dividir: {len(tareas)}")
    print("por clase:", m["clase_mdo"].value_counts().to_dict())
    print("objetos por oración:", m.groupby("texto_norm").size().value_counts().sort_index().to_dict())
    print("avisos:", m["aviso"].replace("", "ninguno").value_counts().to_dict())
    print("pares verbo+objeto idénticos en la misma oración:",
          int(m.duplicated(subset=["texto_norm", "Verbo", "Objeto"], keep=False).sum()))


if __name__ == "__main__":
    main()
