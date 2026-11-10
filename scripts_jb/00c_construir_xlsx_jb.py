"""Paso 3 de la segmentación: valida los segmentos de Gemma y construye los xlsx nuevos.

Validación de cada segmento (inicio, fin): enteros, dentro de la oración y cubriendo todo el tramo verbo→objeto
(si hay una «a» entre ambos queda dentro; si no la hay, no se exige nada). Si no pasa, regla de respaldo: el segmento va
desde justo después del objetivo anterior hasta justo antes del siguiente (en orden de aparición).

Salidas en dataset/segmentacion_jb/:
  completo_jb.xlsx  todas las filas de los dos xlsx originales (mismas celdas); en las oraciones con varios objetos,
                    'Texto oración' es el segmento. Columnas añadidas: clase_mdo, archivo, fila_excel, segmentada, metodo.
  divididas_jb.xlsx solo las filas divididas, con 'Texto completo (referencia)' y la posición de verbo/objeto/segmento.
Uso: python scripts_jb/00c_construir_xlsx_jb.py --tag gemma_v1
"""
import argparse
import json
import re

import pandas as pd

from common_jb import ROOT

SEG = ROOT / "dataset" / "segmentacion_jb"
ORIGINALES = ["Corpus", "ID", "Texto oración", "Verbo", "Objeto", "Animacidad", "Determinante", "Definitud"]


def parsear(respuesta: str) -> dict:
    """{t: (inicio, fin)} a partir del último JSON con 'segmentos' de la respuesta."""
    for m in reversed(list(re.finditer(r"\{.*\}", respuesta, flags=re.S))):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and isinstance(d.get("segmentos"), list):
            return {int(s["t"]): (s["inicio"], s["fin"]) for s in d["segmentos"]
                    if isinstance(s, dict) and {"t", "inicio", "fin"} <= set(s)}
    return {}


def valido(seg, v: int, o: int, n: int) -> bool:
    if not (isinstance(seg, tuple) and all(isinstance(x, int) for x in seg)):
        return False
    i, f = seg
    tramo = [x for x in (v, o) if x >= 0]
    return 0 <= i <= f < n and bool(tramo) and i <= min(tramo) and max(tramo) <= f


def regla(objetivos: list[dict], n: int) -> dict:
    """Respaldo: corta entre objetivos consecutivos (los que comparten tramo se agrupan)."""
    tramos = {o["t"]: (min(x for x in (o["pos_verbo"], o["pos_objeto"]) if x >= 0),
                       max(x for x in (o["pos_verbo"], o["pos_objeto"]) if x >= 0))
              for o in objetivos if max(o["pos_verbo"], o["pos_objeto"]) >= 0}
    orden = sorted(tramos, key=lambda t: tramos[t])
    res = {}
    for k, t in enumerate(orden):
        a, b = tramos[t]
        prev_fin = max([tramos[u][1] for u in orden[:k] if tramos[u][1] < a], default=-1)
        sig_ini = min([tramos[u][0] for u in orden[k + 1:] if tramos[u][0] > b], default=n)
        res[t] = (prev_fin + 1, sig_ini - 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gemma_v1")
    a = ap.parse_args()

    filas = pd.read_csv(SEG / "filas.csv", index_col="fila")
    tareas = {t["oracion_id"]: t for t in map(json.loads, (SEG / "tareas.jsonl").read_text(encoding="utf-8").splitlines())}
    respuestas = {r["oracion_id"]: r for r in map(json.loads, (SEG / f"respuestas_{a.tag}.jsonl").read_text(encoding="utf-8").splitlines())}

    filas["segmentada"], filas["metodo"] = "no", ""
    filas["inicio_segmento"], filas["fin_segmento"] = -1, -1
    filas["Texto completo (referencia)"] = ""
    n_gemma = n_regla = 0
    for k, tarea in tareas.items():
        palabras, objetivos = tarea["palabras"], tarea["objetivos"]
        segs = parsear(respuestas.get(k, {}).get("respuesta", ""))
        respaldo = regla(objetivos, len(palabras))
        for o in objetivos:
            seg = tuple(segs[o["t"]]) if o["t"] in segs else None
            if valido(seg, o["pos_verbo"], o["pos_objeto"], len(palabras)):
                metodo = "gemma"; n_gemma += 1
            elif o["t"] in respaldo:
                seg, metodo = respaldo[o["t"]], "regla"; n_regla += 1
            else:
                seg, metodo = (0, len(palabras) - 1), "sin segmentar (verbo/objeto no encontrados)"
            i = o["fila"]
            filas.loc[i, ["inicio_segmento", "fin_segmento", "metodo", "segmentada"]] = [seg[0], seg[1], metodo, "sí"]
            filas.loc[i, "Texto completo (referencia)"] = filas.at[i, "Texto oración"]
            filas.loc[i, "Texto oración"] = " ".join(palabras[seg[0] : seg[1] + 1])

    completo = filas[ORIGINALES + ["clase_mdo", "archivo", "fila_excel", "segmentada", "metodo"]]
    completo.to_excel(SEG / "completo_jb.xlsx", index=False)
    divididas = filas[filas["segmentada"] == "sí"].sort_values(["oracion_id", "t"])
    divididas = divididas[["oracion_id", "t", "Texto oración", "Texto completo (referencia)"] + ORIGINALES[3:]
                          + ["Corpus", "ID", "clase_mdo", "archivo", "fila_excel", "pos_verbo", "pos_objeto",
                             "inicio_segmento", "fin_segmento", "metodo", "aviso"]]
    divididas.to_excel(SEG / "divididas_jb.xlsx", index=False)
    print(f"completo: {len(completo)} filas | divididas: {len(divididas)} filas de {len(tareas)} oraciones")
    print(f"segmentos de Gemma válidos: {n_gemma} | regla de respaldo: {n_regla}")
    print("palabras por segmento (p50/p95/max):",
          divididas["Texto oración"].str.split().str.len().quantile([.5, .95, 1]).tolist())


if __name__ == "__main__":
    main()
