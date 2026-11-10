"""Paso 3 de la segmentación: valida los segmentos de Gemma y construye los xlsx nuevos.

Validación de cada segmento: Gemma devuelve inicio, fin y las posiciones de verbo/objeto que usó (prompt v2; con v1 se
usan las posiciones del paso 1). Se acepta si en esas posiciones están de verdad el verbo y el objeto de la fila y si el
segmento cubre todo el tramo verbo→objeto (si hay una «a» entre ambos queda dentro; si no la hay, no se exige nada).
Si no pasa, regla de respaldo: desde justo después del objetivo anterior hasta justo antes del siguiente, con un
máximo de VENTANA palabras a cada lado del tramo verbo→objeto.

Salidas en dataset/segmentacion_jb/:
  completo_jb.xlsx  todas las filas de los dos xlsx originales (mismas celdas); en las oraciones con varios objetos,
                    'Texto oración' es el segmento. Columnas añadidas: clase_mdo, archivo, fila_excel, segmentada, metodo.
  divididas_jb.xlsx solo las filas divididas, con 'Texto completo (referencia)', posiciones y avisos para revisar.
Uso: python scripts_jb/00c_construir_xlsx_jb.py --tag gemma_v2
"""
import argparse
import json
import re

import pandas as pd

from common_jb import ROOT

SEG = ROOT / "dataset" / "segmentacion_jb"
ORIGINALES = ["Corpus", "ID", "Texto oración", "Verbo", "Objeto", "Animacidad", "Determinante", "Definitud"]
VENTANA = 10


def norm(w) -> str:
    return re.sub(r"\W", "", str(w).lower())


def coincide(palabra: str, forma: str) -> bool:
    p, f = norm(palabra), norm(forma)
    return bool(f) and (p == f or p.startswith(f))


def entero(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def parsear(respuesta: str) -> dict:
    """{t: {'inicio', 'fin', 'pos_verbo', 'pos_objeto'}} a partir del último JSON con 'segmentos'."""
    for m in reversed(list(re.finditer(r"\{.*\}", respuesta, flags=re.S))):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and isinstance(d.get("segmentos"), list):
            res = {}
            for s in d["segmentos"]:
                t = re.sub(r"\D", "", str(s.get("t", ""))) if isinstance(s, dict) else ""  # admite "T1"
                if t:
                    res[int(t)] = {k: entero(s.get(k)) for k in ("inicio", "fin", "pos_verbo", "pos_objeto")}
            return res
    return {}


def validar(s: dict | None, obj: dict, palabras: list[str]):
    """Devuelve (inicio, fin, pos_verbo, pos_objeto) si el segmento es válido; si no, None."""
    if not s or s["inicio"] is None or s["fin"] is None:
        return None
    n = len(palabras)
    v = s["pos_verbo"] if s.get("pos_verbo") is not None else obj["pos_verbo"]
    o = s["pos_objeto"] if s.get("pos_objeto") is not None else obj["pos_objeto"]
    if not (0 <= v < n and 0 <= o < n and coincide(palabras[v], obj["verbo"]) and coincide(palabras[o], obj["objeto"])):
        return None
    i, f = s["inicio"], s["fin"]
    return (i, f, v, o) if 0 <= i <= min(v, o) and max(v, o) <= f < n else None


def regla(tramos: dict, n: int) -> dict:
    """Respaldo: corta entre objetivos consecutivos, como mucho VENTANA palabras a cada lado del tramo."""
    orden = sorted(tramos, key=lambda t: tramos[t])
    res = {}
    for k, t in enumerate(orden):
        a, b = tramos[t]
        prev_fin = max([tramos[u][1] for u in orden[:k] if tramos[u][1] < a], default=-1)
        sig_ini = min([tramos[u][0] for u in orden[k + 1:] if tramos[u][0] > b], default=n)
        res[t] = (max(prev_fin + 1, a - VENTANA), min(sig_ini - 1, b + VENTANA))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gemma_v2")
    a = ap.parse_args()

    filas = pd.read_csv(SEG / "filas.csv", index_col="fila")
    tareas = {t["oracion_id"]: t for t in map(json.loads, (SEG / "tareas.jsonl").read_text(encoding="utf-8").splitlines())}
    respuestas = {r["oracion_id"]: r for r in map(json.loads, (SEG / f"respuestas_{a.tag}.jsonl").read_text(encoding="utf-8").splitlines())}

    filas["segmentada"], filas["metodo"] = "no", ""
    filas["inicio_segmento"], filas["fin_segmento"] = -1, -1
    filas["Texto completo (referencia)"] = ""
    filas["aviso"] = filas["aviso"].fillna("")
    for k, tarea in tareas.items():
        palabras, objetivos = tarea["palabras"], tarea["objetivos"]
        segs = parsear(respuestas.get(k, {}).get("respuesta", ""))
        elegidos = {o["t"]: validar(segs.get(o["t"]), o, palabras) for o in objetivos}
        # tramos verbo→objeto finales (los de Gemma si son válidos; si no, los del paso 1)
        tramos = {}
        for o in objetivos:
            e = elegidos[o["t"]]
            v, ob = (e[2], e[3]) if e else (o["pos_verbo"], o["pos_objeto"])
            if v >= 0 and ob >= 0:
                tramos[o["t"]] = (min(v, ob), max(v, ob), v, ob)
        respaldo = regla({t: x[:2] for t, x in tramos.items()}, len(palabras))
        for o in objetivos:
            i, t = o["fila"], o["t"]
            if elegidos[t]:
                ini, fin, v, ob = elegidos[t]; metodo = "gemma"
            elif t in tramos:
                (ini, fin), (_, _, v, ob) = respaldo[t], tramos[t]; metodo = "regla"
            else:
                ini, fin, v, ob = 0, len(palabras) - 1, -1, -1; metodo = "sin segmentar (verbo/objeto no encontrados)"
            otros = [u for u, x in tramos.items() if u != t and x[2] != v and ini <= x[0] and x[1] <= fin]
            avisos = [filas.at[i, "aviso"]] if filas.at[i, "aviso"] else []
            if otros:
                avisos.append("contiene otro objetivo: " + ", ".join(f"T{u}" for u in otros))
            if v != o["pos_verbo"] or ob != o["pos_objeto"]:
                avisos.append(f"posición corregida (verbo {o['pos_verbo']}→{v}, objeto {o['pos_objeto']}→{ob})")
            filas.loc[i, ["inicio_segmento", "fin_segmento", "pos_verbo", "pos_objeto", "metodo", "segmentada", "aviso"]] = \
                [ini, fin, v, ob, metodo, "sí", "; ".join(avisos)]
            filas.loc[i, "Texto completo (referencia)"] = filas.at[i, "Texto oración"]
            filas.loc[i, "Texto oración"] = " ".join(palabras[ini : fin + 1])

    completo = filas[ORIGINALES + ["clase_mdo", "archivo", "fila_excel", "segmentada", "metodo"]]
    completo.to_excel(SEG / "completo_jb.xlsx", index=False)
    div = filas[filas["segmentada"] == "sí"].sort_values(["oracion_id", "t"])
    div = div[["oracion_id", "t", "Texto oración", "Texto completo (referencia)"] + ORIGINALES[3:]
              + ["Corpus", "ID", "clase_mdo", "archivo", "fila_excel", "pos_verbo", "pos_objeto",
                 "inicio_segmento", "fin_segmento", "metodo", "aviso"]]
    div.to_excel(SEG / "divididas_jb.xlsx", index=False)
    print(f"completo: {len(completo)} filas | divididas: {len(div)} filas de {len(tareas)} oraciones")
    print("método:", div["metodo"].value_counts().to_dict())
    print("con aviso:", int((div["aviso"] != "").sum()), "| contiene otro objetivo:", int(div["aviso"].str.contains("contiene otro").sum()),
          "| posición corregida:", int(div["aviso"].str.contains("corregida").sum()))
    print("palabras por segmento (p50/p95/max):", div["Texto oración"].str.split().str.len().quantile([.5, .95, 1]).tolist())


if __name__ == "__main__":
    main()
