"""Paso 3 de la segmentación: valida los segmentos de Gemma y construye los xlsx nuevos.

Validación de cada segmento: Gemma devuelve inicio, fin y las posiciones de verbo/objeto que usó (prompt v3).
Se acepta si en esas posiciones están de verdad el verbo y el objeto de la fila y si el segmento cubre todo el tramo
verbo→objeto (si hay una «a» entre ambos queda dentro; si no la hay, no se exige nada).
Si no pasa, regla de respaldo: desde justo después del objetivo anterior hasta justo antes del siguiente, con un
máximo de VENTANA palabras a cada lado del tramo verbo→objeto.

Salidas en dataset/segmentacion/<corpus>/:
  completo_<tag>.xlsx  todas las filas de los Excel originales (mismas celdas); en las oraciones con varios objetos,
                    'Texto oración' es el segmento. Columnas añadidas: clase_mdo, archivo, fila_excel, segmentada, metodo.
  divididas_<tag>.xlsx solo las filas divididas, con 'Texto completo (referencia)', posiciones y avisos para revisar.
Uso: python scripts/00c_construir_xlsx.py --corpus muestra     (o preseea, eslora) [--tag gemma_v3]
"""
import argparse
import json
import re
import unicodedata

import pandas as pd

from common import SEG

ORIGINALES = ["Corpus", "ID", "Texto oración", "Verbo", "Objeto", "Animacidad", "Determinante", "Definitud"]
PIPELINE = ["sent_id", "verbo_id", "objeto_id", "archivo_fuente"]  # claves de los xlsx de PRESEEA/ESLORA
VENTANA = 10
MAX_SEG = 40  # segmento de Gemma más largo que esto → regla de respaldo (con las posiciones de Gemma)
DISTANCIA_MAX = 15  # como en 00a: verbo y objeto más lejos que esto = anotación dudosa
CLITICOS = ["nos", "los", "las", "les", "me", "te", "se", "os", "lo", "la", "le"]
HABER = {"hay", "había", "habia", "hubo", "habrá", "habra", "habría", "habria", "haya", "hubiera", "hubiese", "haber", "habiendo"}


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
                    res[int(t)]["alerta"] = str(s.get("alerta") or "").strip()  # prompt v3
                    res[int(t)]["nota"] = str(s.get("nota") or "").strip()
            return res
    return {}


def validar(s: dict | None, obj: dict, palabras: list[str]):
    """Devuelve (inicio, fin, pos_verbo, pos_objeto) si el segmento es válido; si no, None.

    El objeto devuelto debe ser la palabra anotada. El verbo también, salvo con la alerta "verbo_no_rige" (prompt v3):
    entonces Gemma da la posición del verbo rector real, que solo se exige que esté dentro del segmento.
    """
    if not s or s["inicio"] is None or s["fin"] is None:
        return None
    n = len(palabras)
    v = s["pos_verbo"] if s.get("pos_verbo") is not None else obj["pos_verbo"]
    o = s["pos_objeto"] if s.get("pos_objeto") is not None else obj["pos_objeto"]
    verbo_ok = s.get("alerta") == "verbo_no_rige" or (0 <= v < n and coincide(palabras[v], obj["verbo"]))
    i, f = s["inicio"], s["fin"]
    if not (0 <= v < n and verbo_ok and 0 <= i <= f < n):
        return None
    if not (0 <= o < n and coincide(palabras[o], obj["objeto"])):
        # Gemma a veces señala el inicio del sintagma («un par», «esa casa») y no el núcleo: se acepta la aparición
        # del objeto dentro del segmento más cercana a la posición que dio.
        dentro = [j for j in range(i, f + 1) if coincide(palabras[j], obj["objeto"])]
        if not dentro:
            return None
        o = min(dentro, key=lambda j: abs(j - (o if o is not None else v)))
    if v == o:
        return None
    return (i, f, v, o) if 0 <= i <= min(v, o) and max(v, o) <= f < n else None


def sin_tilde(w) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", norm(w)) if unicodedata.category(c) != "Mn")


def encliticos(objetivos: list[dict], palabras: list[str], ya: set) -> dict:
    """Objetivos que no se encontraron por la forma exacta, casi siempre por clíticos pegados al verbo («verlo»,
    «haciéndolo», «fíjate»: en UD son dos tokens, en el texto una palabra y a veces con tilde nueva). Se busca sin tildes:
    objeto clítico → la palabra verbo+clíticos; objeto nominal → par verbo/objeto más cercano.
    Devuelve {t: (pos_verbo, pos_objeto)}; con varias filas del mismo verbo, en orden de aparición."""
    res, por_verbo = {}, {}
    pal = [sin_tilde(p) for p in palabras]
    for o in sorted(objetivos, key=lambda o: o["t"]):
        if o["t"] in ya:
            continue
        f, c = sin_tilde(o["verbo"]), sin_tilde(o["objeto"])
        if not f or not c:
            continue
        verbos = [j for j, p in enumerate(pal) if p.startswith(f)]
        if c in CLITICOS:
            pares = [(j, j) for j in verbos if c in pal[j][len(f):]
                     and re.fullmatch(f"(?:{'|'.join(CLITICOS)})+", pal[j][len(f):])]
            pares += [(j, k) for j in verbos for k, p in enumerate(pal) if p == c and k != j]
        else:
            pares = [(j, k) for j in verbos for k, p in enumerate(pal) if k != j and (p == c or p.startswith(c))]
        if pares:
            coste = lambda x: (x[1] - x[0]) if x[1] >= x[0] else 3 * (x[0] - x[1])
            mejor = {}
            for p in pares:  # por cada aparición del verbo, el objeto más cercano; luego en orden de aparición
                if p[0] not in mejor or coste(p) < coste(mejor[p[0]]):
                    mejor[p[0]] = p
            pares = [mejor[j] for j in sorted(mejor)]
            n = por_verbo.get(f, 0)
            res[o["t"]] = pares[min(n, len(pares) - 1)]
            por_verbo[f] = n + 1
    return res


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
    ap.add_argument("--tag", default="gemma_v3")
    ap.add_argument("--corpus", choices=["muestra", "preseea", "eslora"], default="muestra")
    a = ap.parse_args()
    seg = SEG / a.corpus

    filas = pd.read_csv(seg / "filas.csv", index_col="fila")
    tareas = {t["oracion_id"]: t for t in map(json.loads, (seg / "tareas.jsonl").read_text(encoding="utf-8").splitlines())}
    respuestas = {r["oracion_id"]: r for r in map(json.loads, (seg / f"respuestas_{a.tag}.jsonl").read_text(encoding="utf-8").splitlines())}

    filas["segmentada"], filas["metodo"] = "no", ""
    filas["inicio_segmento"], filas["fin_segmento"] = -1, -1
    filas["Texto completo (referencia)"] = ""
    filas["aviso"] = filas["aviso"].fillna("")
    filas["alerta_gemma"], filas["nota_gemma"] = "", ""
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
        enc = encliticos(objetivos, palabras, set(tramos))
        for t, (v, ob) in enc.items():
            tramos[t] = (min(v, ob), max(v, ob), v, ob)
        # verbo muy lejos del objeto (anotación dudosa): la regla de respaldo se centra solo en el objeto
        lejos = {t for t, x in tramos.items() if x[1] - x[0] > DISTANCIA_MAX}
        respaldo = regla({t: ((x[3], x[3]) if t in lejos else x[:2]) for t, x in tramos.items()}, len(palabras))
        for o in objetivos:
            i, t = o["fila"], o["t"]
            if elegidos[t] and elegidos[t][1] - elegidos[t][0] < MAX_SEG:
                ini, fin, v, ob = elegidos[t]; metodo = "gemma"
            elif t in tramos:
                (ini, fin), (_, _, v, ob) = respaldo[t], tramos[t]
                metodo = ("regla (forma sin tilde / clítico pegado)" if t in enc else
                          "regla (centrada en el objeto)" if t in lejos else
                          "regla (segmento de Gemma demasiado largo)" if elegidos[t] else "regla")
            else:
                ini, fin, v, ob = 0, len(palabras) - 1, -1, -1; metodo = "sin segmentar (verbo/objeto no encontrados)"
            otros = [u for u, x in tramos.items() if u != t and x[2] != v and ini <= x[0] and x[1] <= fin]
            avisos = [filas.at[i, "aviso"]] if filas.at[i, "aviso"] else []
            if otros:
                avisos.append("contiene otro objetivo: " + ", ".join(f"T{u}" for u in otros))
            if v != o["pos_verbo"] or ob != o["pos_objeto"]:
                avisos.append(f"posición corregida (verbo {o['pos_verbo']}→{v}, objeto {o['pos_objeto']}→{ob})")
            if norm(o["verbo"]) in HABER:
                avisos.append("haber existencial (criterio del revisor: no se contabiliza como posible MDO)")
            g = segs.get(t) or {}
            alerta = g.get("alerta", "") if elegidos[t] else ""
            if alerta == "verbo_no_rige" and 0 <= v < len(palabras):
                alerta += f" (verbo rector: «{palabras[v]}», palabra {v})"
            filas.loc[i, ["inicio_segmento", "fin_segmento", "pos_verbo", "pos_objeto", "metodo", "segmentada", "aviso",
                          "alerta_gemma", "nota_gemma"]] = \
                [ini, fin, v, ob, metodo, "sí", "; ".join(avisos), alerta, g.get("nota", "") if elegidos[t] else ""]
            filas.loc[i, "Texto completo (referencia)"] = filas.at[i, "Texto oración"]
            filas.loc[i, "Texto oración"] = " ".join(palabras[ini : fin + 1])

    hay = lambda cols: [c for c in cols if c in filas.columns]  # PRESEEA/ESLORA no traen Animacidad, etc.
    completo = filas[hay(ORIGINALES + PIPELINE + ["clase_mdo", "archivo", "fila_excel", "segmentada", "metodo"])]
    completo.to_excel(seg / f"completo_{a.tag}.xlsx", index=False)
    div = filas[filas["segmentada"] == "sí"].sort_values(["oracion_id", "t"])
    div = div[hay(["oracion_id", "t", "Texto oración", "Texto completo (referencia)"] + ORIGINALES[3:]
                  + ["Corpus", "ID"] + PIPELINE + ["clase_mdo", "archivo", "fila_excel", "pos_verbo", "pos_objeto",
                     "inicio_segmento", "fin_segmento", "metodo", "aviso", "alerta_gemma", "nota_gemma"])]
    div.to_excel(seg / f"divididas_{a.tag}.xlsx", index=False)
    print(f"completo: {len(completo)} filas | divididas: {len(div)} filas de {len(tareas)} oraciones")
    print("método:", div["metodo"].value_counts().to_dict())
    print("con aviso:", int((div["aviso"] != "").sum()), "| contiene otro objetivo:", int(div["aviso"].str.contains("contiene otro").sum()),
          "| posición corregida:", int(div["aviso"].str.contains("corregida").sum()))
    print("alertas de Gemma:", div["alerta_gemma"].str.split(" ").str[0].replace("", "ninguna").value_counts().to_dict())
    print("palabras por segmento (p50/p95/max):", div["Texto oración"].str.split().str.len().quantile([.5, .95, 1]).tolist())


if __name__ == "__main__":
    main()
