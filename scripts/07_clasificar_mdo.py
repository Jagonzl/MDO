"""Clasifica MDO (con_MDO / sin_MDO) los objetos de un corpus ya segmentado (PRESEEA, ESLORA).

Entrada: dataset/segmentacion/<corpus>/completo_<tag>.xlsx y divididas_<tag>.xlsx (salida de 00c).
- Filas que ya están en la muestra anotada (mismo ID + Verbo + Objeto que en
  dataset/segmentacion/muestra/completo_<tag>.xlsx): no se clasifican; se toma su clase anotada y su segmento.
  En la salida, seguridad_% = "anotado".
- El resto: modelo final de MDO (mayor F1 macro medio en test de resultados/segmentado/mdo/final_*/metricas.json,
  o el que se indique con --modelo: carpeta local o repo de Hugging Face), con la misma entrada que en
  entrenamiento: (segmento, "verbo: X ; objeto: Y").
- seguridad_% = probabilidad softmax de la clase predicha × 100 (no calibrada).

Salidas en resultados/clasificacion/:
  <CORPUS>_mdo.xlsx         3 columnas: texto (segmento con <<verbo>> y [[objeto]] marcados + "| verbo … ; objeto …"),
                            mdo, seguridad_%
  <CORPUS>_mdo_claves.csv   mismas filas y orden: sent_id, verbo_id, objeto_id, Verbo, Objeto, origen (modelo/anotado),
                            mdo, seguridad_%, segmentada, metodo y dom_marcado del pipeline, para cruzar con el original
  <CORPUS>_mdo_modelo.json  modelo usado, recuentos y entorno
Uso: python scripts/07_clasificar_mdo.py --corpus preseea [--modelo <carpeta o repo HF>]   (en GPU: slurm/07_clasificar.sbatch)
"""
import argparse
import importlib.util
import json

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import CLASIFICACION, MAX_LENGTH, RAW, ROOT, SEG, TAREAS, normalizar, recortar
from entrenamiento import entorno, probabilidades

TAREA = "mdo"
CLAVE = ["sent_id", "verbo_id", "objeto_id"]
FINALES = ROOT / "resultados" / "segmentado" / TAREA

_spec = importlib.util.spec_from_file_location("prep", ROOT / "scripts" / "00a_preparar_segmentos.py")
_prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prep)
norm = _prep.norm


def mejor_modelo() -> dict:
    finales = [json.loads(p.read_text(encoding="utf-8")) for p in FINALES.glob("final_*/metricas.json")]
    m = max(finales, key=lambda m: m["test_f1_macro"]["media"])
    return {"modelo": m["modelo"], "ruta": m["ruta_modelo"], "seed": m["seed_guardada"],
            "test_f1_macro": m["test_f1_macro"]["media"]}


def marcar(segmento: str, verbo: str, objeto: str, pv: int | None = None, po: int | None = None) -> str:
    """Marca el verbo y el objeto en el segmento. pv/po: posiciones dentro del segmento si se conocen; si no, el par
    verbo→objeto más plausible (como en 00a). Siempre se añade "| verbo: … ; objeto: …"."""
    p = segmento.split()
    if pv is None or po is None or not (0 <= pv < len(p) and 0 <= po < len(p)):
        cands = _prep.pares_candidatos(p, verbo, objeto, k=1)
        pv, po = (cands[0][1], cands[0][2]) if cands else (-1, -1)
    if 0 <= pv < len(p):
        p[pv] = f"<<{p[pv]}>>"
    if 0 <= po < len(p) and po != pv:
        p[po] = f"[[{p[po]}]]"
    return " ".join(p) + f" | verbo: {verbo} ; objeto: {objeto}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["preseea", "eslora"], required=True)
    ap.add_argument("--tag", default="gemma_v3")
    ap.add_argument("--modelo", help="carpeta local o repo de Hugging Face; por defecto, el de resultados/segmentado")
    args = ap.parse_args()
    d, nombre = SEG / args.corpus, args.corpus.upper()

    df = pd.read_excel(d / f"completo_{args.tag}.xlsx")
    for c in ["Texto oración", "Verbo", "Objeto"]:
        df[c] = df[c].map(normalizar)
    # posiciones de verbo/objeto dentro del segmento (solo filas divididas)
    div = pd.read_excel(d / f"divididas_{args.tag}.xlsx")
    pos = {tuple(k): (v - i, o - i) for k, v, o, i in
           zip(div[CLAVE].values, div["pos_verbo"], div["pos_objeto"], div["inicio_segmento"]) if v >= 0 and o >= 0}

    # filas ya anotadas a mano en la muestra
    ent = pd.read_excel(SEG / "muestra" / f"completo_{args.tag}.xlsx")
    ent["k"] = list(zip(ent["ID"], ent["Verbo"].map(norm), ent["Objeto"].map(norm)))
    gold = ent.drop_duplicates("k").set_index("k")[["clase_mdo", "Texto oración"]]
    df["k"] = list(zip(df["ID"], df["Verbo"].map(norm), df["Objeto"].map(norm)))
    anotado = df["k"].isin(gold.index)
    print(f"{len(df)} objetos | {int(anotado.sum())} ya anotados en la muestra | {int((~anotado).sum())} a clasificar")

    mm = mejor_modelo()
    if args.modelo:
        mm["ruta"] = args.modelo
    tok = AutoTokenizer.from_pretrained(mm["ruta"])
    model = AutoModelForSequenceClassification.from_pretrained(mm["ruta"]).to("cuda" if torch.cuda.is_available() else "cpu")
    x = df[~anotado].copy()
    x["par"] = "verbo: " + x["Verbo"] + " ; objeto: " + x["Objeto"]
    x["texto"] = x["Texto oración"]
    largo = pd.Series([len(t) for t in tok(x["texto"].tolist(), x["par"].tolist())["input_ids"]], index=x.index) > MAX_LENGTH
    x.loc[largo, "texto"] = [recortar(t, o) for t, o in zip(x.loc[largo, "texto"], x.loc[largo, "Objeto"])]
    probs = probabilidades(model, tok, x) if len(x) else []
    clases = TAREAS[TAREA]["clases"]

    df["origen"], df["mdo"], df["seguridad_%"] = "", "", ""
    df.loc[~anotado, "origen"] = "modelo"
    df.loc[~anotado, "mdo"] = [clases[i] for i in probs.argmax(-1)] if len(x) else []
    df.loc[~anotado, "seguridad_%"] = (probs.max(-1) * 100).round(1) if len(x) else []
    df.loc[anotado, "origen"] = "anotado"
    df.loc[anotado, "mdo"] = [gold.at[k, "clase_mdo"] for k in df.loc[anotado, "k"]]
    df.loc[anotado, "seguridad_%"] = "anotado"
    df.loc[anotado, "Texto oración"] = [normalizar(gold.at[k, "Texto oración"]) for k in df.loc[anotado, "k"]]

    df["texto"] = [marcar(s, v, o, *(pos.get(tuple(k), (None, None)) if org == "modelo" else (None, None)))
                   for s, v, o, k, org in zip(df["Texto oración"], df["Verbo"], df["Objeto"], df[CLAVE].values, df["origen"])]

    CLASIFICACION.mkdir(parents=True, exist_ok=True)
    destino = CLASIFICACION / f"{nombre}_mdo.xlsx"
    df[["texto", "mdo", "seguridad_%"]].to_excel(destino, index=False)
    claves = df[CLAVE + ["Verbo", "Objeto", "origen", "mdo", "seguridad_%", "segmentada", "metodo"]]
    orig = pd.concat([pd.read_excel(p, sheet_name="objetos") for p in sorted((RAW / args.corpus).glob("*.xlsx"))
                      if not p.name.startswith("~$")]).drop_duplicates(subset=CLAVE)
    if "dom_marcado" in orig:
        claves = claves.merge(orig[CLAVE + ["dom_marcado"]], on=CLAVE, how="left")
    claves.to_csv(CLASIFICACION / f"{nombre}_mdo_claves.csv", index=False)
    (CLASIFICACION / f"{nombre}_mdo_modelo.json").write_text(json.dumps(
        {**mm, "n": len(df), "anotados": int(anotado.sum()), "clasificados": int((~anotado).sum()),
         "recortadas": int(largo.sum()), "entorno": entorno()}, indent=2, ensure_ascii=False), encoding="utf-8")

    m = df[df["origen"] == "modelo"]
    print(f"modelo {mm['modelo']} (seed {mm['seed']}, test F1 {mm['test_f1_macro']:.3f}) | {int(largo.sum())} recortadas")
    print("mdo (modelo):", m["mdo"].value_counts().to_dict(), "| mdo (anotado):", df.loc[anotado, "mdo"].value_counts().to_dict())
    seg = pd.to_numeric(m["seguridad_%"])
    print(f"seguridad_% (modelo): media {seg.mean():.1f} | < 90: {int((seg < 90).sum())} | < 70: {int((seg < 70).sum())}")
    if "dom_marcado" in claves:
        c = claves[claves["origen"] == "modelo"]
        print("modelo vs dom_marcado del pipeline:")
        print(pd.crosstab(c["mdo"], c["dom_marcado"]).to_string())
    print(f"-> {destino}")


if __name__ == "__main__":
    main()
