"""¿Cuántas oraciones faltan por clase de animacidad y de definitud para que la muestra sea representativa?

Cuenta la muestra de entrenamiento (dataset/processed/segmentado/mdo, las 3 particiones) por clase de Animacidad y de
Definitud, en total y dentro de cada clase de MDO. Calcula cuántas oraciones faltan para llegar a un mínimo por
combinación (por defecto 100 y 200: con el reparto 80/10/10 quedan unas 10 y 20 en test).

Cuenta también los candidatos de PRESEEA y ESLORA para anotar: objetos que no están en la muestra, con la clase MDO
que les dio el clasificador y la animacidad/definitud heurística del pipeline (columnas `referente` y `definitud`).
Son solo una guía para buscar: esas etiquetas no están revisadas. Se dan también sin los clíticos (lo, la, te…),
que casi no aparecen en la muestra. «Por determinar» no existe en el pipeline.

Salida: resultados/representatividad.csv
Uso: python scripts/08_representatividad.py [--minimos 100 200]
"""
import argparse

import pandas as pd

from common import CLASIFICACION, RAW, ROOT

CLAVE = ["sent_id", "verbo_id", "objeto_id"]
CLITICOS = {"lo", "la", "los", "las", "le", "les", "me", "te", "se", "nos", "os"}
EJES = {"Animacidad": ["humano", "inanimado", "animado"], "Definitud": ["Definido", "Indefinido", "Por determinar"]}
PIPELINE = {"Animacidad": ("referente", {"humano": "humano", "inanimado": "inanimado", "animado": "animado"}),
            "Definitud": ("definitud", {"Def": "Definido", "Ind": "Indefinido"})}


def candidatos() -> pd.DataFrame:
    partes = []
    for corpus in ["preseea", "eslora"]:
        obj = pd.concat([pd.read_excel(p, sheet_name="objetos") for p in sorted((RAW / corpus).glob("*.xlsx"))
                         if not p.name.startswith("~$")]).drop_duplicates(CLAVE)
        cl = pd.read_csv(CLASIFICACION / f"{corpus.upper()}_mdo_claves.csv")
        partes.append(cl[cl["origen"] == "modelo"].merge(obj[CLAVE + ["referente", "definitud"]], on=CLAVE, how="left"))
    return pd.concat(partes, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minimos", type=int, nargs="+", default=[100, 200])
    args = ap.parse_args()

    muestra = pd.concat([pd.read_csv(ROOT / "dataset" / "processed" / "segmentado" / "mdo" / f"{s}.csv")
                         for s in ("train", "dev", "test")])
    cand = candidatos()
    cand["clitico"] = cand["Objeto"].astype(str).str.lower().isin(CLITICOS)
    filas = []
    for eje, clases in EJES.items():
        col, mapa = PIPELINE[eje]
        cand[eje] = cand[col].map(mapa)
        for mdo in ["todas", "con_MDO", "sin_MDO"]:
            m = muestra if mdo == "todas" else muestra[muestra["clase_mdo"] == mdo]
            c = cand if mdo == "todas" else cand[cand["mdo"] == mdo]
            for clase in clases:
                n = int((m[eje] == clase).sum())
                fila = {"eje": eje, "clase_mdo": mdo, "clase": clase, "muestra": n}
                fila |= {f"faltan_min{k}": max(0, k - n) for k in args.minimos}
                hay = clase in mapa.values()
                fila["candidatos_corpus"] = int((c[eje] == clase).sum()) if hay else None
                fila["candidatos_sin_cliticos"] = int(((c[eje] == clase) & ~c["clitico"]).sum()) if hay else None
                filas.append(fila)
    tabla = pd.DataFrame(filas)
    for c in ["candidatos_corpus", "candidatos_sin_cliticos"]:
        tabla[c] = tabla[c].astype("Int64")
    tabla.to_csv(ROOT / "resultados" / "representatividad.csv", index=False)
    pd.set_option("display.width", 200)
    print(tabla.to_string(index=False))
    dentro = tabla[tabla["clase_mdo"] != "todas"]
    for k in args.minimos:
        por_eje = dentro.groupby("eje")[f"faltan_min{k}"].sum().to_dict()
        # cada oración nueva tiene a la vez una animacidad y una definitud: el total está entre el máximo y la suma
        print(f"mínimo {k}: faltan {por_eje} -> entre {max(por_eje.values())} y {sum(por_eje.values())} oraciones nuevas")
    print("\nMuestra por MDO × Animacidad × Definitud:")
    print(pd.crosstab([muestra["clase_mdo"], muestra["Animacidad"]], muestra["Definitud"]).to_string())


if __name__ == "__main__":
    main()
