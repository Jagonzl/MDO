"""Paso 2 de la segmentación: Gemma (vLLM) delimita el segmento de cada objeto directo. Corre en un nodo GPU.

Lee dataset/segmentacion_jb/tareas.jsonl y el prompt prompts_jb/<prompt>.txt (secciones ### SISTEMA / ### USUARIO).
Escribe dataset/segmentacion_jb/respuestas_<tag>.jsonl con la respuesta cruda (la validación se hace en el paso 3).
Patrón de vardial_hisp / dom_mdo: anotador_dom_vllm.py.

Uso (dentro del sbatch, con el venv de vLLM):
  $PYV scripts_jb/00b_segmentar_gemma_jb.py --modelo gemma-4-31B-it --prompt segmentar_mdo_v1 --tag gemma_v1
"""
import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEG = ROOT / "dataset" / "segmentacion_jb"
MODELOS = os.environ.get("MODELOS", "/lustre/romanistik/bonillaj/vardial_hisp/models/hf")


def leer_prompt(nombre: str) -> tuple[str, str]:
    texto = (ROOT / "prompts_jb" / f"{nombre}.txt").read_text(encoding="utf-8")
    sistema, usuario = re.split(r"^### USUARIO\s*$", texto, flags=re.M)
    return re.sub(r"^### SISTEMA\s*$", "", sistema, flags=re.M).strip(), usuario.strip()


def mensaje_usuario(plantilla: str, tarea: dict) -> str:
    numeradas = " ".join(f"{i}:{p}" for i, p in enumerate(tarea["palabras"]))
    objetivos = "\n".join(
        f"T{o['t']}: verbo «{o['verbo']}» (palabra {o['pos_verbo']}) — objeto «{o['objeto']}» (palabra {o['pos_objeto']})"
        for o in tarea["objetivos"])
    return plantilla.format(palabras_numeradas=numeradas, objetivos=objetivos)


def quitar_razonamiento(texto: str) -> str:
    texto = re.sub(r"<think>.*?</think>|\[THINK\].*?\[/THINK\]", "", texto, flags=re.S)
    return re.sub(r"<\|channel>thought.*?<channel\|>", "", texto, flags=re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="gemma-4-31B-it")
    ap.add_argument("--prompt", default="segmentar_mdo_v1")
    ap.add_argument("--tag", default="gemma_v1")
    ap.add_argument("--chat-kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=8192)  # 16384 no cabe en A100 80 GB; el prompt más largo ~3k tokens
    ap.add_argument("--dir", type=Path, default=SEG, help="carpeta con tareas.jsonl (p. ej. dataset/segmentacion_jb/preseea)")
    a = ap.parse_args()
    seg = a.dir if a.dir.is_absolute() else ROOT / a.dir

    tareas = [json.loads(l) for l in (seg / "tareas.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    sistema, plantilla = leer_prompt(a.prompt)
    conv = [[{"role": "system", "content": sistema}, {"role": "user", "content": mensaje_usuario(plantilla, t)}]
            for t in tareas]
    print(f"{len(tareas)} oraciones | prompt {a.prompt} | modelo {a.modelo}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=os.path.join(MODELOS, a.modelo), max_model_len=a.max_model_len,
              gpu_memory_utilization=0.92, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, seed=42)
    outs = llm.chat(conv, sp, chat_template_kwargs=json.loads(a.chat_kwargs))

    destino = seg / f"respuestas_{a.tag}.jsonl"
    with open(destino, "w", encoding="utf-8") as fh:
        for t, c, o in zip(tareas, conv, outs):
            fh.write(json.dumps({"oracion_id": t["oracion_id"], "modelo": a.modelo, "prompt": a.prompt,
                                 "mensaje_usuario": c[1]["content"],
                                 "respuesta": quitar_razonamiento(o.outputs[0].text).strip()}, ensure_ascii=False) + "\n")
    print(f"-> {destino}", flush=True)


if __name__ == "__main__":
    main()
