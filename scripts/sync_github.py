"""Sube dataset/, scripts/ y resultados/ al repositorio de GitHub (sin modelos ni token).

Uso: python scripts/sync_github.py "mensaje del commit"
"""
import base64
import json
import sys
import urllib.error
import urllib.request

from common import ROOT

REPO = "Jagonzl/MDO"
RAMA = "main"
CARPETAS = ["dataset", "scripts", "resultados"]
EXCLUIR = {"__pycache__", "checkpoints", ".ipynb_checkpoints"}
GITIGNORE = "modelos/\ngithubtoken.txt\n__pycache__/\ncheckpoints/\n*.safetensors\n*.bin\n"

TOKEN = (ROOT / "githubtoken.txt").read_text().strip()


def api(metodo: str, ruta: str, datos=None):
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}{ruta}",
        data=json.dumps(datos).encode() if datos is not None else None,
        method=metodo,
        headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def archivos():
    for carpeta in CARPETAS:
        for p in sorted((ROOT / carpeta).rglob("*")):
            if p.is_file() and not EXCLUIR & set(p.relative_to(ROOT).parts):
                yield p.relative_to(ROOT).as_posix(), p.read_bytes()
    for extra in ["README.md"]:
        if (ROOT / extra).exists():
            yield extra, (ROOT / extra).read_bytes()
    yield ".gitignore", GITIGNORE.encode()


def main():
    mensaje = sys.argv[1] if len(sys.argv) > 1 else "Actualizar dataset, scripts y resultados"
    estado, repo = api("GET", "")
    if estado != 200:
        sys.exit(f"Sin acceso a {REPO} ({estado}). Revisa los permisos del token.")

    estado, ref = api("GET", f"/git/ref/heads/{RAMA}")
    if estado != 200:
        contenido = base64.b64encode(b"# MDO\n").decode()
        api("PUT", "/contents/README.md", {"message": "Inicializar repositorio", "content": contenido, "branch": RAMA})
        estado, ref = api("GET", f"/git/ref/heads/{RAMA}")
    padre = ref["object"]["sha"]

    arbol = []
    for ruta, datos in archivos():
        _, blob = api("POST", "/git/blobs", {"content": base64.b64encode(datos).decode(), "encoding": "base64"})
        arbol.append({"path": ruta, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    _, tree = api("POST", "/git/trees", {"tree": arbol})

    _, commit_padre = api("GET", f"/git/commits/{padre}")
    if commit_padre["tree"]["sha"] == tree["sha"]:
        print("Sin cambios que subir.")
        return
    _, commit = api("POST", "/git/commits", {"message": mensaje, "tree": tree["sha"], "parents": [padre]})
    estado, _ = api("PATCH", f"/git/refs/heads/{RAMA}", {"sha": commit["sha"]})
    print(f"{len(arbol)} archivos subidos a {REPO}@{RAMA} ({commit['sha'][:7]})" if estado == 200 else f"Error {estado}")


if __name__ == "__main__":
    main()
