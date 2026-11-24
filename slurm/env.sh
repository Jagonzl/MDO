# Entorno del clúster de la HU. Cargar en cada sbatch y sesión: source /lustre/romanistik/bonillaj/mdo/MDO/slurm/env.sh
export MJ=/lustre/romanistik/bonillaj/mdo
export REPO=$MJ/MDO
export PY=$MJ/venv/bin/python
export HF_HOME=$MJ/cache/hf
export TMPDIR=$MJ/cache/tmp
export XDG_CACHE_HOME=$MJ/cache
export MDO_MODELOS=$MJ/modelos
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# En los trabajos los modelos base ya están en $HF_HOME (descargados desde el login).
if [ -n "$SLURM_JOB_ID" ]; then export HF_HUB_OFFLINE=1; fi
mkdir -p $TMPDIR $MJ/logs $MDO_MODELOS
cd $REPO
