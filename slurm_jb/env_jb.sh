# Cargar en cada sbatch y sesión: source /lustre/romanistik/bonillaj/mdo_jb/MDO/slurm_jb/env_jb.sh
export MJ=/lustre/romanistik/bonillaj/mdo_jb
export REPO=$MJ/MDO
export PY=$MJ/venv/bin/python
export HF_HOME=$MJ/cache/hf
export TMPDIR=$MJ/cache/tmp
export XDG_CACHE_HOME=$MJ/cache
export MODELOS_DIR_JB=$MJ/modelos_jb
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# En los jobs los modelos base ya están en $HF_HOME (descargados desde el login).
if [ -n "$SLURM_JOB_ID" ]; then export HF_HUB_OFFLINE=1; fi
mkdir -p $TMPDIR $MJ/logs $MODELOS_DIR_JB
cd $REPO
