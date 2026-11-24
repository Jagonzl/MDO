#!/bin/bash
# Búsqueda de hiperparámetros (03_busqueda.py) para los 5 modelos de common.PARES:
#   0 = mrbert_es, 1 = mrbert, 2 = mbert, 3 = beto_cased, 4 = beto_uncased.
# Uso: sbatch slurm/03_busqueda.sbatch        (o --array=3 para un solo par)
#SBATCH --job-name=mdo_03
#SBATCH --partition=gpu_a100,gpu_l40,standard
#SBATCH --gres=gpu:1
#SBATCH --exclude=hpc-cms01-004,hpc-cms01-006,hpc-cms01-010,hpc-cms01-029
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/romanistik/bonillaj/mdo/logs/03_%A_%a.log
#SBATCH --array=0-4
source /lustre/romanistik/bonillaj/mdo/MDO/slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Si la GPU del nodo está averiada, torch caería a CPU sin avisar: abortar.
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "SIN GPU en $(hostname)"; exit 1; }
$PY scripts/03_busqueda.py --par $SLURM_ARRAY_TASK_ID "$@"
