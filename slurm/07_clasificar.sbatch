#!/bin/bash
# Clasificación MDO de un corpus segmentado (07_clasificar_mdo.py). Antes: 00a, 00 (Gemma) y 00c para ese corpus.
# Uso: sbatch slurm/07_clasificar.sbatch --corpus preseea      (o eslora)
#SBATCH --job-name=mdo_07
#SBATCH --partition=gpu_a100,gpu_l40,standard
#SBATCH --gres=gpu:1
#SBATCH --exclude=hpc-cms01-004,hpc-cms01-006,hpc-cms01-010,hpc-cms01-029
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/romanistik/bonillaj/mdo/logs/07_%j.log
source /lustre/romanistik/bonillaj/mdo/MDO/slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Si la GPU del nodo está averiada, torch caería a CPU sin avisar: abortar.
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "SIN GPU en $(hostname)"; exit 1; }
$PY scripts/07_clasificar_mdo.py "$@"
