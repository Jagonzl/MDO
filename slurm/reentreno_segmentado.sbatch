#!/bin/bash
# Reentrena BETO cased (MDO) con el dataset segmentado: líneas base, búsqueda y entrenamiento final.
# Antes: 01_build_dataset.py --version segmentado en el login. Salida en resultados/segmentado/.
# Uso: sbatch slurm/reentreno_segmentado.sbatch
#SBATCH --job-name=mdo_seg_beto
#SBATCH --partition=gpu_a100,gpu_l40,standard
#SBATCH --gres=gpu:1
#SBATCH --exclude=hpc-cms01-004,hpc-cms01-006,hpc-cms01-010,hpc-cms01-029
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/romanistik/bonillaj/mdo/logs/reentreno_%j.log
source /lustre/romanistik/bonillaj/mdo/MDO/slurm/env.sh
export MDO_VERSION=segmentado
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Si la GPU del nodo está averiada, torch caería a CPU sin avisar: abortar.
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "SIN GPU en $(hostname)"; exit 1; }
set -e
$PY scripts/02_baseline.py --task mdo
$PY scripts/03_busqueda.py --task mdo --model beto_cased
$PY scripts/04_final.py --task mdo --model beto_cased
