#!/bin/bash
# Paso 3 del plan: reentrena beto_cased (MDO) con el dataset de oraciones segmentadas (dataset/processed_jb).
# Antes: 01_build_dataset_jb.py en el login. Salidas en resultados_jb_seg/ y $MJ/modelos_jb_seg/.
# Uso: sbatch slurm_jb/seg_beto_jb.sbatch
#SBATCH --job-name=mdo_jb_seg_beto
#SBATCH --partition=gpu_a100,gpu_l40,standard
#SBATCH --gres=gpu:1
#SBATCH --exclude=hpc-cms01-004,hpc-cms01-010,hpc-cms01-029,hpc-cms01-006
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/romanistik/bonillaj/mdo_jb/logs/seg_beto_%j.log
source /lustre/romanistik/bonillaj/mdo_jb/MDO/slurm_jb/env_jb.sh
export MDO_JB_PROCESSED=$REPO/dataset/processed_jb
export MDO_JB_RESULTADOS=$REPO/resultados_jb_seg
export MODELOS_DIR_JB=$MJ/modelos_jb_seg
mkdir -p $MODELOS_DIR_JB
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "SIN GPU en $(hostname)"; exit 1; }
set -e
$PY scripts_jb/02_baseline_jb.py --task mdo
$PY scripts_jb/03_small_test_jb.py --task mdo --model beto_cased
$PY scripts_jb/04_final_jb.py --task mdo --model beto_cased
