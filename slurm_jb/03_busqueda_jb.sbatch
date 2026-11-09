#!/bin/bash
# Búsqueda de hiperparámetros (03_small_test_jb.py) para los 10 pares (tarea, modelo) de common_jb.PARES:
#   0-4 = mdo × {mrbert_es, mrbert, mbert, beto_cased, beto_uncased}; 5-9 = definitud × (los mismos).
# Uso: sbatch slurm_jb/03_busqueda_jb.sbatch        (o --array=0 para un solo par)
#SBATCH --job-name=mdo_jb_03
#SBATCH --partition=gpu_l40,gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=/lustre/romanistik/bonillaj/mdo_jb/logs/03_%A_%a.log
source /lustre/romanistik/bonillaj/mdo_jb/MDO/slurm_jb/env_jb.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scripts_jb/03_small_test_jb.py --par $SLURM_ARRAY_TASK_ID "$@"
