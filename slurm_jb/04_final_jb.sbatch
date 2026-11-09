#!/bin/bash
# Entrenamiento final + test (04_final_jb.py) para los 10 pares; cada elemento espera a su par del 03:
#   sbatch --dependency=aftercorr:<jobid_03> slurm_jb/04_final_jb.sbatch
#SBATCH --job-name=mdo_jb_04
#SBATCH --partition=gpu_l40,gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=/lustre/romanistik/bonillaj/mdo_jb/logs/04_%A_%a.log
source /lustre/romanistik/bonillaj/mdo_jb/MDO/slurm_jb/env_jb.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scripts_jb/04_final_jb.py --par $SLURM_ARRAY_TASK_ID "$@"
