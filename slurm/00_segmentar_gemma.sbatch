#!/bin/bash
# Segmentación con Gemma (00b, vLLM). Antes: 00a en el login; después: 00c en el login.
# Uso: sbatch slurm/00_segmentar_gemma.sbatch --corpus muestra
#      sbatch --time=04:00:00 slurm/00_segmentar_gemma.sbatch --corpus eslora --max-tokens 4096
#SBATCH --job-name=mdo_seg
#SBATCH --partition=gpu_a100,gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --exclude=hpc-cms01-004,hpc-cms01-006,hpc-cms01-010,hpc-cms01-029
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/romanistik/bonillaj/mdo/logs/seg_%j.log
# Entorno de vLLM ya montado en el clúster (venv y pesos de Gemma, solo lectura).
source /lustre/romanistik/bonillaj/dom_mdo/env.sh
export TMPDIR=/lustre/romanistik/bonillaj/mdo/cache/tmp
cd /lustre/romanistik/bonillaj/mdo/MDO
$PYV scripts/00b_segmentar_gemma.py "$@"
