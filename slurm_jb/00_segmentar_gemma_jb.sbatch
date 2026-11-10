#!/bin/bash
# Paso 2 de la segmentación (Gemma con vLLM). Antes: 00a en el login; después: 00c en el login.
# Uso: sbatch slurm_jb/00_segmentar_gemma_jb.sbatch [--prompt segmentar_mdo_v1 --tag gemma_v1]
#SBATCH --job-name=mdo_jb_seg
#SBATCH --partition=gpu_a100,gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --exclude=hpc-cms01-004,hpc-cms01-010,hpc-cms01-029,hpc-cms01-006
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/romanistik/bonillaj/mdo_jb/logs/seg_%j.log
# Entorno de vLLM ya montado para dom_mdo (venv y pesos de vardial_hisp, solo lectura).
source /lustre/romanistik/bonillaj/dom_mdo/env.sh
export TMPDIR=/lustre/romanistik/bonillaj/mdo_jb/cache/tmp
cd /lustre/romanistik/bonillaj/mdo_jb/MDO
$PYV scripts_jb/00b_segmentar_gemma_jb.py "$@"
