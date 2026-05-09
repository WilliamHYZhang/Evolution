#!/bin/bash
#SBATCH --job-name=evo_eval_pre
#SBATCH --output=logs/eval_pretrain_%j.out
#SBATCH --error=logs/eval_pretrain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --partition=gpu_h200
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# One-shot pre-finetuning accuracy (ANLI, WSC, MATH) with the same caps/splits as
# training runs. Run this before redoing the triplet job if you need a clean MATH
# baseline row (outputs/eval_pretrain_<jobid>/metrics.csv).

set -euo pipefail
ROOT=/n/home08/whyz/Evolution
cd "$ROOT"
mkdir -p logs
module load python/3.12.11-fasrc01

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"

PY="$ROOT/venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing $PY — run: bash $ROOT/scripts/bootstrap_venv.sh"
  exit 1
fi
if [[ ! -f "$ROOT/.env" ]]; then
  echo "Missing $ROOT/.env (HF_TOKEN required)."
  exit 1
fi

OUT="${OUT:-outputs/eval_pretrain_${SLURM_JOB_ID}}"
echo "=== Evolution eval_pretrain OUT=${OUT} ==="
date
nvidia-smi -L 2>/dev/null || true

"$PY" "$ROOT/scripts/eval_pretrain.py" --output_dir "${OUT}"
