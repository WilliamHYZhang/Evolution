#!/bin/bash
#SBATCH --job-name=evo_debug_math
#SBATCH --output=logs/debug_one_math_%j.out
#SBATCH --error=logs/debug_one_math_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --partition=gpu_h200
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
ROOT=/n/home08/whyz/Evolution
cd "$ROOT"
mkdir -p logs
module load python/3.12.11-fasrc01

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"

PY="$ROOT/venv/bin/python"
ADAPTER_DIR="${ADAPTER_DIR:-outputs/run_5825378/adapter}"
INDEX="${INDEX:-0}"

echo "=== Evolution debug one math ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "ADAPTER_DIR=${ADAPTER_DIR} INDEX=${INDEX}"
date
nvidia-smi -L 2>/dev/null || true

"$PY" "$ROOT/scripts/debug_one_math.py" \
  --adapter_dir "${ADAPTER_DIR}" \
  --index "${INDEX}"
