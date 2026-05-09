#!/bin/bash
#SBATCH --job-name=evo_smoke_load
#SBATCH --output=logs/smoke_load_%j.out
#SBATCH --error=logs/smoke_load_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --partition=gpu_h200
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
ROOT=/n/home08/whyz/Evolution
cd "$ROOT"
mkdir -p logs
module load python/3.12.11-fasrc01
PY="$ROOT/venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing $PY — run: bash $ROOT/scripts/bootstrap_venv.sh"
  exit 1
fi
export PYTHONPATH="$ROOT"
"$PY" "$ROOT/scripts/smoke_load.py"
