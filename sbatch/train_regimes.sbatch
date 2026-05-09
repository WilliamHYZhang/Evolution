#!/bin/bash
#SBATCH --job-name=evo_train
#SBATCH --output=logs/train_regimes_%j.out
#SBATCH --error=logs/train_regimes_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --partition=gpu_h200
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Phased QLoRA (default): max_cycles=3 × structured_dynamic ⇒ 9 phases × 200 steps
# (=1800 optimizer steps) plus baseline + 9 full eval rounds. Recent runs finished in
# ~1h; 3h is a comfortable default. Override if needed, e.g. sbatch --time=10:00:00 ...
#
# Optional environment (exported before sbatch, or use #SBATCH cannot read env):
#   REGIME            structured_dynamic | unstructured_dynamic | static |
#                     structured_pair | unstructured_pair
#   OUT               output directory (default outputs/run_${SLURM_JOB_ID})
#   STATIC_TASK       anli | wsc | math — only matters for REGIME=static
#   SEED              default 42
#   BATCH_SIZE        default 4
#   GRAD_ACCUM        gradient accumulation steps, default 2
#   HF_CACHE_ON_TMP   set to 1 to use SLURM_TMPDIR for HF_HOME (node-local cache)

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
  echo "Missing $ROOT/.env (HF_TOKEN required for model/tokenizer download)."
  exit 1
fi

REGIME="${REGIME:-structured_dynamic}"
OUT="${OUT:-outputs/run_${SLURM_JOB_ID}}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

echo "=== Evolution train_regimes ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "REGIME=${REGIME} OUT=${OUT}"
echo "SEED=${SEED} BATCH_SIZE=${BATCH_SIZE} GRAD_ACCUM=${GRAD_ACCUM}"
date
nvidia-smi -L 2>/dev/null || true

if [[ -n "${SLURM_TMPDIR:-}" && "${HF_CACHE_ON_TMP:-0}" == "1" ]]; then
  export HF_HOME="${SLURM_TMPDIR}/hf_home"
  export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
  mkdir -p "$TRANSFORMERS_CACHE"
  echo "HF_CACHE_ON_TMP=1 HF_HOME=${HF_HOME}"
fi

ARGS=(
  "$ROOT/scripts/train_regimes.py"
  --regime "${REGIME}"
  --output_dir "${OUT}"
  --seed "${SEED}"
  --phase_steps 200
  --max_cycles 3
  --max_train_samples_per_task 2000
  --max_val_samples_per_task 256
  --max_seq_length 2048
  --batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
)
if [[ -n "${STATIC_TASK:-}" ]]; then
  ARGS+=(--static_task "${STATIC_TASK}")
fi

"$PY" "${ARGS[@]}"
