#!/bin/bash
#SBATCH --job-name=evo_static_anli
#SBATCH --output=logs/train_static_anli_%j.out
#SBATCH --error=logs/train_static_anli_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --partition=gpu_h200
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Static baseline: ANLI only for every phase. Same budget as train_regimes.sbatch:
# max_cycles=3 ⇒ n=9 phases × phase_steps=200 ⇒ 1800 optimizer steps + baseline + 9 evals.

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

OUT="${OUT:-outputs/static_anli_${SLURM_JOB_ID}}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

echo "=== Evolution static ANLI (matched compute to 3-cycle triplet) ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-} OUT=${OUT}"
echo "SEED=${SEED} BATCH_SIZE=${BATCH_SIZE} GRAD_ACCUM=${GRAD_ACCUM}"
date
nvidia-smi -L 2>/dev/null || true

if [[ -n "${SLURM_TMPDIR:-}" && "${HF_CACHE_ON_TMP:-0}" == "1" ]]; then
  export HF_HOME="${SLURM_TMPDIR}/hf_home"
  export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
  mkdir -p "$TRANSFORMERS_CACHE"
  echo "HF_CACHE_ON_TMP=1 HF_HOME=${HF_HOME}"
fi

"$PY" "$ROOT/scripts/train_regimes.py" \
  --regime static \
  --static_task anli \
  --output_dir "${OUT}" \
  --seed "${SEED}" \
  --phase_steps 200 \
  --max_cycles 3 \
  --max_train_samples_per_task 2000 \
  --max_val_samples_per_task 256 \
  --max_seq_length 2048 \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}"
