"""GPU eval only: Qwen QLoRA base (no training) on ANLI / WSC / MATH val slices.

Writes metrics.csv with a single baseline block (phase_index=-1) so MATH pre-finetune
accuracy is recorded with the same data split and eval caps as phased training.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import set_seed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo.config import TrainRunConfig
from evo.data import load_all_tasks
from evo.env_setup import load_hf_env
from evo.eval_run import eval_all_tasks, log_phase_metrics
from evo.model import build_tokenizer_and_model, require_hf_token


def main():
    load_hf_env()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/eval_pretrain"),
    )
    ap.add_argument("--max_train_samples_per_task", type=int, default=2000)
    ap.add_argument("--max_val_samples_per_task", type=int, default=256)
    ap.add_argument("--math_subject", default="algebra")
    ap.add_argument("--anli_config", default="r3")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = TrainRunConfig(
        max_train_samples_per_task=args.max_train_samples_per_task,
        max_val_samples_per_task=args.max_val_samples_per_task,
        math_subject=args.math_subject,
        anli_config=args.anli_config,
        seed=args.seed,
    )
    set_seed(cfg.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for eval.")
    token = require_hf_token()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, model = build_tokenizer_and_model(
        cfg.model_id,
        token,
        cfg.lora_r,
        cfg.lora_alpha,
        cfg.lora_dropout,
    )
    splits = load_all_tasks(cfg, cfg.seed)
    for name in ("anli", "wsc", "math"):
        tr, va = splits[name]
        print(f"  data {name}: train_rows={len(tr)} val_rows={len(va)}")
    eval_cap = min(128, cfg.max_val_samples_per_task)
    t0 = time.time()
    baseline = eval_all_tasks(model, tokenizer, splits, cfg, eval_cap)
    log_phase_metrics(
        out / "metrics.csv",
        0,
        -1,
        "baseline",
        baseline,
        time.time() - t0,
    )
    print("pretrain baseline accuracy:", baseline)


if __name__ == "__main__":
    main()
