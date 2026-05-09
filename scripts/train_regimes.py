"""Phased QLoRA training for static / structured / unstructured regimes.

Regimes (max_cycles K ⇒ n = 3K phases, each phase_steps S, cumulative max_steps):
  static — one task (static_task) for all n phases.
  structured_dynamic — (anli,wsc,math) repeated K times (n=3K).
  unstructured_dynamic — (math,anli,wsc) repeated K times.
  structured_pair — (anli,wsc) alternating for n phases (two related NLU tasks).
  unstructured_pair — (anli,math) alternating for n phases (NLU vs symbolic math).

Smoke (--smoke): tiny data, few steps. Writes metrics.csv with phase_index=-1
baseline accuracies (pre-train) then one row block per phase after training.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo.config import TrainRunConfig
from evo.env_setup import load_hf_env
from evo.train_run import run_training


def main() -> None:
    load_hf_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/run"))
    ap.add_argument(
        "--regime",
        choices=[
            "static",
            "structured_dynamic",
            "unstructured_dynamic",
            "structured_pair",
            "unstructured_pair",
        ],
        default="structured_dynamic",
    )
    ap.add_argument("--static_task", choices=["anli", "wsc", "math"], default="anli")
    ap.add_argument("--anli_config", default="r3")
    ap.add_argument("--math_subject", default="algebra")
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--max_train_samples_per_task", type=int, default=2000)
    ap.add_argument("--max_val_samples_per_task", type=int, default=256)
    ap.add_argument("--phase_steps", type=int, default=200)
    ap.add_argument("--max_cycles", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = TrainRunConfig(
        output_dir=args.output_dir,
        regime=args.regime,
        static_task=args.static_task,
        anli_config=args.anli_config,
        math_subject=args.math_subject,
        max_seq_length=args.max_seq_length,
        max_train_samples_per_task=args.max_train_samples_per_task,
        max_val_samples_per_task=args.max_val_samples_per_task,
        phase_steps=args.phase_steps,
        max_cycles=args.max_cycles,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        smoke=args.smoke,
    )
    if args.smoke:
        cfg.smoke = True
        cfg.max_train_samples_per_task = 16
        cfg.max_val_samples_per_task = 8
        cfg.phase_steps = 2
        cfg.max_cycles = 1
        cfg.max_seq_length = 512
        cfg.batch_size = 1
        cfg.gradient_accumulation_steps = 1
        cfg.warmup_steps = 0
        cfg.eval_max_new_tokens_math = 64

    run_training(cfg)


if __name__ == "__main__":
    main()
