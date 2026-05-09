import os
import time
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint

from evo.collator import PaddingCollator
from evo.config import TrainRunConfig, Regime
from evo.data import load_all_tasks
from evo.eval_run import eval_all_tasks, log_phase_metrics
from evo.model import build_tokenizer_and_model, require_hf_token
from evo.sft import SFTDataset


class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        ep = logs.get("epoch")
        parts = [f"[step {state.global_step}"]
        if ep is not None:
            parts.append(f"epoch {ep:.2f}")
        if loss is not None:
            parts.append(f"loss {loss:.4f}")
        if lr is not None:
            parts.append(f"lr {lr:.6f}")
        print(" | ".join(parts) + "]")


def phase_task_list(regime: Regime, static_task: str, cycles: int) -> list[str]:
    struct = ["anli", "wsc", "math"]
    unstruct = ["math", "anli", "wsc"]
    n = cycles * 3
    if regime == "static":
        return [static_task] * n
    if regime == "structured_dynamic":
        return (struct * cycles)[:n] if cycles else []
    if regime == "unstructured_dynamic":
        return (unstruct * cycles)[:n] if cycles else []
    if regime == "structured_pair":
        pair = ["anli", "wsc"]
        if not cycles:
            return []
        return (pair * ((n + 1) // 2))[:n]
    if regime == "unstructured_pair":
        pair = ["anli", "math"]
        if not cycles:
            return []
        return (pair * ((n + 1) // 2))[:n]
    raise ValueError(regime)


def run_training(config: TrainRunConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible to PyTorch. On FASRC gpu_h200 this is usually a "
            "torch-vs-driver mismatch: reinstall from Evolution/requirements.txt "
            "(torch pinned like GTOGoblin). See stderr for the NVIDIA driver warning."
        )
    if config.smoke:
        print(
            "smoke: CUDA enabled, device_count=",
            torch.cuda.device_count(),
            "device_0=",
            torch.cuda.get_device_name(0),
        )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    token = require_hf_token()
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer, model = build_tokenizer_and_model(
        config.model_id,
        token,
        config.lora_r,
        config.lora_alpha,
        config.lora_dropout,
    )

    splits = load_all_tasks(config, config.seed)
    for name in ("anli", "wsc", "math"):
        tr, va = splits[name]
        print(f"  data {name}: train_rows={len(tr)} val_rows={len(va)}")
    phases = phase_task_list(config.regime, config.static_task, config.max_cycles)
    if not phases:
        raise ValueError("No phases (max_cycles=0?)")

    metrics_path = out / "metrics.csv"
    cumulative = 0
    t0 = time.time()
    eval_cap = 8 if config.smoke else min(128, config.max_val_samples_per_task)

    baseline = eval_all_tasks(model, tokenizer, splits, config, eval_cap)
    log_phase_metrics(metrics_path, 0, -1, "baseline", baseline, time.time() - t0)
    print("baseline (pre-train) accuracy:", baseline)

    for phase_idx, task in enumerate(phases):
        cumulative += config.phase_steps
        train_hf, _ = splits[task]
        train_ds = SFTDataset(
            train_hf,
            task,
            tokenizer,
            config.max_seq_length,
            config.math_train_target,
        )
        collator = PaddingCollator(tokenizer, pad_to_multiple_of=8)
        training_args = TrainingArguments(
            output_dir=str(out),
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_steps=cumulative,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            logging_steps=5,
            logging_strategy="steps",
            logging_first_step=True,
            save_strategy="steps",
            save_steps=config.phase_steps,
            save_total_limit=3,
            bf16=torch.cuda.is_available(),
            tf32=torch.cuda.is_available(),
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=torch.cuda.is_available(),
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            processing_class=tokenizer,
            data_collator=collator,
            callbacks=[ProgressCallback()],
        )
        resume = get_last_checkpoint(str(out)) if phase_idx > 0 else None
        trainer.train(resume_from_checkpoint=resume)
        gs = int(trainer.state.global_step)

        metrics = eval_all_tasks(model, tokenizer, splits, config, eval_cap)
        log_phase_metrics(
            metrics_path,
            gs,
            phase_idx,
            task,
            metrics,
            time.time() - t0,
        )
        print(f"phase {phase_idx} train={task} step={gs} metrics={metrics}")

        snap = out / "snapshots" / f"phase_{phase_idx:03d}_{task}"
        snap.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(snap)
        tokenizer.save_pretrained(snap)
        print(f"Saved phase snapshot (LoRA + tokenizer) to {snap}")

    adapter_dir = out / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved final adapter to {adapter_dir}")
