import csv
import re
from pathlib import Path
from typing import Any

import torch

from evo.sft import row_to_messages


def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


def extract_after_answer(gen: str) -> str:
    """Text after the last 'ANSWER:' prefix (case-insensitive)."""
    matches = list(re.finditer(r"ANSWER:\s*", gen, flags=re.IGNORECASE | re.DOTALL))
    if not matches:
        return ""
    m = matches[-1]
    return gen[m.end() :].strip()


def gold_answer(task: str, row: dict[str, Any]) -> str:
    if task == "anli":
        lab = row["label"]
        if hasattr(lab, "item"):
            lab = int(lab.item())
        return ["entailment", "neutral", "contradiction"][lab]
    if task == "wsc":
        lab = row["label"]
        if hasattr(lab, "item"):
            lab = int(lab.item())
        return "true" if lab == 1 else "false"
    if task == "math":
        return row.get("solution") or ""
    raise ValueError(task)


def eval_accuracy(
    model,
    tokenizer,
    hf_val,
    task: str,
    math_target: str,
    max_new_tokens: int,
    max_examples: int,
) -> float:
    model.eval()
    n = min(max_examples, len(hf_val))
    correct = 0
    device = next(model.parameters()).device
    for i in range(n):
        row = hf_val[i]
        messages = row_to_messages(task, row, math_target)
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
        gold = gold_answer(task, row)
        rest = extract_after_answer(gen)
        if task in ("anli", "wsc"):
            pred = rest.split()[0] if rest else ""
            pred = pred.lower().strip(".,;:!?\"'")
            ok = _norm(pred) == _norm(gold)
        else:
            ok = _norm(rest) == _norm(gold)
        if ok:
            correct += 1
    return correct / max(n, 1)


METRIC_FIELDS = [
    "global_step",
    "phase_index",
    "train_task",
    "eval_task",
    "metric",
    "value",
    "elapsed_s",
]


def append_metrics_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def log_phase_metrics(
    path: Path,
    global_step: int,
    phase_index: int,
    train_task: str,
    metrics: dict[str, float],
    elapsed_s: float,
) -> None:
    for eval_task, value in metrics.items():
        append_metrics_row(
            path,
            {
                "global_step": global_step,
                "phase_index": phase_index,
                "train_task": train_task,
                "eval_task": eval_task,
                "metric": "accuracy",
                "value": value,
                "elapsed_s": round(elapsed_s, 2),
            },
        )


def eval_all_tasks(model, tokenizer, splits: dict, config, eval_cap: int) -> dict[str, float]:
    """Run greedy accuracy on val slice for anli, wsc, math (same caps as post-phase eval)."""
    metrics: dict[str, float] = {}
    for et in ("anli", "wsc", "math"):
        _, val_hf = splits[et]
        mtoks = (
            config.eval_max_new_tokens_anli
            if et == "anli"
            else config.eval_max_new_tokens_wsc
            if et == "wsc"
            else config.eval_max_new_tokens_math
        )
        metrics[et] = eval_accuracy(
            model,
            tokenizer,
            val_hf,
            et,
            config.math_train_target,
            mtoks,
            eval_cap,
        )
    return metrics
