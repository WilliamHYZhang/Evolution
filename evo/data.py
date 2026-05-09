from typing import Literal, Tuple

from datasets import Dataset, load_dataset

TaskName = Literal["anli", "wsc", "math"]


def load_task_splits(
    task: TaskName,
    anli_config: str,
    math_subject: str,
    max_train: int,
    max_val: int,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    if task == "anli":
        ds = load_dataset("anli", "plain_text")
        train_key = f"train_{anli_config}"
        val_key = f"dev_{anli_config}"
        train = ds[train_key].shuffle(seed=seed).select(
            range(min(max_train, len(ds[train_key])))
        )
        val = ds[val_key].shuffle(seed=seed + 1).select(
            range(min(max_val, len(ds[val_key]))))
        return train, val
    if task == "wsc":
        ds = load_dataset("super_glue", "wsc")
        train = ds["train"].shuffle(seed=seed).select(
            range(min(max_train, len(ds["train"])))
        )
        val = ds["validation"].shuffle(seed=seed + 1).select(
            range(min(max_val, len(ds["validation"]))))
        return train, val
    if task == "math":
        ds = load_dataset("EleutherAI/hendrycks_math", math_subject)
        full = ds["train"].shuffle(seed=seed)
        n = len(full)
        nv = min(max_val, n)
        nt = min(max_train, n - nv)
        if nt == 0 and n > 0:
            nv = max(0, n - 1)
            nt = min(max_train, n - nv)
        train = full.select(range(nt))
        if nv == 0 or nt >= n:
            val = full.select([])
        else:
            val = full.select(range(nt, min(nt + nv, n)))
        return train, val
    raise ValueError(task)


def load_all_tasks(config, seed: int):
    out = {}
    for t in ("anli", "wsc", "math"):
        out[t] = load_task_splits(
            t,
            config.anli_config,
            config.math_subject,
            config.max_train_samples_per_task,
            config.max_val_samples_per_task,
            seed,
        )
    return out
