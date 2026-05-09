from typing import Any

import torch
from torch.utils.data import Dataset

ANSWER_INSTRUCTION = (
    "Always start your assistant message with the exact prefix ANSWER: followed by your answer."
)


def row_to_messages(task: str, row: dict[str, Any], math_target: str) -> list[dict[str, str]]:
    if task == "anli":
        label = row["label"]
        if hasattr(label, "item"):
            label = int(label.item())
        names = ["entailment", "neutral", "contradiction"]
        ans = names[label]
        return [
            {"role": "system", "content": ANSWER_INSTRUCTION + " One word: entailment, neutral, or contradiction."},
            {
                "role": "user",
                "content": f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}",
            },
            {"role": "assistant", "content": f"ANSWER: {ans}"},
        ]
    if task == "wsc":
        lab = row["label"]
        if hasattr(lab, "item"):
            lab = int(lab.item())
        ans = "true" if lab == 1 else "false"
        return [
            {"role": "system", "content": ANSWER_INSTRUCTION + ' Answer with true or false after the prefix.'},
            {
                "role": "user",
                "content": (
                    f"Passage: {row['text']}\n"
                    f"Span1: {row['span1_text']}\nSpan2: {row['span2_text']}\n"
                    f"Does span1 refer to span2?"
                ),
            },
            {"role": "assistant", "content": f"ANSWER: {ans}"},
        ]
    if task == "math":
        sol = row["solution"] or ""
        if math_target == "answer_only":
            sol = sol.strip()
            if "\\boxed{" in sol:
                i = sol.rfind("\\boxed{")
                sol = sol[i : i + 200]
        return [
            {"role": "system", "content": ANSWER_INSTRUCTION + " Put your full solution after the prefix."},
            {"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": f"ANSWER: {sol}"},
        ]
    raise ValueError(task)


def encode_messages(tokenizer, messages: list[dict[str, str]], max_length: int) -> dict[str, torch.Tensor]:
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_enc = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_enc = tokenizer(
        prompt_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = full_enc["input_ids"].squeeze(0)
    attention_mask = full_enc["attention_mask"].squeeze(0)
    prompt_ids = prompt_enc["input_ids"].squeeze(0)
    pl = min(prompt_ids.shape[0], input_ids.shape[0])
    while pl > 0 and not torch.equal(input_ids[:pl], prompt_ids[:pl]):
        pl -= 1
    labels = input_ids.clone()
    labels[:pl] = -100
    labels[attention_mask == 0] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class SFTDataset(Dataset):
    def __init__(self, hf_ds, task: str, tokenizer, max_length: int, math_target: str) -> None:
        self.rows = hf_ds
        self.task = task
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.math_target = math_target

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        messages = row_to_messages(self.task, row, self.math_target)
        return encode_messages(self.tokenizer, messages, self.max_length)
