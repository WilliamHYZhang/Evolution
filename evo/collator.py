import torch
import torch.nn.functional as F
from transformers import DataCollatorWithPadding


class PaddingCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.pad_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    def __call__(self, features):
        labels = [feat.pop("labels") for feat in features]
        batch = self.pad_collator(features)
        max_len = batch["input_ids"].shape[1]
        padded = []
        for lab in labels:
            if lab.numel() < max_len:
                lab = F.pad(lab, (0, max_len - lab.numel()), value=-100)
            else:
                lab = lab[:max_len]
            padded.append(lab)
        batch["labels"] = torch.stack(padded)
        return batch
