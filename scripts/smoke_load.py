"""One forward pass: tokenizer + 4-bit Qwen + LoRA."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from evo.env_setup import load_hf_env
from evo.model import build_tokenizer_and_model, require_hf_token


def main() -> None:
    load_hf_env()
    token = require_hf_token()
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch.cuda.get_device_name(0) =", torch.cuda.get_device_name(0))
    else:
        raise RuntimeError("smoke_load expects a GPU; CUDA is not available to PyTorch.")
    tok, model = build_tokenizer_and_model(
        "Qwen/Qwen3.5-0.8B",
        token,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    device = next(model.parameters()).device
    print("model primary device =", device)
    ids = torch.tensor([[1, 2, 3, 4]], device=device, dtype=torch.long)
    att = torch.ones_like(ids)
    out = model(input_ids=ids, attention_mask=att, labels=ids)
    print("smoke_load ok, loss=", float(out.loss.detach()))


if __name__ == "__main__":
    main()
