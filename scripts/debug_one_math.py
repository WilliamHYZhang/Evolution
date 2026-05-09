import argparse
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo.config import TrainRunConfig
from evo.data import load_all_tasks
from evo.env_setup import load_hf_env
from evo.eval_run import _norm, extract_after_answer
from evo.model import require_hf_token
from evo.sft import row_to_messages


def last_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return matches[-1].strip() if matches else ""


def main() -> None:
    load_hf_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", type=Path, default=Path("outputs/run_5825378/adapter"))
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = TrainRunConfig(seed=args.seed)
    set_seed(cfg.seed)
    token = require_hf_token()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, token=token)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        token=token,
        device_map="auto",
        quantization_config=quant,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    splits = load_all_tasks(cfg, cfg.seed)
    _, math_val = splits["math"]
    row = math_val[args.index]
    messages = row_to_messages("math", row, cfg.math_train_target)
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=cfg.eval_max_new_tokens_math,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen = tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
    rest = extract_after_answer(gen)
    gold = row["solution"] or ""

    print("ADAPTER:", args.adapter_dir)
    print("VAL_INDEX:", args.index)
    print("\nPROMPT:\n", prompt)
    print("\nGENERATED:\n", gen)
    print("\nEXTRACT_AFTER_ANSWER:\n", rest)
    print("\nGOLD_SOLUTION:\n", gold)
    print("\nPRED_BOXED:", last_boxed(rest) or last_boxed(gen))
    print("GOLD_BOXED:", last_boxed(gold))
    print("FULL_SOLUTION_EXACT_MATCH:", _norm(rest) == _norm(gold))
    print("BOXED_EXACT_MATCH:", _norm(last_boxed(rest) or last_boxed(gen)) == _norm(last_boxed(gold)))


if __name__ == "__main__":
    main()
