from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Regime = Literal[
    "static",
    "structured_dynamic",
    "unstructured_dynamic",
    "structured_pair",
    "unstructured_pair",
]
TaskName = Literal["anli", "wsc", "math"]


@dataclass
class TrainRunConfig:
    model_id: str = "Qwen/Qwen3.5-0.8B"
    output_dir: Path = Path("outputs/run")
    regime: Regime = "structured_dynamic"
    static_task: TaskName = "anli"
    anli_config: str = "r3"
    math_subject: str = "algebra"
    max_seq_length: int = 2048
    max_train_samples_per_task: int = 2000
    max_val_samples_per_task: int = 256
    phase_steps: int = 200
    max_cycles: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_steps: int = 20
    weight_decay: float = 0.01
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 42
    math_train_target: Literal["solution", "answer_only"] = "solution"
    eval_max_new_tokens_anli: int = 8
    eval_max_new_tokens_wsc: int = 8
    eval_max_new_tokens_math: int = 256
    smoke: bool = False
