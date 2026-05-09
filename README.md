# Evolution

Small research codebase for testing how LLM fine-tuning behaves across static and changing task environments.

The code trains QLoRA adapters on phased task schedules using ANLI, WSC, and MATH data, then logs accuracy after each phase.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` with your Hugging Face token:

```bash
HF_TOKEN=your_token_here
```

## Common Commands

Smoke test:

```bash
python scripts/train_regimes.py --smoke --output_dir outputs/smoke
```

Train a structured dynamic run:

```bash
python scripts/train_regimes.py --regime structured_dynamic --output_dir outputs/run
```

Evaluate the pretrained baseline:

```bash
python scripts/eval_pretrain.py --output_dir outputs/eval_pretrain
```

SLURM job files are in `sbatch/`.
