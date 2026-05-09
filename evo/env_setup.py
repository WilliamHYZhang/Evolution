"""Load HF_TOKEN from Evolution/.env only (standalone project)."""

from pathlib import Path

from dotenv import load_dotenv

EVOLUTION_ROOT = Path(__file__).resolve().parents[1]


def load_hf_env() -> None:
    load_dotenv(EVOLUTION_ROOT / ".env")
