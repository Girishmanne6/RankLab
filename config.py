"""Central configuration for RankLab, loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data" / "processed"))
RAW_DIR = Path(os.getenv("RAW_DIR", ROOT_DIR / "data" / "raw"))
# Local EB-NeRD small extract (train/ + validation/ + articles.parquet).
EBNERD_RAW_PATH = Path(
    os.getenv("EBNERD_RAW_PATH", ROOT_DIR / "data" / "raw" / "ebnerd_small")
)
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT_DIR / "models" / "checkpoints"))

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT_DIR / 'ranklab.db'}")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# "small" = real EB-NeRD small subset, "demo" = bundled synthetic generator
# with an identical schema (keeps the stack fully runnable offline).
EBNERD_VARIANT = os.getenv("EBNERD_VARIANT", "demo")

EPOCHS = int(os.getenv("EPOCHS", "50"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "64"))
DEVICE = os.getenv("DEVICE", "cpu")

SEED = 42
