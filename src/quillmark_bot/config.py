import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

CASES_PATH = ROOT / "cases.jsonl"
HANDBOOK_PATH = ROOT / "policy_handbook.md"
SCHEMA_PATH = ROOT / "db" / "schema.sql"

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3.5-4B")
# The model already quantized to 4-bit, saved locally by `python -m quillmark_bot.model`
QUANTIZED_MODEL_DIR = ROOT / "data" / "models" / "qwen3.5-4b-nf4"

INDEX_DIR = ROOT / "data" / "index"
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "BAAI/bge-base-en-v1.5")
RERANK_MODEL_ID = os.getenv("RERANK_MODEL_ID", "BAAI/bge-reranker-base")


def database_url() -> str:
    user = os.getenv("POSTGRES_USER", "quillmark")
    password = os.getenv("POSTGRES_PASSWORD", "quillmark")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5434")
    db = os.getenv("POSTGRES_DB", "quillmark")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"
