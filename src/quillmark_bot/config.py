import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

CASES_PATH = ROOT / "cases.jsonl"
HANDBOOK_PATH = ROOT / "policy_handbook.md"
SCHEMA_PATH = ROOT / "db" / "schema.sql"


def database_url() -> str:
    user = os.getenv("POSTGRES_USER", "quillmark")
    password = os.getenv("POSTGRES_PASSWORD", "quillmark")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5434")
    db = os.getenv("POSTGRES_DB", "quillmark")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"
