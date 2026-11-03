"""Build the RAG vector index: normalized embeddings of the knowledge documents and past cases.

5,037 vectors fit in a NumPy array and are searched exactly with one matrix product, so there is no
vector database to run. Each collection is saved as <name>.npy plus <name>.ids.json in INDEX_DIR.

Usage: python -m quillmark_bot.rag.index
"""
import json
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from quillmark_bot.config import EMBED_MODEL_ID, INDEX_DIR
from quillmark_bot.rag.documents import case_documents, knowledge_documents


def build_index() -> None:
    embedder = SentenceTransformer(EMBED_MODEL_ID, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    collections = {
        "knowledge": [(d["id"], d["text"]) for d in knowledge_documents()],
        "cases": [(d["id"], d["search_text"]) for d in case_documents()],
    }
    for name, docs in collections.items():
        start = time.perf_counter()
        ids, texts = zip(*docs)
        embeddings = embedder.encode(list(texts), batch_size=64, normalize_embeddings=True)
        np.save(INDEX_DIR / f"{name}.npy", embeddings.astype(np.float32))
        (INDEX_DIR / f"{name}.ids.json").write_text(json.dumps(ids), encoding="utf-8")
        print(f"{name:<10} {len(ids):>5,} documents indexed in {time.perf_counter() - start:.1f}s")
    print(f"Index written to {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
