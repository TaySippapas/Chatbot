"""Find the policies, incidents and past cases relevant to a question.

Knowledge (32 policies, 5 incidents): vector search, plus any policy or incident ID named in the
question. On test questions, vector search alone put the right documents in its top 2 every time;
BM25 matched common words ("charged", "statement") and the cross-encoder didn't improve the order.

Past cases: vector search and BM25 keyword search (exact IDs and amounts such as INC-2026-03 or $29),
merged with reciprocal rank fusion, re-scored by a cross-encoder on the GPU, then collapsed to one case
per scenario: many cases share the same agent-note template, and without collapsing the top 5 would
often tell the same story five times.
"""
import json
import re
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from quillmark_bot.config import EMBED_MODEL_ID, INDEX_DIR, RERANK_MODEL_ID
from quillmark_bot.rag.documents import case_documents, knowledge_documents

# bge v1.5 embeds short queries with this instruction; documents are embedded without it
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# Keeps IDs (pol-dsp-04, inc-2026-03, qm-c-000123) and amounts (29.00, 1,500) as single tokens
TOKEN = re.compile(r"[a-z]+-[a-z0-9]+-\d+|\d[\d,.]*\d|\w+")
KNOWLEDGE_ID = re.compile(r"\b(?:POL-[A-Z]{3}-\d{2}|INC-\d{4}-\d{2})\b", re.IGNORECASE)
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass
class Hit:
    id: str
    score: float  # knowledge: cosine similarity; cases: cross-encoder relevance
    text: str     # what the model is shown


@dataclass
class Retrieval:
    knowledge: list[Hit] = field(default_factory=list)
    cases: list[Hit] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ids(self) -> list[str]:
        return [h.id for h in self.knowledge + self.cases]


class Retriever:
    def __init__(self, device: str = "cuda"):
        half = {"torch_dtype": torch.float16}
        self.embedder = SentenceTransformer(EMBED_MODEL_ID, device=device, model_kwargs=half)
        self.reranker = CrossEncoder(RERANK_MODEL_ID, device=device, model_kwargs=half)

        # (search text, display text, template id) per document id
        self.docs = {
            "knowledge": {d["id"]: (d["text"], d["text"], None) for d in knowledge_documents()},
            "cases": {d["id"]: (d["search_text"], d["display_text"], d["metadata"]["template_id"])
                      for d in case_documents()},
        }
        self.ids, self.embeddings, self.bm25 = {}, {}, {}
        for name, docs in self.docs.items():
            ids_path = INDEX_DIR / f"{name}.ids.json"
            if not ids_path.exists():
                raise FileNotFoundError(f"No index at {INDEX_DIR}; run: python -m quillmark_bot.rag.index")
            self.ids[name] = json.loads(ids_path.read_text(encoding="utf-8"))
            if self.ids[name] != list(docs):
                raise RuntimeError(f"The {name} index is out of date; run: python -m quillmark_bot.rag.index")
            self.embeddings[name] = np.load(INDEX_DIR / f"{name}.npy")
        self.bm25["cases"] = BM25Okapi([tokenize(self.docs["cases"][i][0]) for i in self.ids["cases"]])

    def _candidates(self, name: str, query: str, query_embedding: np.ndarray, n: int) -> list[str]:
        """Vector and keyword results merged by reciprocal rank fusion, best first."""
        ids = self.ids[name]
        n = min(n, len(ids))
        similarity = self.embeddings[name] @ query_embedding  # cosine: both sides are normalized
        vector_ids = [ids[i] for i in similarity.argsort()[::-1][:n]]
        scores = self.bm25[name].get_scores(tokenize(query))
        keyword_ids = [ids[i] for i in scores.argsort()[::-1][:n] if scores[i] > 0]
        fused = {}
        for ranking in (vector_ids, keyword_ids):
            for rank, doc_id in enumerate(ranking):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1 / (RRF_K + rank)
        return sorted(fused, key=fused.get, reverse=True)[:n]

    def _knowledge(self, query: str, query_embedding: np.ndarray, n: int) -> list[tuple[str, float]]:
        """IDs named in the question first, then the rest by cosine similarity."""
        ids = self.ids["knowledge"]
        similarity = self.embeddings["knowledge"] @ query_embedding
        named = [m.upper() for m in KNOWLEDGE_ID.findall(query) if m.upper() in self.docs["knowledge"]]
        ranked = list(dict.fromkeys(named + [ids[i] for i in similarity.argsort()[::-1]]))[:n]
        return [(doc_id, float(similarity[ids.index(doc_id)])) for doc_id in ranked]

    def _rerank(self, name: str, query: str, ids: list[str]) -> list[tuple[str, float]]:
        scores = self.reranker.predict([(query, self.docs[name][i][0]) for i in ids], batch_size=32)
        return sorted(zip(ids, scores.tolist()), key=lambda pair: pair[1], reverse=True)

    def search(self, query: str, n_knowledge: int = 4, n_cases: int = 5, candidates: int = 100) -> Retrieval:
        start = time.perf_counter()
        query_embedding = self.embedder.encode(QUERY_INSTRUCTION + query, normalize_embeddings=True)

        knowledge = self._knowledge(query, query_embedding, n_knowledge)
        cases = self._rerank("cases", query, self._candidates("cases", query, query_embedding, candidates))

        seen_templates, distinct_cases = set(), []
        for case_id, score in cases:  # best-scored case per scenario
            template = self.docs["cases"][case_id][2]
            if template not in seen_templates:
                seen_templates.add(template)
                distinct_cases.append((case_id, score))

        return Retrieval(
            knowledge=[Hit(i, s, self.docs["knowledge"][i][1]) for i, s in knowledge[:n_knowledge]],
            cases=[Hit(i, s, self.docs["cases"][i][1]) for i, s in distinct_cases[:n_cases]],
            seconds=time.perf_counter() - start,
        )
