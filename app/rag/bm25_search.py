from __future__ import annotations
from rank_bm25 import BM25Okapi
from app.rag.retriever import get_collection


def bm25_search(query: str, n_results: int = 20, topic: str | None = None) -> list[dict]:
    where = {"topic": topic} if topic else None
    all_docs = get_collection().get(where=where, include=["documents", "metadatas"])
    if not all_docs["documents"]:
        return []
    tokenized = [d.lower().split() for d in all_docs["documents"]]
    scores = BM25Okapi(tokenized).get_scores(query.lower().split())
    ranked = sorted(
        zip(all_docs["documents"], all_docs["metadatas"], scores),
        key=lambda x: x[2], reverse=True,
    )
    return [{"text": d, "metadata": m, "score": float(s)} for d, m, s in ranked[:n_results]]
