from __future__ import annotations
from rank_bm25 import BM25Okapi
from app.rag.retriever import get_collection
from app.core.logger import get_logger

logger = get_logger(__name__)


def bm25_search(query: str, n_results: int = 20, topic: str | None = None) -> list[dict]:
    # topic=None means no where filter — searches ALL documents
    where = {"topic": topic} if topic else None
    logger.debug("bm25_search | n_results=%d where=%s", n_results, where)

    all_docs = get_collection().get(where=where, include=["documents", "metadatas"])
    if not all_docs["documents"]:
        logger.debug("bm25_search | collection empty or no matching docs")
        return []

    tokenized = [d.lower().split() for d in all_docs["documents"]]
    scores = BM25Okapi(tokenized).get_scores(query.lower().split())
    ranked = sorted(
        zip(all_docs["documents"], all_docs["metadatas"], scores),
        key=lambda x: x[2],
        reverse=True,
    )
    results = [{"text": d, "metadata": m, "score": float(s)} for d, m, s in ranked[:n_results]]
    logger.debug("bm25_search | returned %d results", len(results))
    return results
