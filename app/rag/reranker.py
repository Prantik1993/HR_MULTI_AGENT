from __future__ import annotations
from sentence_transformers import CrossEncoder
from app.core.logger import get_logger

logger = get_logger(__name__)

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("reranker | loading cross-encoder/ms-marco-MiniLM-L-6-v2")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("reranker | model loaded")
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        logger.debug("reranker | no candidates to rerank")
        return []
    logger.debug("reranker | scoring %d candidates top_k=%d", len(candidates), top_k)
    scores = get_reranker().predict([(query, c["text"]) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    results = [c for c, _ in ranked[:top_k]]
    logger.debug("reranker | top scores: %s", [round(float(s), 3) for _, s in ranked[:top_k]])
    return results
