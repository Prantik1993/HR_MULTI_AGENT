from __future__ import annotations
from app.rag.retriever import semantic_search
from app.rag.bm25_search import bm25_search
from app.rag.reranker import rerank
from app.core.logger import get_logger

logger = get_logger(__name__)

NO_CONTEXT = "__NO_CONTEXT__"


def retrieve(query: str, topic: str | None = None, top_k: int = 5) -> list[dict]:
    logger.debug("pipeline.retrieve | query=%r topic=%s top_k=%d", query[:80], topic, top_k)

    semantic = semantic_search(query, 20, topic)
    bm25     = bm25_search(query, 20, topic)
    logger.debug("pipeline.retrieve | semantic=%d bm25=%d", len(semantic), len(bm25))

    seen: set[str] = set()
    merged: list[dict] = []
    for chunk in semantic + bm25:
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            merged.append(chunk)

    ranked = rerank(query, merged, top_k)
    logger.debug("pipeline.retrieve | after rerank=%d", len(ranked))
    return ranked


def format_context(chunks: list[dict]) -> tuple[str, list[str]]:
    if not chunks:
        return NO_CONTEXT, []
    context = (chr(10) * 2).join(c["text"] for c in chunks)
    sources = list(dict.fromkeys(c["metadata"].get("source", "Unknown") for c in chunks))
    return context, sources
