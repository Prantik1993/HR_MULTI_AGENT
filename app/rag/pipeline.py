from __future__ import annotations
from app.rag.retriever import semantic_search
from app.rag.bm25_search import bm25_search
from app.rag.reranker import rerank

NO_CONTEXT = "__NO_CONTEXT__"


def retrieve(query: str, topic: str | None = None, top_k: int = 5) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []
    for chunk in semantic_search(query, 20, topic) + bm25_search(query, 20, topic):
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            merged.append(chunk)
    return rerank(query, merged, top_k)


def format_context(chunks: list[dict]) -> tuple[str, list[str]]:
    if not chunks:
        return NO_CONTEXT, []
    context = (chr(10) * 2).join(c["text"] for c in chunks)
    sources = list(dict.fromkeys(c["metadata"].get("source", "Unknown") for c in chunks))
    return context, sources
