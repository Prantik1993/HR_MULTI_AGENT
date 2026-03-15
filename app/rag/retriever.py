from __future__ import annotations
import chromadb
from app.config import settings
from app.rag.embedder import embed
from app.core.logger import get_logger

logger = get_logger(__name__)

_client: chromadb.PersistentClient | None = None


def get_collection() -> chromadb.Collection:
    global _client
    if _client is None:
        logger.info("retriever | connecting to ChromaDB at %s", settings.chroma_path)
        _client = chromadb.PersistentClient(path=settings.chroma_path)
    return _client.get_or_create_collection(settings.collection_name)


def semantic_search(query: str, n_results: int = 20, topic: str | None = None) -> list[dict]:
    # topic=None means no where filter — searches ALL documents
    where = {"topic": topic} if topic else None
    logger.debug("retriever.semantic_search | n_results=%d where=%s", n_results, where)

    r = get_collection().query(
        query_embeddings=embed([query]),
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    if not r["documents"] or not r["documents"][0]:
        logger.debug("retriever.semantic_search | no results")
        return []

    results = [
        {"text": d, "metadata": m, "score": 1 - s}
        for d, m, s in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]
    logger.debug("retriever.semantic_search | returned %d results", len(results))
    return results
