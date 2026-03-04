from __future__ import annotations
import chromadb
from app.config import settings
from app.rag.embedder import embed

_client: chromadb.PersistentClient | None = None


def get_collection() -> chromadb.Collection:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_path)
    return _client.get_or_create_collection(settings.collection_name)


def semantic_search(query: str, n_results: int = 20, topic: str | None = None) -> list[dict]:
    where = {"topic": topic} if topic else None
    r = get_collection().query(
        query_embeddings=embed([query]),
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    if not r["documents"] or not r["documents"][0]:
        return []
    return [
        {"text": d, "metadata": m, "score": 1 - s}
        for d, m, s in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]
