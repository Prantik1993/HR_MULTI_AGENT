from __future__ import annotations
from sentence_transformers import SentenceTransformer
from app.core.logger import get_logger

logger = get_logger(__name__)

_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("embedder | loading all-MiniLM-L6-v2")
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("embedder | model loaded")
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    logger.debug("embedder | encoding %d texts", len(texts))
    return get_embedder().encode(texts, convert_to_numpy=True).tolist()
