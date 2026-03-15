from __future__ import annotations
from pathlib import Path
import re
import chromadb
from app.config import settings
from app.ingestion.loader import load_directory
from app.ingestion.chunker import chunk_text
from app.rag.embedder import embed
from app.core.logger import get_logger

logger = get_logger(__name__)

MIN_WORDS = 30
_NOISE_RE = re.compile(r"[^a-zA-Z0-9 ]")


def _is_valid_chunk(text: str) -> bool:
    s = text.strip()
    if len(s.split()) < MIN_WORDS:
        return False
    if s.count(".") / max(len(s), 1) > 0.3:
        return False
    if len(_NOISE_RE.findall(s)) / max(len(s), 1) > 0.35:
        return False
    if sum(c.isdigit() for c in s) / max(len(s), 1) > 0.4:
        return False
    return True


def ingest(docs_dir: str = "./data/docs") -> None:
    logger.info("ingest | starting — docs_dir=%s", docs_dir)
    client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = client.get_or_create_collection(settings.collection_name)

    docs = load_directory(Path(docs_dir))
    if not docs:
        logger.warning("ingest | no documents found in %s", docs_dir)
        print(f"No documents found in {docs_dir}")
        return

    logger.info("ingest | found %d documents", len(docs))
    chunks, metas, ids, skipped = [], [], [], 0

    for i, (text, fp) in enumerate(docs):
        logger.debug("ingest | processing %s", fp.name)
        for j, chunk in enumerate(chunk_text(text)):
            if _is_valid_chunk(chunk):
                chunks.append(chunk)
                # FIX [05]: no 'topic' field — source is the only metadata
                metas.append({"source": fp.name, "chunk_index": j})
                ids.append(f"{fp.stem}_{i}_{j}")
            else:
                skipped += 1

    if not chunks:
        logger.error("ingest | no valid chunks produced — check document quality")
        print("No valid chunks found.")
        return

    logger.info("ingest | embedding %d chunks (%d skipped)", len(chunks), skipped)
    print(f"Embedding {len(chunks)} chunks from {len(docs)} files ({skipped} garbage skipped)...")
    collection.upsert(documents=chunks, embeddings=embed(chunks), metadatas=metas, ids=ids)
    logger.info("ingest | done — %d chunks stored in collection '%s'", len(chunks), settings.collection_name)
    print(f"Done. {len(chunks)} chunks stored.")


if __name__ == "__main__":
    ingest()
