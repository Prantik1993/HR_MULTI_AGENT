from __future__ import annotations
from pathlib import Path
import re
import chromadb
from app.config import settings
from app.ingestion.loader import load_directory
from app.ingestion.chunker import chunk_text
from app.rag.embedder import embed

TOPIC_MAP = {
    "leave": "policy", "handbook": "policy", "benefit": "policy", "policy": "policy",
    "grievance": "grievance", "complaint": "grievance", "dispute": "grievance",
    "talent": "talent", "hiring": "talent", "job": "talent", "recruit": "talent",
}

MIN_WORDS = 30
_NOISE_RE = re.compile(r"[^a-zA-Z0-9 ]")


def _guess_topic(name: str) -> str:
    n = name.lower()
    return next((t for k, t in TOPIC_MAP.items() if k in n), "general")


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
    client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = client.get_or_create_collection(settings.collection_name)
    docs = load_directory(Path(docs_dir))
    if not docs:
        print(f"No documents found in {docs_dir}")
        return
    chunks, metas, ids, skipped = [], [], [], 0
    for i, (text, fp) in enumerate(docs):
        topic = _guess_topic(fp.name)
        for j, chunk in enumerate(chunk_text(text)):
            if _is_valid_chunk(chunk):
                chunks.append(chunk)
                metas.append({"source": fp.name, "topic": topic, "chunk_index": j})
                ids.append(f"{fp.stem}_{i}_{j}")
            else:
                skipped += 1
    if not chunks:
        print("No valid chunks found. Check your documents.")
        return
    print(f"Embedding {len(chunks)} chunks from {len(docs)} files ({skipped} garbage skipped)...")
    collection.upsert(documents=chunks, embeddings=embed(chunks), metadatas=metas, ids=ids)
    print(f"Done. {len(chunks)} chunks stored.")


if __name__ == "__main__":
    ingest()
