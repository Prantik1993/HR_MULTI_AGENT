from app.ingestion.chunker import chunk_text
from app.cache.query_cache import QueryCache


def test_chunker_basic():
    words = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(words, chunk_size=10, overlap=2)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_chunker_overlap():
    words = " ".join(f"w{i}" for i in range(20))
    chunks = chunk_text(words, chunk_size=10, overlap=5)
    assert len(chunks) >= 2


def test_cache_hit_miss():
    c = QueryCache()
    assert c.get("hello") is None
    c.set("hello", "world")
    assert c.get("hello") == "world"
    assert c.get("HELLO") == "world"


def test_cache_eviction():
    c = QueryCache(max_size=2)
    c.set("a", "1")
    c.set("b", "2")
    c.set("c", "3")
    assert len(c) == 2
