import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from app.graph.state import HRState
from app.graph.nodes.intake_node import VALID
from app.ingestion.chunker import chunk_text
from app.cache.query_cache import QueryCache
from api.main import _build_messages


# ── Intake node ───────────────────────────────────────────────────────────────

def test_valid_intents():
    assert VALID == {"policy", "grievance", "talent"}


def test_state_structure():
    state: HRState = {
        "messages": [HumanMessage(content="test")],
        "intent": "policy",
        "answer": "",
        "sources": [],
    }
    assert state["intent"] == "policy"
    assert isinstance(state["messages"], list)
    assert isinstance(state["sources"], list)


def test_history_message_types():
    history = [
        {"role": "user",      "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    msgs = _build_messages(history, "new question")
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)
    assert isinstance(msgs[2], HumanMessage)
    assert msgs[2].content == "new question"


def test_empty_history():
    msgs = _build_messages([], "first question")
    assert len(msgs) == 1
    assert msgs[0].content == "first question"


def test_history_trimmed_to_max():
    from api.main import MAX_HISTORY_TURNS
    long_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(30)
    ]
    msgs = _build_messages(long_history, "final question")
    # trimmed history + 1 for the current query
    assert len(msgs) <= MAX_HISTORY_TURNS + 1


# ── Chunker ───────────────────────────────────────────────────────────────────

def test_chunker_basic():
    words = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(words, chunk_size=10, overlap=2)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_chunker_overlap():
    words = " ".join(f"w{i}" for i in range(20))
    chunks = chunk_text(words, chunk_size=10, overlap=5)
    assert len(chunks) >= 2


def test_chunker_default_size_300():
    """Confirm default chunk_size is 300 (not the old 200)."""
    import inspect
    sig = inspect.signature(chunk_text)
    assert sig.parameters["chunk_size"].default == 300
    assert sig.parameters["overlap"].default == 50


def test_chunker_preserves_long_section():
    """A 250-word section should fit in a single 300-word chunk."""
    section = " ".join([f"word{i}" for i in range(250)])
    chunks = chunk_text(section, chunk_size=300, overlap=50)
    assert len(chunks) == 1


# ── Cache ─────────────────────────────────────────────────────────────────────

def test_cache_hit_miss():
    c = QueryCache()
    assert c.get("hello") is None
    c.set("hello", "world")
    assert c.get("hello") == "world"


def test_cache_case_insensitive():
    c = QueryCache()
    c.set("hello", "world")
    assert c.get("HELLO") == "world"


def test_cache_eviction():
    c = QueryCache(max_size=2)
    c.set("a", "1")
    c.set("b", "2")
    c.set("c", "3")
    assert len(c) == 2


def test_cache_ttl_expiry():
    """Expired entries should not be returned."""
    c = QueryCache(max_size=10, ttl_seconds=0)
    c.set("key", "value")
    import time; time.sleep(0.01)
    assert c.get("key") is None


# ── RAG pipeline ──────────────────────────────────────────────────────────────

def test_format_context_empty_returns_no_context():
    from app.rag.pipeline import format_context, NO_CONTEXT
    ctx, sources = format_context([])
    assert ctx == NO_CONTEXT
    assert sources == []


def test_format_context_deduplicates_sources():
    from app.rag.pipeline import format_context
    chunks = [
        {"text": "chunk one",   "metadata": {"source": "policy.pdf"}},
        {"text": "chunk two",   "metadata": {"source": "policy.pdf"}},
        {"text": "chunk three", "metadata": {"source": "handbook.pdf"}},
    ]
    _, sources = format_context(chunks)
    assert sources == ["policy.pdf", "handbook.pdf"]
    assert len(sources) == 2


def test_format_context_joins_chunks():
    from app.rag.pipeline import format_context
    chunks = [
        {"text": "first chunk",  "metadata": {"source": "a.pdf"}},
        {"text": "second chunk", "metadata": {"source": "b.pdf"}},
    ]
    ctx, _ = format_context(chunks)
    assert "first chunk" in ctx
    assert "second chunk" in ctx


def test_no_context_sentinel_value():
    from app.rag.pipeline import NO_CONTEXT
    assert NO_CONTEXT == "__NO_CONTEXT__"


# ── Chunk quality filter ──────────────────────────────────────────────────────

def test_valid_chunk_passes():
    from app.ingestion.ingest import _is_valid_chunk
    good = " ".join(["This is a normal sentence with real HR policy words."] * 5)
    assert _is_valid_chunk(good) is True


def test_short_chunk_rejected():
    from app.ingestion.ingest import _is_valid_chunk
    assert _is_valid_chunk("too short") is False


def test_numeric_heavy_chunk_rejected():
    from app.ingestion.ingest import _is_valid_chunk
    noisy = "1234 5678 9012 3456 7890 " * 10
    assert _is_valid_chunk(noisy) is False


def test_noise_heavy_chunk_rejected():
    from app.ingestion.ingest import _is_valid_chunk
    noisy = "!@#$ %^&* ()_+ " * 20
    assert _is_valid_chunk(noisy) is False


# ── Supervisor routing ────────────────────────────────────────────────────────

def test_route_after_intake_greeting_goes_to_end():
    from app.graph.supervisor import _route_after_intake
    from langgraph.graph import END
    state: HRState = {
        "messages": [HumanMessage(content="hi")],
        "intent": "greeting",
        "answer": "Hello!",
        "sources": [],
    }
    assert _route_after_intake(state) == END


def test_route_after_intake_offtopic_goes_to_end():
    from app.graph.supervisor import _route_after_intake
    from langgraph.graph import END
    state: HRState = {
        "messages": [HumanMessage(content="what is the weather?")],
        "intent": "offtopic",
        "answer": "I only handle HR topics.",
        "sources": [],
    }
    assert _route_after_intake(state) == END


def test_route_after_intake_policy_goes_to_rewriter():
    from app.graph.supervisor import _route_after_intake
    state: HRState = {
        "messages": [HumanMessage(content="leave policy")],
        "intent": "policy",
        "answer": "",
        "sources": [],
    }
    assert _route_after_intake(state) == "query_rewriter"


def test_route_or_fallback_no_context_goes_to_fallback():
    from app.graph.supervisor import _route_or_fallback
    from app.rag.pipeline import NO_CONTEXT
    state: HRState = {
        "messages": [HumanMessage(content="test")],
        "intent": "policy",
        "answer": NO_CONTEXT,
        "sources": [],
    }
    assert _route_or_fallback(state) == "fallback"


def test_route_or_fallback_with_answer_goes_to_end():
    from app.graph.supervisor import _route_or_fallback
    from langgraph.graph import END
    state: HRState = {
        "messages": [HumanMessage(content="test")],
        "intent": "policy",
        "answer": "You get 21 days annual leave.",
        "sources": ["policy.pdf"],
    }
    assert _route_or_fallback(state) == END