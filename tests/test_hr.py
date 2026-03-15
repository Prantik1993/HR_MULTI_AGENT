import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from app.graph.state import HRState
from app.graph.nodes.intake_node import VALID
from app.ingestion.chunker import chunk_text
from app.cache.query_cache import QueryCache
from app.config import settings
from api.main import _build_messages


# ── Intake node ───────────────────────────────────────────────────────────────

def test_valid_intents():
    assert VALID == {"policy", "grievance", "talent"}


def test_state_has_autonomous_fields():
    state: HRState = {
        "messages": [HumanMessage(content="test")],
        "intent": "policy",
        "answer": "",
        "sources": [],
        "sub_tasks": [],
        "retry_count": 0,
        "critic_score": 0.0,
        "memory": [],
    }
    assert state["sub_tasks"] == []
    assert state["retry_count"] == 0
    assert state["critic_score"] == 0.0
    assert state["memory"] == []


def test_history_message_types():
    history = [
        {"role": "user", "content": "hello"},
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


# ── Planner node ──────────────────────────────────────────────────────────────

def test_planner_splits_multi_task_question():
    from app.graph.nodes.planner_node import planner_node

    state: HRState = {
        "messages": [HumanMessage(content="check cataract bill 25000 and leave policy")],
        "intent": "policy",
        "answer": "",
        "sources": [],
        "sub_tasks": [],
        "retry_count": 0,
        "critic_score": 0.0,
        "memory": [],
    }

    mock_response = MagicMock()
    mock_response.content = "medical reimbursement approval policy\nleave approval policy"

    with patch("app.graph.nodes.planner_node._get_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm
        result = planner_node(state)

    assert len(result["sub_tasks"]) == 2
    assert "medical reimbursement approval policy" in result["sub_tasks"]
    assert "leave approval policy" in result["sub_tasks"]


def test_planner_single_task_question():
    from app.graph.nodes.planner_node import planner_node

    state: HRState = {
        "messages": [HumanMessage(content="how many days annual leave do I get")],
        "intent": "policy",
        "answer": "",
        "sources": [],
        "sub_tasks": [],
        "retry_count": 0,
        "critic_score": 0.0,
        "memory": [],
    }

    mock_response = MagicMock()
    mock_response.content = "annual leave entitlement policy"

    with patch("app.graph.nodes.planner_node._get_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm
        result = planner_node(state)

    assert len(result["sub_tasks"]) == 1


def test_planner_max_4_subtasks():
    from app.graph.nodes.planner_node import planner_node

    state: HRState = {
        "messages": [HumanMessage(content="complex multi part question")],
        "intent": "policy",
        "answer": "",
        "sources": [],
        "sub_tasks": [],
        "retry_count": 0,
        "critic_score": 0.0,
        "memory": [],
    }

    mock_response = MagicMock()
    mock_response.content = "query1\nquery2\nquery3\nquery4\nquery5\nquery6"

    with patch("app.graph.nodes.planner_node._get_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm
        result = planner_node(state)

    assert len(result["sub_tasks"]) <= 4


# ── Critic node ───────────────────────────────────────────────────────────────

def test_critic_scores_good_answer():
    from app.graph.nodes.critic_node import critic_node

    state: HRState = {
        "messages": [HumanMessage(content="how many days leave?")],
        "intent": "policy",
        "answer": "You are entitled to 21 days annual leave per Section 3.1 of the Leave Policy.",
        "sources": ["leave_policy.pdf"],
        "sub_tasks": [],
        "retry_count": 0,
        "critic_score": 0.0,
        "memory": [],
    }

    mock_response = MagicMock()
    mock_response.content = "0.92"

    with patch("app.graph.nodes.critic_node._get_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm
        result = critic_node(state)

    assert result["critic_score"] == pytest.approx(0.92)


def test_critic_scores_empty_answer_zero():
    from app.graph.nodes.critic_node import critic_node

    state: HRState = {
        "messages": [HumanMessage(content="test?")],
        "intent": "policy",
        "answer": "",
        "sources": [],
        "sub_tasks": [],
        "retry_count": 0,
        "critic_score": 0.0,
        "memory": [],
    }
    result = critic_node(state)
    assert result["critic_score"] == 0.0


def test_critic_clamps_score_between_0_and_1():
    from app.graph.nodes.critic_node import critic_node

    state: HRState = {
        "messages": [HumanMessage(content="test?")],
        "intent": "policy",
        "answer": "some answer",
        "sources": [],
        "sub_tasks": [],
        "retry_count": 0,
        "critic_score": 0.0,
        "memory": [],
    }

    mock_response = MagicMock()
    mock_response.content = "1.5"

    with patch("app.graph.nodes.critic_node._get_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm
        result = critic_node(state)

    assert result["critic_score"] <= 1.0


def test_critic_handles_invalid_response():
    from app.graph.nodes.critic_node import critic_node

    state: HRState = {
        "messages": [HumanMessage(content="test?")],
        "intent": "policy",
        "answer": "some answer here",
        "sources": [],
        "sub_tasks": [],
        "retry_count": 0,
        "critic_score": 0.0,
        "memory": [],
    }

    mock_response = MagicMock()
    mock_response.content = "looks good to me"

    with patch("app.graph.nodes.critic_node._get_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm
        result = critic_node(state)

    assert 0.0 <= result["critic_score"] <= 1.0


# ── Memory node ───────────────────────────────────────────────────────────────

def test_memory_node_appends_summary():
    from app.graph.nodes.memory_node import memory_node

    state: HRState = {
        "messages": [HumanMessage(content="how many days leave?")],
        "intent": "policy",
        "answer": "21 days annual leave.",
        "sources": [],
        "sub_tasks": [],
        "retry_count": 0,
        "critic_score": 0.9,
        "memory": ["User asked about probation — 3 months probation period"],
    }

    mock_response = MagicMock()
    mock_response.content = "User asked about leave — 21 days annual leave entitlement"

    with patch("app.graph.nodes.memory_node._get_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm
        result = memory_node(state)

    assert len(result["memory"]) == 2
    assert "21 days annual leave" in result["memory"][-1]


def test_memory_node_caps_at_10():
    from app.graph.nodes.memory_node import memory_node

    state: HRState = {
        "messages": [HumanMessage(content="test?")],
        "intent": "policy",
        "answer": "test answer",
        "sources": [],
        "sub_tasks": [],
        "retry_count": 0,
        "critic_score": 0.9,
        "memory": [f"memory item {i}" for i in range(10)],
    }

    mock_response = MagicMock()
    mock_response.content = "new memory item"

    with patch("app.graph.nodes.memory_node._get_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm
        result = memory_node(state)

    assert len(result["memory"]) <= 10


# ── Retry routing logic ───────────────────────────────────────────────────────

def test_retry_increments_counter():
    from app.graph.supervisor import _increment_retry

    state: HRState = {
        "messages": [],
        "intent": "policy",
        "answer": "",
        "sources": [],
        "sub_tasks": [],
        "retry_count": 1,
        "critic_score": 0.3,
        "memory": [],
    }
    result = _increment_retry(state)
    assert result["retry_count"] == 2


def test_route_after_critic_proceeds_when_score_high():
    from app.graph.supervisor import _route_after_critic

    state: HRState = {
        "messages": [],
        "intent": "policy",
        "answer": "good answer",
        "sources": [],
        "sub_tasks": [],
        "retry_count": 0,
        "critic_score": 0.9,
        "memory": [],
    }
    assert _route_after_critic(state) == "synthesiser"


def test_route_after_critic_retries_when_score_low():
    from app.graph.supervisor import _route_after_critic

    state: HRState = {
        "messages": [],
        "intent": "policy",
        "answer": "vague answer",
        "sources": [],
        "sub_tasks": [],
        "retry_count": 0,
        "critic_score": 0.3,
        "memory": [],
    }
    assert _route_after_critic(state) == "retry"


def test_route_after_critic_stops_retrying_at_max():
    from app.graph.supervisor import _route_after_critic

    state: HRState = {
        "messages": [],
        "intent": "policy",
        "answer": "still vague",
        "sources": [],
        "sub_tasks": [],
        "retry_count": settings.max_retries,
        "critic_score": 0.2,
        "memory": [],
    }
    assert _route_after_critic(state) == "synthesiser"


# ── Chunker ───────────────────────────────────────────────────────────────────

def test_chunker_basic():
    words = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(words, chunk_size=10, overlap=2)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_chunker_uses_config_defaults():
    words = " ".join(f"w{i}" for i in range(settings.chunk_size * 3))
    chunks = chunk_text(words)
    assert len(chunks) > 1


# ── Cache ─────────────────────────────────────────────────────────────────────

def test_cache_hit_miss():
    c = QueryCache(max_size=10)
    assert c.get("hello") is None
    c.set("hello", "world")
    assert c.get("hello") == "world"


def test_cache_case_insensitive():
    c = QueryCache(max_size=10)
    c.set("hello", "world")
    assert c.get("HELLO") == "world"


def test_cache_eviction():
    c = QueryCache(max_size=2)
    c.set("a", "1")
    c.set("b", "2")
    c.set("c", "3")
    assert len(c) == 2


def test_cache_uses_config_default():
    c = QueryCache()
    assert c._max_size == settings.cache_max_size


# ── RAG pipeline ──────────────────────────────────────────────────────────────

def test_format_context_empty_returns_no_context():
    from app.rag.pipeline import format_context, NO_CONTEXT
    ctx, sources = format_context([])
    assert ctx == NO_CONTEXT
    assert sources == []


def test_format_context_deduplicates_sources():
    from app.rag.pipeline import format_context
    chunks = [
        {"text": "chunk one", "metadata": {"source": "policy.pdf"}},
        {"text": "chunk two", "metadata": {"source": "policy.pdf"}},
        {"text": "chunk three", "metadata": {"source": "handbook.pdf"}},
    ]
    _, sources = format_context(chunks)
    assert sources == ["policy.pdf", "handbook.pdf"]
    assert len(sources) == 2


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
