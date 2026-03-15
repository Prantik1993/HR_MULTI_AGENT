"""
app/graph/supervisor.py
------------------------
FIX [04]: Added 'greeting' and 'offtopic' conditional edges to END.
          Without these, LangGraph raises KeyError at runtime when
          intake_node returns those intents.
"""
from __future__ import annotations
from langgraph.graph import StateGraph, END
from app.graph.state import HRState
from app.graph.nodes.intake_node import intake_node, VALID
from app.graph.nodes.query_rewriter_node import query_rewriter_node
from app.graph.nodes.policy_node import policy_node
from app.graph.nodes.grievance_node import grievance_node
from app.graph.nodes.talent_node import talent_node
from app.graph.nodes.fallback_node import fallback_node
from app.rag.pipeline import NO_CONTEXT
from app.core.logger import get_logger

logger = get_logger(__name__)


def _route_after_intake(state: HRState) -> str:
    intent = state["intent"]
    # FIX [04]: greeting/offtopic short-circuit to END — no RAG needed
    if intent in ("greeting", "offtopic"):
        logger.info("supervisor | %s → END (no RAG)", intent)
        return END
    logger.info("supervisor | %s → query_rewriter", intent)
    return "query_rewriter"


def _route_intent(state: HRState) -> str:
    return state["intent"]


def _route_or_fallback(state: HRState) -> str:
    if state["answer"] == NO_CONTEXT:
        logger.info("supervisor | no context → fallback")
        return "fallback"
    return END


def build_graph() -> StateGraph:
    builder = StateGraph(HRState)

    builder.add_node("intake",          intake_node)
    builder.add_node("query_rewriter",  query_rewriter_node)
    builder.add_node("policy",          policy_node)
    builder.add_node("grievance",       grievance_node)
    builder.add_node("talent",          talent_node)
    builder.add_node("fallback",        fallback_node)

    builder.set_entry_point("intake")

    # FIX [04]: greeting/offtopic go directly to END; HR intents continue to rewriter
    builder.add_conditional_edges(
        "intake",
        _route_after_intake,
        {"query_rewriter": "query_rewriter", END: END},
    )

    builder.add_conditional_edges(
        "query_rewriter",
        _route_intent,
        {"policy": "policy", "grievance": "grievance", "talent": "talent"},
    )

    builder.add_conditional_edges("policy",    _route_or_fallback, {"fallback": "fallback", END: END})
    builder.add_conditional_edges("grievance", _route_or_fallback, {"fallback": "fallback", END: END})
    builder.add_conditional_edges("talent",    _route_or_fallback, {"fallback": "fallback", END: END})
    builder.add_edge("fallback", END)

    return builder.compile()


graph = build_graph()
logger.info("supervisor | graph compiled OK")
