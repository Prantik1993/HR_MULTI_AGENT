"""
app/graph/nodes/_base.py
-------------------------
Shared specialist node logic.

FIX [03]: retrieve() now called with topic=None — searches ALL documents
          regardless of PDF filename or ingestion metadata.
"""
from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.config import settings
from app.rag.pipeline import retrieve, format_context, NO_CONTEXT
from app.graph.state import HRState
from app.core.logger import get_logger

logger = get_logger(__name__)

_llm = ChatOpenAI(
    model=settings.openai_model,
    api_key=settings.openai_api_key,
    temperature=0,
)


def _specialist_node(state: HRState, node_name: str, system_prompt: str) -> dict:
    query = state["messages"][-1].content
    logger.info("%s | query: %r", node_name, query[:120])

    # FIX [03]: topic=None — no filename-based filtering
    chunks = retrieve(query, topic=None)
    logger.info("%s | retrieved %d chunks", node_name, len(chunks))

    context, sources = format_context(chunks)

    if context == NO_CONTEXT:
        logger.warning("%s | no context found → routing to fallback", node_name)
        return {"answer": NO_CONTEXT, "sources": [], "messages": []}

    logger.info("%s | sources: %s", node_name, sources)
    system = system_prompt.format(context=context)
    response = _llm.invoke([SystemMessage(content=system)] + list(state["messages"]))
    logger.info("%s | answer length: %d chars", node_name, len(response.content))
    return {"answer": response.content, "sources": sources, "messages": [response]}
