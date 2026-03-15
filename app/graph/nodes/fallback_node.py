from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.config import settings
from app.prompts.prompts import FALLBACK_SYSTEM
from app.rag.pipeline import retrieve, format_context, NO_CONTEXT
from app.graph.state import HRState
from app.core.logger import get_logger

logger = get_logger(__name__)

_llm = ChatOpenAI(
    model=settings.openai_model,
    api_key=settings.openai_api_key,
    temperature=0,
)


def fallback_node(state: HRState) -> dict:
    query = state["messages"][-1].content
    logger.info("fallback_node | broad retrieval for: %r", query[:120])

    chunks = retrieve(query, topic=None, top_k=5)
    logger.info("fallback_node | retrieved %d chunks", len(chunks))

    context, sources = format_context(chunks)
    if context == NO_CONTEXT:
        context = "No relevant HR documents found."
        sources = []
        logger.warning("fallback_node | no context — using generic guidance")

    system = FALLBACK_SYSTEM.format(context=context)
    response = _llm.invoke([SystemMessage(content=system)] + list(state["messages"]))
    logger.info("fallback_node | answer length: %d chars", len(response.content))
    return {"answer": response.content, "sources": sources, "messages": [response]}
