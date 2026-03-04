from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.config import settings
from app.prompts.prompts import FALLBACK_SYSTEM
from app.rag.pipeline import retrieve, format_context, NO_CONTEXT
from app.graph.state import HRState

_llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0)


def fallback_node(state: HRState) -> dict:
    """Broad retrieval across all topics when specialist found nothing."""
    query = state["messages"][-1].content
    chunks = retrieve(query, topic=None, top_k=5)
    context, sources = format_context(chunks)
    if context == NO_CONTEXT:
        context = "No relevant HR documents found."
        sources = []
    system = FALLBACK_SYSTEM.format(context=context)
    response = _llm.invoke([SystemMessage(content=system)] + list(state["messages"]))
    return {"answer": response.content, "sources": sources, "messages": [response]}
