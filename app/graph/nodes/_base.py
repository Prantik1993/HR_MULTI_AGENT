from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.config import settings
from app.rag.pipeline import retrieve, format_context, NO_CONTEXT
from app.graph.state import HRState

_llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0)


def _specialist_node(state: HRState, topic: str, system_prompt: str) -> dict:
    query = state["messages"][-1].content
    chunks = retrieve(query, topic=topic)
    context, sources = format_context(chunks)
    if context == NO_CONTEXT:
        return {"answer": NO_CONTEXT, "sources": [], "messages": []}
    system = system_prompt.format(context=context)
    response = _llm.invoke([SystemMessage(content=system)] + list(state["messages"]))
    return {"answer": response.content, "sources": sources, "messages": [response]}
