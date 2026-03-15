from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import settings
from app.prompts.prompts import REWRITE_PROMPT
from app.graph.state import HRState
from app.core.logger import get_logger

logger = get_logger(__name__)

_llm = ChatOpenAI(
    model=settings.openai_model,
    api_key=settings.openai_api_key,
    temperature=0,
)


def query_rewriter_node(state: HRState) -> dict:
    original = state["messages"][-1].content
    logger.info("query_rewriter_node | original: %r", original[:120])

    response = _llm.invoke([SystemMessage(content=REWRITE_PROMPT), HumanMessage(content=original)])
    rewritten = response.content.strip()
    logger.info("query_rewriter_node | rewritten: %r", rewritten[:120])

    new_messages = list(state["messages"])[:-1] + [HumanMessage(content=rewritten)]
    return {"messages": new_messages}
