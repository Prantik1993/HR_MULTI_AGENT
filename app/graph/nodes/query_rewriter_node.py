from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import settings
from app.prompts.prompts import REWRITE_PROMPT
from app.graph.state import HRState

_llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0)


def query_rewriter_node(state: HRState) -> dict:
    original = state["messages"][-1].content
    response = _llm.invoke([SystemMessage(content=REWRITE_PROMPT), HumanMessage(content=original)])
    rewritten = response.content.strip()
    # preserve history, replace only last (current) query with rewritten version
    new_messages = list(state["messages"])[:-1] + [HumanMessage(content=rewritten)]
    return {"messages": new_messages}
