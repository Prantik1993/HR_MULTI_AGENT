from __future__ import annotations
from langchain_core.messages import HumanMessage
from app.prompts.prompts import MEMORY_PROMPT
from app.graph.nodes._base import _get_llm
from app.graph.state import HRState


def memory_node(state: HRState) -> dict:
    """Save a one-line Q&A summary to session memory for follow-up questions."""
    question = state["messages"][-1].content
    answer = state.get("answer", "")

    if not answer:
        return {}

    prompt = MEMORY_PROMPT.format(question=question, answer=answer)
    response = _get_llm().invoke([HumanMessage(content=prompt)])

    existing = list(state.get("memory", []) or [])
    existing.append(response.content.strip())
    return {"memory": existing[-10:]}
