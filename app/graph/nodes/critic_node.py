from __future__ import annotations
from langchain_core.messages import HumanMessage
from app.prompts.prompts import CRITIC_PROMPT
from app.graph.nodes._base import _get_llm
from app.graph.state import HRState


def critic_node(state: HRState) -> dict:
    """Score answer quality 0.0–1.0. Low score triggers replan in supervisor."""
    question = state["messages"][-1].content
    answer = state.get("answer", "")

    if not answer or answer == "__NO_CONTEXT__":
        return {"critic_score": 0.0}

    prompt = CRITIC_PROMPT.format(question=question, answer=answer)
    response = _get_llm().invoke([HumanMessage(content=prompt)])

    try:
        score = max(0.0, min(1.0, float(response.content.strip())))
    except ValueError:
        score = 0.5

    return {"critic_score": score}
