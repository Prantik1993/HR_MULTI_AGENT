from __future__ import annotations
from langchain_core.messages import HumanMessage
from app.prompts.prompts import PLANNER_PROMPT, REPLAN_PROMPT
from app.graph.nodes._base import _get_llm
from app.graph.state import HRState


def planner_node(state: HRState) -> dict:
    """Break the question into sub-queries. On retry, rephrase with different keywords."""
    question = state["messages"][-1].content
    retry_count = state.get("retry_count", 0)
    existing = state.get("sub_tasks", [])

    if retry_count > 0 and existing:
        prompt = REPLAN_PROMPT.format(
            question=question,
            previous_queries="\n".join(existing),
            retry_count=retry_count,
        )
    else:
        prompt = PLANNER_PROMPT.format(question=question)

    response = _get_llm().invoke([HumanMessage(content=prompt)])
    lines = [l.strip() for l in response.content.strip().splitlines() if l.strip()]
    return {"sub_tasks": lines[:4] if lines else [question]}
