from __future__ import annotations
from langchain_core.messages import HumanMessage
from app.prompts.prompts import SYNTHESISER_PROMPT
from app.graph.nodes._base import _get_llm, _build_memory_block
from app.graph.state import HRState


def synthesiser_node(state: HRState) -> dict:
    """Merge partial answers and draft emails/letters if the user asked for them."""
    question = state["messages"][-1].content
    partial_answers = state.get("answer", "")
    memory_block = _build_memory_block(state)

    prompt = SYNTHESISER_PROMPT.format(
        question=question,
        partial_answers=partial_answers,
        memory=memory_block,
    )
    response = _get_llm().invoke([HumanMessage(content=prompt)])
    return {"answer": response.content, "messages": [response]}
