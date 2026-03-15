from __future__ import annotations
from app.graph.nodes._base import _run_query, _build_memory_block
from app.prompts.prompts import RAG_SYSTEM
from app.rag.pipeline import NO_CONTEXT
from app.graph.state import HRState


def multi_specialist_node(state: HRState) -> dict:
    """Run one retrieval pass per planner sub-task across all PDFs."""
    sub_tasks = state.get("sub_tasks") or [state["messages"][-1].content]
    memory_block = _build_memory_block(state)

    answers: list[str] = []
    all_sources: list[str] = []

    for sub_query in sub_tasks:
        result = _run_query(
            query=sub_query,
            system_prompt=RAG_SYSTEM,
            memory_block=memory_block,
        )
        if result["answer"] != NO_CONTEXT:
            answers.append(result["answer"])
            all_sources.extend(result["sources"])

    if not answers:
        return {"answer": NO_CONTEXT, "sources": [], "messages": []}

    return {
        "answer": "\n\n---\n\n".join(answers),
        "sources": list(dict.fromkeys(all_sources)),
        "messages": [],
    }
