from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.config import settings
from app.prompts.prompts import INTAKE_PROMPT
from app.graph.state import HRState

_llm = ChatOpenAI(
    model=settings.openai_model,
    api_key=settings.openai_api_key,
    temperature=0,
    max_tokens=10,
)

VALID = {"policy", "grievance", "talent"}


def intake_node(state: HRState) -> dict:
    messages = [SystemMessage(content=INTAKE_PROMPT), state["messages"][-1]]
    response = _llm.invoke(messages)
    intent = response.content.strip().lower()
    return {"intent": intent if intent in VALID else "policy"}
