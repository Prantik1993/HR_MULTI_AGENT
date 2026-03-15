"""
app/graph/nodes/intake_node.py
-------------------------------
Classifies user intent. Handles greeting / off-topic BEFORE hitting RAG.

FIXES:
  [01] max_tokens raised 10 → 150  (greeting replies were truncated)
  [02] greeting/offtopic parsed and returned as final answer → END
  [11] greeting/offtopic never reaches query_rewriter (saves 2 LLM calls)
"""
from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.config import settings
from app.prompts.prompts import INTAKE_PROMPT
from app.graph.state import HRState
from app.core.logger import get_logger

logger = get_logger(__name__)

_llm = ChatOpenAI(
    model=settings.openai_model,
    api_key=settings.openai_api_key,
    temperature=0,
    max_tokens=150,          # FIX [01]: was 10
)

VALID = {"policy", "grievance", "talent"}


def intake_node(state: HRState) -> dict:
    user_msg = state["messages"][-1].content
    logger.info("intake_node | input: %r", user_msg[:120])

    response = _llm.invoke([SystemMessage(content=INTAKE_PROMPT), state["messages"][-1]])
    raw = response.content.strip()
    logger.debug("intake_node | LLM raw: %r", raw)

    lower = raw.lower()

    # FIX [02][11]: short-circuit greeting/offtopic — never enters RAG pipeline
    if lower.startswith("greeting:"):
        reply = raw[len("greeting:"):].strip()
        logger.info("intake_node | intent=greeting → END")
        return {"intent": "greeting", "answer": reply, "sources": []}

    if lower.startswith("offtopic:"):
        reply = raw[len("offtopic:"):].strip()
        logger.info("intake_node | intent=offtopic → END")
        return {"intent": "offtopic", "answer": reply, "sources": []}

    intent = lower
    if intent not in VALID:
        logger.warning("intake_node | unknown intent %r → default policy", intent)
        intent = "policy"

    logger.info("intake_node | intent=%s", intent)
    return {"intent": intent}
