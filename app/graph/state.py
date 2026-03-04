from __future__ import annotations
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class HRState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    answer: str
    sources: list[str]
