from __future__ import annotations
from app.prompts.prompts import TALENT_SYSTEM
from app.graph.nodes._base import _specialist_node
from app.graph.state import HRState


def talent_node(state: HRState) -> dict:
    return _specialist_node(state, "talent_node", TALENT_SYSTEM)
