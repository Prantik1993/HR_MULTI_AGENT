from __future__ import annotations
from app.prompts.prompts import GRIEVANCE_SYSTEM
from app.graph.nodes._base import _specialist_node
from app.graph.state import HRState


def grievance_node(state: HRState) -> dict:
    return _specialist_node(state, "grievance", GRIEVANCE_SYSTEM)
