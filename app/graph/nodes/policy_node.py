from __future__ import annotations
from app.prompts.prompts import POLICY_SYSTEM
from app.graph.nodes._base import _specialist_node
from app.graph.state import HRState


def policy_node(state: HRState) -> dict:
    return _specialist_node(state, "policy_node", POLICY_SYSTEM)
