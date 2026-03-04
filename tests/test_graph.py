from langchain_core.messages import HumanMessage, AIMessage
from app.graph.state import HRState
from app.graph.nodes.intake_node import VALID


def test_valid_intents():
    assert VALID == {"policy", "grievance", "talent"}


def test_state_structure():
    state: HRState = {
        "messages": [HumanMessage(content="test")],
        "intent": "policy",
        "answer": "",
        "sources": [],
    }
    assert state["intent"] == "policy"
    assert isinstance(state["messages"], list)
    assert isinstance(state["sources"], list)


def test_history_message_types():
    from api.main import _build_messages
    history = [
        {"role": "user",      "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    msgs = _build_messages(history, "new question")
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)
    assert isinstance(msgs[2], HumanMessage)
    assert msgs[2].content == "new question"
