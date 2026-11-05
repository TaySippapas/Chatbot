"""The web API with a stand-in chatbot: no GPU, no model. The real bot is checked by running the server."""
import json

import pytest
from fastapi.testclient import TestClient

from quillmark_bot import server
from quillmark_bot.analytics import DataRequest, DataResult
from quillmark_bot.answer import Answer
from quillmark_bot.hybrid.engine import Trace
from quillmark_bot.rag.retriever import Hit, Retrieval
from quillmark_bot.router import Route


class FakeBot:
    """Goes through the same stages as HybridChatbot.ask, instantly."""

    def __init__(self):
        self.last_trace = None
        self.calls = []

    def ask(self, question, history, today=None, previous_request=None, on_text=None, on_event=None,
            should_stop=None):
        self.calls.append({"question": question, "history": history, "previous_request": previous_request})
        route = Route(needs_cases=True, needs_data=True, reason="data question",
                      data_request=DataRequest("count", {"branch": ["Bayside"]}))
        data = DataResult(description="number of cases, where branch = Bayside", sql="SELECT ...",
                          rows=[{"cases": 57}], text="Counted: number of cases.\nResult: 57 cases")
        retrieval = Retrieval(cases=[Hit("QM-C-000001", 0.91, "[QM-C-000001] opened 2025-01-01 | Spark Card\nCustomer: ...")],
                              seconds=0.1)
        on_event("route", route)
        on_event("data", data)
        on_event("cases", retrieval)
        on_event("writing", None)
        for word in ("57 ", "cases ", "[case records]."):
            on_text(word)
        self.last_trace = Trace(route=route, data=data, retrieval=retrieval, seconds_data=0.02)
        return Answer(text="57 cases [case records].", prompt_tokens=6500, cached_tokens=6300, output_tokens=3,
                      seconds_to_first_token=0.4, seconds_total=1.2, sources=["QM-C-000001"])


def events_of(response) -> list[tuple[str, dict]]:
    """Parse a server-sent event stream into (event, data) pairs."""
    events, name = [], None
    for line in response.iter_lines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            events.append((name, json.loads(line.split(":", 1)[1])))
    return events


@pytest.fixture
def client(monkeypatch):
    bot = FakeBot()
    monkeypatch.setattr(server.state, "bot", bot)
    monkeypatch.setattr(server.state, "status", "ready")
    monkeypatch.setattr(server.state, "logging_on", False)
    return TestClient(server.app), bot  # no `with`: the startup model load doesn't run


def test_chat_streams_stages_then_words_then_the_answer(client):
    http, bot = client
    body = {"question": "How many in Bayside?", "today": "2026-03-15",
            "previous_request": {"metric": "count", "filters": {"branch": ["Riverton"]}},
            "history": [{"role": "user", "content": "How many in Riverton?"}, {"role": "assistant", "content": "57."}]}
    with http.stream("POST", "/api/chat", json=body) as response:
        assert response.status_code == 200
        events = events_of(response)

    assert [name for name, _ in events] == ["status", "route", "data", "cases", "writing", "token", "token", "token", "done"]
    route, data, cases, done = events[1][1], events[2][1], events[3][1], events[-1][1]
    assert route["needs_data"] and route["needs_cases"] and route["decided_by"] == "rules"
    assert data["rows"] == [{"cases": 57}]
    assert cases["cases"][0]["id"] == "QM-C-000001"
    assert done["text"] == "57 cases [case records]."
    assert done["request"] == {"metric": "count", "filters": {"branch": ["Bayside"]}, "date_from": None,
                               "date_to": None, "group_by": None}
    assert done["stats"]["cached_tokens"] == 6300
    # The previous query and the history reach the bot
    assert bot.calls[0]["previous_request"] == DataRequest("count", {"branch": ["Riverton"]})
    assert bot.calls[0]["history"][0] == {"role": "user", "content": "How many in Riverton?"}


def test_chat_is_refused_while_the_model_loads(monkeypatch):
    monkeypatch.setattr(server.state, "bot", None)
    response = TestClient(server.app).post("/api/chat", json={"question": "Hi"})
    assert response.status_code == 503


def test_empty_question_is_rejected(client):
    assert client[0].post("/api/chat", json={"question": ""}).status_code == 422


def test_documents_for_citations(client):
    http, _ = client
    assert "Overdraft" in http.get("/api/documents/POL-ACC-02").json()["text"]
    assert http.get("/api/documents/POL-XXX-99").status_code == 404


def test_feedback_is_saved(client, monkeypatch):
    saved = []
    monkeypatch.setattr(server.monitoring, "set_feedback", lambda i, r: saved.append((i, r)) or True)
    assert client[0].post("/api/feedback", json={"interaction_id": 7, "rating": -1}).json() == {"saved": True}
    assert saved == [(7, -1)]
