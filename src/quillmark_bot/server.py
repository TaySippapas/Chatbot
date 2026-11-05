"""Web API for the chatbot, and the React UI (web/dist) once it is built.

The model loads in the background as soon as the server starts, so the page shows "starting up" instead
of the first question paying for the load. /api/chat streams server-sent events: each stage as it
finishes (route, database result, similar cases), then the answer as it is written, then the checked
answer with everything the UI shows under it. Closing the stream (the Stop button, a closed tab) ends
generation at the next token.

Usage: python -m quillmark_bot.server   -> http://localhost:8000
       development: also `npm run dev` in web/ -> http://localhost:5173 (proxies /api here)
"""
import asyncio
import json
import os
import threading
from datetime import date
from functools import cache
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from quillmark_bot import monitoring
from quillmark_bot.analytics import DataRequest
from quillmark_bot.config import MODEL_ID, ROOT
from quillmark_bot.rag.documents import case_documents, knowledge_documents

WEB_DIST = ROOT / "web" / "dist"


class State:
    """The one chatbot this server runs. One GPU: answers take turns through `lock`."""
    bot = None
    status: Literal["loading", "ready", "error"] = "loading"
    message = "Loading the model onto the GPU..."
    logging_on = False
    lock = threading.Lock()


state = State()


def load_bot() -> None:
    """Load the model, retriever and caches (runs in a background thread at startup)."""
    try:
        # Imported here so the server answers /api/health while torch and the model load
        from quillmark_bot.hybrid.engine import HybridChatbot
        from quillmark_bot.model import load_model
        from quillmark_bot.rag.retriever import Retriever

        model, processor = load_model()
        state.message = "Reading the policy handbook into the cache..."
        bot = HybridChatbot(model, processor, Retriever())
        bot.build_cache()
        state.bot, state.status, state.message = bot, "ready", "Ready"
    except Exception as error:  # shown on the page instead of a server that silently never gets ready
        state.status, state.message = "error", f"The model could not be loaded: {type(error).__name__}: {error}"


async def lifespan(app: FastAPI):
    state.logging_on = monitoring.ensure_schema()
    threading.Thread(target=load_bot, daemon=True).start()
    yield


app = FastAPI(title="Quillmark support assistant", lifespan=lifespan)


@cache
def documents() -> dict[str, str]:
    """Text of every policy, incident and case, for showing what an answer cited."""
    docs = {d["id"]: d["text"] for d in knowledge_documents()}
    docs.update({d["id"]: d["display_text"] for d in case_documents()})
    return docs


# ---------------------------------------------------------------- shapes sent to the browser

def event_payload(name: str, value) -> dict:
    """What the UI needs from each stage reported by HybridChatbot.ask(on_event=...)."""
    if name == "route":
        return {"decided_by": value.decided_by, "reason": value.reason, "llm_output": value.llm_output,
                "seconds": value.seconds, "needs_data": value.needs_data or bool(value.case_id),
                "needs_cases": value.needs_cases, "case_id": value.case_id}
    if name == "data":
        return {"description": value.description, "text": value.text, "sql": value.sql, "params": value.params,
                "rows": value.rows, "error": value.error}
    if name == "cases":
        return {"cases": [{"id": h.id, "score": round(h.score, 3), "title": h.text.splitlines()[0]}
                          for h in value.cases], "seconds": value.seconds}
    return {}


def answer_record(answer, trace) -> dict:
    """The finished answer and how it was made: what the UI keeps and shows under the answer."""
    route = trace.route
    return {
        "text": answer.text,
        "cited": answer.cited_ids,
        "stopped": answer.stopped,
        "unverified_numbers": answer.unverified_numbers,
        "case_only_numbers": answer.case_only_numbers,
        "invalid_citations": answer.invalid_citations,
        "request": route.data_request.to_dict() if route.data_request else None,
        "stats": {"seconds_route": route.seconds, "seconds_data": trace.seconds_data if trace.data else None,
                  "seconds_retrieval": answer.seconds_retrieval or None,
                  "seconds_first_token": answer.seconds_to_first_token, "seconds_answer": answer.seconds_total,
                  "prompt_tokens": answer.prompt_tokens, "cached_tokens": answer.cached_tokens,
                  "output_tokens": answer.output_tokens},
    }


# ---------------------------------------------------------------- endpoints

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = []
    today: date | None = None
    previous_request: dict | None = None  # the previous answer's database query, for follow-ups
    session_id: str | None = None


class Feedback(BaseModel):
    interaction_id: int
    rating: Literal[1, -1] | None


@app.get("/api/health")
def health() -> dict:
    return {"status": state.status, "message": state.message, "model": MODEL_ID, "logging": state.logging_on}


@app.post("/api/chat")
async def chat(body: ChatRequest):
    if state.bot is None:
        raise HTTPException(503, state.message)
    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()

    def put(name, data):
        loop.call_soon_threadsafe(events.put_nowait, (name, data))

    def work():
        if not state.lock.acquire(blocking=False):
            put("status", {"stage": "queued"})
            state.lock.acquire()
        try:
            put("status", {"stage": "routing"})
            previous = DataRequest(**body.previous_request) if body.previous_request else None
            answer = state.bot.ask(
                body.question, [t.model_dump() for t in body.history], today=body.today, previous_request=previous,
                on_text=lambda text: put("token", {"text": text}),
                on_event=lambda name, value: put(name, event_payload(name, value)),
                should_stop=stop.is_set)
            trace = state.bot.last_trace
            record = answer_record(answer, trace)
            if state.logging_on and not answer.stopped:  # a stopped answer is incomplete: not a measurement
                record["interaction_id"] = monitoring.log_interaction("app", body.question, answer, trace,
                                                                      body.session_id)
            put("done", record)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            if state.logging_on:
                monitoring.log_interaction("app", body.question, session_id=body.session_id, error=message)
            put("error", {"message": message})
        finally:
            state.lock.release()
            put(None, None)

    threading.Thread(target=work, daemon=True).start()

    async def stream():
        try:
            while (item := await events.get())[0] is not None:
                yield {"event": item[0], "data": json.dumps(jsonable_encoder(item[1]))}
        finally:
            stop.set()  # the browser went away (Stop, closed tab): generation ends at the next token

    return EventSourceResponse(stream())


@app.post("/api/feedback")
def feedback(body: Feedback) -> dict:
    return {"saved": monitoring.set_feedback(body.interaction_id, body.rating)}


@app.get("/api/documents/{doc_id}")
def document(doc_id: str) -> dict:
    text = documents().get(doc_id)
    if text is None:
        raise HTTPException(404, f"No policy, incident or case {doc_id}")
    return {"id": doc_id, "text": text}


if WEB_DIST.exists():  # after the API routes, so /api/* is never shadowed
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
