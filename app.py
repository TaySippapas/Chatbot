"""Streamlit chat UI for the Quillmark Bank support assistant.

Each question is routed automatically: the handbook and incidents are always available (cached), and
similar past cases and figures from the case database are added when the question needs them.

Usage: streamlit run app.py
(Build the RAG index first: python -m quillmark_bot.rag.index. For figures: docker compose up -d)
"""
import gc
import queue
import sys
import threading
from datetime import date
from pathlib import Path

import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from quillmark_bot.analytics import DataRequest  # noqa: E402
from quillmark_bot.hybrid.engine import HybridChatbot  # noqa: E402
from quillmark_bot.model import load_model  # noqa: E402
from quillmark_bot.rag.documents import case_documents, knowledge_documents  # noqa: E402
from quillmark_bot.rag.retriever import Retriever  # noqa: E402

EXAMPLES = [
    "I was charged twice for the same purchase on March 15. What happens now?",
    "What is the overdraft fee and how can I avoid it?",
    "How often do goodwill requests get approved?",
    "Which branch had the most fraud complaints last year?",
]


# ---------------------------------------------------------------- shared resources (once per server)

@st.cache_resource(show_spinner="Loading Qwen3.5-4B onto the GPU, caching the handbook and the router...")
def get_bot() -> HybridChatbot:
    model, processor = load_model()
    bot = HybridChatbot(model, processor, Retriever())
    bot.build_cache()
    return bot


def load_bot() -> HybridChatbot:
    """get_bot(), but a Windows out-of-memory failure becomes a clear message instead of a traceback.

    A failed load is not cached, so it can be retried; the partly loaded model is released first,
    otherwise it keeps holding the memory the retry needs.
    """
    try:
        return get_bot()
    except OSError as error:
        if "1455" not in str(error):  # "The paging file is too small for this operation to complete"
            raise
        gc.collect()
        torch.cuda.empty_cache()
        st.error("Windows ran out of memory while loading the model: it needs about 6 GB of free virtual "
                 "memory (RAM + page file). Close a few programs, or give Windows a larger page file, "
                 "then ask again.")
        st.stop()


@st.cache_resource
def gpu_lock() -> threading.Lock:
    """One GPU: answers from different browser tabs wait their turn."""
    return threading.Lock()


@st.cache_resource
def documents_by_id() -> dict[str, str]:
    """Text of every policy, incident and case, to show what an answer cited."""
    docs = {d["id"]: d["text"] for d in knowledge_documents()}
    docs.update({d["id"]: d["display_text"] for d in case_documents()})
    return docs


# ---------------------------------------------------------------- helpers

def md(text: str) -> str:
    """Streamlit renders text between two $ signs as math; amounts like "$29 ... $50" must stay text."""
    return text.replace("$", "\\$")


def stream_answer(bot, question: str, history: list[dict], today: date, previous_request, outcome: dict):
    """Run bot.ask in a worker thread and yield its text as it is generated.

    The bot streams through a callback, while st.write_stream wants a generator; a queue joins the two.
    The finished Answer and its trace land in `outcome`.
    """
    chunks = queue.Queue()

    def work():
        try:
            with gpu_lock():
                outcome["answer"] = bot.ask(question, history, on_text=chunks.put, today=today,
                                            previous_request=previous_request)
                outcome["trace"] = bot.last_trace
        except Exception as error:  # surfaced in the UI rather than dying silently in the thread
            outcome["error"] = error
        finally:
            chunks.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (chunk := chunks.get()) is not None:
        yield md(chunk)


def make_record(outcome: dict) -> dict:
    """What gets kept in the conversation for one answer (plain values: it lives in session state)."""
    a, trace = outcome["answer"], outcome["trace"]
    route = trace.route
    record = {
        "text": a.text,
        "cited": a.cited_ids,
        "unverified_numbers": a.unverified_numbers,
        "case_only_numbers": a.case_only_numbers,
        "invalid_citations": a.invalid_citations,
        "request": route.data_request.to_dict() if route.data_request else None,
        "stats": (f"route {route.seconds:.1f} s · read {a.computed_prompt_tokens:,} new tokens "
                  f"({a.cached_tokens:,} from cache) · first word {a.seconds_to_first_token:.1f} s · "
                  f"{a.seconds_total:.1f} s answering"),
        "retrieved": [(h.id, h.score) for h in trace.retrieval.cases] if trace.retrieval else [],
        "route": {
            "decided_by": route.decided_by, "reason": route.reason, "llm_output": route.llm_output,
            "uses": ["handbook (cached)"] + (["database"] if trace.data else [])
                    + (["past cases"] if trace.retrieval else []),
        },
    }
    if trace.data:
        record["data"] = {"description": trace.data.description, "sql": trace.data.sql,
                          "params": trace.data.params, "rows": trace.data.rows, "error": trace.data.error,
                          "seconds": trace.seconds_data}
    return record


def show_details(record: dict) -> None:
    if record.get("unverified_numbers"):
        st.warning("Check these figures: they aren't in the handbook, the past cases or the database "
                   f"results this answer was given: {md(', '.join(record['unverified_numbers']))}")
    if record.get("case_only_numbers"):
        st.warning("These figures appear only in other customers' past cases, so they shouldn't be read as "
                   f"this customer's own amounts: {md(', '.join(record['case_only_numbers']))}")
    if record.get("invalid_citations"):
        st.warning(f"These citations weren't among the documents provided: {', '.join(record['invalid_citations'])}")
    st.caption(record["stats"])
    route = record["route"]
    with st.expander(f"How this was answered: {' + '.join(route['uses'])}"):
        st.markdown(f"**Route** decided by {route['decided_by']}: {md(route['reason'])}")
        if route["llm_output"]:
            st.markdown("The router model's decision form:")
            st.code(route["llm_output"].strip(), language="json")
        data = record.get("data")
        if data:
            st.markdown(f"**Database query** ({data['seconds']:.2f} s): {md(data['description'])}")
            if data["sql"]:
                st.code(data["sql"], language="sql")
                st.caption(f"Parameters: {data['params']}")
            if data["error"]:
                st.error(f"The database could not be reached: {data['error']}")
            elif data["rows"]:
                st.dataframe(data["rows"], hide_index=True)
        if record["retrieved"]:
            docs = documents_by_id()
            st.markdown(f"**Similar past cases** ({len(record['retrieved'])})")
            for doc_id, score in record["retrieved"]:
                first_line = docs.get(doc_id, "").splitlines()[0]
                st.markdown(f"`{doc_id}` · score {score:.2f} · {md(first_line)}")
    if record["cited"]:
        docs = documents_by_id()
        with st.expander(f"Cited: {', '.join(record['cited'])}"):
            for doc_id in record["cited"]:
                st.markdown(f"**{doc_id}**")
                st.text(docs.get(doc_id, "Not found in the handbook, incidents or cases."))


def history() -> list[dict]:
    """The conversation so far as plain question/answer turns."""
    turns = []
    for message in st.session_state.messages:
        if message["role"] == "user":
            turns.append({"role": "user", "content": message["content"]})
        elif message.get("record"):
            turns.append({"role": "assistant", "content": message["record"]["text"]})
        else:
            turns.pop()  # a question whose answer failed is left out
    return turns


# ---------------------------------------------------------------- page

st.set_page_config(page_title="Quillmark Support", page_icon="🏦")
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("Settings")
    today = st.date_input(
        "Today's date", value=date.today(), key="today",
        help="Incidents only apply within their dates. Set a past date to test one, "
             "e.g. 2026-03-15 for the duplicate-charge outage.",
    )
    if st.button("New conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown(
        "**How answers are made.** The policy handbook and known incidents are always available, read "
        "once and cached. For each question a router decides whether to also look up similar past cases "
        "and figures from the case database. Open *How this was answered* under any answer to see what "
        "was used."
    )
    if torch.cuda.is_available():
        st.caption(f"GPU memory in use: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

st.title("Quillmark Support Assistant")
st.caption("Fictional bank · answers come from the policy handbook, known incidents, past cases and the case "
           "database · Qwen3.5-4B running locally")
bot = load_bot()  # on page open rather than at the first question, so the first answer isn't the slow one

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(md(message["content"]))
        elif message.get("record"):
            st.markdown(md(message["record"]["text"]))
            show_details(message["record"])
        else:
            st.error(message["error"])

question = st.chat_input("Ask about fees, cards, disputes, loans...")
if "pending" in st.session_state:
    question = st.session_state.pop("pending")

if not st.session_state.messages and not question:
    st.markdown("Ask a question, or try one of these:")
    columns = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if columns[i % 2].button(example, key=f"example-{i}", width="stretch"):
            st.session_state.pending = example
            st.rerun()

if question:
    turns = history()
    # The previous answer's database query lets a short follow-up ("and in Riverton?") reuse it
    last = st.session_state.messages[-1] if st.session_state.messages else {}
    previous = (last.get("record") or {}).get("request")
    previous_request = DataRequest(**previous) if previous else None
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(md(question))

    with st.chat_message("assistant"):
        outcome = {}
        slot = st.empty()
        with slot:
            st.write_stream(stream_answer(bot, question, turns, today, previous_request, outcome))
        if "error" in outcome:
            message = {"role": "assistant", "error": f"Something went wrong: {outcome['error']}"}
            st.error(message["error"])
        else:
            message = {"role": "assistant", "record": make_record(outcome)}
            slot.markdown(md(message["record"]["text"]))  # the checked text, e.g. with database tags unified
            show_details(message["record"])
    st.session_state.messages.append(message)
