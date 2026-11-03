"""Streamlit chat UI for the Quillmark Bank support bots: CAG, RAG, or both side by side.

Usage: streamlit run app.py
(For RAG, build the index first: python -m quillmark_bot.rag.index)
"""
import queue
import sys
import threading
from datetime import date
from pathlib import Path

import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from quillmark_bot.cag.engine import CAGChatbot  # noqa: E402
from quillmark_bot.cag.knowledge import build_knowledge  # noqa: E402
from quillmark_bot.model import load_model  # noqa: E402
from quillmark_bot.rag.documents import case_documents, knowledge_documents  # noqa: E402
from quillmark_bot.rag.engine import RAGChatbot  # noqa: E402
from quillmark_bot.rag.retriever import Retriever  # noqa: E402

MODES = {"CAG": "CAG: whole handbook cached", "RAG": "RAG: retrieved documents", "Both": "Compare both"}
EXAMPLES = [
    "I was charged twice for the same purchase on March 15. What happens now?",
    "What is the overdraft fee and how can I avoid it?",
    "I want to dispute a charge but my statement was 75 days ago. Can it still be reviewed?",
    "How do I activate my new debit card?",
]


# ---------------------------------------------------------------- shared resources (once per server)

@st.cache_resource(show_spinner="Loading Qwen3.5-4B onto the GPU...")
def get_model():
    return load_model()


@st.cache_resource(show_spinner="Caching the handbook and incidents...")
def get_cag() -> CAGChatbot:
    model, processor = get_model()
    bot = CAGChatbot(model, processor, build_knowledge())
    bot.build_cache()
    return bot


@st.cache_resource(show_spinner="Loading the retriever...")
def get_rag() -> RAGChatbot:
    model, processor = get_model()
    return RAGChatbot(model, processor, Retriever())


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


BOTS = {"CAG": get_cag, "RAG": get_rag}


# ---------------------------------------------------------------- helpers

def md(text: str) -> str:
    """Streamlit renders text between two $ signs as math; amounts like "$29 ... $50" must stay text."""
    return text.replace("$", "\\$")


def stream_answer(bot, question: str, history: list[dict], today: date, outcome: dict):
    """Run bot.ask in a worker thread and yield its text as it is generated.

    The bots stream through a callback, while st.write_stream wants a generator; a queue joins the two.
    The finished Answer (and RAG's retrieval) lands in `outcome`.
    """
    chunks = queue.Queue()

    def work():
        try:
            with gpu_lock():
                outcome["answer"] = bot.ask(question, history, on_text=chunks.put, today=today)
                outcome["retrieval"] = getattr(bot, "last_retrieval", None)
        except Exception as error:  # surfaced in the UI rather than dying silently in the thread
            outcome["error"] = error
        finally:
            chunks.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (chunk := chunks.get()) is not None:
        yield md(chunk)


def make_record(name: str, outcome: dict) -> dict:
    """What gets kept in the conversation for one bot's answer."""
    a = outcome["answer"]
    if name == "CAG":
        stats = (f"CAG · read {a.computed_prompt_tokens} new tokens ({a.cached_tokens:,} from cache) · "
                 f"first word {a.seconds_to_first_token:.1f} s · {a.seconds_total:.1f} s total")
    else:
        stats = (f"RAG · retrieval {a.seconds_retrieval:.2f} s · read {a.prompt_tokens:,} tokens · "
                 f"first word {a.seconds_to_first_token:.1f} s · {a.seconds_total:.1f} s total")
    retrieval = outcome.get("retrieval")
    retrieved = [(h.id, h.score) for h in retrieval.knowledge + retrieval.cases] if retrieval else []
    return {"text": a.text, "cited": a.cited_ids, "stats": stats, "retrieved": retrieved}


def show_details(record: dict) -> None:
    st.caption(record["stats"])
    docs = documents_by_id()
    if record["cited"]:
        with st.expander(f"Cited: {', '.join(record['cited'])}"):
            for doc_id in record["cited"]:
                st.markdown(f"**{doc_id}**")
                st.text(docs.get(doc_id, "Not found in the handbook, incidents or cases."))
    if record["retrieved"]:
        with st.expander(f"Retrieved for this question ({len(record['retrieved'])} documents)"):
            for doc_id, score in record["retrieved"]:
                first_line = docs.get(doc_id, "").splitlines()[0]
                st.markdown(f"`{doc_id}` · score {score:.2f} · {md(first_line)}")


def show_answers(answers: dict) -> None:
    columns = st.columns(len(answers)) if len(answers) > 1 else [st.container()]
    for column, (name, record) in zip(columns, answers.items()):
        with column:
            if len(answers) > 1:
                st.markdown(f"#### {name}")
            st.markdown(md(record["text"]))
            show_details(record)


def history_for(name: str) -> list[dict]:
    """This bot's side of the conversation so far: each question with its answer, if it gave one."""
    turns, question = [], None
    for message in st.session_state.messages:
        if message["role"] == "user":
            question = message["content"]
        elif name in message["answers"]:
            turns += [{"role": "user", "content": question},
                      {"role": "assistant", "content": message["answers"][name]["text"]}]
    return turns


# ---------------------------------------------------------------- page

st.set_page_config(page_title="Quillmark Support", page_icon="🏦", layout="wide")
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Bot", list(MODES), format_func=MODES.get, key="mode")
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
        "**CAG** reads the whole policy handbook and the known incidents once and keeps them cached, "
        "so each question only adds a few tokens.\n\n"
        "**RAG** searches the policies, incidents and 5,000 past cases for each question and gives "
        "the model only the best matches."
    )
    if torch.cuda.is_available():
        st.caption(f"GPU memory in use: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

st.title("Quillmark Support Assistant")
st.caption("Fictional bank · answers come from the policy handbook, known incidents and past cases · "
           "Qwen3.5-4B running locally")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(md(message["content"]))
        else:
            show_answers(message["answers"])

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
    names = ["CAG", "RAG"] if mode == "Both" else [mode]
    histories = {name: history_for(name) for name in names}
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(md(question))

    answers = {}
    with st.chat_message("assistant"):
        columns = st.columns(len(names)) if len(names) > 1 else [st.container()]
        for name, column in zip(names, columns):
            with column:
                if len(names) > 1:
                    st.markdown(f"#### {name}")
                bot = BOTS[name]()
                outcome = {}
                st.write_stream(stream_answer(bot, question, histories[name], today, outcome))
                if "error" in outcome:
                    st.error(f"{name} failed: {outcome['error']}")
                    continue
                answers[name] = make_record(name, outcome)
                show_details(answers[name])
    st.session_state.messages.append({"role": "assistant", "answers": answers})
