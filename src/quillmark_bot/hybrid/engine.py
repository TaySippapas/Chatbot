"""One chat that combines all three sources, with no mode to choose.

- CAG, always: rules, worked examples, the whole handbook and the known incidents, read once and cached.
- RAG, when the router asks: the 5 most relevant past cases, one per scenario.
- SQL, when the router asks: counts and totals from the case database, through fixed templates.

The router decides per question (rules first, the LLM only when they can't), and the model writes one
answer from the cached knowledge plus whatever was fetched.
"""
import time
from dataclasses import dataclass
from datetime import date

import torch

from quillmark_bot import analytics, checks
from quillmark_bot.answer import ID_PATTERN, Answer
from quillmark_bot.cag.engine import CAGChatbot
from quillmark_bot.cag.knowledge import build_knowledge
from quillmark_bot.rag.retriever import Retrieval, Retriever
from quillmark_bot.router import Route, Router
from quillmark_bot.router.llm import LLMRouter

MAX_HISTORY_MESSAGES = 6  # last 3 exchanges

RULES = """\
You are the customer support assistant for Quillmark Bank. You answer customers' questions using the \
policy handbook and known incidents below, plus any past cases and database results given with the question.

Grounding
- Every fact you state (fee, rate, number, time limit, step, menu path, phone number) must appear in the \
handbook, the known incidents, the past cases or the database results. Do not use general banking \
knowledge and do not guess how something works.
- If none of them covers the question, say "I don't have information about that" and offer to connect the \
customer with an agent. Do not describe a process they don't describe, and do not list what other \
policies say instead.
- A policy's conditions apply only to what that policy covers. Never move a waiver, limit or exception \
from one policy to another. When asked how to avoid or remove a fee, give only the ways stated in the \
policy that sets that fee.
- Only send the customer to a team, channel or department that the handbook names for that purpose.

Past cases
- Past cases show how similar situations were resolved. Use them as precedent, but the handbook decides: \
if a past case and a policy disagree, follow the policy.
- Past cases belong to other customers. Never present their amounts, dates or details as this customer's.
- Their own amounts are hidden as [amount]. Describe what happened without them, and never fill one in.

Database results
- Figures about many cases (counts, totals, averages, shares) come only from the database results. Use \
them exactly as given: never estimate, adjust or invent a figure, and never work out new statistics from \
the past cases.
- The database counts exactly what its "Counted:" line says. If that is broader or narrower than what the \
customer asked about, give the figure for what was counted and say what it covers.
- Cite database figures as [case records].
- If the database is unavailable, say you can't give figures right now and answer the rest.

Known incidents
- Incidents only matter when the customer reports a problem. For general questions about a fee, rate or \
rule, answer from the policies without mentioning incidents or asking for a branch or date.
- An incident applies only if all of these match: the customer's product is listed, the customer's branch \
is covered (an incident marked "only" does not apply to any other branch), and the problem happened \
within the incident's reported dates.
- If an incident applies, follow its resolution instead of the routine policy. Tell the customer what is \
reversed, refunded or credited, the amounts, and whether it happens automatically or needs action from them.
- If an incident does not apply, answer with the routine policy as if the incident did not exist. Do not \
mention the incident, not even to say that it does not apply.
- If an incident might apply but the branch or date is missing, ask the customer for it.

Citations
- Cite each policy, incident or past case you rely on by its ID in square brackets, e.g. [POL-DSP-01], \
[INC-2026-03] or [QM-C-000123]. Cite only IDs that directly state the fact you use.

Style
- Answer in a few short sentences without headings: the answer first, then the reason with citations, \
then what the customer should do next.
- Never ask for passwords, PINs, one-time passcodes or full account numbers.
- Only if the question has nothing to do with banking at Quillmark, say you can only help with Quillmark \
Bank questions."""


@dataclass
class Trace:
    """How one answer was produced, for display and debugging."""
    route: Route
    data: analytics.DataResult | None = None
    retrieval: Retrieval | None = None
    case_texts: list[str] | None = None  # the past cases as the model saw them, amounts masked
    seconds_data: float = 0.0


class HybridChatbot:
    def __init__(self, model, processor, retriever: Retriever, max_new_tokens: int = 512):
        self.retriever = retriever
        self.answerer = CAGChatbot(model, processor, build_knowledge(RULES), max_new_tokens=max_new_tokens)
        self.llm_router = LLMRouter(model, processor)
        self.router = Router(llm=self.llm_router)
        vocab = self.router.vocab
        self.coverage = f"cases opened {vocab.first_date} to {vocab.last_date}"
        self.knowledge_ids = set(ID_PATTERN.findall(self.answerer.knowledge))
        # Amounts the handbook and incidents state; any other amount in a past case is that customer's own
        self.policy_figures = checks.numbers_in(self.answerer.knowledge)
        self.last_trace: Trace | None = None

    def build_cache(self) -> None:
        """Read the answer knowledge and the router prompt once each."""
        self.answerer.build_cache()
        self.llm_router.build_cache()

    @torch.inference_mode()
    def ask(self, question: str, history: list[dict] | None = None, on_text=None,
            today: date | None = None, previous_request: analytics.DataRequest | None = None) -> Answer:
        """`previous_request`: the database query behind the previous answer, so a short follow-up
        ("and in Riverton?") can reuse it."""
        today = today or date.today()
        history = (history or [])[-MAX_HISTORY_MESSAGES:]
        route = self.router.route(question, history, today, previous_request)
        trace = Trace(route=route)
        blocks = []

        start = time.perf_counter()
        if route.case_id:
            trace.data = analytics.lookup_case(route.case_id)
        elif route.needs_data and route.data_request:
            trace.data = analytics.run(route.data_request, self.coverage)
        trace.seconds_data = time.perf_counter() - start
        if trace.data:
            blocks.append(f"<database_results>\n{trace.data.text}\n</database_results>")

        if route.needs_cases:
            previous_question = [m["content"] for m in history if m["role"] == "user"][-1:]
            trace.retrieval = self.retriever.search(" ".join(previous_question + [question]), n_knowledge=0)
            trace.case_texts = [checks.mask_amounts(hit.text, self.policy_figures) for hit in trace.retrieval.cases]
            cases = "\n\n".join(trace.case_texts)
            blocks.append(f"<past_cases>\n{cases}\n</past_cases>")

        content = "\n\n".join(blocks + [f"Customer question: {question}"])
        answer = self.answerer.ask(content, history, on_text, today)
        answer.sources = trace.retrieval.ids if trace.retrieval else []
        answer.seconds_retrieval = trace.retrieval.seconds if trace.retrieval else 0.0
        self._check(answer, trace, question, history, today)
        self.last_trace = trace
        return answer

    def _check(self, answer: Answer, trace: Trace, question: str, history: list[dict], today: date) -> None:
        """Flag figures and citations the model wasn't given (see checks.py)."""
        answer.text = checks.normalize_database_tags(answer.text)
        trusted = [self.answerer.knowledge, question, today.isoformat()]
        trusted += [m["content"] for m in history if m["role"] == "user"]  # not earlier answers: they could carry a slip
        if trace.data:
            trusted.append(trace.data.text)
        cases = trace.case_texts or []
        answer.unverified_numbers = checks.unverified_numbers(answer.text, trusted + cases)
        answer.case_only_numbers = checks.case_only_numbers(answer.text, trusted, cases)

        provided = self.knowledge_ids | set(answer.sources)
        if trace.data:
            provided |= set(ID_PATTERN.findall(trace.data.text))
        answer.invalid_citations = checks.invalid_citations(answer.cited_ids, provided)
