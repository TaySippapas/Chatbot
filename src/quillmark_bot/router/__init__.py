"""Decide, per question, whether to retrieve past cases and/or query the case database.

Rules decide when the text is clear; otherwise the LLM fills the same decision form. The cached
handbook and incidents are always in the answer prompt, so the router never has to choose them.
"""
import time
from dataclasses import dataclass
from datetime import date

from quillmark_bot.analytics import DataRequest
from quillmark_bot.router.rules import Findings, analyze
from quillmark_bot.router.vocab import Vocab, load_vocab

FOLLOW_UP_MAX_WORDS = 8


@dataclass
class Route:
    needs_cases: bool = True
    needs_data: bool = False
    data_request: DataRequest | None = None
    case_id: str | None = None
    decided_by: str = "rules"      # "rules", "llm", or "rules (llm failed)"
    reason: str = ""
    llm_output: str | None = None  # the raw form the LLM wrote, for display
    seconds: float = 0.0


def request_from_findings(found: Findings) -> DataRequest:
    filters = {name: values for name, values in found.values.items()}
    group_by = found.group_by
    # Two values of the same field ("Bayside vs Riverton") means compare them
    if not group_by:
        group_by = next((name for name, values in filters.items() if len(values) > 1), None)
    return DataRequest(found.metric or "count", filters, found.date_from, found.date_to, group_by)


class Router:
    def __init__(self, llm=None, vocab: Vocab | None = None):
        """`llm` is an LLMRouter (router.llm); without one, unclear questions get the default route."""
        self.llm = llm
        self.vocab = vocab or load_vocab()

    def route(self, question: str, history: list[dict] | None = None, today: date | None = None,
              previous_request: DataRequest | None = None) -> Route:
        """`previous_request` is the database query the previous answer used, if any, however it was routed."""
        start = time.perf_counter()
        today = today or date.today()
        previous = next((m["content"] for m in reversed(history or []) if m["role"] == "user"), None)
        found = analyze(question, today, self.vocab)
        route = self._decide(found, question, previous, today, previous_request)
        route.seconds = time.perf_counter() - start
        return route

    def _decide(self, found: Findings, question: str, previous: str | None, today: date,
                previous_request: DataRequest | None = None) -> Route:
        # 1. A case ID: look that case up
        if found.case_ids:
            return Route(needs_cases=True, case_id=found.case_ids[0], reason=f"case ID {found.case_ids[0]}")

        ask = lambda reason: self._ask_llm(question, previous, today, found, reason, previous_request)

        # 2-3. Data word + something counted across cases
        if found.is_data:
            why = f"data question ({found.data_words[0]!r} + {found.counted[0]!r})"
            if found.personal:
                return ask(f"{why} with a personal part")
            # Nothing recognised to filter, group or date by: the subject is in words the rules don't know
            # ("collectors calling at work"), and a bare count would cover all cases
            subject_known = found.values or found.group_by or found.date_from
            if found.unclear or found.vague_time or not found.metric or not subject_known:
                missing = ("unclear wording" if found.unclear else "vague dates" if found.vague_time
                           else "no metric recognised" if not found.metric else "subject not recognised")
                return ask(f"{why}, but {missing}")
            return Route(needs_cases=False, needs_data=True, data_request=request_from_findings(found), reason=why)

        # 4. Words that may or may not want numbers
        if found.unclear:
            return ask(f"unclear wording ({found.unclear[0]!r})")

        # 5. Short follow-up to an answer that used the database ("and in Riverton?"), whether the rules
        # or the LLM routed it
        if previous and len(question.split()) <= FOLLOW_UP_MAX_WORDS and (found.values or found.date_from):
            if previous_request or analyze(previous, today, self.vocab).is_data:
                return ask("short follow-up to a data question")

        # 6. Default: the handbook (always there) plus similar past cases
        return Route(needs_cases=True, reason="no data signals: handbook + similar cases")

    def _ask_llm(self, question, previous, today, found: Findings, reason: str,
                 previous_request: DataRequest | None = None) -> Route:
        if self.llm is None:
            return Route(needs_cases=True, decided_by="rules (no llm)", reason=f"{reason}; default route")
        route = self.llm.decide(question, previous, today, found.hints(), previous_request)
        route.reason = f"{reason} → {route.reason}" if route.reason else reason
        return route
