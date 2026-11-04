"""LLM half of the router: for questions the rules can't settle, the model fills a small JSON form.

The model only writes the form; code validates every field against the allowed values and runs the
fixed SQL templates. The prompt (instructions, allowed values, examples) never changes, so it is
cached once like the CAG knowledge and each decision only reads the question.
"""
import json
import re
from datetime import date

from quillmark_bot.analytics import GROUP_COLUMNS, METRICS, DataRequest
from quillmark_bot.cag.engine import CAGChatbot
from quillmark_bot.config import CASES_PATH
from quillmark_bot.data.incidents import incident_entries
from quillmark_bot.data.load import load_cases
from quillmark_bot.router import Route
from quillmark_bot.router.vocab import Vocab, load_vocab

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

INSTRUCTIONS = """\
You are the router for Quillmark Bank's customer support assistant. You do not answer the customer. \
You decide what the assistant needs to answer: (a) figures counted across the bank's past cases, from \
its case database, and (b) examples of similar past cases. The policy handbook is always available, so \
you never need to ask for it.

Reply with one line of JSON and nothing else, in this shape:
{"needs_data": true|false, "needs_cases": true|false, "data_request": null or {"metric": "...", \
"filters": {"field": ["value"]}, "date_from": "YYYY-MM-DD" or null, "date_to": "YYYY-MM-DD" or null, \
"group_by": "..." or null}}

- needs_data: true only if the answer needs numbers about many past cases: how many, how often, totals, \
averages, shares, comparisons, trends. A question about a rule, fee, limit or deadline does not need \
data, even if it says "how many" or "how much"; the handbook answers it.
- needs_cases: true if examples of how similar customer situations were resolved would help the \
customer. False for questions that only ask for statistics.
- metric: count, total_refunded, avg_refunded, avg_days_open, avg_satisfaction, or outcome_breakdown \
(how cases ended: favorable, partial, unfavorable, info; use it for "how often are they approved").
- filters: only fields and exact values from the allowed values below, only what the question mentions.
- group_by: branch, product, category, issue, channel, outcome, incident_id, month, or null.
- Dates: turn relative dates into date_from and date_to using today's date. "Recently" means the last \
3 months. Leave both null when the question gives no period.
- "Rule matcher found" lists values already recognised in the question; use them when they fit.
- "Previous query" is the database query behind the previous answer. A short follow-up usually keeps it \
and changes only what the new question names."""

EXAMPLES = [
    ("2026-11-03", None, "product = Spark Card", "What's the late fee on the Spark Card?",
     {"needs_data": False, "needs_cases": True, "data_request": None}),
    ("2026-11-03", None, "issue = Goodwill request; metric = outcome_breakdown",
     "How often do goodwill requests get approved?",
     {"needs_data": True, "needs_cases": True, "data_request": {
         "metric": "outcome_breakdown", "filters": {"issue": ["Goodwill request"]},
         "date_from": None, "date_to": None, "group_by": None}}),
    ("2026-04-20", None, "incident_id = INC-2026-03; metric = count",
     "How many complaints about the March outage came in recently?",
     {"needs_data": True, "needs_cases": False, "data_request": {
         "metric": "count", "filters": {"incident_id": ["INC-2026-03"]},
         "date_from": "2026-01-20", "date_to": "2026-04-20", "group_by": None}}),
    ("2026-11-03", None, "issue = Credit limit increase; metric = count",
     "How many credit limit increase requests can I make in a year?",
     {"needs_data": False, "needs_cases": True, "data_request": None}),
    ("2026-11-03", None, "category = Disputes; metric = outcome_breakdown", "Is it common for disputes to be won?",
     {"needs_data": True, "needs_cases": True, "data_request": {
         "metric": "outcome_breakdown", "filters": {"category": ["Disputes"]},
         "date_from": None, "date_to": None, "group_by": None}}),
    ("2026-11-03", "How many duplicate-charge complaints came from Bayside?", "branch = Riverton",
     "And in Riverton?",
     {"needs_data": True, "needs_cases": False, "data_request": {
         "metric": "count", "filters": {"issue": ["Duplicate charge"], "branch": ["Riverton"]},
         "date_from": None, "date_to": None, "group_by": None}}),
    ("2026-11-03", None, "category = Disputes", "How long do disputes usually take to close?",
     {"needs_data": True, "needs_cases": False, "data_request": {
         "metric": "avg_days_open", "filters": {"category": ["Disputes"]},
         "date_from": None, "date_to": None, "group_by": None}}),
    ("2026-11-03", None, "issue = App lockout",
     "Lots of people seem to get locked out of the app. Why does that happen and how do I fix it?",
     {"needs_data": False, "needs_cases": True, "data_request": None}),
    ("2026-11-03", None, "product = Horizon Rewards Card; issue = Card late fee; dates 2026-01-01 to 2026-11-03; "
     "metric = count; group_by = month",
     "Which months had the most Horizon card late fee complaints this year?",
     {"needs_data": True, "needs_cases": False, "data_request": {
         "metric": "count", "filters": {"product": ["Horizon Rewards Card"], "issue": ["Card late fee"]},
         "date_from": "2026-01-01", "date_to": "2026-11-03", "group_by": "month"}}),
    ("2026-11-03", None, "metric = count",
     "How many customers complained about the way debt collectors contacted them, and what are my rights?",
     {"needs_data": True, "needs_cases": True, "data_request": {
         "metric": "count", "filters": {"issue": ["Collection contact practices", "Third-party collector conduct"]},
         "date_from": None, "date_to": None, "group_by": None}}),
]


# The database query behind the previous answer in the follow-up example, so the model learns to carry it over
EXAMPLE_PREVIOUS_QUERY = {"metric": "count", "filters": {"issue": ["Duplicate charge"], "branch": ["Bayside"]}}


def compact(request: dict) -> str:
    return json.dumps({k: v for k, v in request.items() if v})


def user_message(question: str, previous: str | None, hints: str, previous_query: dict | None = None) -> str:
    previous_lines = f"Previous question: {previous}\n" if previous else ""
    if previous_query:
        previous_lines += f"Previous query: {compact(previous_query)}\n"
    return f"{previous_lines}Rule matcher found: {hints}\nQuestion: {question}"


def router_prompt(vocab: Vocab) -> str:
    incident_labels = {iid: text.splitlines()[0].split("] ", 1)[1]
                       for iid, text in incident_entries(load_cases(CASES_PATH))}
    allowed = "\n".join(
        f"{name}: " + (", ".join(f"{v} ({incident_labels.get(v, '')})" for v in values) if name == "incident_id"
                       else ", ".join(values))
        for name, values in vocab.values.items())
    examples = "\n\n".join(
        f"Today's date: {today}\n"
        f"{user_message(question, previous, hints, EXAMPLE_PREVIOUS_QUERY if previous else None)}\n"
        f"{json.dumps(form)}"
        for today, previous, hints, question, form in EXAMPLES)
    return f"{INSTRUCTIONS}\n\nAllowed values\n{allowed}\n\nExamples\n\n{examples}"


def parse_decision(text: str, vocab: Vocab) -> Route:
    """Turn the model's form into a Route, keeping only valid fields and values."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        form = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        form = None
    if not isinstance(form, dict):
        return Route(needs_cases=True, decided_by="rules (llm failed)", reason="unreadable form; default route",
                     llm_output=text)

    notes = []
    request = None
    spec = form.get("data_request")
    if form.get("needs_data") and isinstance(spec, dict):
        filters = {}
        for name, values in (spec.get("filters") or {}).items():
            values = values if isinstance(values, list) else [values]
            valid = [v for v in values if name in vocab.values and v in vocab.values[name]]
            notes += [f"dropped {name}={v!r}" for v in values if v not in valid]
            if valid:
                filters[name] = valid
        metric = spec.get("metric") if spec.get("metric") in METRICS else "count"
        group_by = spec.get("group_by") if spec.get("group_by") in GROUP_COLUMNS else None
        dates = [d if isinstance(d, str) and ISO_DATE.match(d) else None
                 for d in (spec.get("date_from"), spec.get("date_to"))]
        request = DataRequest(metric, filters, dates[0], dates[1], group_by)

    reason = f"needs_data={request is not None}, needs_cases={bool(form.get('needs_cases'))}"
    if notes:
        reason += f" ({'; '.join(notes)})"
    return Route(needs_cases=bool(form.get("needs_cases")), needs_data=request is not None, data_request=request,
                 decided_by="llm", reason=reason, llm_output=text)


class LLMRouter:
    def __init__(self, model, processor, vocab: Vocab | None = None):
        self.vocab = vocab or load_vocab()
        self.bot = CAGChatbot(model, processor, router_prompt(self.vocab), max_new_tokens=200)

    def build_cache(self) -> None:
        self.bot.build_cache()

    def decide(self, question: str, previous: str | None, today: date, hints: str,
               previous_request: DataRequest | None = None) -> Route:
        previous_query = previous_request.to_dict() if previous_request else None
        answer = self.bot.ask(user_message(question, previous, hints, previous_query), today=today)
        return parse_decision(answer.text, self.vocab)
