"""Rule-based first pass of the router: plain pattern matching, no model, about a millisecond.

The rules only decide when the text makes the route clear. A question is a database question only
when it has both a data word ("how many", "total", "most") and something being counted across cases
("complaints", "customers", "which branch"): a data word alone is usually a policy question
("how many days do I have to dispute?", "what's the total cost of a wire?"). Anything doubtful is
handed to the LLM together with what the rules found.
"""
import re
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date

from quillmark_bot.router.vocab import Vocab

CASE_ID = re.compile(r"\bqm-c-\d{6}\b")
DATA_WORDS = re.compile(
    r"\b(how many|number of|count|total|sum|average|avg|mean|percent|percentage|proportion|share of|"
    r"rate of|most|least|fewest|top|highest|lowest|compare|comparison|breakdown|statistics|stats|"
    r"trend|how much)\b|%")
# Plural nouns only: the singular is usually the customer's own ("dispute a charge", "my complaint")
COUNTED = re.compile(
    r"\b(cases|complaints|customers|requests|disputes|tickets|people|refunds|refunded|claims)\b"
    r"|\bwhich (branch(es)?|products?|categor(y|ies)|issues?|channels?|months?)\b"
    r"|\b(per|by|each|every) (branch|product|category|issue|channel|month)\b"
    r"|\bmost common (issues?|complaints?|problems?|categor(y|ies))\b")
UNCLEAR = re.compile(
    r"\b(how often|how common|is (it|this|that) common|usually|typically|normally|a lot of|lots of|"
    r"(?<!how )many people|frequently|likely|chances?|odds)\b")
PERSONAL = re.compile(r"\b(i|i'm|i've|my|me|can i|should i|do i|will i)\b")
VAGUE_TIME = re.compile(r"\b(recently|lately|these days|nowadays|a while ago)\b")

OUTCOME_WORDS = re.compile(r"\b(approved|denied|granted|won|lost|declined|rejected|success(ful)?|approval|outcomes?)\b")
REFUND = re.compile(r"\brefund(s|ed)?\b")
GROUP_PATTERNS = [
    (re.compile(r"\b(per|by|each|every) month\b|\bmonthly\b|\bover time\b|\btrend\b"), "month"),
    (re.compile(r"\bwhich branch(es)?\b|\b(per|by|each) branch\b|\bbranch(es)? with the most\b"), "branch"),
    (re.compile(r"\bwhich products?\b|\b(per|by|each) product\b"), "product"),
    (re.compile(r"\bwhich categor(y|ies)\b|\b(per|by|each) category\b|\bmost common categor(y|ies)\b"), "category"),
    (re.compile(r"\bwhich issues?\b|\b(per|by|each) issue\b|\bmost common (issues?|complaints?|problems?)\b"), "issue"),
    (re.compile(r"\bwhich channels?\b|\b(per|by|each) channel\b"), "channel"),
]
COUNT_WORDS = re.compile(r"\b(how many|number of|count|most|least|fewest|top|highest|lowest|compare|comparison)\b")
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]
MONTH = re.compile(r"\b(" + "|".join(MONTHS) + r")(?:\s+(20\d\d))?\b")
YEAR = re.compile(r"\b(20\d\d)\b")
RELATIVE = re.compile(r"\b(this|last|past) (year|month)\b|\b(last|past) (\d+) (days|weeks|months)\b")


@dataclass
class Findings:
    """Everything the rules spotted in a question."""
    case_ids: list[str] = field(default_factory=list)
    values: dict[str, list[str]] = field(default_factory=dict)   # field -> official values
    date_from: str | None = None
    date_to: str | None = None
    vague_time: bool = False
    data_words: list[str] = field(default_factory=list)
    counted: list[str] = field(default_factory=list)
    unclear: list[str] = field(default_factory=list)
    personal: bool = False
    metric: str | None = None
    group_by: str | None = None

    @property
    def is_data(self) -> bool:
        return bool(self.data_words and self.counted)

    def hints(self) -> str:
        """What the rules found, in a form the LLM router can use."""
        parts = [f"{name} = {' or '.join(values)}" for name, values in self.values.items()]
        if self.date_from or self.date_to:
            parts.append(f"dates {self.date_from or '?'} to {self.date_to or '?'}")
        if self.metric:
            parts.append(f"metric = {self.metric}")
        if self.group_by:
            parts.append(f"group_by = {self.group_by}")
        return "; ".join(parts) or "nothing"


def _month_range(year: int, month: int) -> tuple[str, str]:
    return date(year, month, 1).isoformat(), date(year, month, monthrange(year, month)[1]).isoformat()


def _dates(text: str, today: date) -> tuple[str | None, str | None, bool]:
    """(date_from, date_to, vague). More than one period, or a vague one, is left to the LLM."""
    periods = []
    for m in RELATIVE.finditer(text):
        if m.group(2) == "year":
            year = today.year - (m.group(1) != "this")
            periods.append((f"{year}-01-01", today.isoformat() if m.group(1) == "this" else f"{year}-12-31"))
        elif m.group(2) == "month":
            if m.group(1) == "this":
                periods.append((date(today.year, today.month, 1).isoformat(), today.isoformat()))
            else:
                y, mo = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
                periods.append(_month_range(y, mo))
        else:
            days = int(m.group(4)) * {"days": 1, "weeks": 7, "months": 30}[m.group(5)]
            periods.append((date.fromordinal(today.toordinal() - days).isoformat(), today.isoformat()))
    text_without_months = text
    for m in MONTH.finditer(text):
        month = MONTHS.index(m.group(1)) + 1
        if m.group(1) == "may" and not m.group(2):
            continue  # "may" is far more often a verb
        year = int(m.group(2)) if m.group(2) else today.year - (month > today.month)
        periods.append(_month_range(year, month))
        text_without_months = text_without_months.replace(m.group(0), " ")
    for m in YEAR.finditer(text_without_months):
        periods.append((f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"))

    vague = bool(VAGUE_TIME.search(text)) or len(set(periods)) > 1
    if len(set(periods)) == 1:
        return periods[0][0], periods[0][1], vague
    return None, None, vague


def _metric(text: str, group_by: str | None) -> str | None:
    if OUTCOME_WORDS.search(text):
        return "outcome_breakdown"
    if REFUND.search(text):
        if re.search(r"\b(average|avg|mean|typical)\b", text):
            return "avg_refunded"
        if re.search(r"\b(total|sum|how much|altogether)\b", text):
            return "total_refunded"
    if re.search(r"\b(satisfaction|satisfied|rating|ratings)\b", text):
        return "avg_satisfaction"
    if re.search(r"\b(average|avg|mean)\b", text) and re.search(r"\b(days|time|how long|close|resolve)\b", text):
        return "avg_days_open"
    if COUNT_WORDS.search(text) or group_by:
        return "count"
    return None


def analyze(question: str, today: date, vocab: Vocab) -> Findings:
    text = " ".join(question.lower().split())
    found = Findings(case_ids=[c.upper() for c in CASE_ID.findall(text)])
    found.values = vocab.match(text)
    # IDs and names contain dates or month words ("inc-2026-03", "march outage"); drop them first
    found.date_from, found.date_to, found.vague_time = _dates(vocab.remove(CASE_ID.sub(" ", text)), today)
    found.data_words = [m.group(0) for m in DATA_WORDS.finditer(text)]
    found.counted = [m.group(0) for m in COUNTED.finditer(text)]
    found.unclear = [m.group(0) for m in UNCLEAR.finditer(text)]
    found.personal = bool(PERSONAL.search(text))
    found.group_by = next((g for pattern, g in GROUP_PATTERNS if pattern.search(text)), None)
    found.metric = _metric(text, found.group_by)
    return found
