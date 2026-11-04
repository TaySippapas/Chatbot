"""How an answer and its route are checked against an item of eval/questions.jsonl.

Answer checks: `must` (every group must appear; a group is a phrase or a list of alternatives,
case-insensitive), `must_not`, `cite` (all), `cite_any` (one at least), `no_cite` ("*": nothing),
`handoff` (says it isn't covered, or offers an agent). Route checks compare only the fields an
item's expected route lists.
"""
import re

from quillmark_bot.analytics import DataRequest

FIGURE = re.compile(r"^\$?[\d,]+(?:\.\d+)?%?$")
HANDOFF_PHRASES = ["agent", "don't have", "do not have", "not covered", "isn't covered", "doesn't cover",
                   "does not cover", "no information", "only help", "can't help", "cannot help", "not able to help"]
ASK_LLM = "rules (no llm)"  # what a router without its LLM reports where it would have handed over


def _found(option: str, lower: str) -> bool:
    """A figure must match whole ("$20" is not found in "$20.50"); other phrases match as text."""
    if FIGURE.match(option):
        return re.search(rf"(?<![\d.,]){re.escape(option.lower())}(?!\.?\d|,\d)", lower) is not None
    return option.lower() in lower


def answer_problems(item: dict, text: str, cited: list[str]) -> list[str]:
    lower = text.lower()
    problems = []
    for group in item.get("must", []):
        options = [group] if isinstance(group, str) else group
        if not any(_found(option, lower) for option in options):
            problems.append("missing " + " / ".join(repr(o) for o in options))
    problems += [f"contains {p!r}" for p in item.get("must_not", []) if p.lower() in lower]
    if item.get("handoff") and not any(p in lower for p in HANDOFF_PHRASES):
        problems.append("doesn't say it isn't covered or offer an agent")
    cited_set = set(cited)
    problems += [f"did not cite {c}" for c in item.get("cite", []) if c not in cited_set]
    if item.get("cite_any") and not cited_set & set(item["cite_any"]):
        problems.append(f"cited none of {', '.join(item['cite_any'])}")
    for c in item.get("no_cite", []):
        if c == "*" and cited_set:
            problems.append(f"should cite nothing, cited {', '.join(sorted(cited_set))}")
        elif c in cited_set:
            problems.append(f"should not cite {c}")
    return problems


def route_problems(route, expected: dict | None, rules_only: bool = False) -> list[str]:
    """`rules_only`: the router ran without its LLM, so an expected LLM route only has to be handed over."""
    if not expected:
        return []
    if expected["by"] == "llm":
        if rules_only:
            return [] if route.decided_by == ASK_LLM else [f"the rules decided ({route.reason}) instead of the LLM"]
        if route.decided_by != "llm":
            return [f"expected the LLM to decide, got {route.decided_by} ({route.reason})"]
    elif route.decided_by != "rules":
        return [f"expected the rules to decide, got {route.decided_by} ({route.reason})"]

    problems = []
    if "data" in expected and route.needs_data != expected["data"]:
        problems.append(f"needs_data {route.needs_data}, expected {expected['data']}")
    if "case_id" in expected and route.case_id != expected["case_id"]:
        problems.append(f"case_id {route.case_id}, expected {expected['case_id']}")
    request = route.data_request
    if expected.get("metric") and (not request or request.metric != expected["metric"]):
        problems.append(f"metric {request.metric if request else None}, expected {expected['metric']}")
    for name, values in expected.get("filters", {}).items():
        got = sorted(request.filters.get(name, [])) if request else []
        if got != sorted(values):
            problems.append(f"filter {name} = {got}, expected {sorted(values)}")
    for name in ("group_by", "date_from", "date_to"):
        if name in expected and getattr(request, name, None) != expected[name]:
            problems.append(f"{name} {getattr(request, name, None)}, expected {expected[name]}")
    return problems


def expected_request(expected: dict | None) -> DataRequest | None:
    """The query an expected route stands for, to feed a follow-up when the real one isn't available."""
    if not expected or not expected.get("data"):
        return None
    return DataRequest(expected.get("metric", "count"), expected.get("filters", {}), expected.get("date_from"),
                       expected.get("date_to"), expected.get("group_by"))
