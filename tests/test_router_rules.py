"""Rule-based routing: question in, expected route out. No GPU, no database."""
from datetime import date

import pytest

from quillmark_bot.router import Router

TODAY = date(2026, 11, 3)
ASK_LLM = "rules (no llm)"  # what the router reports when it would hand over to the LLM


@pytest.fixture(scope="module")
def router():
    return Router(llm=None)


@pytest.mark.parametrize("question", [
    "What's the overdraft fee?",
    "How many days do I have to dispute a charge?",
    "What's the total cost of an international wire?",
    "I was charged twice on March 15. What happens now?",
    "What is the average APR on personal loans?",
    "Write me a short poem about the ocean.",
])
def test_default_route_handbook_and_cases(router, question):
    route = router.route(question, today=TODAY)
    assert route.decided_by == "rules"
    assert route.needs_cases and not route.needs_data


def test_case_id_lookup(router):
    route = router.route("Tell me about QM-C-000996", today=TODAY)
    assert route.case_id == "QM-C-000996" and route.decided_by == "rules"


def test_count_with_filters_and_month(router):
    route = router.route("How many duplicate-charge complaints came from Bayside in March 2026?", today=TODAY)
    assert route.decided_by == "rules" and route.needs_data and not route.needs_cases
    r = route.data_request
    assert r.metric == "count"
    assert r.filters == {"issue": ["Duplicate charge"], "branch": ["Bayside"]}
    assert (r.date_from, r.date_to) == ("2026-03-01", "2026-03-31")


def test_group_by_branch_last_year(router):
    r = router.route("Which branch had the most fraud complaints last year?", today=TODAY).data_request
    assert (r.metric, r.group_by, r.filters) == ("count", "branch", {"category": ["Fraud"]})
    assert (r.date_from, r.date_to) == ("2025-01-01", "2025-12-31")


def test_total_refunded_for_incident_alias(router):
    r = router.route("What was the total refunded to customers for the March outage?", today=TODAY).data_request
    assert (r.metric, r.filters) == ("total_refunded", {"incident_id": ["INC-2026-03"]})
    assert r.date_from is None  # "march" belongs to the incident name, not a date


def test_outcome_breakdown(router):
    r = router.route("What percentage of goodwill requests were approved?", today=TODAY).data_request
    assert (r.metric, r.filters) == ("outcome_breakdown", {"issue": ["Goodwill request"]})


def test_two_branches_are_compared(router):
    r = router.route("Compare the number of lockout complaints in Bayside and Riverton", today=TODAY).data_request
    assert r.group_by == "branch" and r.filters["branch"] == ["Bayside", "Riverton"]


@pytest.mark.parametrize("question", [
    "How often do goodwill requests get approved?",                     # unclear wording
    "How many complaints did we get recently?",                         # vague dates
    "How many complaints were there, and should I file one too?",       # data + personal part
    "How many people complained about collectors calling them at work?",  # subject not in the vocabulary
])
def test_unclear_questions_go_to_llm(router, question):
    assert router.route(question, today=TODAY).decided_by == ASK_LLM


def test_short_follow_up_goes_to_llm(router):
    history = [{"role": "user", "content": "How many duplicate-charge complaints came from Bayside?"},
               {"role": "assistant", "content": "57."}]
    assert router.route("And in Riverton?", history, TODAY).decided_by == ASK_LLM
