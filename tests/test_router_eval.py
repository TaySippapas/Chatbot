"""Every question of the evaluation set must be routed by the rules as expected. No GPU, no database.

Questions whose expected route is the LLM's only have to be handed over to it here; what the LLM then
decides is checked by eval/run_eval.py.
"""
import json
import sys
from datetime import date
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from quillmark_bot.router import Router  # noqa: E402
from scoring import expected_request, route_problems  # noqa: E402

ITEMS = [json.loads(line) for line in (EVAL_DIR / "questions.jsonl").open(encoding="utf-8")]
DEFAULT_TODAY = date(2026, 11, 3)


@pytest.fixture(scope="module")
def router():
    return Router(llm=None)


@pytest.mark.parametrize("item", ITEMS, ids=[i["id"] for i in ITEMS])
def test_rules_route_as_expected(router, item):
    today = date.fromisoformat(item["today"]) if "today" in item else DEFAULT_TODAY
    history, previous = [], None
    for turn in item["turns"]:
        route = router.route(turn["question"], history, today, previous)
        assert route_problems(route, turn.get("route"), rules_only=True) == [], turn["question"]
        history += [{"role": "user", "content": turn["question"]}, {"role": "assistant", "content": "(answer)"}]
        previous = expected_request(turn.get("route"))
