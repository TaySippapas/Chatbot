"""Monitoring records: written and read back in Postgres, then removed. Skipped when Postgres is down."""
from types import SimpleNamespace

import psycopg
import pytest

from quillmark_bot import monitoring
from quillmark_bot.answer import Answer
from quillmark_bot.config import database_url


def _database_up() -> bool:
    try:
        psycopg.connect(database_url(), connect_timeout=2).close()
        return True
    except psycopg.Error:
        return False


pytestmark = pytest.mark.skipif(not _database_up(), reason="Postgres is not running (docker compose up -d)")


@pytest.fixture(scope="module")
def db():
    assert monitoring.ensure_schema()
    with psycopg.connect(database_url(), autocommit=True) as conn:
        yield conn


def test_interaction_with_route_and_flags(db):
    answer = Answer(text="57 cases [case records].", prompt_tokens=6400, cached_tokens=6300, output_tokens=12,
                    seconds_to_first_token=0.3, seconds_total=2.1, unverified_numbers=["57"])
    trace = SimpleNamespace(route=SimpleNamespace(decided_by="rules", reason="data question", seconds=0.001),
                            data=SimpleNamespace(description="number of cases", error=None),
                            retrieval=None, seconds_data=0.02)
    interaction_id = monitoring.log_interaction("cli", "How many?", answer, trace, session_id="test")
    try:
        row = db.execute("SELECT decided_by, used_database, used_cases, unverified_figures, cached_tokens "
                         "FROM interactions WHERE id = %s", (interaction_id,)).fetchone()
        assert row == ("rules", True, False, ["57"], 6300)
        assert monitoring.set_feedback(interaction_id, -1)
        assert db.execute("SELECT feedback FROM interactions WHERE id = %s", (interaction_id,)).fetchone() == (-1,)
    finally:
        db.execute("DELETE FROM interactions WHERE id = %s", (interaction_id,))


def test_failed_question_is_recorded(db):
    interaction_id = monitoring.log_interaction("app", "Hello?", error="CUDA out of memory")
    try:
        assert db.execute("SELECT error, answer FROM interactions WHERE id = %s",
                          (interaction_id,)).fetchone() == ("CUDA out of memory", None)
    finally:
        db.execute("DELETE FROM interactions WHERE id = %s", (interaction_id,))


def test_evaluation_run_with_results(db):
    summary = {"questions": 2, "answers_correct": 1, "routes_correct": 2, "answers_with_unverified_figures": 0,
               "answers_with_case_only_figures": 0, "answers_with_invalid_citations": 0, "median_seconds_per_item": 5.0}
    rows = [{"id": "a", "category": "policy", "answer_ok": True, "route_ok": True, "answer_problems": [],
             "route_problems": [], "seconds": 4.0},
            {"id": "b", "category": "policy", "answer_ok": False, "route_ok": True, "answer_problems": ["missing '$20'"],
             "route_problems": [], "seconds": 6.0}]
    run_id = monitoring.log_eval_run(summary, rows, "test.jsonl")
    try:
        assert db.execute("SELECT count(*) FROM eval_results WHERE run_id = %s", (run_id,)).fetchone() == (2,)
    finally:
        db.execute("DELETE FROM eval_runs WHERE id = %s", (run_id,))  # cascades to eval_results
