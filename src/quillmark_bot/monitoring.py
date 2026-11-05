"""Record how the chatbot is used and how it scores, for the Grafana dashboard (monitoring/grafana).

Every call is best effort: if Postgres is down the chatbot keeps answering and the record is skipped.
Grafana reads these tables through its own login (grafana_reader), which can only SELECT.

Usage: python -m quillmark_bot.monitoring   (create the tables and the Grafana login; safe to repeat)
"""
import logging
import os
import subprocess

import psycopg
from psycopg import sql

from quillmark_bot.config import MODEL_ID, ROOT, database_url

log = logging.getLogger(__name__)

SCHEMA_PATH = ROOT / "db" / "monitoring.sql"
GRAFANA_ROLE = "grafana_reader"


def _connect() -> psycopg.Connection:
    return psycopg.connect(database_url(), connect_timeout=2, autocommit=True)


def ensure_schema() -> bool:
    """Create the monitoring tables and Grafana's read-only login. Returns False if Postgres is unreachable."""
    try:
        with _connect() as conn:
            conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
            role = sql.Identifier(GRAFANA_ROLE)
            password = sql.Literal(os.getenv("GRAFANA_DB_PASSWORD", GRAFANA_ROLE))
            if conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (GRAFANA_ROLE,)).fetchone():
                conn.execute(sql.SQL("ALTER ROLE {} LOGIN PASSWORD {}").format(role, password))
            else:
                conn.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(role, password))
            conn.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(conn.info.dbname), role))
            conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role))
            conn.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(role))
            # Tables build_db recreates later stay readable too
            conn.execute(sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {}").format(role))
        return True
    except psycopg.Error as error:
        log.warning("Monitoring is unavailable: %s", error)
        return False


def log_interaction(source: str, question: str, answer=None, trace=None, session_id: str | None = None,
                    error: str | None = None) -> int | None:
    """Store one question with its answer (an Answer) and how it was made (a hybrid Trace). Returns its id."""
    row = {"source": source, "session_id": session_id, "model": MODEL_ID, "question": question, "error": error,
           "answer": None, "decided_by": None, "route_reason": None, "used_database": None, "used_cases": None,
           "data_query": None, "data_error": None, "seconds_route": None, "seconds_data": None,
           "seconds_retrieval": None, "seconds_first_token": None, "seconds_answer": None,
           "prompt_tokens": None, "cached_tokens": None, "output_tokens": None, "cited": None,
           "unverified_figures": None, "case_only_figures": None, "invalid_citations": None}
    if answer is not None:
        row.update(answer=answer.text, seconds_first_token=answer.seconds_to_first_token,
                   seconds_answer=answer.seconds_total, seconds_retrieval=answer.seconds_retrieval or None,
                   prompt_tokens=answer.prompt_tokens, cached_tokens=answer.cached_tokens,
                   output_tokens=answer.output_tokens, cited=answer.cited_ids,
                   unverified_figures=answer.unverified_numbers, case_only_figures=answer.case_only_numbers,
                   invalid_citations=answer.invalid_citations)
    if trace is not None:
        route = trace.route
        row.update(decided_by=route.decided_by, route_reason=route.reason, used_database=trace.data is not None,
                   used_cases=trace.retrieval is not None, seconds_route=route.seconds,
                   data_query=trace.data.description if trace.data else None,
                   data_error=trace.data.error if trace.data else None,
                   seconds_data=trace.seconds_data if trace.data else None)
    columns = sql.SQL(", ").join(map(sql.Identifier, row))
    values = sql.SQL(", ").join(map(sql.Placeholder, row))
    try:
        with _connect() as conn:
            return conn.execute(sql.SQL("INSERT INTO interactions ({}) VALUES ({}) RETURNING id").format(columns, values),
                                row).fetchone()[0]
    except psycopg.Error as error_:
        log.warning("Could not record the interaction: %s", error_)
        return None


def set_feedback(interaction_id: int, rating: int | None) -> bool:
    """rating: 1 (helpful), -1 (not helpful) or None (cleared)."""
    try:
        with _connect() as conn:
            conn.execute("UPDATE interactions SET feedback = %s, feedback_at = CASE WHEN %s IS NULL THEN NULL "
                         "ELSE now() END WHERE id = %s", (rating, rating, interaction_id))
        return True
    except psycopg.Error as error:
        log.warning("Could not record feedback: %s", error)
        return False


def git_commit() -> str | None:
    """The checked-out commit, marked "-dirty" when there are uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return None


def log_eval_run(summary: dict, rows: list[dict], question_file: str) -> int | None:
    """Store an evaluation run (eval/run_eval.py) and its per-question results. Returns the run id."""
    try:
        with _connect() as conn, conn.transaction():
            run_id = conn.execute(
                "INSERT INTO eval_runs (git_commit, question_file, questions, answers_correct, routes_correct, "
                "answers_with_unverified_figures, answers_with_case_only_figures, answers_with_invalid_citations, "
                "median_seconds) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (git_commit(), question_file, summary["questions"], summary["answers_correct"],
                 summary["routes_correct"], summary["answers_with_unverified_figures"],
                 summary["answers_with_case_only_figures"], summary["answers_with_invalid_citations"],
                 summary["median_seconds_per_item"])).fetchone()[0]
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO eval_results (run_id, item_id, category, answer_ok, route_ok, problems, seconds) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    [(run_id, r["id"], r["category"], r["answer_ok"], r["route_ok"],
                      r["answer_problems"] + r["route_problems"], r["seconds"]) for r in rows])
        return run_id
    except psycopg.Error as error:
        log.warning("Could not record the evaluation run: %s", error)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if ensure_schema():
        print(f"Monitoring tables and the {GRAFANA_ROLE} login are ready.")
    else:
        raise SystemExit("Postgres is unreachable; start it with: docker compose up -d postgres")
