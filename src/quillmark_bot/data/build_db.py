"""Rebuild the Postgres tables from cases.jsonl and the policy handbook.

Usage: python -m quillmark_bot.data.build_db
"""
import psycopg
from psycopg.types.json import Jsonb

from quillmark_bot import monitoring
from quillmark_bot.config import CASES_PATH, HANDBOOK_PATH, SCHEMA_PATH, database_url
from quillmark_bot.data.incidents import REFRESH_INCIDENT_STATS, ROOT_CAUSES
from quillmark_bot.data.load import load_cases, load_policies

CASE_COLUMNS = [
    "case_id", "opened_date", "closed_date", "channel", "branch", "product", "category", "issue",
    "customer_tone", "resolution", "outcome", "satisfaction", "amount_refunded", "incident_id",
    "meta", "customer_narrative", "agent_notes",
]


def main() -> None:
    cases = load_cases(CASES_PATH)
    policies = load_policies(HANDBOOK_PATH)

    missing = {c["incident_id"] for c in cases if c["incident_id"]} - ROOT_CAUSES.keys()
    if missing:
        raise ValueError(f"No root cause written for incidents: {sorted(missing)}")

    with psycopg.connect(database_url()) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))

        cur.executemany(
            "INSERT INTO policies (policy_id, area, title, body) "
            "VALUES (%(policy_id)s, %(area)s, %(title)s, %(body)s)",
            policies,
        )
        cur.executemany(
            "INSERT INTO incidents (incident_id, root_cause) VALUES (%s, %s)",
            list(ROOT_CAUSES.items()),
        )
        cur.executemany(
            f"INSERT INTO cases ({', '.join(CASE_COLUMNS)}) "
            f"VALUES ({', '.join(f'%({c})s' for c in CASE_COLUMNS)})",
            [{**c, "meta": Jsonb(c["meta"])} for c in cases],
        )
        cur.executemany(
            "INSERT INTO case_policies (case_id, policy_id) VALUES (%s, %s)",
            [(c["case_id"], p) for c in cases for p in c["related_policies"]],
        )
        cur.execute(REFRESH_INCIDENT_STATS)

        for table in ("policies", "incidents", "cases", "case_policies"):
            cur.execute(f"SELECT count(*) FROM {table}")
            print(f"{table:<14} {cur.fetchone()[0]:>6,} rows")

    # The monitoring tables live alongside (and survive this rebuild); make sure they and Grafana's login exist
    monitoring.ensure_schema()


if __name__ == "__main__":
    main()
