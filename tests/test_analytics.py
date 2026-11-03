"""SQL templates: safe by construction, and correct against the raw data when Postgres is running."""
import json
from collections import Counter

import psycopg
import pytest

from quillmark_bot.analytics import DataRequest, build_sql, run
from quillmark_bot.config import CASES_PATH, database_url


def test_values_are_parameters_not_sql():
    sql, params = build_sql(DataRequest("count", {"branch": ["Bayside'; DROP TABLE cases; --"]}))
    assert "DROP" not in sql and params["branch"] == ["Bayside'; DROP TABLE cases; --"]


@pytest.mark.parametrize("request_", [
    DataRequest("delete_everything"),
    DataRequest("count", {"password": ["x"]}),
    DataRequest("count", group_by="customer_narrative"),
])
def test_unknown_metric_filter_or_grouping_is_rejected(request_):
    with pytest.raises(ValueError):
        build_sql(request_)


def _database_up() -> bool:
    try:
        psycopg.connect(database_url(), connect_timeout=2).close()
        return True
    except psycopg.Error:
        return False


needs_db = pytest.mark.skipif(not _database_up(), reason="Postgres is not running (docker compose up -d)")


@pytest.fixture(scope="module")
def cases():
    return [json.loads(line) for line in CASES_PATH.open(encoding="utf-8")]


@needs_db
def test_count_and_dates_match_raw_data(cases):
    rows = run(DataRequest("count", {"category": ["Fraud"]}, "2025-01-01", "2025-12-31", "branch")).rows
    expected = Counter(c["branch"] for c in cases if c["category"] == "Fraud" and c["opened_date"].startswith("2025"))
    assert {r["branch"]: r["cases"] for r in rows} == expected


@needs_db
def test_refund_total_keeps_cents(cases):
    rows = run(DataRequest("total_refunded", {"incident_id": ["INC-2026-03"]})).rows
    expected = round(sum(c["amount_refunded"] for c in cases if c["incident_id"] == "INC-2026-03"), 2)
    assert rows[0]["value"] == expected


@needs_db
def test_policy_filter_uses_case_policies(cases):
    rows = run(DataRequest("count", {"policy_id": ["POL-DSP-04"]})).rows
    assert rows[0]["cases"] == sum("POL-DSP-04" in c["related_policies"] for c in cases)
