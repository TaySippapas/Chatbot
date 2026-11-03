"""Answer counting questions from Postgres with fixed, parameterised SQL templates.

The router fills a DataRequest. Only whitelisted metrics, filter columns and groupings become SQL,
and filter values travel as query parameters, so text written by the model is never executed.
Queries run in a read-only transaction with a short timeout.
"""
from dataclasses import dataclass, field
from decimal import Decimal

import psycopg
from psycopg.rows import dict_row

from quillmark_bot.config import database_url

# metric -> SQL expression for its value (None: the case count is the value)
METRICS = {
    "count": None,
    "total_refunded": "sum(amount_refunded)",
    "avg_refunded": "avg(amount_refunded)",
    "avg_days_open": "avg(days_open)",
    "avg_satisfaction": "avg(satisfaction)",
    "outcome_breakdown": None,
}
METRIC_LABELS = {
    "count": "number of cases", "total_refunded": "total refunded", "avg_refunded": "average refund",
    "avg_days_open": "average days to close", "avg_satisfaction": "average satisfaction (1-5)",
    "outcome_breakdown": "outcomes",
}
FILTER_COLUMNS = ("branch", "product", "category", "issue", "channel", "outcome", "incident_id")
GROUP_COLUMNS = {"branch": "branch", "product": "product", "category": "category", "issue": "issue",
                 "channel": "channel", "outcome": "outcome", "incident_id": "incident_id",
                 "month": "to_char(opened_month, 'YYYY-MM')"}
MAX_GROUPS = 15
STATEMENT_TIMEOUT = "2s"


@dataclass
class DataRequest:
    metric: str = "count"
    filters: dict[str, list[str]] = field(default_factory=dict)  # OR within a field, AND across fields
    date_from: str | None = None   # opened_date, YYYY-MM-DD, inclusive
    date_to: str | None = None
    group_by: str | None = None

    def describe(self) -> str:
        parts = [METRIC_LABELS[self.metric]]
        conditions = [f"{name} = {' or '.join(values)}" for name, values in self.filters.items()]
        if conditions:
            parts.append("where " + " and ".join(conditions))
        if self.date_from or self.date_to:
            parts.append(f"opened {self.date_from or 'start'} to {self.date_to or 'today'}")
        if self.group_by:
            parts.append(f"by {self.group_by}")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {"metric": self.metric, "filters": self.filters, "date_from": self.date_from,
                "date_to": self.date_to, "group_by": self.group_by}


@dataclass
class DataResult:
    description: str
    sql: str = ""
    params: dict = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    text: str = ""      # what the answering model reads
    error: str | None = None


def build_sql(request: DataRequest) -> tuple[str, dict]:
    if request.metric not in METRICS:
        raise ValueError(f"Unknown metric {request.metric!r}")
    if request.group_by and request.group_by not in GROUP_COLUMNS:
        raise ValueError(f"Unknown grouping {request.group_by!r}")

    groups = []
    if request.group_by:
        groups.append(f"{GROUP_COLUMNS[request.group_by]} AS {request.group_by}")
    if request.metric == "outcome_breakdown" and request.group_by != "outcome":
        groups.append("outcome")
    select = groups + ["count(*) AS cases"]
    if METRICS[request.metric]:
        select.append(f"round({METRICS[request.metric]}, 2) AS value")

    where, params = [], {}
    for name, values in request.filters.items():
        if name == "policy_id":
            where.append("EXISTS (SELECT 1 FROM case_policies cp WHERE cp.case_id = cases.case_id "
                         "AND cp.policy_id = ANY(%(policy_id)s))")
        elif name in FILTER_COLUMNS:
            where.append(f"{name} = ANY(%({name})s)")
        else:
            raise ValueError(f"Unknown filter {name!r}")
        params[name] = list(values)
    if request.date_from:
        where.append("opened_date >= %(date_from)s")
        params["date_from"] = request.date_from
    if request.date_to:
        where.append("opened_date <= %(date_to)s")
        params["date_to"] = request.date_to

    sql = f"SELECT {', '.join(select)}\nFROM cases"
    if where:
        sql += "\nWHERE " + "\n  AND ".join(where)
    if groups:
        sql += f"\nGROUP BY {', '.join(str(i + 1) for i in range(len(groups)))}"
        if request.group_by == "month":
            sql += "\nORDER BY 1"
        else:
            sort = "value" if METRICS[request.metric] else "cases"
            sql += f"\nORDER BY {'1, ' if request.metric == 'outcome_breakdown' and request.group_by else ''}{sort} DESC"
        sql += f"\nLIMIT {MAX_GROUPS * (4 if request.metric == 'outcome_breakdown' and request.group_by else 1)}"
    return sql, params


def _query(sql: str, params: dict) -> list[dict]:
    with psycopg.connect(database_url(), connect_timeout=3, row_factory=dict_row) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [{k: float(v) if isinstance(v, Decimal) else v for k, v in row.items()} for row in rows]


def _format(request: DataRequest, rows: list[dict], coverage: str) -> str:
    money = request.metric in ("total_refunded", "avg_refunded")
    fmt = (lambda v: f"${v:,.2f}") if money else (lambda v: f"{v:,.2f}".rstrip("0").rstrip("."))
    lines = [f"Counted: {request.describe()}."]
    if coverage and not (request.date_from or request.date_to):
        lines.append(f"Period: every case on record ({coverage}).")
    if not rows or (len(rows) == 1 and rows[0]["cases"] == 0):
        return "\n".join(lines + ["Result: no matching cases (0)."])

    total = sum(r["cases"] for r in rows)
    if request.metric == "outcome_breakdown":
        for r in rows:
            prefix = f"{r[request.group_by]} / " if request.group_by and request.group_by != "outcome" else ""
            lines.append(f"- {prefix}{r['outcome']}: {r['cases']} cases")
        if not request.group_by or request.group_by == "outcome":
            header = len(lines) - len(rows)
            lines = lines[:header] + [f"{line} ({r['cases'] / total:.0%} of {total})"
                                      for line, r in zip(lines[header:], rows)]
    elif request.group_by:
        for r in rows:
            value = f"{r['cases']} cases" if request.metric == "count" else f"{fmt(r['value'])} ({r['cases']} cases)"
            share = f", {r['cases'] / total:.0%}" if request.metric == "count" else ""
            lines.append(f"- {r[request.group_by]}: {value}{share}")
        if len(rows) == MAX_GROUPS and request.group_by != "month":
            lines.append(f"(top {MAX_GROUPS} shown)")
    else:
        r = rows[0]
        lines.append(f"Result: {r['cases']} cases" if request.metric == "count"
                     else f"Result: {fmt(r['value'])} across {r['cases']} cases")
    return "\n".join(lines)


def run(request: DataRequest, coverage: str = "") -> DataResult:
    """Run a DataRequest. Database problems come back as a DataResult with `error` set, never raise."""
    result = DataResult(description=request.describe())
    try:
        result.sql, result.params = build_sql(request)
        result.rows = _query(result.sql, result.params)
    except psycopg.Error as error:
        result.error = f"{type(error).__name__}: {error}".strip()
        result.text = "The case database is unavailable right now, so no figures could be retrieved."
        return result
    result.text = _format(request, result.rows, coverage)
    return result


CASE_SQL = """SELECT case_id, opened_date, closed_date, channel, branch, product, category, issue,
       customer_narrative, agent_notes, resolution, amount_refunded, outcome, incident_id,
       (SELECT array_agg(policy_id ORDER BY policy_id) FROM case_policies cp
        WHERE cp.case_id = cases.case_id) AS policies
FROM cases
WHERE case_id = %(case_id)s"""


def lookup_case(case_id: str) -> DataResult:
    result = DataResult(description=f"case {case_id}", sql=CASE_SQL, params={"case_id": case_id})
    try:
        result.rows = _query(CASE_SQL, result.params)
    except psycopg.Error as error:
        result.error = f"{type(error).__name__}: {error}".strip()
        result.text = "The case database is unavailable right now, so the case could not be looked up."
        return result
    if not result.rows:
        result.text = f"No case with ID {case_id} exists."
        return result
    c = result.rows[0]
    result.text = (
        f"Case [{c['case_id']}]: opened {c['opened_date']}, closed {c['closed_date']}, {c['channel']}, "
        f"{c['branch']} branch, {c['product']}, {c['category']} / {c['issue']}\n"
        f"Customer: {c['customer_narrative']}\n"
        f"Agent notes: {c['agent_notes']}\n"
        f"Resolution: {c['resolution']} ({c['outcome']}), refunded ${c['amount_refunded']:,.2f}"
        + (f", incident {c['incident_id']}" if c["incident_id"] else "")
        + (f", policies {', '.join(c['policies'])}" if c["policies"] else "")
    )
    return result
