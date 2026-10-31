-- Quillmark Bank case database. Rebuilt from scratch by `python -m quillmark_bot.data.build_db`.

DROP TABLE IF EXISTS case_policies, cases, incidents, policies;

-- One row per handbook policy (### [POL-...] block)
CREATE TABLE policies (
    policy_id   TEXT PRIMARY KEY,
    area        TEXT NOT NULL,          -- handbook section: Accounts, Cards, ...
    title       TEXT NOT NULL,
    body        TEXT NOT NULL
);

-- No incidents file exists: rows are rebuilt from cases, root causes come from agent notes
CREATE TABLE incidents (
    incident_id     TEXT PRIMARY KEY,
    issue           TEXT,
    branches        TEXT[],             -- all 8 branches = bank-wide incident
    first_opened    DATE,
    last_opened     DATE,
    case_count      INTEGER,
    total_refunded  INTEGER,
    root_cause      TEXT NOT NULL
);

CREATE TABLE cases (
    case_id             TEXT PRIMARY KEY,
    opened_date         DATE NOT NULL,
    closed_date         DATE NOT NULL,
    days_open           INTEGER GENERATED ALWAYS AS (closed_date - opened_date) STORED,
    opened_month        DATE GENERATED ALWAYS AS (date_trunc('month', opened_date::timestamp)::date) STORED,
    channel             TEXT NOT NULL,
    branch              TEXT NOT NULL,
    product             TEXT NOT NULL,
    category            TEXT NOT NULL,
    issue               TEXT NOT NULL,
    customer_tone       TEXT NOT NULL,
    resolution          TEXT NOT NULL,
    outcome             TEXT NOT NULL CHECK (outcome IN ('favorable', 'partial', 'unfavorable', 'info')),
    satisfaction        INTEGER NOT NULL CHECK (satisfaction BETWEEN 1 AND 5),
    amount_refunded     INTEGER NOT NULL CHECK (amount_refunded >= 0),   -- dollars
    incident_id         TEXT REFERENCES incidents,
    is_incident         BOOLEAN GENERATED ALWAYS AS (incident_id IS NOT NULL) STORED,
    meta                JSONB NOT NULL DEFAULT '{}',                     -- sparse, issue-specific keys
    customer_narrative  TEXT NOT NULL,
    agent_notes         TEXT NOT NULL,
    CHECK (closed_date >= opened_date)
);

-- related_policies list, one row per (case, policy)
CREATE TABLE case_policies (
    case_id     TEXT NOT NULL REFERENCES cases ON DELETE CASCADE,
    policy_id   TEXT NOT NULL REFERENCES policies,
    PRIMARY KEY (case_id, policy_id)
);

CREATE INDEX ON cases (opened_date);
CREATE INDEX ON cases (branch);
CREATE INDEX ON cases (product);
CREATE INDEX ON cases (category, issue);
CREATE INDEX ON cases (incident_id) WHERE incident_id IS NOT NULL;
CREATE INDEX ON case_policies (policy_id);
