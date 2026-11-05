-- How the chatbot is used and how it scores, for the Grafana dashboard (monitoring/grafana).
-- Kept apart from schema.sql, which build_db drops and recreates: rebuilding the case data never
-- deletes this history. Safe to run again: python -m quillmark_bot.monitoring

-- One row per answered (or failed) question
CREATE TABLE IF NOT EXISTS interactions (
    id                   bigserial PRIMARY KEY,
    created_at           timestamptz NOT NULL DEFAULT now(),
    source               text NOT NULL CHECK (source IN ('app', 'cli', 'eval')),
    session_id           text,
    model                text,
    question             text NOT NULL,
    answer               text,
    error                text,
    decided_by           text,           -- rules, llm, rules (llm failed)
    route_reason         text,
    used_database        boolean,
    used_cases           boolean,
    data_query           text,
    data_error           text,
    seconds_route        real,
    seconds_data         real,
    seconds_retrieval    real,
    seconds_first_token  real,
    seconds_answer       real,
    prompt_tokens        integer,
    cached_tokens        integer,
    output_tokens        integer,
    cited                text[],
    unverified_figures   text[],
    case_only_figures    text[],
    invalid_citations    text[],
    feedback             smallint CHECK (feedback IN (-1, 1)),
    feedback_at          timestamptz
);
CREATE INDEX IF NOT EXISTS interactions_created_at ON interactions (created_at);

-- One row per evaluation run (eval/run_eval.py), and one per question in it
CREATE TABLE IF NOT EXISTS eval_runs (
    id                              bigserial PRIMARY KEY,
    created_at                      timestamptz NOT NULL DEFAULT now(),
    git_commit                      text,
    question_file                   text NOT NULL,
    questions                       integer NOT NULL,
    answers_correct                 integer NOT NULL,
    routes_correct                  integer NOT NULL,
    answers_with_unverified_figures integer NOT NULL,
    answers_with_case_only_figures  integer NOT NULL,
    answers_with_invalid_citations  integer NOT NULL,
    median_seconds                  real
);

CREATE TABLE IF NOT EXISTS eval_results (
    run_id      bigint NOT NULL REFERENCES eval_runs ON DELETE CASCADE,
    item_id     text NOT NULL,
    category    text,
    answer_ok   boolean NOT NULL,
    route_ok    boolean NOT NULL,
    problems    text[],
    seconds     real,
    PRIMARY KEY (run_id, item_id)
);
