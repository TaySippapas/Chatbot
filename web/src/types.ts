// Shapes sent by src/quillmark_bot/server.py

export type Health = { status: "loading" | "ready" | "error"; message: string; model: string; logging: boolean };

export type RouteInfo = {
  decided_by: string; // "rules", "llm", "rules (llm failed)"
  reason: string;
  llm_output: string | null;
  seconds: number;
  needs_data: boolean;
  needs_cases: boolean;
  case_id: string | null;
};

export type DataInfo = {
  description: string;
  text: string;
  sql: string;
  params: Record<string, unknown>;
  rows: Record<string, unknown>[];
  error: string | null;
};

export type CaseHit = { id: string; score: number; title: string };

export type DataRequest = Record<string, unknown>;

export type AnswerRecord = {
  text: string;
  cited: string[];
  stopped: boolean;
  unverified_numbers: string[];
  case_only_numbers: string[];
  invalid_citations: string[];
  request: DataRequest | null;
  interaction_id?: number | null;
  stats: {
    seconds_route: number;
    seconds_data: number | null;
    seconds_retrieval: number | null;
    seconds_first_token: number;
    seconds_answer: number;
    prompt_tokens: number;
    cached_tokens: number;
    output_tokens: number;
  };
};

export type StepState = "pending" | "active" | "done" | "failed";
export type StepKind = "read" | "data" | "cases" | "write";
export type Step = { kind: StepKind; state: StepState; detail?: string };

export type UserMessage = { role: "user"; id: string; text: string };

export type AssistantMessage = {
  role: "assistant";
  id: string;
  steps: Step[];
  route?: RouteInfo;
  data?: DataInfo;
  cases?: CaseHit[];
  streamed: string; // words as they arrive
  record?: AnswerRecord; // the checked answer, once done
  error?: string;
  stoppedByUser?: boolean;
  queued?: boolean;
  startedAt: number;
  writingAt?: number;
  firstWordAt?: number;
  finishedAt?: number;
  feedback?: 1 | -1 | null;
};

export type Message = UserMessage | AssistantMessage;
