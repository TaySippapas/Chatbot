import type { AnswerRecord, CaseHit, DataInfo, DataRequest, Health, RouteInfo } from "./types";

export async function getHealth(): Promise<Health> {
  const response = await fetch("/api/health");
  if (!response.ok) throw new Error(`Server answered ${response.status}`);
  return response.json();
}

export async function getDocument(id: string): Promise<string> {
  const response = await fetch(`/api/documents/${encodeURIComponent(id)}`);
  if (!response.ok) throw new Error(response.status === 404 ? `${id} wasn't found` : `Server answered ${response.status}`);
  return (await response.json()).text;
}

export async function sendFeedback(interactionId: number, rating: 1 | -1 | null): Promise<void> {
  await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ interaction_id: interactionId, rating }),
  });
}

export type ChatHandlers = {
  onQueued: () => void;
  onRoute: (route: RouteInfo) => void;
  onData: (data: DataInfo) => void;
  onCases: (cases: CaseHit[]) => void;
  onWriting: () => void;
  onToken: (text: string) => void;
  onDone: (record: AnswerRecord) => void;
  onError: (message: string) => void;
};

export type ChatBody = {
  question: string;
  history: { role: "user" | "assistant"; content: string }[];
  today: string;
  previous_request: DataRequest | null;
  session_id: string;
};

/** POST a question and dispatch the server-sent events as they arrive. Abort `signal` to stop. */
export async function streamChat(body: ChatBody, handlers: ChatHandlers, signal: AbortSignal): Promise<void> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Server answered ${response.status}`);
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let end: number;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      dispatch(buffer.slice(0, end), handlers);
      buffer = buffer.slice(end + 2);
    }
  }
}

function dispatch(block: string, h: ChatHandlers) {
  let name = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return; // keep-alive comments
  const payload = JSON.parse(data.join("\n"));
  switch (name) {
    case "status":
      if (payload.stage === "queued") h.onQueued();
      break;
    case "route": return h.onRoute(payload);
    case "data": return h.onData(payload);
    case "cases": return h.onCases(payload.cases);
    case "writing": return h.onWriting();
    case "token": return h.onToken(payload.text);
    case "done": return h.onDone(payload);
    case "error": return h.onError(payload.message);
  }
}
