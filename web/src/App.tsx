import { useCallback, useEffect, useRef, useState } from "react";
import { getHealth, sendFeedback, streamChat } from "./api";
import { AssistantMessage } from "./components/AssistantMessage";
import { CitationDrawer } from "./components/CitationDrawer";
import { Composer } from "./components/Composer";
import { EmptyState } from "./components/EmptyState";
import { Sidebar } from "./components/Sidebar";
import type { AssistantMessage as Answer, DataInfo, Health, Message, RouteInfo, Step, StepKind } from "./types";

const localToday = () => new Date(Date.now() - new Date().getTimezoneOffset() * 60000).toISOString().slice(0, 10);
const newId = () => crypto.randomUUID();

function routeDetail(route: RouteInfo): string {
  if (route.decided_by === "rules") return `Clear question, matched instantly by the rules (${Math.max(1, Math.round(route.seconds * 1000))} ms)`;
  if (route.decided_by === "llm") return `Worked out by the AI router (${route.seconds.toFixed(1)} s)`;
  return "Couldn't be worked out, so the handbook and similar cases were used";
}

/** One line for the progress list: the figure itself where there is one ("54 cases"). */
function dataHeadline(data: DataInfo): string {
  if (data.error) return "Case records are unavailable right now";
  const lines = data.text.split("\n");
  const result = lines.find((l) => l.startsWith("Result:"));
  if (result) return result.replace("Result: ", "");
  const items = lines.filter((l) => l.startsWith("- ")).map((l) => l.slice(2));
  if (items.length) return items.slice(0, 2).join(" · ") + (items.length > 2 ? ` · +${items.length - 2} more` : "");
  return lines[0];
}

const setStep = (steps: Step[], kind: StepKind, change: Partial<Step>) =>
  steps.map((s) => (s.kind === kind ? { ...s, ...change } : s));

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const [today, setToday] = useState(localToday);
  const [citation, setCitation] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const sessionId = useRef(newId());
  const bottom = useRef<HTMLDivElement>(null);
  const ready = health?.status === "ready";

  // Poll until the model is loaded (the server loads it at startup)
  useEffect(() => {
    let timer = 0;
    const poll = async () => {
      try {
        const h = await getHealth();
        setHealth(h);
        if (h.status === "loading") timer = window.setTimeout(poll, 1500);
      } catch {
        setHealth({ status: "loading", message: "Waiting for the server…", model: "", logging: false });
        timer = window.setTimeout(poll, 2000);
      }
    };
    poll();
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (messages.length) bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const update = useCallback((id: string, change: (m: Answer) => Answer) => {
    setMessages((all) => all.map((m) => (m.id === id && m.role === "assistant" ? change(m) : m)));
  }, []);

  const send = async (question: string) => {
    if (busy || !ready) return;
    // Only finished answers go back as history: a stopped or failed question is left out
    const history: { role: "user" | "assistant"; content: string }[] = [];
    messages.forEach((m, i) => {
      const next = messages[i + 1];
      if (m.role === "user" && next?.role === "assistant" && next.record && !next.record.stopped) {
        history.push({ role: "user", content: m.text }, { role: "assistant", content: next.record.text });
      }
    });
    const last = [...messages].reverse().find((m): m is Answer => m.role === "assistant");
    const answerId = newId();
    setMessages((all) => [
      ...all,
      { role: "user", id: newId(), text: question },
      { role: "assistant", id: answerId, steps: [{ kind: "read", state: "active" }], streamed: "", startedAt: Date.now() },
    ]);
    setBusy(true);
    controller.current = new AbortController();

    try {
      await streamChat(
        { question, history, today, previous_request: last?.record?.request ?? null, session_id: sessionId.current },
        {
          onQueued: () => update(answerId, (m) => ({ ...m, queued: true })),
          onRoute: (route) =>
            update(answerId, (m) => {
              const steps: Step[] = [{ kind: "read", state: "done", detail: routeDetail(route) }];
              if (route.needs_data) steps.push({ kind: "data", state: "active" });
              if (route.needs_cases) steps.push({ kind: "cases", state: route.needs_data ? "pending" : "active" });
              steps.push({ kind: "write", state: "pending" });
              return { ...m, route, steps };
            }),
          onData: (data) =>
            update(answerId, (m) => ({
              ...m,
              data,
              steps: setStep(setStep(m.steps, "data", { state: data.error ? "failed" : "done", detail: dataHeadline(data) }), "cases", { state: "active" }),
            })),
          onCases: (cases) =>
            update(answerId, (m) => ({
              ...m,
              cases,
              steps: setStep(m.steps, "cases", { state: "done", detail: `${cases.length} cases with different outcomes` }),
            })),
          onWriting: () => update(answerId, (m) => ({ ...m, writingAt: Date.now(), steps: setStep(m.steps, "write", { state: "active" }) })),
          onToken: (text) =>
            update(answerId, (m) => ({ ...m, streamed: m.streamed + text, firstWordAt: m.firstWordAt ?? Date.now() })),
          onDone: (record) =>
            update(answerId, (m) => ({ ...m, record, finishedAt: Date.now(), steps: m.steps.map((s) => ({ ...s, state: "done" as const })) })),
          onError: (message) =>
            update(answerId, (m) => ({ ...m, error: message, steps: m.steps.map((s) => (s.state === "done" ? s : { ...s, state: "failed" as const })) })),
        },
        controller.current.signal,
      );
    } catch (error) {
      const stopped = (error as Error).name === "AbortError";
      update(answerId, (m) => ({
        ...m,
        ...(stopped ? { stoppedByUser: true } : { error: (error as Error).message }),
        // Nothing is running any more: no step should keep spinning
        steps: m.steps.map((s) => (s.state === "done" ? s : { ...s, state: stopped ? ("pending" as const) : ("failed" as const) })),
      }));
    } finally {
      setBusy(false);
      controller.current = null;
    }
  };

  const feedback = (message: Answer, rating: 1 | -1 | null) => {
    update(message.id, (m) => ({ ...m, feedback: rating }));
    if (message.record?.interaction_id) sendFeedback(message.record.interaction_id, rating);
  };

  return (
    <div className="layout">
      <Sidebar
        health={health}
        today={today}
        onToday={setToday}
        onNewConversation={() => setMessages([])}
        canReset={!busy && messages.length > 0}
      />
      <main className="main">
        {health?.status !== "ready" && (
          <div className={`banner banner-${health?.status ?? "loading"}`} role="status">
            {health?.status === "error" ? health.message : (health?.message ?? "Connecting to the server…")}
          </div>
        )}
        <div className="thread">
          {messages.length === 0 ? (
            <EmptyState ready={ready} onPick={send} />
          ) : (
            messages.map((m) =>
              m.role === "user" ? (
                <article key={m.id} className="message message-user"><p>{m.text}</p></article>
              ) : (
                <AssistantMessage key={m.id} message={m} onCite={setCitation} onFeedback={feedback} />
              ),
            )
          )}
          <div ref={bottom} />
        </div>
        <Composer ready={ready} busy={busy} onSend={send} onStop={() => controller.current?.abort()} />
      </main>
      <CitationDrawer id={citation} onClose={() => setCitation(null)} />
    </div>
  );
}
