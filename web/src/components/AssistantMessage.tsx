import { useState } from "react";
import type { AssistantMessage as Message } from "../types";
import { AnswerText } from "./AnswerText";
import { Details } from "./Details";
import { Steps } from "./Steps";

type Props = {
  message: Message;
  onCite: (id: string) => void;
  onFeedback: (message: Message, rating: 1 | -1 | null) => void;
};

function Flags({ message }: { message: Message }) {
  const r = message.record;
  if (!r) return null;
  const notes = [];
  if (r.unverified_numbers.length)
    notes.push(`${r.unverified_numbers.join(", ")} isn't stated in the handbook or our records. It may be a calculation, so please check it.`);
  if (r.case_only_numbers.length)
    notes.push(`${r.case_only_numbers.join(", ")} comes from another customer's case, not yours.`);
  if (r.invalid_citations.length)
    notes.push(`${r.invalid_citations.join(", ")} was cited without being one of the documents used.`);
  if (!notes.length) return null;
  return (
    <div className="flags" role="note">
      <strong>Please double-check</strong>
      <ul>{notes.map((n) => <li key={n}>{n}</li>)}</ul>
    </div>
  );
}

export function AssistantMessage({ message, onCite, onFeedback }: Props) {
  const [copied, setCopied] = useState(false);
  const done = Boolean(message.record || message.error || message.stoppedByUser);
  const writing = message.steps.find((s) => s.kind === "write");
  const text = message.record?.text ?? message.streamed;
  const waitingForFirstWord = writing?.state === "active" && !message.streamed;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard refused: nothing to do */
    }
  };

  const firstWord = message.firstWordAt ? (message.firstWordAt - message.startedAt) / 1000 : null;
  const total = message.finishedAt ? (message.finishedAt - message.startedAt) / 1000 : null;

  return (
    <article className="message message-assistant" aria-busy={!done}>
      <div className="avatar" aria-hidden="true">Q</div>
      <div className="message-body">
        {message.queued && !message.route && <p className="queued">Another answer is being written. Yours is next.</p>}
        {!message.record && !message.stoppedByUser && <Steps steps={message.steps} writingSince={message.writingAt} />}

        {waitingForFirstWord && (
          <div className="skeleton" aria-hidden="true">
            <span /><span /><span />
          </div>
        )}

        {text && <AnswerText text={text} typing={!done} onCite={onCite} />}

        {message.stoppedByUser && <p className="stopped">Stopped. Ask again whenever you like.</p>}
        {message.error && (
          <div className="error-box" role="alert">
            <strong>Something went wrong.</strong> {message.error}
          </div>
        )}

        <Flags message={message} />

        {message.record && (
          <>
            <Details message={message} onCite={onCite} />
            <footer className="message-actions">
              {message.record.interaction_id ? (
                <span className="feedback" role="group" aria-label="Was this helpful?">
                  <button
                    type="button"
                    className={message.feedback === 1 ? "is-on" : ""}
                    aria-pressed={message.feedback === 1}
                    title="Helpful"
                    onClick={() => onFeedback(message, message.feedback === 1 ? null : 1)}
                  >
                    👍
                  </button>
                  <button
                    type="button"
                    className={message.feedback === -1 ? "is-on" : ""}
                    aria-pressed={message.feedback === -1}
                    title="Not helpful"
                    onClick={() => onFeedback(message, message.feedback === -1 ? null : -1)}
                  >
                    👎
                  </button>
                </span>
              ) : null}
              <button type="button" className="text-button" onClick={copy}>{copied ? "Copied" : "Copy"}</button>
              {firstWord != null && total != null && (
                <span className="timing-note">
                  first word {firstWord.toFixed(1)} s · done in {total.toFixed(1)} s
                </span>
              )}
            </footer>
          </>
        )}
      </div>
    </article>
  );
}
