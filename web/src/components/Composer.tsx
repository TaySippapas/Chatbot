import { useEffect, useRef, useState } from "react";

type Props = { ready: boolean; busy: boolean; onSend: (text: string) => void; onStop: () => void };

/** The message box: Enter sends, Shift+Enter adds a line; while an answer is written it offers Stop. */
export function Composer({ ready, busy, onSend, onStop }: Props) {
  const [text, setText] = useState("");
  const box = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [text]);

  useEffect(() => {
    if (ready && !busy) box.current?.focus();
  }, [ready, busy]);

  const send = () => {
    const question = text.trim();
    if (!question || busy || !ready) return;
    onSend(question);
    setText("");
  };

  return (
    <div className="composer">
      <div className="composer-box">
        <textarea
          ref={box}
          id="question"
          rows={1}
          value={text}
          placeholder={ready ? "Ask about fees, cards, disputes, loans or case statistics…" : "Starting up, one moment…"}
          disabled={!ready}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          aria-label="Your question"
        />
        {busy ? (
          <button type="button" className="send stop" onClick={onStop} aria-label="Stop the answer">
            <span className="stop-square" aria-hidden="true" />
          </button>
        ) : (
          <button type="button" className="send" onClick={send} disabled={!ready || !text.trim()} aria-label="Send">
            ↑
          </button>
        )}
      </div>
      <p className="composer-hint">Enter to send · Shift+Enter for a new line · answers can be wrong, so check important details</p>
    </div>
  );
}
