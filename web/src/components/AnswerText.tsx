import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

const CITATION = /\[(POL-[A-Z]{3}-\d{2}|INC-\d{4}-\d{2}|QM-C-\d{6})\]/g;
const CASE_RECORDS = /\[case records\]/gi;
const reducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/**
 * Reveals `text` at a steady pace as it grows. The model sends words in uneven bursts; revealing a few
 * characters per frame (faster when it falls behind) reads as calm, continuous typing instead.
 */
function useTypewriter(text: string): string {
  const [shown, setShown] = useState("");
  const target = useRef(text);
  target.current = text;

  useEffect(() => {
    if (reducedMotion()) {
      setShown(text);
      return;
    }
    let frame = 0;
    const tick = () => {
      setShown((current) => {
        const goal = target.current;
        if (!goal.startsWith(current)) return goal; // the checked answer differs slightly: show it as is
        const backlog = goal.length - current.length;
        return backlog > 0 ? goal.slice(0, current.length + Math.max(1, Math.ceil(backlog / 10))) : current;
      });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return shown;
}

/** The answer, typed out, with citations as buttons that open the cited policy, incident or case. */
export function AnswerText({ text, typing, onCite }: { text: string; typing: boolean; onCite: (id: string) => void }) {
  const shown = useTypewriter(text);
  const linked = shown.replace(CITATION, "[$1](#cite-$1)").replace(CASE_RECORDS, "[case records](#records)");
  const caret = typing || shown.length < text.length;

  return (
    <div className={`answer-text${caret ? " is-typing" : ""}`}>
      <ReactMarkdown
        components={{
          a: ({ href, children }) => {
            if (href?.startsWith("#cite-")) {
              const id = href.slice(6);
              return (
                <button type="button" className="cite" onClick={() => onCite(id)} title={`Open ${id}`}>
                  {id}
                </button>
              );
            }
            if (href === "#records") return <span className="cite cite-records">case records</span>;
            return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
          },
        }}
      >
        {linked}
      </ReactMarkdown>
    </div>
  );
}
