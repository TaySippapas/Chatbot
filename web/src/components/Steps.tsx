import { useEffect, useState } from "react";
import type { Step, StepKind } from "../types";

const LABELS: Record<StepKind, { active: string; done: string }> = {
  read: { active: "Reading your question", done: "Understood your question" },
  data: { active: "Checking our case records", done: "Checked our case records" },
  cases: { active: "Finding similar past cases", done: "Found similar past cases" },
  write: { active: "Writing the answer", done: "Answer written" },
};

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 100);
    return () => window.clearInterval(timer);
  }, []);
  return <span className="step-elapsed">{((now - since) / 1000).toFixed(1)} s</span>;
}

/** The live checklist shown while an answer is being prepared: each stage ticks off as it finishes. */
export function Steps({ steps, writingSince }: { steps: Step[]; writingSince?: number }) {
  return (
    <ol className="steps" aria-label="Progress">
      {steps.map((step) => (
        <li key={step.kind} className={`step step-${step.state}`}>
          <span className="step-icon" aria-hidden="true">
            {step.state === "done" ? "✓" : step.state === "failed" ? "!" : ""}
          </span>
          <span className="step-body">
            <span className="step-label">
              {step.state === "done" ? LABELS[step.kind].done : LABELS[step.kind].active}
              {step.kind === "write" && step.state === "active" && writingSince && <Elapsed since={writingSince} />}
            </span>
            {step.detail && <span className="step-detail">{step.detail}</span>}
          </span>
        </li>
      ))}
    </ol>
  );
}
