import { useEffect, useState } from "react";
import { getDocument } from "../api";

const kindOf = (id: string) =>
  id.startsWith("POL-") ? "Policy" : id.startsWith("INC-") ? "Known incident" : "Past case";

/** Side panel with the full text of a cited policy, incident or past case. */
export function CitationDrawer({ id, onClose }: { id: string | null; onClose: () => void }) {
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    setText(null);
    setError(null);
    getDocument(id).then(setText, (e: Error) => setError(e.message));
  }, [id]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!id) return null;
  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-label={`${kindOf(id)} ${id}`}>
        <header>
          <div>
            <span className="drawer-kind">{kindOf(id)}</span>
            <h3>{id}</h3>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Close">✕</button>
        </header>
        {error && <p className="details-error">{error}</p>}
        {!error && text === null && <div className="skeleton"><span /><span /><span /></div>}
        {text && <pre className="drawer-text">{text.replace(/^#+\s*/gm, "")}</pre>}
      </aside>
    </>
  );
}
