import type { Health } from "../types";

type Props = {
  health: Health | null;
  today: string;
  onToday: (value: string) => void;
  onNewConversation: () => void;
  canReset: boolean;
};

function Status({ health }: { health: Health | null }) {
  if (!health) return <span className="status status-loading">Connecting…</span>;
  if (health.status === "ready") return <span className="status status-ready">Ready</span>;
  if (health.status === "error") return <span className="status status-error">Not available</span>;
  return <span className="status status-loading">Starting up…</span>;
}

export function Sidebar({ health, today, onToday, onNewConversation, canReset }: Props) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">Q</span>
        <span className="brand-name">
          Quillmark<span>Support</span>
        </span>
      </div>

      <button type="button" className="new-chat" onClick={onNewConversation} disabled={!canReset}>
        <span aria-hidden="true">＋</span> New conversation
      </button>

      <div className="sidebar-section">
        <label htmlFor="today" className="sidebar-label">Today's date</label>
        <input id="today" type="date" value={today} onChange={(e) => onToday(e.target.value)} />
        <p className="sidebar-hint">Known incidents only apply within their dates. Set a past date to test one, such as 2026-03-15 for the duplicate-charge outage.</p>
      </div>

      <div className="sidebar-foot">
        <Status health={health} />
        <a href="http://localhost:3000" target="_blank" rel="noreferrer">Dashboard ↗</a>
        <p className="sidebar-small">Fictional bank · answers come from the policy handbook, known incidents, past cases and case records · runs locally</p>
      </div>
    </aside>
  );
}
