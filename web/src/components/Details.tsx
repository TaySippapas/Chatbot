import type { AssistantMessage } from "../types";

const seconds = (s: number | null | undefined) => (s == null ? "–" : s < 0.001 ? "under 1 ms" : s < 0.1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`);

function routeLabel(decidedBy: string) {
  if (decidedBy === "rules") return "the rules";
  if (decidedBy === "llm") return "the AI router";
  return "the default route (the AI router's answer couldn't be read)";
}

/** "How this was answered": route, database query and rows, similar cases, timings. Folded by default. */
export function Details({ message, onCite }: { message: AssistantMessage; onCite: (id: string) => void }) {
  const { route, data, cases, record } = message;
  if (!route || !record) return null;
  const used = ["Handbook", data && "Case records", cases?.length && `${cases.length} similar cases`].filter(Boolean);
  const columns = data?.rows.length ? Object.keys(data.rows[0]) : [];

  return (
    <details className="details">
      <summary>
        <span>How this was answered</span>
        <span className="details-used">{used.join(" · ")}</span>
      </summary>
      <div className="details-body">
        <section>
          <h4>Route</h4>
          <p>
            Decided by {routeLabel(route.decided_by)} in {seconds(route.seconds)}: {route.reason}
          </p>
          {route.llm_output && <pre className="code">{route.llm_output.trim()}</pre>}
        </section>

        {data && (
          <section>
            <h4>Case records</h4>
            <p>{data.description}</p>
            {data.error ? (
              <p className="details-error">The case database couldn't be reached.</p>
            ) : (
              columns.length > 0 && (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
                    </thead>
                    <tbody>
                      {data.rows.slice(0, 15).map((row, i) => (
                        <tr key={i}>{columns.map((c) => <td key={c}>{String(row[c] ?? "")}</td>)}</tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )
            )}
            {data.sql && <pre className="code">{data.sql}</pre>}
          </section>
        )}

        {cases && cases.length > 0 && (
          <section>
            <h4>Similar past cases</h4>
            <ul className="case-list">
              {cases.map((c) => (
                <li key={c.id}>
                  <button type="button" className="cite" onClick={() => onCite(c.id)}>{c.id}</button>
                  <span>{c.title.replace(/^\[[^\]]+\]\s*/, "")}</span>
                </li>
              ))}
            </ul>
          </section>
        )}

        <section>
          <h4>Timing</h4>
          <dl className="timing">
            <div><dt>Routing</dt><dd>{seconds(record.stats.seconds_route)}</dd></div>
            <div><dt>Case records</dt><dd>{seconds(record.stats.seconds_data)}</dd></div>
            <div><dt>Similar cases</dt><dd>{seconds(record.stats.seconds_retrieval)}</dd></div>
            <div><dt>First word</dt><dd>{seconds(record.stats.seconds_first_token)}</dd></div>
            <div><dt>Writing</dt><dd>{seconds(record.stats.seconds_answer)}</dd></div>
            <div><dt>Read from cache</dt><dd>{record.stats.cached_tokens.toLocaleString()} tokens</dd></div>
          </dl>
        </section>
      </div>
    </details>
  );
}
