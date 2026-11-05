const EXAMPLES = [
  { tag: "Known incident", text: "I was charged twice for the same purchase on March 15. What happens now?" },
  { tag: "Policy", text: "What is the overdraft fee and how can I avoid it?" },
  { tag: "Case statistics", text: "How often do goodwill requests get approved?" },
  { tag: "Case statistics", text: "Which branch had the most fraud complaints last year?" },
];

export function EmptyState({ ready, onPick }: { ready: boolean; onPick: (text: string) => void }) {
  return (
    <section className="empty">
      <h1>How can we help?</h1>
      <p>
        Ask about Quillmark's fees, cards, disputes, loans and known incidents, or about what happened in
        past cases. Every answer shows where it came from.
      </p>
      <div className="examples">
        {EXAMPLES.map((example) => (
          <button key={example.text} type="button" className="example" disabled={!ready} onClick={() => onPick(example.text)}>
            <span className="example-tag">{example.tag}</span>
            <span>{example.text}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
