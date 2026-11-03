"""Documents the RAG index is built from: handbook policies, known incidents and past cases."""
import hashlib
import re
from collections import Counter

from quillmark_bot.config import CASES_PATH, HANDBOOK_PATH
from quillmark_bot.data.incidents import incident_entries
from quillmark_bot.data.load import load_cases, load_policies

CHANNEL_PREFIX = re.compile(r"^\[.*?\]\s*")  # e.g. "[Call summary in customer's words] "


def template_id(agent_notes: str) -> str:
    """Agent notes are fill-in-the-blank templates; masking the numbers groups cases by scenario."""
    masked = re.sub(r"\d[\d,.]*", "#", agent_notes)
    return hashlib.sha1(masked.encode()).hexdigest()[:12]


def knowledge_documents() -> list[dict]:
    """The 32 policies and 5 incidents, one document each. `text` is both searched and shown."""
    docs = [{"id": p["policy_id"], "kind": "policy", "text": f"### [{p['policy_id']}] {p['title']}\n\n{p['body']}"}
            for p in load_policies(HANDBOOK_PATH)]
    docs += [{"id": incident_id, "kind": "incident", "text": text}
             for incident_id, text in incident_entries(load_cases(CASES_PATH))]
    return docs


def case_documents() -> list[dict]:
    """One document per past case: what to search on, what to show the model, and filterable metadata."""
    cases = load_cases(CASES_PATH)
    template_sizes = Counter(template_id(c["agent_notes"]) for c in cases)
    docs = []
    for c in cases:
        tid = template_id(c["agent_notes"])
        narrative = CHANNEL_PREFIX.sub("", c["customer_narrative"])
        docs.append({
            "id": c["case_id"],
            "search_text": f"{c['issue']}. {narrative} {c['agent_notes']}",
            "display_text": (
                f"[{c['case_id']}] opened {c['opened_date']} | {c['product']} | {c['branch']} branch | "
                f"Issue: {c['issue']}\n"
                f"Customer: {narrative}\n"
                f"Agent notes: {c['agent_notes']}\n"
                f"Resolution: {c['resolution']} | Refunded: ${c['amount_refunded']:,.2f} | "
                f"{template_sizes[tid]} past cases were resolved the same way"
            ),
            "metadata": {
                "category": c["category"], "issue": c["issue"], "product": c["product"],
                "branch": c["branch"], "outcome": c["outcome"],
                "opened": int(c["opened_date"].replace("-", "")),  # Chroma filters need numbers
                "incident_id": c["incident_id"] or "",              # and no None values
                "template_id": tid, "template_size": template_sizes[tid],
            },
        })
    return docs
