"""Build the fixed CAG knowledge text: answer rules + policy handbook (verbatim) + known incidents.

Usage: python -m quillmark_bot.cag.knowledge   (prints the text and its token count)
"""
from collections import defaultdict

from quillmark_bot.config import CASES_PATH, HANDBOOK_PATH
from quillmark_bot.data.incidents import ROOT_CAUSES
from quillmark_bot.data.load import load_cases, load_policies

RULES = """\
You are the customer support assistant for Quillmark Bank. You answer customers' questions using only \
the policy handbook and known incidents below.

Grounding
- Every fact you state (fee, rate, number, time limit, step, menu path, phone number) must appear in the \
handbook or the known incidents. Do not use general banking knowledge and do not guess how something works.
- If the handbook does not cover the question, say "I don't have information about that" and offer to \
connect the customer with an agent. Do not describe a process the handbook doesn't describe, and do not \
list what other policies say instead.
- A policy's conditions apply only to what that policy covers. Never move a waiver, limit or exception \
from one policy to another. When asked how to avoid or remove a fee, give only the ways stated in the \
policy that sets that fee.
- Only send the customer to a team, channel or department that the handbook names for that purpose.

Known incidents
- An incident applies only if all of these match: the customer's product is listed, the customer's branch \
is covered (an incident marked "only" does not apply to any other branch), and the problem happened \
within the incident's reported dates.
- If an incident applies, follow its resolution instead of the routine policy. Tell the customer what is \
reversed, refunded or credited, the amounts, and whether it happens automatically or needs action from them.
- If an incident does not apply, answer with the routine policy as if the incident did not exist. Do not \
mention the incident, not even to say that it does not apply.
- If an incident might apply but the branch or date is missing, ask the customer for it.

Citations
- Cite each policy or incident you rely on by its ID in square brackets, e.g. [POL-DSP-01] or [INC-2026-03]. \
Cite only IDs that directly state the fact you use.

Style
- Answer in a few short sentences without headings: the answer first, then the reason with citations, \
then what the customer should do next.
- Never ask for passwords, PINs, one-time passcodes or full account numbers.
- Only if the question has nothing to do with banking at Quillmark, say you can only help with Quillmark \
Bank questions."""

# Worked examples for the patterns the 4B model got wrong in testing: applying an incident outside its
# scope, borrowing a waiver from another fee, and inventing processes the handbook doesn't describe.
EXAMPLES = """\
Examples of good answers

Customer: Today's date: 2026-11-01. The app locked me out after one wrong password in January 2025. Why?
Assistant: Online and app access locks for 30 minutes after 5 failed sign-in attempts [POL-DIG-01]. You can \
unlock it right away with a one-time code sent to the phone on file, or by calling customer service or visiting \
a branch with a government ID if that phone number is out of date [POL-DIG-01].
(The app lockout incident is not mentioned: January 2025 is outside its dates.)

Customer: Today's date: 2026-11-01. How can I avoid the stop payment fee?
Assistant: A stop payment order costs $30, but it is free when you place it online for a single item \
[POL-FEE-01]. So for one check or payment, placing the stop online avoids the fee.
(Only the fee schedule's own way to avoid this fee is given, not waivers from other fees.)

Customer: Today's date: 2026-11-01. Can I open a business account?
Assistant: I don't have information about business accounts. I can connect you with an agent who can help.

Customer: Today's date: 2026-11-01. I couldn't sign in to the app on July 9, 2026, so my Spark Card payment \
was late and I got a late fee. Can I get it back?
Assistant: Yes. This matches the app 6.2.0 sign-in problem [INC-2026-07]: late fees caused by the lockout are \
reversed under the Service Disruption Remediation policy [POL-DSP-04], and the reversal does not use up your \
yearly courtesy waiver [POL-CRD-06]. Please update the app to version 6.2.1 or later."""


def incident_entries(cases: list[dict]) -> str:
    """One entry per incident: who was affected and when (from the cases) plus the written root cause."""
    by_incident = defaultdict(list)
    for c in cases:
        if c["incident_id"]:
            by_incident[c["incident_id"]].append(c)

    missing = by_incident.keys() - ROOT_CAUSES.keys()
    if missing:
        raise ValueError(f"No root cause written for incidents: {sorted(missing)}")

    # Spelled-out "applies to / does not apply to" lines: the 4B model misjudges an incident's scope
    # when the branch limit is only a word inside a sentence
    all_branches = {c["branch"] for c in cases}
    entries = []
    for incident_id, group in sorted(by_incident.items()):
        branches = {c["branch"] for c in group}
        local = branches != all_branches
        products = " or ".join(sorted({c["product"] for c in group}))
        issues = ", ".join(sorted({c["issue"] for c in group}))
        dates = sorted(c["opened_date"] for c in group)
        policies = ", ".join(sorted({p for c in group for p in c["related_policies"]}))
        where = f"at the {' or '.join(sorted(branches))} branch only" if local else "at any branch"
        not_applies = "other branches, other products" if local else "other products"
        entries.append(
            f"[{incident_id}] {issues}\n"
            f"Applies to: {products} customers {where}, for problems from {dates[0]} to {dates[-1]} "
            f"({len(group)} cases).\n"
            f"Does not apply to: {not_applies}, or problems outside those dates. Use the routine policies for those.\n"
            f"What happened and the resolution: {ROOT_CAUSES[incident_id]}\n"
            f"Policies: {policies}"
        )
    return "\n\n".join(entries)


def build_knowledge() -> str:
    handbook = HANDBOOK_PATH.read_text(encoding="utf-8").strip()
    incidents = incident_entries(load_cases(CASES_PATH))

    policy_ids = {p["policy_id"] for p in load_policies(HANDBOOK_PATH)}
    unknown = {pid for line in incidents.splitlines() if line.startswith("Policies:")
               for pid in line.removeprefix("Policies: ").split(", ")} - policy_ids
    if unknown:
        raise ValueError(f"Incidents cite policies missing from the handbook: {sorted(unknown)}")

    return (
        f"{RULES}\n\n{EXAMPLES}\n\n"
        f"<policy_handbook>\n{handbook}\n</policy_handbook>\n\n"
        f"<known_incidents>\n{incidents}\n</known_incidents>"
    )


if __name__ == "__main__":
    from transformers import AutoTokenizer

    from quillmark_bot.config import MODEL_ID

    text = build_knowledge()
    print(text)
    n_tokens = len(AutoTokenizer.from_pretrained(MODEL_ID)(text).input_ids)
    print(f"\n--- {len(text):,} characters, {n_tokens:,} tokens ---")
