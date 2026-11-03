"""Known values the router can recognise in a question, and the everyday wording that maps to them.

Official values (branches, products, issues, ...) come from the data, so they stay in sync with it.
Aliases are curated: many started from phrases that are far more common in one issue's customer
narratives than in the rest (see notebooks/01_eda.ipynb for the data).
"""
import re
from dataclasses import dataclass
from functools import lru_cache

from quillmark_bot.config import CASES_PATH, HANDBOOK_PATH
from quillmark_bot.data.load import load_cases, load_policies

FILTER_FIELDS = ("branch", "product", "category", "issue", "channel", "outcome", "incident_id", "policy_id")

PRODUCT_ALIASES = {
    "checking account": "Everyday Checking", "checking": "Everyday Checking",
    "spark credit card": "Spark Card", "spark": "Spark Card",
    "horizon card": "Horizon Rewards Card", "horizon": "Horizon Rewards Card",
    "personal loans": "Personal Loan",
    "home loans": "Home Loan", "mortgages": "Home Loan", "mortgage": "Home Loan",
    "savings account": "Quillmark Saver", "savings": "Quillmark Saver", "saver": "Quillmark Saver",
    "online banking": "Digital Banking", "mobile banking": "Digital Banking",
}
# Singular/plural forms of category names; bare words that are too generic ("card", "fee") are left out
CATEGORY_ALIASES = {
    "dispute": "Disputes", "fraud cases": "Fraud", "collection": "Collections",
    "credit reports": "Credit reporting", "loans": "Loans", "mobile deposits": "Deposits",
    "rewards points": "Rewards", "points": "Rewards",
}
ISSUE_ALIASES = {
    "charged twice": "Duplicate charge", "charged me twice": "Duplicate charge",
    "double charged": "Duplicate charge", "double charges": "Duplicate charge", "billed twice": "Duplicate charge",
    "duplicate charges": "Duplicate charge", "duplicates": "Duplicate charge",
    "locked out": "App lockout", "locked me out": "App lockout", "lockouts": "App lockout", "lockout": "App lockout",
    "credit report errors": "Incorrect information on report", "credit report error": "Incorrect information on report",
    "overdraft fees": "Overdraft fee", "overdrafts": "Overdraft fee",
    "monthly fees": "Monthly maintenance fee", "monthly fee": "Monthly maintenance fee",
    "maintenance fees": "Monthly maintenance fee",
    "collection agency": "Third-party collector conduct", "collection agencies": "Third-party collector conduct",
    "collection calls": "Collection contact practices", "collector calls": "Collection contact practices",
    "card late fees": "Card late fee", "credit card late fee": "Card late fee",
    "late statements": "Late fee - delayed statement", "late statement": "Late fee - delayed statement",
    "delayed statements": "Late fee - delayed statement",
    "unauthorized debit": "Unauthorized debit transaction",
    "unauthorized credit card": "Unauthorized credit card charge", "unauthorized card charges": "Unauthorized credit card charge",
    "validate the debt": "Debt validation", "validation requests": "Debt validation",
    "goodwill requests": "Goodwill request", "goodwill removals": "Goodwill request", "goodwill": "Goodwill request",
    "sign-in problems": "Sign-in problem", "login problems": "Sign-in problem", "verification code": "Sign-in problem",
    "escrow increases": "Escrow increase", "payment jumped": "Escrow increase",
    "lost cards": "Lost card replacement", "lost card": "Lost card replacement",
    "stolen cards": "Lost card replacement", "replacement cards": "Lost card replacement",
    "loan late fees": "Loan late fee",
    "hardship requests": "Hardship assistance", "hardship": "Hardship assistance", "steady path": "Hardship assistance",
    "wrong recipient": "QuickSend to wrong recipient", "wrong person": "QuickSend to wrong recipient",
    "charged after cancelling": "Charged after cancellation", "still charged": "Charged after cancellation",
    "credit limit increases": "Credit limit increase", "limit increases": "Credit limit increase",
    "impersonation scams": "QuickSend impersonation scam", "impersonation": "QuickSend impersonation scam",
    "never arrived": "Item not received", "not received": "Item not received", "never showed up": "Item not received",
    "early closure fees": "Early closure fee", "foreign transaction fees": "Foreign transaction fee",
    "dormancy fees": "Dormancy fee", "settlements": "Settlement", "payment plans": "Payment plan",
    "paper statement fees": "Paper statement fee", "annual fees": "Annual fee", "wire fees": "Wire transfer fee",
    "card activations": "Card activation", "address changes": "Address change",
}
INCIDENT_ALIASES = {
    "march outage": "INC-2026-03", "processing outage": "INC-2026-03", "card outage": "INC-2026-03",
    "outage": "INC-2026-03",
    "app bug": "INC-2026-07", "6.2.0": "INC-2026-07", "app version 6.2.0": "INC-2026-07",
    "escrow error": "INC-2025-06", "escrow incident": "INC-2025-06",
    "print vendor": "INC-2025-11", "ashgrove": "INC-2025-11",
    "brightway": "INC-2026-02", "brightway recovery partners": "INC-2026-02",
}
CHANNEL_ALIASES = {
    "by phone": "phone", "over the phone": "phone", "phone calls": "phone", "via chat": "chat",
    "by chat": "chat", "by email": "email", "via email": "email", "in branch": "branch",
    "in-branch": "branch", "app messages": "app message",
}


def normalize(text: str) -> str:
    """Lowercase, and treat a hyphen between words as a space ("duplicate-charge" = "duplicate charge")."""
    return re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text.lower())


@dataclass(frozen=True)
class Vocab:
    values: dict           # field -> sorted list of official values
    aliases: dict          # lowercase phrase -> (field, official value)
    matcher: re.Pattern    # every alias, longest first, on word boundaries
    first_date: str        # data coverage, for describing results
    last_date: str

    def match(self, text: str) -> dict[str, list[str]]:
        """Official values mentioned in `text`, by field, in order of appearance."""
        found: dict[str, list[str]] = {}
        for phrase in self.matcher.findall(normalize(text)):
            field, value = self.aliases[phrase]
            if value not in found.setdefault(field, []):
                found[field].append(value)
        return found

    def remove(self, text: str) -> str:
        """`text` with every recognised value blanked out."""
        return self.matcher.sub(" ", normalize(text))


@lru_cache(maxsize=1)
def load_vocab() -> Vocab:
    cases = load_cases(CASES_PATH)
    values = {f: sorted({c[f] for c in cases if c.get(f)}) for f in FILTER_FIELDS if f != "policy_id"}
    values["policy_id"] = sorted(p["policy_id"] for p in load_policies(HANDBOOK_PATH))

    aliases = {}
    for field in ("branch", "product", "category", "issue"):
        for value in values[field]:
            aliases[value.lower()] = (field, value)
            if field == "branch":
                aliases[f"{value.lower()} branch"] = (field, value)
    for table, field in ((PRODUCT_ALIASES, "product"), (CATEGORY_ALIASES, "category"), (ISSUE_ALIASES, "issue"),
                         (INCIDENT_ALIASES, "incident_id"), (CHANNEL_ALIASES, "channel")):
        for phrase, value in table.items():
            if value not in values[field]:
                raise ValueError(f"Alias {phrase!r} points to unknown {field} {value!r}")
            aliases[phrase] = (field, value)
    for field in ("incident_id", "policy_id"):
        for value in values[field]:
            aliases[value.lower()] = (field, value)

    aliases = {normalize(phrase): target for phrase, target in aliases.items()}
    phrases = sorted(aliases, key=len, reverse=True)
    matcher = re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, phrases)) + r")(?![\w-])")
    dates = sorted(c["opened_date"] for c in cases)
    return Vocab(values, aliases, matcher, dates[0], dates[-1])
