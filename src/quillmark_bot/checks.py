"""Code checks on a finished answer, so grounding doesn't rest on the prompt's rules alone.

- Numbers: every figure in the answer must appear somewhere the model was given (handbook, incidents,
  past cases, database results, the question). Anything else is flagged, not removed: it may be
  honest arithmetic ("5% of $400 is $20"), but it may also be invented ("180 cases"), and the reader
  should be able to tell which figures are backed. A figure found only in past cases is flagged too:
  those are other customers' amounts, and the model has presented one as the customer's own.
- Citations: an ID is only a valid citation if that document was in the prompt.
- Database tags: the model tends to invent a bracket tag for database figures ("[DATABASE-001]");
  they all become "[case records]".
"""
import re

from quillmark_bot.answer import ID_PATTERN

# A number not glued to a word or an ID ("POL-DSP-04", "QM-C-000123" don't count), with an optional
# leading $, thousands commas ("1,500" but not the comma after "31,") and trailing %
NUMBER = re.compile(r"(?<![\w\-./:])\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\w\-])")
# Dates the model may restate in words ("2025-12-31" -> "December 31, 2025")
ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
US_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
DATABASE_TAG = re.compile(r"\[(?:[^\[\]]*database[^\[\]]*|case records?)\]", re.IGNORECASE)
CASE_RECORDS = "[case records]"


def _normalize(number: str) -> str:
    """"$1,500.00" and "1500" are the same figure; so are "35%" and "35"."""
    n = number.strip("$%").replace(",", "")
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n


def numbers_in(text: str) -> set[str]:
    found = {_normalize(m.group(0)) for m in NUMBER.finditer(ID_PATTERN.sub(" ", text))}
    for m in ISO_DATE.finditer(text):
        found |= {m.group(1), str(int(m.group(2))), str(int(m.group(3)))}
    for m in US_DATE.finditer(text):
        found |= {str(int(m.group(1))), str(int(m.group(2))), m.group(3)}
    return found


def _figures(answer: str) -> list[str]:
    figures = []
    for match in NUMBER.finditer(ID_PATTERN.sub(" ", answer)):
        if match.group(0) not in figures:
            figures.append(match.group(0))
    return figures


def unverified_numbers(answer: str, sources: list[str]) -> list[str]:
    """Figures in the answer, as written, that appear in none of the sources."""
    known = set().union(*(numbers_in(s) for s in sources)) if sources else set()
    return [f for f in _figures(answer) if _normalize(f) not in known]


def case_only_numbers(answer: str, trusted: list[str], cases: list[str]) -> list[str]:
    """Figures found in past cases but nowhere else: other customers' amounts, not this customer's."""
    trusted_known = set().union(*(numbers_in(s) for s in trusted)) if trusted else set()
    case_known = set().union(*(numbers_in(s) for s in cases)) if cases else set()
    return [f for f in _figures(answer) if _normalize(f) in case_known - trusted_known]


AMOUNT_PLACEHOLDER = "[amount]"


def mask_amounts(text: str, keep: set[str]) -> str:
    """Replace dollar amounts with [amount], except those in `keep` (normalized figures, e.g. the handbook's).

    Past cases are other customers' records: their purchase and refund amounts mean nothing for the
    customer asking, and the model copied them into answers as the customer's own. Policy amounts
    ($29 fee, $25 credit) are kept, since they are the same for everyone.
    """
    def replace(match: re.Match) -> str:
        figure = match.group(0)
        return figure if not figure.startswith("$") or _normalize(figure) in keep else AMOUNT_PLACEHOLDER
    return NUMBER.sub(replace, text)


def invalid_citations(cited: list[str], provided: set[str]) -> list[str]:
    return [doc_id for doc_id in cited if doc_id not in provided]


def normalize_database_tags(text: str) -> str:
    return DATABASE_TAG.sub(CASE_RECORDS, text)
