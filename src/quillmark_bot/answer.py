import re
from dataclasses import dataclass, field

ID_PATTERN = re.compile(r"\b(?:POL-[A-Z]{3}-\d{2}|INC-\d{4}-\d{2}|QM-C-\d{6})\b")


@dataclass
class Answer:
    text: str
    prompt_tokens: int              # full prompt length
    cached_tokens: int              # served from the knowledge cache, not recomputed
    output_tokens: int
    seconds_to_first_token: float
    seconds_total: float
    sources: list[str] = field(default_factory=list)  # IDs of documents retrieved into the prompt (RAG)
    seconds_retrieval: float = 0.0
    unverified_numbers: list[str] = field(default_factory=list)  # figures found in nothing the model was given
    case_only_numbers: list[str] = field(default_factory=list)   # figures found only in other customers' cases
    invalid_citations: list[str] = field(default_factory=list)   # IDs cited that were not in the prompt
    stopped: bool = False  # generation was ended early on request

    @property
    def computed_prompt_tokens(self) -> int:
        """Prompt tokens the model actually had to read for this question."""
        return self.prompt_tokens - self.cached_tokens

    @property
    def cited_ids(self) -> list[str]:
        return sorted(set(ID_PATTERN.findall(self.text)))
