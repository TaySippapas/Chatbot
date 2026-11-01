import re
from dataclasses import dataclass

ID_PATTERN = re.compile(r"\b(?:POL-[A-Z]{3}-\d{2}|INC-\d{4}-\d{2}|QM-C-\d{6})\b")


@dataclass
class Answer:
    text: str
    prompt_tokens: int              # full prompt length
    cached_tokens: int              # served from the knowledge cache, not recomputed
    output_tokens: int
    seconds_to_first_token: float
    seconds_total: float

    @property
    def computed_prompt_tokens(self) -> int:
        """Prompt tokens the model actually had to read for this question."""
        return self.prompt_tokens - self.cached_tokens

    @property
    def cited_ids(self) -> list[str]:
        return sorted(set(ID_PATTERN.findall(self.text)))
