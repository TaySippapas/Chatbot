"""Cache-augmented generation: read the knowledge once, reuse a copy of its cache for every question.

Qwen3.5 mixes full-attention layers (KV cache, one entry per token) with linear-attention layers
(a running state that can't be rolled back), so the knowledge cache is never trimmed after a
question. Each question gets a fresh deep copy of it instead.
"""
import copy
import time
from datetime import date

import torch

from quillmark_bot.answer import Answer
from quillmark_bot.generation import generate_answer, prefill, render_chat, tokenize

SYSTEM_TURN_END = "<|im_end|>\n"
MAX_HISTORY_MESSAGES = 6  # last 3 exchanges


class CAGChatbot:
    def __init__(self, model, processor, knowledge: str, enable_thinking: bool = False,
                 max_new_tokens: int = 512):
        self.model = model
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.knowledge = knowledge
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self.prefix_ids = None
        self.prefix_cache = None
        self.cache_build_seconds = None

    def _render(self, messages: list[dict]) -> str:
        return render_chat(self.tokenizer, messages, self.enable_thinking)

    def _tokenize(self, text: str) -> torch.Tensor:
        return tokenize(self.tokenizer, text, self.model.device)

    @torch.inference_mode()
    def build_cache(self) -> None:
        """Read the system turn (rules + handbook + incidents) once and keep the resulting cache."""
        # The template refuses a system message on its own, so render with a placeholder user turn
        # and cut right after the system turn's end marker.
        full = self._render([{"role": "system", "content": self.knowledge}, {"role": "user", "content": "-"}])
        end = full.index(self.knowledge) + len(self.knowledge) + len(SYSTEM_TURN_END)
        if full[end - len(SYSTEM_TURN_END):end] != SYSTEM_TURN_END:
            raise RuntimeError("Unexpected chat template: system turn does not end where expected")

        self.prefix_ids = self._tokenize(full[:end])
        start = time.perf_counter()
        self.prefix_cache = prefill(self.model, self.prefix_ids)
        torch.cuda.synchronize()
        self.cache_build_seconds = time.perf_counter() - start

    @property
    def cached_tokens(self) -> int:
        return self.prefix_ids.shape[1]

    @torch.inference_mode()
    def ask(self, question: str, history: list[dict] | None = None, on_text=None,
            today: date | None = None, use_cache: bool = True, should_stop=None) -> Answer:
        """Answer one question. `history` is prior {"role", "content"} turns, oldest first.

        use_cache=False reads the whole prompt from scratch, for comparing against plain prompting.
        should_stop: optional callable; once it returns true, generation ends at the next token.
        """
        if use_cache and self.prefix_cache is None:
            self.build_cache()

        # The date sits in the user turn, after the cached part, so the cache stays valid every day
        user_turn = f"Today's date: {(today or date.today()).isoformat()}\n\n{question}"
        messages = [{"role": "system", "content": self.knowledge},
                    *(history or [])[-MAX_HISTORY_MESSAGES:],
                    {"role": "user", "content": user_turn}]
        input_ids = self._tokenize(self._render(messages))

        cached = 0
        cache = None
        start = time.perf_counter()
        if use_cache:
            cached = self.cached_tokens
            if not torch.equal(input_ids[:, :cached], self.prefix_ids):
                raise RuntimeError("Prompt no longer starts with the cached knowledge; rebuild the cache")
            cache = copy.deepcopy(self.prefix_cache)
            # The new part can be long too (the hybrid bot adds past cases and database results), so it is
            # read in chunks as well; generate() then continues from the last token
            cache = prefill(self.model, input_ids[:, cached:-1], cache)

        return generate_answer(self.model, self.tokenizer, input_ids, cache, cached,
                               self.max_new_tokens, on_text, start, should_stop)
