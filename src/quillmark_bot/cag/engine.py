"""Cache-augmented generation: read the knowledge once, reuse a copy of its cache for every question.

Qwen3.5 mixes full-attention layers (KV cache, one entry per token) with linear-attention layers
(a running state that can't be rolled back), so the knowledge cache is never trimmed after a
question. Each question gets a fresh deep copy of it instead.
"""
import copy
import time
from datetime import date

import torch
from transformers import TextStreamer

from quillmark_bot.answer import Answer

SYSTEM_TURN_END = "<|im_end|>\n"
MAX_HISTORY_MESSAGES = 6  # last 3 exchanges
# Reading ~5K tokens in one pass needs more than 8 GB of activations; chunks keep it on the GPU
PREFILL_CHUNK_TOKENS = 1024


class _TimedStreamer(TextStreamer):
    """Records when the first answer token arrives and forwards decoded text to a callback."""

    def __init__(self, tokenizer, on_text=None):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.on_text = on_text
        self.first_token_at = None

    def put(self, value):
        if not self.next_tokens_are_prompt and self.first_token_at is None:
            self.first_token_at = time.perf_counter()
        super().put(value)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if self.on_text and text:
            self.on_text(text)


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
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking
        )

    def _tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)

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
        cache = None
        for i in range(0, self.prefix_ids.shape[1], PREFILL_CHUNK_TOKENS):
            # logits_to_keep=1: skip computing vocabulary logits for every knowledge position
            out = self.model(input_ids=self.prefix_ids[:, i:i + PREFILL_CHUNK_TOKENS],
                             past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
        torch.cuda.synchronize()
        self.prefix_cache = cache
        self.cache_build_seconds = time.perf_counter() - start

    @property
    def cached_tokens(self) -> int:
        return self.prefix_ids.shape[1]

    @torch.inference_mode()
    def ask(self, question: str, history: list[dict] | None = None, on_text=None,
            today: date | None = None, use_cache: bool = True) -> Answer:
        """Answer one question. `history` is prior {"role", "content"} turns, oldest first.

        use_cache=False reads the whole prompt from scratch, for comparing against plain prompting.
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
        if use_cache:
            cached = self.cached_tokens
            if not torch.equal(input_ids[:, :cached], self.prefix_ids):
                raise RuntimeError("Prompt no longer starts with the cached knowledge; rebuild the cache")
            cache = copy.deepcopy(self.prefix_cache)

        streamer = _TimedStreamer(self.tokenizer, on_text)
        start = time.perf_counter()
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            past_key_values=cache,
            max_new_tokens=self.max_new_tokens,
            do_sample=False, temperature=None, top_p=None, top_k=None,
            streamer=streamer,
        )
        total = time.perf_counter() - start

        new_tokens = output[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = text.split("</think>")[-1].strip()  # drop reasoning if thinking is enabled
        return Answer(
            text=text,
            prompt_tokens=input_ids.shape[1],
            cached_tokens=cached,
            output_tokens=len(new_tokens),
            seconds_to_first_token=(streamer.first_token_at or time.perf_counter()) - start,
            seconds_total=total,
        )
