"""Generation steps shared by the CAG and RAG bots: chat rendering, chunked prompt reading, streaming."""
import time

import torch
from transformers import TextStreamer

from quillmark_bot.answer import Answer

# Reading ~5K tokens in one pass needs more than 8 GB of activations; chunks keep it on the GPU
PREFILL_CHUNK_TOKENS = 1024


class TimedStreamer(TextStreamer):
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


def render_chat(tokenizer, messages: list[dict], enable_thinking: bool = False) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )


def tokenize(tokenizer, text: str, device) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


@torch.inference_mode()
def prefill(model, input_ids: torch.Tensor, cache=None):
    """Read `input_ids` in chunks and return the resulting cache."""
    for i in range(0, input_ids.shape[1], PREFILL_CHUNK_TOKENS):
        # logits_to_keep=1: skip computing vocabulary logits for every prompt position
        out = model(input_ids=input_ids[:, i:i + PREFILL_CHUNK_TOKENS],
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
    return cache


@torch.inference_mode()
def generate_answer(model, tokenizer, input_ids: torch.Tensor, cache, cached_tokens: int,
                    max_new_tokens: int, on_text=None, start: float | None = None) -> Answer:
    """Generate greedily from `input_ids`, reusing `cache` for the tokens it already holds.

    `cached_tokens` is what gets reported as served from a reusable cache; `start` lets the caller
    count its own prompt reading in the time to first token.
    """
    streamer = TimedStreamer(tokenizer, on_text)
    start = start or time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        past_key_values=cache,
        max_new_tokens=max_new_tokens,
        do_sample=False, temperature=None, top_p=None, top_k=None,
        streamer=streamer,
    )
    total = time.perf_counter() - start

    new_tokens = output[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    text = text.split("</think>")[-1].strip()  # drop reasoning if thinking is enabled
    return Answer(
        text=text,
        prompt_tokens=input_ids.shape[1],
        cached_tokens=cached_tokens,
        output_tokens=len(new_tokens),
        seconds_to_first_token=(streamer.first_token_at or time.perf_counter()) - start,
        seconds_total=total,
    )
