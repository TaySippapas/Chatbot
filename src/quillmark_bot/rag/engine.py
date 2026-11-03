"""Retrieval-augmented generation: find the relevant policies, incidents and past cases for each
question and put only those in the prompt. Nothing is cached between questions."""
import time
from datetime import date

import torch

from quillmark_bot.answer import Answer
from quillmark_bot.generation import generate_answer, prefill, render_chat, tokenize
from quillmark_bot.rag.retriever import Retrieval, Retriever

MAX_HISTORY_MESSAGES = 6  # last 3 exchanges

RULES = """\
You are the customer support assistant for Quillmark Bank. With each question you are given the most \
relevant policies, known incidents and past cases. Answer using only those.

Grounding
- Every fact you state (fee, rate, number, time limit, step, menu path, phone number) must appear in the \
policies, incidents or past cases provided. Do not use general banking knowledge and do not guess how \
something works.
- If they do not cover the question, say "I don't have information about that" and offer to connect the \
customer with an agent. Do not describe a process they don't describe.
- A policy's conditions apply only to what that policy covers. Never move a waiver, limit or exception \
from one policy to another. When asked how to avoid or remove a fee, give only the ways stated in the \
policy that sets that fee.
- Only send the customer to a team, channel or department that a policy names for that purpose.

Past cases
- Past cases show how similar situations were resolved. Use them as precedent, but the policies decide: \
if a past case and a policy disagree, follow the policy.
- Past cases belong to other customers. Never present their amounts, dates or details as this customer's.

Known incidents
- Incidents only matter when the customer reports a problem. For general questions about a fee, rate or \
rule, answer from the policies without mentioning incidents or asking for a branch or date.
- An incident applies only if all of these match: the customer's product is listed, the customer's branch \
is covered (an incident marked "only" does not apply to any other branch), and the problem happened \
within the incident's reported dates.
- If an incident applies, follow its resolution instead of the routine policy. Tell the customer what is \
reversed, refunded or credited, the amounts, and whether it happens automatically or needs action from them.
- If an incident does not apply, answer with the routine policy as if the incident did not exist. Do not \
mention the incident, not even to say that it does not apply.
- If an incident might apply but the branch or date is missing, ask the customer for it.

Citations
- Cite each policy, incident or past case you rely on by its ID in square brackets, e.g. [POL-DSP-01], \
[INC-2026-03] or [QM-C-000123]. Cite only IDs from the material provided that directly state the fact you use.

Style
- Answer in a few short sentences without headings: the answer first, then the reason with citations, \
then what the customer should do next.
- Never ask for passwords, PINs, one-time passcodes or full account numbers.
- Only if the question has nothing to do with banking at Quillmark, say you can only help with Quillmark \
Bank questions."""


def user_turn(question: str, retrieval: Retrieval, today: date) -> str:
    knowledge = "\n\n".join(hit.text for hit in retrieval.knowledge)
    cases = "\n\n".join(hit.text for hit in retrieval.cases)
    return (
        f"Today's date: {today.isoformat()}\n\n"
        f"<policies_and_incidents>\n{knowledge}\n</policies_and_incidents>\n\n"
        f"<past_cases>\n{cases}\n</past_cases>\n\n"
        f"Customer question: {question}"
    )


class RAGChatbot:
    def __init__(self, model, processor, retriever: Retriever, enable_thinking: bool = False,
                 max_new_tokens: int = 512):
        self.model = model
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.retriever = retriever
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self.last_retrieval: Retrieval | None = None

    @torch.inference_mode()
    def ask(self, question: str, history: list[dict] | None = None, on_text=None,
            today: date | None = None) -> Answer:
        """Answer one question. `history` is prior {"role", "content"} turns, oldest first.

        Earlier turns are kept as plain question/answer text; only the current question gets retrieved
        material. The previous question joins the search query so follow-ups like "what about Horizon?"
        still find the right documents.
        """
        history = (history or [])[-MAX_HISTORY_MESSAGES:]
        previous_question = [m["content"] for m in history if m["role"] == "user"][-1:]
        retrieval = self.retriever.search(" ".join(previous_question + [question]))
        self.last_retrieval = retrieval

        messages = [{"role": "system", "content": RULES}, *history,
                    {"role": "user", "content": user_turn(question, retrieval, today or date.today())}]
        input_ids = tokenize(self.tokenizer, render_chat(self.tokenizer, messages, self.enable_thinking),
                             self.model.device)

        # Read the prompt in chunks, then let generate() continue from the last token
        start = time.perf_counter()
        cache = prefill(self.model, input_ids[:, :-1])
        answer = generate_answer(self.model, self.tokenizer, input_ids, cache, cached_tokens=0,
                                 max_new_tokens=self.max_new_tokens, on_text=on_text, start=start)
        answer.sources = retrieval.ids
        answer.seconds_retrieval = retrieval.seconds
        return answer
