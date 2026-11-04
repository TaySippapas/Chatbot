"""Chat with the hybrid bot (handbook cache + past cases + case database) in the terminal.

Usage:
    python -m quillmark_bot.hybrid                   # interactive; /reset clears history, /quit exits
    python -m quillmark_bot.hybrid -q "How many duplicate-charge complaints came from Bayside?"

Needs the RAG index (python -m quillmark_bot.rag.index) and, for figures, Postgres (docker compose up -d).
"""
import argparse

from quillmark_bot.hybrid.engine import HybridChatbot
from quillmark_bot.model import load_model
from quillmark_bot.rag.retriever import Retriever


def describe(trace) -> str:
    route = trace.route
    lines = [f"[route by {route.decided_by} in {route.seconds:.2f}s: {route.reason}]"]
    if trace.data:
        lines.append(f"[database: {trace.data.description} ({trace.seconds_data:.2f}s)"
                     + (f", ERROR {trace.data.error}" if trace.data.error else "") + "]")
    if trace.retrieval:
        lines.append(f"[cases: {', '.join(h.id for h in trace.retrieval.cases)}]")
    return "\n".join(lines)


def stats(answer) -> str:
    lines = [f"[read {answer.computed_prompt_tokens:,} new tokens, {answer.cached_tokens:,} from cache | "
             f"first token {answer.seconds_to_first_token:.2f}s | {answer.output_tokens} tokens in "
             f"{answer.seconds_total:.1f}s | cited {', '.join(answer.cited_ids) or 'nothing'}]"]
    if answer.unverified_numbers:
        lines.append(f"[CHECK: figures not found in what the model was given: {', '.join(answer.unverified_numbers)}]")
    if answer.case_only_numbers:
        lines.append(f"[CHECK: figures only in other customers' cases: {', '.join(answer.case_only_numbers)}]")
    if answer.invalid_citations:
        lines.append(f"[CHECK: cited but not provided: {', '.join(answer.invalid_citations)}]")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    print("Loading model, retriever and caches...")
    model, processor = load_model()
    bot = HybridChatbot(model, processor, Retriever(), max_new_tokens=args.max_new_tokens)
    bot.build_cache()
    print("Ready.\n")

    last_request = None  # the previous answer's database query, for follow-ups

    def answer(question: str, history: list[dict]):
        nonlocal last_request
        result = bot.ask(question, history, on_text=lambda t: print(t, end="", flush=True),
                         previous_request=last_request)
        last_request = bot.last_trace.route.data_request
        print(f"\n{describe(bot.last_trace)}\n{stats(result)}\n")
        return result

    if args.question:
        answer(args.question, [])
        return

    history: list[dict] = []
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question in {"/quit", "/exit"}:
            break
        if question == "/reset":
            history.clear()
            last_request = None
            print("History cleared.\n")
            continue
        if question:
            print("bot> ", end="")
            result = answer(question, history)
            history += [{"role": "user", "content": question}, {"role": "assistant", "content": result.text}]


if __name__ == "__main__":
    main()
