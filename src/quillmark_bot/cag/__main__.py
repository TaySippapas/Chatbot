"""Chat with the CAG bot in the terminal.

Usage:
    python -m quillmark_bot.cag                      # interactive; /reset clears history, /quit exits
    python -m quillmark_bot.cag -q "What is the overdraft fee?"
"""
import argparse

from quillmark_bot.cag.engine import CAGChatbot
from quillmark_bot.cag.knowledge import build_knowledge
from quillmark_bot.model import load_model


def stats(answer) -> str:
    return (f"[read {answer.computed_prompt_tokens} new prompt tokens, {answer.cached_tokens:,} from cache | "
            f"first token {answer.seconds_to_first_token:.2f}s | {answer.output_tokens} tokens in "
            f"{answer.seconds_total:.1f}s | cited {', '.join(answer.cited_ids) or 'nothing'}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument("--thinking", action="store_true", help="let the model reason before answering (slower)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    print("Loading model...")
    model, processor = load_model()
    bot = CAGChatbot(model, processor, build_knowledge(), enable_thinking=args.thinking,
                     max_new_tokens=args.max_new_tokens)
    bot.build_cache()
    print(f"Knowledge cached: {bot.cached_tokens:,} tokens in {bot.cache_build_seconds:.1f}s\n")

    def answer(question: str, history: list[dict]):
        result = bot.ask(question, history, on_text=lambda t: print(t, end="", flush=True))
        print(f"\n{stats(result)}\n")
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
            print("History cleared.\n")
            continue
        if question:
            print("bot> ", end="")
            result = answer(question, history)
            history += [{"role": "user", "content": question}, {"role": "assistant", "content": result.text}]


if __name__ == "__main__":
    main()
