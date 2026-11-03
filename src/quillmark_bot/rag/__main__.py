"""Chat with the RAG bot in the terminal.

Usage:
    python -m quillmark_bot.rag                      # interactive; /reset clears history, /quit exits
    python -m quillmark_bot.rag -q "What is the overdraft fee?"
    python -m quillmark_bot.rag --show-sources       # also list what was retrieved for each question

Build the index first: python -m quillmark_bot.rag.index
"""
import argparse

from quillmark_bot.model import load_model
from quillmark_bot.rag.engine import RAGChatbot
from quillmark_bot.rag.retriever import Retriever


def stats(answer) -> str:
    return (f"[retrieval {answer.seconds_retrieval:.2f}s | read {answer.prompt_tokens:,} prompt tokens | "
            f"first token {answer.seconds_to_first_token:.2f}s | {answer.output_tokens} tokens in "
            f"{answer.seconds_total:.1f}s | cited {', '.join(answer.cited_ids) or 'nothing'}]")


def sources(retrieval) -> str:
    lines = ["Retrieved:"]
    for hit in retrieval.knowledge + retrieval.cases:
        lines.append(f"  {hit.id:<12} {hit.score:5.2f}  {hit.text.splitlines()[0][:90]}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument("--show-sources", action="store_true", help="list retrieved documents per question")
    parser.add_argument("--thinking", action="store_true", help="let the model reason before answering (slower)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    print("Loading model and retriever...")
    model, processor = load_model()
    bot = RAGChatbot(model, processor, Retriever(), enable_thinking=args.thinking,
                     max_new_tokens=args.max_new_tokens)
    print("Ready.\n")

    def answer(question: str, history: list[dict]):
        result = bot.ask(question, history, on_text=lambda t: print(t, end="", flush=True))
        print(f"\n{stats(result)}")
        if args.show_sources:
            print(sources(bot.last_retrieval))
        print()
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
