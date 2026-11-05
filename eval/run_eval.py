"""Run the chatbot on every question of eval/questions.jsonl and score it.

Per item: the answer checks and the route checks of scoring.py on each turn (the LLM's routes
included), plus the answer safeguards: figures not found in what the model was given, figures found
only in other customers' past cases, and citations of documents it wasn't given. Safeguard flags are
counted separately: a flagged figure can be honest arithmetic ("5% of $400 is $20").

Needs the GPU, the RAG index and Postgres (docker compose up -d).
Usage:
    python eval/run_eval.py [--questions FILE] [--only ID ...] [--category NAME ...]
    python eval/run_eval.py --rescore eval/results/<timestamp>.jsonl   # re-check saved answers, no GPU
Results: eval/results/<timestamp>.jsonl and <timestamp>.summary.json (gitignored).
"""
import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "src"))
sys.path.insert(0, str(EVAL_DIR))

from quillmark_bot import monitoring  # noqa: E402
from quillmark_bot.hybrid.engine import HybridChatbot  # noqa: E402
from quillmark_bot.model import load_model  # noqa: E402
from quillmark_bot.rag.retriever import Retriever  # noqa: E402
from scoring import answer_problems, route_problems  # noqa: E402

DEFAULT_TODAY = date(2026, 11, 3)


def item_problems(item: dict, text: str, cited: list[str], case_only: list[str]) -> list[str]:
    """scoring.answer_problems, plus `no_case_figures`: no figure taken from another customer's case."""
    problems = answer_problems(item, text, cited)
    if item.get("no_case_figures") and case_only:
        problems.append(f"used other customers' figures: {', '.join(case_only)}")
    return problems


def run_item(bot: HybridChatbot, item: dict, session_id: str | None = None) -> dict:
    """`session_id`: when given, each turn is also recorded for the dashboard (source "eval")."""
    today = date.fromisoformat(item["today"]) if "today" in item else DEFAULT_TODAY
    history, previous, route_issues, seconds = [], None, [], 0.0
    for turn in item["turns"]:
        answer = bot.ask(turn["question"], history, today=today, previous_request=previous)
        route = bot.last_trace.route
        if session_id:
            monitoring.log_interaction("eval", turn["question"], answer, bot.last_trace, session_id)
        route_issues += route_problems(route, turn.get("route"))
        seconds += route.seconds + answer.seconds_total
        history += [{"role": "user", "content": turn["question"]}, {"role": "assistant", "content": answer.text}]
        previous = route.data_request
    problems = item_problems(item, answer.text, answer.cited_ids, answer.case_only_numbers)
    return {
        "id": item["id"], "category": item["category"],
        "answer_ok": not problems, "route_ok": not route_issues,
        "answer_problems": problems, "route_problems": route_issues,
        "unverified_numbers": answer.unverified_numbers, "case_only_numbers": answer.case_only_numbers,
        "invalid_citations": answer.invalid_citations,
        "answer": answer.text, "cited": answer.cited_ids,
        "route": {"decided_by": route.decided_by, "reason": route.reason,
                  "request": route.data_request.to_dict() if route.data_request else None},
        "seconds": round(seconds, 1),
    }


def summarize(rows: list[dict], stamp: str, minutes: float | None) -> dict:
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    return {
        "when": stamp, "questions": len(rows),
        "answers_correct": sum(r["answer_ok"] for r in rows),
        "routes_correct": sum(r["route_ok"] for r in rows),
        "answers_with_unverified_figures": sum(bool(r["unverified_numbers"]) for r in rows),
        "answers_with_case_only_figures": sum(bool(r["case_only_numbers"]) for r in rows),
        "answers_with_invalid_citations": sum(bool(r["invalid_citations"]) for r in rows),
        "by_category": {c: {"answers": f"{sum(r['answer_ok'] for r in rs)}/{len(rs)}",
                            "routes": f"{sum(r['route_ok'] for r in rs)}/{len(rs)}"}
                        for c, rs in by_category.items()},
        "median_seconds_per_item": statistics.median(r["seconds"] for r in rows),
        "total_minutes": minutes,
    }


def rescore(path: Path, items: dict) -> None:
    """Re-apply the current answer checks to saved answers; routes and flags are kept as they ran."""
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    for row in rows:
        row["answer_problems"] = item_problems(items[row["id"]], row["answer"], row["cited"], row["case_only_numbers"])
        row["answer_ok"] = not row["answer_problems"]
        if row["answer_problems"]:
            print(f"FAIL  {row['id']:<24} {'; '.join(row['answer_problems'])}")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary_path = path.with_name(path.stem + ".summary.json")
    old = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary = summarize(rows, old.get("when", path.stem), old.get("total_minutes"))
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="item IDs to run")
    parser.add_argument("--category", nargs="*", help="categories to run")
    parser.add_argument("--rescore", type=Path, help="re-check a saved results file with the current checks")
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl",
                        help="question file (default: eval/questions.jsonl)")
    args = parser.parse_args()

    items = [json.loads(line) for line in args.questions.open(encoding="utf-8")]
    if args.rescore:
        rescore(args.rescore, {i["id"]: i for i in items})  # pass --questions for a fresh-set results file
        return
    items = [i for i in items if (not args.only or i["id"] in args.only)
             and (not args.category or i["category"] in args.category)]

    model, processor = load_model()
    bot = HybridChatbot(model, processor, Retriever(), max_new_tokens=400)
    bot.build_cache()

    results_dir = EVAL_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}-{args.questions.stem}"
    session_id = f"eval-{stamp}" if monitoring.ensure_schema() else None
    rows = []
    start = time.perf_counter()
    with (results_dir / f"{stamp}.jsonl").open("w", encoding="utf-8") as out:
        for n, item in enumerate(items, 1):
            row = run_item(bot, item, session_id)
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            flags = (row["unverified_numbers"] + [f"{n} (cases only)" for n in row["case_only_numbers"]]
                     + row["invalid_citations"])
            print(f"[{n:>2}/{len(items)}] answer {'PASS' if row['answer_ok'] else 'FAIL'}  "
                  f"route {'PASS' if row['route_ok'] else 'FAIL'}  {row['id']:<24} "
                  f"{'; '.join(row['answer_problems'] + row['route_problems'])}"
                  + (f"  [flagged: {', '.join(flags)}]" if flags else ""), flush=True)

    summary = summarize(rows, stamp, round((time.perf_counter() - start) / 60, 1))
    (results_dir / f"{stamp}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if session_id and not (args.only or args.category):  # only full runs are comparable over time
        monitoring.log_eval_run(summary, rows, args.questions.name)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
