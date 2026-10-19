"""
benchmark.py
Does the multi-teacher pipeline beat a single model call? Measure it.

For every query in eval/queries.json it runs:
  baseline : one General-teacher call (same retrieved context)
  mopd     : the full pipeline (route -> teachers in parallel -> distil)
and records latency and token usage. With --judge, a model compares the two
answers pairwise (positions randomised to avoid position bias).

Caveat: an LLM judge from the same model family is a useful signal, not ground truth.

Usage:
    python benchmark.py [--queries eval/queries.json] [--judge] [--limit N]
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def parse_verdict(text: str) -> Optional[str]:
    """Extract A / B / TIE from the judge's JSON reply; None if it cannot be parsed."""
    match = re.search(r'"winner"\s*:\s*"(A|B|TIE)"', text or "")
    return match.group(1) if match else None


def load_queries(path) -> List[Dict]:
    """eval/queries.json holds plain strings or {"query", "all_of", "any_of"} objects with regex checks."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [{"query": item} if isinstance(item, str) else dict(item) for item in raw]


def check_answer(text: str, spec: Dict) -> Optional[bool]:
    """Deterministic correctness: every `all_of` regex must match, and at least one `any_of` regex (if given).

    Returns None when the query has no check. A regex check is strict and can reject a correct answer that is
    phrased unexpectedly, so treat accuracy as a floor and read a few answers by hand.
    """
    all_of, any_of = spec.get("all_of") or [], spec.get("any_of") or []
    if not all_of and not any_of:
        return None
    flags = re.IGNORECASE | re.DOTALL
    if not all(re.search(p, text or "", flags) for p in all_of):
        return False
    return bool(any(re.search(p, text or "", flags) for p in any_of)) if any_of else True


def summarize(rows: List[Dict]) -> Dict:
    out: Dict = {"queries": len(rows)}
    for mode in ("baseline", "mopd"):
        lat = [r[mode]["latency_s"] for r in rows]
        tok = [r[mode]["tokens"] for r in rows]
        out[mode] = {
            "avg_latency_s": round(sum(lat) / len(lat), 2) if lat else 0.0,
            "avg_tokens": round(sum(tok) / len(tok)) if tok else 0,
        }
        checked = [r[mode]["correct"] for r in rows if r[mode].get("correct") is not None]
        if checked:
            out[mode]["accuracy"] = f"{sum(checked)}/{len(checked)}"
        costs = [r[mode]["cost_usd"] for r in rows if r[mode].get("cost_usd") is not None]
        if costs:
            out[mode]["avg_cost_usd"] = round(sum(costs) / len(costs), 6)
    wins = Counter(r.get("winner") for r in rows if r.get("winner"))
    if wins:
        out["judge"] = {
            "mopd": wins.get("mopd", 0),
            "baseline": wins.get("baseline", 0),
            "tie": wins.get("tie", 0),
        }
    return out


def judge_pair(client, model: str, query: str, answer_a: str, answer_b: str) -> Optional[str]:
    prompt = (
        "You are an impartial judge. Compare two answers to the question and pick the one "
        "that is more accurate, complete and clear.\n\n"
        f"Question: {query}\n\n=== Answer A ===\n{answer_a}\n\n=== Answer B ===\n{answer_b}\n\n"
        'Reply with JSON only, for example {"winner": "A"}. Allowed values: "A", "B", "TIE".'
    )
    response = client.messages.create(
        model=model, max_tokens=50, messages=[{"role": "user", "content": prompt}]
    )
    return parse_verdict(response.content[0].text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="eval/queries.json")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-tools", action="store_true", help="ablation: take the calculator away from the math teacher")
    args = parser.parse_args()

    import config
    from core.distiller import StudentDistiller
    from core.pipeline import run_query
    from core.rag import RAGEngine
    from core.router import QueryRouter
    from teachers import build_teachers

    if not config.ANTHROPIC_API_KEY:
        sys.exit("ANTHROPIC_API_KEY not set.")

    from core.cost import estimate_cost

    queries = load_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]

    rag, router, distiller, teachers = RAGEngine(), QueryRouter(), StudentDistiller(), build_teachers()
    if args.no_tools:
        for teacher in teachers.values():
            teacher.tools = []
    rag.load_documents()
    rng = random.Random(42)

    rows: List[Dict] = []
    for spec in queries:
        query = spec["query"]
        context = rag.retrieve(query)
        start = time.perf_counter()
        base = teachers["general"].ask(query, context)
        base_latency = time.perf_counter() - start

        result = run_query(query, rag, router, teachers, distiller)

        row = {
            "query": query,
            "baseline": {
                "answer": base.text,
                "latency_s": round(base_latency, 2),
                "tokens": base.input_tokens + base.output_tokens,
                "cost_usd": estimate_cost(config.MODEL, base.input_tokens, base.output_tokens),
                "correct": check_answer(base.text, spec),
            },
            "mopd": {
                "answer": result.final_answer,
                "latency_s": result.stats["total_latency_s"],
                "tokens": result.stats["input_tokens"] + result.stats["output_tokens"],
                "teachers": result.teachers_activated,
                "cost_usd": result.stats.get("cost_usd"),
                "tool_calls": result.stats.get("tool_calls", 0),
                "correct": check_answer(result.final_answer, spec),
            },
            "winner": None,
        }

        if args.judge:
            first, second = ("baseline", "mopd") if rng.random() < 0.5 else ("mopd", "baseline")
            verdict = judge_pair(
                distiller.get_client(), config.MODEL, query, row[first]["answer"], row[second]["answer"]
            )
            if verdict == "TIE":
                row["winner"] = "tie"
            elif verdict == "A":
                row["winner"] = first
            elif verdict == "B":
                row["winner"] = second

        rows.append(row)
        print(f"done: {query[:60]}")

    summary = summarize(rows)
    print("\n" + json.dumps(summary, indent=2))

    out = Path("outputs/benchmark.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
