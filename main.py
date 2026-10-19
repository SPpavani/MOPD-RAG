"""
main.py  -  MOPD-RAG Pipeline
Inspired by Nemotron-3-Ultra's Multi-Teacher On-Policy Distillation.

Run:
    python main.py                      # interactive
    python main.py --query "Why does ice float?"   # one-shot
"""
import argparse
import datetime
import json
import logging
import os
import sys
from dataclasses import asdict

import config
from core.cache import ResponseCache
from core.distiller import StudentDistiller
from core.pipeline import PipelineResult, run_query
from core.rag import RAGEngine
from core.router import QueryRouter
from teachers import build_teachers


def check_api_key() -> None:
    if not config.ANTHROPIC_API_KEY:
        print("\n[ERROR] ANTHROPIC_API_KEY not set.")
        print("  Export it first:  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)


def log_session(result: PipelineResult) -> None:
    os.makedirs(os.path.dirname(config.LOG_FILE) or ".", exist_ok=True)
    record = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), **asdict(result)}
    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_result(result: PipelineResult) -> None:
    print(f"\n{'─' * 60}\n  FINAL SYNTHESIZED ANSWER:\n{'─' * 60}")
    print(result.final_answer)
    s = result.stats
    print(
        f"\n[{', '.join(result.teachers_activated)}] "
        f"{s['total_latency_s']}s | tokens in/out: {s['input_tokens']}/{s['output_tokens']}"
    )
    cost = s.get("cost_usd")
    extras = [f"cost ~${cost:.5f}" if cost is not None else "cost unknown for this model"]
    if s.get("tool_calls"):
        extras.append(f"{s['tool_calls']} calculator call(s)")
    if s.get("cache_hit"):
        extras.append("answered from cache")
    print("[" + ", ".join(extras) + "]")
    for name, err in result.errors.items():
        print(f"[warning] {name} failed: {err}")
    for warning in result.security_warnings:
        print(f"[security] {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MOPD-RAG: multi-teacher RAG pipeline")
    parser.add_argument("--query", help="answer one question and exit")
    args = parser.parse_args()

    check_api_key()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    print("\n╔══════════════════════════════════════════════╗")
    print("║   MOPD-RAG  |  Multi-Teacher RAG Pipeline   ║")
    print("║   Inspired by NVIDIA Nemotron-3-Ultra        ║")
    print("╚══════════════════════════════════════════════╝\n")

    rag = RAGEngine()
    router = QueryRouter()
    distiller = StudentDistiller()
    teachers = build_teachers()
    cache = ResponseCache()
    rag.load_documents()

    def handle(query: str) -> None:
        print(f"\n{'═' * 60}\n  QUERY: {query}\n{'═' * 60}")
        result = run_query(query, rag, router, teachers, distiller, cache=cache)
        print_result(result)
        log_session(result)
        print(f"[Logged to {config.LOG_FILE}]")

    if args.query:
        handle(args.query)
        return

    print("\nType your query (or 'quit' to exit):\n")
    while True:
        try:
            query = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        handle(query)


if __name__ == "__main__":
    main()
