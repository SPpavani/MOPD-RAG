"""
main.py  —  MOPD-RAG Pipeline
Inspired by Nemotron-3-Ultra's Multi-Teacher On-Policy Distillation.

Run:
    python main.py
"""

import os
import json
import datetime

from config import LOG_FILE, ANTHROPIC_API_KEY
from core.rag import RAGEngine
from core.router import QueryRouter
from core.distiller import StudentDistiller
from teachers.math_teacher import MathTeacher
from teachers.code_teacher import CodeTeacher
from teachers.science_teacher import ScienceTeacher
from teachers.general_teacher import GeneralTeacher


# ── Registry: name → teacher instance ────────────────────────────
TEACHER_REGISTRY = {
    "math":    MathTeacher(),
    "code":    CodeTeacher(),
    "science": ScienceTeacher(),
    "general": GeneralTeacher(),
}


def check_api_key():
    if not ANTHROPIC_API_KEY:
        print("\n[ERROR] ANTHROPIC_API_KEY not set.")
        print("  Export it first:  export ANTHROPIC_API_KEY='sk-ant-...'")
        exit(1)


def log_session(query: str, teacher_outputs: dict, final_answer: str):
    os.makedirs("outputs", exist_ok=True)
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "query": query,
        "teachers_activated": list(teacher_outputs.keys()),
        "teacher_outputs": teacher_outputs,
        "final_answer": final_answer,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_pipeline(query: str, rag: RAGEngine, router: QueryRouter, distiller: StudentDistiller):
    print(f"\n{'═'*60}")
    print(f"  QUERY: {query}")
    print(f"{'═'*60}")

    # Step 1: Route — find which teachers to activate
    active_teacher_names = router.route(query)

    # Step 2: For each teacher, retrieve context + generate answer
    teacher_outputs: dict[str, str] = {}
    for name in active_teacher_names:
        teacher = TEACHER_REGISTRY.get(name)
        if not teacher:
            continue
        context = rag.retrieve(query)
        answer  = teacher.answer(query, context)
        teacher_outputs[name] = answer

    # Step 3: Distill all teacher answers → final answer
    final = distiller.distill(query, teacher_outputs)

    # Step 4: Print & log
    print(f"\n{'─'*60}")
    print("  FINAL SYNTHESIZED ANSWER:")
    print(f"{'─'*60}")
    print(final)

    log_session(query, teacher_outputs, final)
    print(f"\n[Logged to {LOG_FILE}]")
    return final


def main():
    check_api_key()

    print("\n╔══════════════════════════════════════════════╗")
    print("║   MOPD-RAG  |  Multi-Teacher RAG Pipeline   ║")
    print("║   Inspired by NVIDIA Nemotron-3-Ultra        ║")
    print("╚══════════════════════════════════════════════╝\n")

    # Initialise components
    rag       = RAGEngine()
    router    = QueryRouter()
    distiller = StudentDistiller()

    rag.load_documents()

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

        run_pipeline(query, rag, router, distiller)


if __name__ == "__main__":
    main()
