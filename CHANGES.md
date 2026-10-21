# What changed and why

1. **Retrieval bug found and fixed.** The scorer rewarded stopwords, so every unrelated question received context
   (12/12 off-topic questions). Now BM25 over content words with a term-coverage gate: 0/12, and 1033 -> 264 chars of context per question.
2. **Router rebuilt and measured.** Ambiguous words need a second cue. Held-out test: exact-set accuracy 69% -> 94%, F1 0.80 -> 0.96, precision 1.00.
   Trade-off, reported honestly: model calls per question rose 1.90 -> 2.30 because math is now routed correctly.
3. **Calculator tool** for the math teacher (safe `ast` evaluator, bounded tool loop).
4. **Prompt-injection quarantine** for retrieved chunks, wrapper and header neutralisation.
5. **Benchmark upgraded**: deterministic regex answer checks, USD cost, calculator ablation (`--no-tools`).
6. **Cost accounting and an LRU response cache.**
7. **Quality gates in CI**: retrieval and routing evals fail the build if they regress; ruff, 85% coverage gate, pip-audit, Dependabot.
8. **Cleanup (do this by hand, see below)**: four stale stub files, an empty root `__init__.py` and a stale `Project Structure.docx` (it describes files that do not exist) are safe to delete. Run `git rm router.py rag.py distiller.py __init__.py "Project Structure.docx"` on this branch. Tests and lint pass with or without them.

Two bugs in my own new code were caught by tests before delivery (comma handling and big-integer growth in the calculator) and one by the offline demo (cache hits reported calculator calls).
