# MOPD-RAG: Multi-Teacher RAG Pipeline

A RAG pipeline where a **router** sends each question to specialist "teacher" models (math, code, science, general), and a **distiller** merges their answers into one final response.
Inspired by the multi-teacher on-policy distillation idea from NVIDIA's Nemotron-3-Ultra.

## How it works

```
question
   │
   ▼
QueryRouter ──► picks which teachers to activate
   │
   ▼
for each active teacher:  RAGEngine.retrieve(question) ──► Teacher.answer(question, context)
   │
   ▼
StudentDistiller ──► merges all teacher answers into one final answer
   │
   ▼
printed + logged as JSON lines (timestamp, query, teachers used, outputs, final answer)
```

| Component | Role |
|---|---|
| `router.py` | Decides which teachers a query needs |
| `rag.py` | Loads documents and retrieves context |
| Teachers (`math`, `code`, `science`, `general`) | Each answers using its own specialist prompt plus the retrieved context |
| `distiller.py` | Synthesises the teacher answers into one response |
| `main.py` | Interactive CLI loop and session logging |

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"
python main.py
```

Type a question at the `You >` prompt, or `quit` to exit.

## Status: work in progress

The pipeline logic is in place. The repository layout is being cleaned up so that `core/` (`rag.py`, `router.py`, `distiller.py`) and `teachers/` (the four teacher modules) match the imports in `main.py`.

## Roadmap
- [ ] Finish the `core/` and `teachers/` package layout
- [ ] Add a small evaluation set comparing multi-teacher answers against a single-model baseline
- [ ] Track latency and token cost per query
- [ ] Add tests and a GitHub Actions workflow
