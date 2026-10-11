# MOPD-RAG: Multi-Teacher RAG Pipeline

[![CI](https://github.com/SPpavani/MOPD-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/SPpavani/MOPD-RAG/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)

A RAG pipeline where a **router** sends each question to specialist "teachers" (math, code, science, general), the teachers answer **in parallel** using retrieved context, and a **distiller** merges their answers into one response. It is built to answer one question honestly: **does a multi-specialist pipeline beat a single model call, and at what cost in latency and tokens?** `benchmark.py` measures exactly that.

## How this relates to MOPD (read this first)

The name comes from *Multi-Teacher On-Policy Distillation* (NVIDIA Nemotron-3-Ultra), a **training** technique in which one student model learns from several specialist teachers.

This project is an **inference-time analogue**, not a re-implementation:

- No model is trained or fine-tuned.
- The "teachers" are the same base model with different specialist system prompts.
- The "student" is a synthesis call that merges the teachers' answers.

The teacher / student vocabulary is a metaphor for the architecture. The engineering question it explores is routing and ensembling at inference time.

## Architecture

```
question
   │
   ▼
QueryRouter ──► which specialists does this need? ("general" always participates)
   │
   ▼
RAGEngine.retrieve()   ◄── runs ONCE, TF-IDF over your .txt / .md files
   │
   ▼
┌──────────┬──────────┬──────────┬──────────┐
│  math    │  code    │ science  │ general  │   teachers run IN PARALLEL
└────┬─────┴────┬─────┴────┬─────┴────┬─────┘   (a failing teacher is isolated)
     └──────────┴────┬─────┴──────────┘
                     ▼
            StudentDistiller ──► one synthesised answer
                     │             (skipped when only one teacher answered)
                     ▼
     answer + tokens + latency, logged as JSON lines
```

| Path | Role |
|---|---|
| `core/router.py` | Keyword router. Phrase-aware (`"how many"`), plural-tolerant |
| `core/rag.py` | Chunking and TF-IDF retrieval with a from-scratch scorer. No vector DB. Drops zero-relevance chunks |
| `core/pipeline.py` | `run_query()`: route → retrieve once → parallel teachers → distil. Returns answer, errors and token/latency stats |
| `core/distiller.py` | Synthesis call. Passes a single teacher's answer straight through to save a call |
| `teachers/` | `BaseTeacher` plus four specialists, each a system prompt on the same model |
| `benchmark.py` | Baseline (single call) vs multi-teacher: latency, tokens, optional pairwise LLM judge |
| `main.py` | Interactive CLI and one-shot `--query` mode, with JSONL session logging |
| `tests/` | Offline test suite (fake clients, no network, no API key) |

## Run

```bash
git clone https://github.com/SPpavani/MOPD-RAG.git
cd MOPD-RAG
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"

python main.py                                   # interactive
python main.py --query "Why does ice float on water?"
```

Put your own `.txt` / `.md` files in `data/documents/` (a small sample is included). With no documents, teachers answer from general knowledge. Set `MOPD_MODEL` to change the model.

## Benchmark

```bash
python benchmark.py --judge        # baseline vs multi-teacher on eval/queries.json
```

Reports average latency and tokens for both modes, and (with `--judge`) pairwise wins from an LLM judge with randomised answer positions. Results are saved to `outputs/benchmark.json`.

An LLM judge from the same model family is a useful signal, not ground truth. Treat small win-rate differences with caution; the query set is small.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Design decisions

- **Retrieve once**: all teachers share the same context, so retrieval runs once instead of once per teacher.
- **Parallel teachers**: latency tracks the slowest teacher, not the sum of all teachers.
- **Failure isolation**: one failed teacher call is recorded and the pipeline continues. If the distiller fails, it falls back to the general teacher's answer.
- **Skip pointless synthesis**: with one teacher there is nothing to merge.
- **Drop irrelevant context**: chunks with zero relevance are not injected into prompts.
- **Observability**: every run records tokens in and out, per-teacher latency and errors.

## Known limitations

- Router is keyword-based: fast and predictable, but brittle on ambiguous words.
- Retrieval is lexical (TF-IDF), so paraphrases can be missed.
- Specialists differ only by prompt; they are not separately trained or tool-equipped.
- Plain-text sources only (`.txt`, `.md`); no PDF ingestion.
- Benchmark query set is small.

## Roadmap

- [x] Runnable package layout, tests and CI
- [x] Baseline vs multi-teacher benchmark with token and latency tracking
- [ ] LLM-based router with structured output, compared against the keyword router
- [ ] Embedding or hybrid retrieval, compared against TF-IDF
- [ ] Tool use: a calculator / code-execution tool for the math and code teachers so answers are verified, not just generated
- [ ] Confidence-aware distillation (weight teachers by self-reported confidence)
- [ ] Response streaming and a FastAPI wrapper
- [ ] Prompt caching to cut cost on repeated context
