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
RAGEngine.retrieve()   ◄── runs ONCE, BM25 over your .txt / .md files
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
| `core/router.py` | Weighted router: strong cues and arithmetic patterns activate a specialist; ambiguous words (`class`, `cell`, `error`) need a second cue. The original router is kept as `mode="keyword_legacy"` |
| `core/rag.py` | Chunking and BM25 retrieval over content words (stopwords removed) with a term-coverage gate, so unrelated questions get no context. Injected chunks are quarantined. Original TF-IDF scorer kept as `tfidf_legacy` |
| `core/calc.py` | Safe `ast`-based calculator tool for the math teacher (no `eval`, size-limited) |
| `core/security.py` | Prompt-injection tripwires for retrieved text; wrapper and header neutralisation |
| `core/cost.py`, `core/cache.py` | USD cost per query; LRU cache for repeated questions |
| `evals/` | Retrieval and routing evaluation sets (no API key needed) |
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

## Measured results (all reproducible, no API key)

Two evaluation sets live in `evals/` and run in CI as quality gates. **Both were written by the project author**, so they share the author's blind spots; treat them as a regression suite and a direction, not as a general benchmark.

### Retrieval: `python evals/run_retrieval_eval.py --scorer tfidf_legacy|bm25`

A bug was found while auditing: the original scorer gave a positive score to any chunk sharing *any* word with the question, including "is", "the" and "of", so **every unrelated question received RAG context** ("Who is the president of the United States?" scored highest). The README then claimed zero-relevance chunks were dropped; they were not.

| 26 answerable + 12 off-topic questions | original (TF-IDF) | now (BM25 + gate) |
|---|---:|---:|
| Exact-wording questions, correct doc ranked first | 17/18 | **18/18** |
| Paraphrased questions, correct doc in top 3 | 7/8 | 7/8 |
| Off-topic questions that still received context | **12/12** | **0/12** |
| Average context injected per question | 1033 chars | **264 chars** |

The coverage threshold (a chunk must contain at least a third of the question's content words) was chosen **after** looking at this sweep on this same small set, so it is mildly tuned to it:

| min coverage | exact hit@1 | paraphrase hit@3 | off-topic leaks |
|---:|---:|---:|---:|
| 0.00 | 18/18 | 8/8 | 3/12 |
| 0.25 | 18/18 | 7/8 | 3/12 |
| **0.34** | 18/18 | 7/8 | **0/12** |
| 0.50 | 18/18 | 4/8 | 0/12 |

Remaining limit: this is lexical. A paraphrase with no shared words is missed (the sweep's 8/8 at 0.00 only holds because the corpus is tiny). Embedding or hybrid retrieval is the real fix.

### Routing: `python evals/run_routing_eval.py --split dev|test --router keyword_legacy|weighted`

The original router activated a specialist when *any* listed word appeared: "social class" -> code, "prison cell" -> science, "margin of error" -> code, and "What is 15% of 240?" -> **no math teacher at all**. The new router gives ambiguous words half weight. The 48-question **dev** set was used while building it; the 48-question **test** set was written first and run once at the end.

| | exact-set accuracy | precision | recall | F1 |
|---|---:|---:|---:|---:|
| original, dev | 56% | 0.75 | 0.66 | 0.70 |
| original, **test** | 69% | 0.86 | 0.75 | 0.80 |
| weighted, dev (tuned on; do not quote) | 100% | 1.00 | 1.00 | 1.00 |
| weighted, **test** (held out) | **94%** | **1.00** | 0.93 | **0.96** |

The three test misses are all science questions in everyday words ("Why is the sky blue?"). That is the ceiling of keyword routing; an LLM router with structured output is the next step. A missed specialist degrades gracefully (the general teacher still answers), while a wrongly added one costs a paid call, which is why precision matters most here.

### What it does to cost: honest accounting

Over the 20 benchmark questions (`data/documents` corpus): context characters sent per question fell **98%** (1334 -> 22), but model calls per question **rose from 1.90 to 2.30**, because the new router correctly activates the math teacher on arithmetic that the old one missed. This is a deliberate accuracy-for-cost trade, not a saving. Net token cost needs a real run (below). Ideas to claw it back (skip the general teacher when a tool-verified specialist answered; skip the distiller) are on the roadmap.

## Verified arithmetic: the calculator tool

Models are unreliable at multi-digit arithmetic. The math teacher now calls `core/calc.py` instead of computing in its head: an `ast`-based evaluator that allows only numbers, `+ - * / // % **`, parentheses and a short list of functions. It rejects `__import__`, attribute access, comprehensions, `9**9**9` and `factorial(10**7)`, and it refuses ambiguous input like `1,234` rather than guessing (a silent wrong answer is worse than an error). The tool loop is bounded (4 rounds), tolerates unknown tools and tool crashes, and is tested with a scripted client shaped like the Anthropic response. **Not yet verified against the live API**; run the benchmark below to check.

## Security: indirect prompt injection

Retrieved documents are written by someone else yet pasted into the prompt. Chunks that look like injection ("ignore all previous instructions...", tag break-outs, invisible characters) are quarantined and reported in `result.security_warnings`; a document cannot close the `<context>` wrapper; a teacher cannot fake the distiller's `=== NAME TEACHER ===` headers; prompts declare context and teacher text to be data. This is a tripwire, not a guarantee.

## Run without an API key

```bash
python examples/demo_offline.py     # whole pipeline with a scripted model: routing, calculator, cache, quarantine
python evals/run_retrieval_eval.py
python evals/run_routing_eval.py --split test --misses
```

## Benchmark with a real model

```bash
python benchmark.py --judge               # baseline vs multi-teacher
python benchmark.py --no-tools            # ablation: take the calculator away
```

`eval/queries.json` holds 20 questions, 18 with **deterministic regex answer checks** (e.g. `1234 * 5678` must contain `7,006,652`), so accuracy no longer depends only on an LLM judge from the same model family. It reports accuracy, latency, tokens and USD cost for both modes. Prices in `core/cost.py` are a snapshot; verify them at anthropic.com/pricing or set `MOPD_PRICE_IN` / `MOPD_PRICE_OUT`. **Real-model results: pending** (no key was available when this was written). Regex checks can reject a correct answer phrased unexpectedly, so read a few answers by hand.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q        # 110 tests, no network or API key; CI enforces 85% coverage (currently ~98%)
```

## Design decisions

- **Retrieve once**: all teachers share the same context, so retrieval runs once instead of once per teacher.
- **Parallel teachers**: latency tracks the slowest teacher, not the sum of all teachers.
- **Failure isolation**: one failed teacher call is recorded and the pipeline continues. If the distiller fails, it falls back to the general teacher's answer.
- **Skip pointless synthesis**: with one teacher there is nothing to merge.
- **Drop irrelevant context**: chunks with zero relevance are not injected into prompts.
- **Observability**: every run records tokens in and out, per-teacher latency and errors.

## Known limitations

- Router is lexical: it misses questions phrased in everyday words and is tuned by hand to a lexicon.
- Retrieval is lexical (BM25), so paraphrases without shared words are missed. Chunking cuts at fixed character positions and can split a word.
- The evaluation sets are small and author-written; the coverage threshold was tuned on the retrieval set.
- Specialists differ by prompt, not training; only the math teacher has a tool.
- The calculator tool loop is tested against a scripted client, not the live API. No real-model benchmark results have been recorded yet.
- Injection detection is a tripwire (pattern lists), not a complete defence.
- Plain-text sources only (`.txt`, `.md`); no PDF ingestion.
- Cost uses a price snapshot and is `None` for unknown models.

## Roadmap

- [x] Runnable package layout, tests and CI
- [x] Baseline vs multi-teacher benchmark with token and latency tracking
- [x] Retrieval fix (stopwords, BM25, coverage gate) with before/after measurement
- [x] Weighted router with held-out evaluation
- [x] Calculator tool for the math teacher; deterministic answer checks in the benchmark
- [x] Prompt-injection quarantine; USD cost tracking; response cache
- [ ] Record real-model results (accuracy with vs without the calculator, cost per query)
- [ ] LLM-based router with structured output, compared against the weighted router on `evals/`
- [ ] Hybrid retrieval (BM25 + embeddings) compared on `evals/`
- [ ] Cost-aware cascade: skip the general teacher or the distiller when a tool-verified specialist answered; cheap model first
- [ ] Confidence-aware distillation / abstention when teachers disagree
- [ ] Streaming, a FastAPI wrapper, prompt caching
