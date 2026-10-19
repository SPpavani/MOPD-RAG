"""
core/pipeline.py
The MOPD-RAG pipeline as a pure, testable function.

  route -> retrieve ONCE -> run teachers IN PARALLEL -> distil -> result

A teacher that fails does not kill the answer: its error is recorded and the
distiller works with the teachers that succeeded.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
from core.cache import ResponseCache
from core.cost import estimate_cost

logger = logging.getLogger(__name__)


@dataclass
class TeacherRun:
    name: str
    text: str = ""
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    tool_calls: int = 0


@dataclass
class PipelineResult:
    query: str
    teachers_activated: List[str]
    teacher_outputs: Dict[str, str]
    errors: Dict[str, str]
    final_answer: str
    context_used: bool
    stats: Dict = field(default_factory=dict)
    security_warnings: List[str] = field(default_factory=list)


def _run_teacher(name: str, teacher, query: str, context: str) -> TeacherRun:
    try:
        resp = teacher.ask(query, context)
        return TeacherRun(
            name=name,
            text=resp.text,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            latency_s=resp.latency_s,
            tool_calls=getattr(resp, "tool_calls", 0),
        )
    except Exception as exc:  # one failing teacher must not break the pipeline
        logger.exception("Teacher '%s' failed", name)
        return TeacherRun(name=name, error=f"{type(exc).__name__}: {exc}")


def run_query(
    query: str,
    rag,
    router,
    teachers: Dict[str, object],
    distiller,
    max_workers: Optional[int] = None,
    cache: Optional[ResponseCache] = None,
) -> PipelineResult:
    start = time.perf_counter()

    names = [n for n in router.route(query) if n in teachers]
    logger.info("Activated teachers: %s", names)

    context = rag.retrieve(query)   # once, shared by all teachers
    warnings = list(getattr(rag, "last_warnings", []) or [])
    for w in warnings:
        logger.warning("Security: %s", w)
    logger.info("Retrieved context: %s", "yes" if context else "none (general knowledge)")

    cache_key = None
    if cache is not None:
        cache_key = cache.key(query, context, config.MODEL, names)
        hit = cache.get(cache_key)
        if hit is not None:
            cache.tokens_saved += hit.stats.get("input_tokens", 0) + hit.stats.get("output_tokens", 0)
            stats = dict(hit.stats, cache_hit=True, total_latency_s=round(time.perf_counter() - start, 3),
                         input_tokens=0, output_tokens=0, tool_calls=0, cost_usd=0.0 if hit.stats.get("cost_usd") is not None else None)
            return PipelineResult(**{**hit.__dict__, "stats": stats, "security_warnings": warnings})

    runs: Dict[str, TeacherRun] = {}
    if names:
        workers = max(1, min(len(names), max_workers or config.MAX_WORKERS))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {n: pool.submit(_run_teacher, n, teachers[n], query, context) for n in names}
            for n in names:
                runs[n] = futures[n].result()

    outputs = {n: r.text for n, r in runs.items() if r.error is None}
    errors = {n: r.error for n, r in runs.items() if r.error is not None}

    input_tokens = sum(r.input_tokens for r in runs.values())
    output_tokens = sum(r.output_tokens for r in runs.values())

    if not outputs:
        final = "All teachers failed: " + "; ".join(f"{n}: {e}" for n, e in errors.items())
        if not errors:
            final = "No teachers were activated for this query."
    else:
        try:
            resp = distiller.distill_response(query, outputs)
            final = resp.text
            input_tokens += resp.input_tokens
            output_tokens += resp.output_tokens
        except Exception as exc:
            logger.exception("Distiller failed; falling back to a teacher answer")
            errors["distiller"] = f"{type(exc).__name__}: {exc}"
            final = outputs.get("general") or next(iter(outputs.values()))

    stats = {
        "total_latency_s": round(time.perf_counter() - start, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": estimate_cost(config.MODEL, input_tokens, output_tokens),
        "tool_calls": sum(r.tool_calls for r in runs.values()),
        "cache_hit": False,
        "teacher_latency_s": {n: r.latency_s for n, r in runs.items()},
    }
    result = PipelineResult(
        query=query,
        teachers_activated=names,
        teacher_outputs=outputs,
        errors=errors,
        final_answer=final,
        context_used=bool(context),
        stats=stats,
        security_warnings=warnings,
    )
    if cache is not None and cache_key and not errors and outputs:
        cache.put(cache_key, result)
    return result
