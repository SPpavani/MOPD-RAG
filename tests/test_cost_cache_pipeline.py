
from core.cache import ResponseCache
from core.cost import estimate_cost
from core.pipeline import run_query
from tests.test_pipeline import FakeDistiller, FakeRAG, FakeRouter, FakeTeacher


def test_cost_for_known_model_and_unknown_model(monkeypatch):
    monkeypatch.delenv("MOPD_PRICE_IN", raising=False)
    monkeypatch.delenv("MOPD_PRICE_OUT", raising=False)
    assert estimate_cost("claude-haiku-4-5-20251001", 1_000_000, 1_000_000) == 6.0
    assert estimate_cost("some-unknown-model", 100, 100) is None


def test_price_override_from_environment(monkeypatch):
    monkeypatch.setenv("MOPD_PRICE_IN", "2")
    monkeypatch.setenv("MOPD_PRICE_OUT", "10")
    assert estimate_cost("anything", 1_000_000, 500_000) == 7.0


def test_cache_is_lru_and_counts_hits():
    cache = ResponseCache(max_size=2)
    cache.put("a", 1); cache.put("b", 2); cache.get("a"); cache.put("c", 3)   # noqa: E702
    assert cache.get("b") is None and cache.get("a") == 1 and cache.get("c") == 3
    assert cache.stats()["hits"] == 3 and cache.stats()["misses"] == 1


def test_cache_key_depends_on_query_context_and_teachers():
    k = ResponseCache.key
    assert k("What is X?", "ctx", "m", ["a"]) == k("  what is   x? ", "ctx", "m", ["a"])
    assert k("q", "ctx1", "m", ["a"]) != k("q", "ctx2", "m", ["a"])
    assert k("q", "c", "m", ["a"]) != k("q", "c", "m", ["a", "b"])


def _teachers():
    return {"math": FakeTeacher("m"), "general": FakeTeacher("g")}


def test_second_identical_query_is_answered_from_cache_with_zero_tokens():
    cache, d1, d2 = ResponseCache(), FakeDistiller(), FakeDistiller()
    first = run_query("question", FakeRAG(), FakeRouter(["general", "math"]), _teachers(), d1, cache=cache)
    second = run_query("question", FakeRAG(), FakeRouter(["general", "math"]), _teachers(), d2, cache=cache)
    assert first.stats["cache_hit"] is False and first.stats["input_tokens"] > 0
    assert second.stats["cache_hit"] is True and second.stats["input_tokens"] == 0
    assert d2.received is None                       # the distiller was never called
    assert second.final_answer == first.final_answer
    assert second.stats["tool_calls"] == 0 and second.stats["cost_usd"] in (0.0, None)
    assert cache.stats()["tokens_saved"] == first.stats["input_tokens"] + first.stats["output_tokens"]


def test_failed_runs_are_not_cached():
    cache = ResponseCache()
    teachers = {"math": FakeTeacher(error=RuntimeError("boom")), "general": FakeTeacher("g")}
    run_query("q", FakeRAG(), FakeRouter(["general", "math"]), teachers, FakeDistiller(), cache=cache)
    assert len(cache) == 0


def test_cost_tool_calls_and_security_warnings_are_reported():
    class WarnRAG(FakeRAG):
        last_warnings = ["quarantined a chunk from bad.md: override_instructions"]
    result = run_query("q", WarnRAG(), FakeRouter(["general"]), {"general": FakeTeacher("g")}, FakeDistiller())
    assert result.security_warnings == WarnRAG.last_warnings
    assert "cost_usd" in result.stats and result.stats["tool_calls"] == 0
