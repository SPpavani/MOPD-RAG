"""Quality gates: if retrieval or routing regress below the measured level, CI fails."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

import run_retrieval_eval as retrieval  # noqa: E402
import run_routing_eval as routing  # noqa: E402


def test_unrelated_questions_receive_no_context():
    result = retrieval.evaluate("bm25")
    assert result["off_topic"]["false_context"] == 0


def test_retrieval_still_finds_the_right_document():
    result = retrieval.evaluate("bm25")
    assert result["exact"]["hit@1"] >= 17 and result["paraphrase"]["hit@3"] >= 6


def test_legacy_scorer_still_has_the_bug_this_project_measured():
    # documents the baseline: the original scorer injected context into every off-topic question
    assert retrieval.evaluate("tfidf_legacy")["off_topic"]["false_context"] == 12


def test_weighted_router_beats_legacy_on_the_held_out_test_split():
    weighted = routing.evaluate("test", "weighted")
    legacy = routing.evaluate("test", "keyword_legacy")
    assert weighted["exact"] >= 43 and weighted["precision"] >= 0.95
    assert weighted["f1"] > legacy["f1"] + 0.10


def test_router_does_not_buy_accuracy_by_activating_everything():
    assert routing.evaluate("test", "weighted")["avg_specialists"] < 1.2
