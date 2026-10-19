from benchmark import parse_verdict, summarize


def test_parse_verdict():
    assert parse_verdict('{"winner": "A"}') == "A"
    assert parse_verdict('Sure! {"winner":"TIE"}') == "TIE"
    assert parse_verdict("I think B") is None
    assert parse_verdict(None) is None


def _row(b_lat, b_tok, m_lat, m_tok, winner=None):
    return {
        "baseline": {"latency_s": b_lat, "tokens": b_tok},
        "mopd": {"latency_s": m_lat, "tokens": m_tok},
        "winner": winner,
    }


def test_summarize_averages_and_counts_judge_wins():
    rows = [_row(1.0, 100, 3.0, 400, "mopd"), _row(2.0, 200, 5.0, 600, "baseline"), _row(1.5, 150, 4.0, 500, "mopd")]
    s = summarize(rows)
    assert s["queries"] == 3
    assert s["baseline"] == {"avg_latency_s": 1.5, "avg_tokens": 150}
    assert s["mopd"] == {"avg_latency_s": 4.0, "avg_tokens": 500}
    assert s["judge"] == {"mopd": 2, "baseline": 1, "tie": 0}


def test_summarize_without_judge_has_no_judge_key():
    assert "judge" not in summarize([_row(1.0, 10, 2.0, 20)])


def test_summarize_empty():
    s = summarize([])
    assert s["queries"] == 0 and s["baseline"]["avg_tokens"] == 0


# ---- deterministic answer checks ----
from benchmark import check_answer, load_queries  # noqa: E402


def test_check_answer_all_of_and_any_of():
    spec = {"all_of": [r"7[, ]?006[, ]?652"]}
    assert check_answer("It is 7,006,652.", spec) is True
    assert check_answer("About 7 million.", spec) is False
    assert check_answer("anything", {"query": "no check"}) is None
    spec2 = {"all_of": ["density"], "any_of": ["hydrogen bond", "lattice"]}
    assert check_answer("Ice has lower density because of its lattice", spec2) is True
    assert check_answer("Ice has lower density.", spec2) is False


def test_summarize_reports_accuracy_and_cost_when_present():
    rows = [
        {"baseline": {"latency_s": 1, "tokens": 10, "correct": False, "cost_usd": 0.001},
         "mopd": {"latency_s": 2, "tokens": 40, "correct": True, "cost_usd": 0.004}, "winner": None},
        {"baseline": {"latency_s": 1, "tokens": 10, "correct": True, "cost_usd": 0.001},
         "mopd": {"latency_s": 2, "tokens": 40, "correct": True, "cost_usd": 0.004}, "winner": None},
    ]
    s = summarize(rows)
    assert s["baseline"]["accuracy"] == "1/2" and s["mopd"]["accuracy"] == "2/2"
    assert s["mopd"]["avg_cost_usd"] == 0.004


def test_query_file_loads_and_every_check_is_valid_regex():
    import re
    queries = load_queries("eval/queries.json")
    assert len(queries) >= 15
    for q in queries:
        for pattern in (q.get("all_of") or []) + (q.get("any_of") or []):
            re.compile(pattern)
