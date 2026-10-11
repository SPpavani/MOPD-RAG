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
