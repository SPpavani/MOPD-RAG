from core.router import QueryRouter

router = QueryRouter(mode="keyword_legacy")   # these tests describe the ORIGINAL router
weighted = QueryRouter()


def test_general_is_always_active():
    assert "general" in router.route("calculate the integral of x")
    assert router.route("tell me a joke") == ["general"]


def test_math_keyword():
    assert router.route("calculate the integral") == ["general", "math"]


def test_multi_word_keyword_is_matched_as_phrase():
    # Regression: "how many" could never match when the query was split into words
    assert "math" in router.route("how many apples are in the basket")


def test_plural_forms_match():
    assert "math" in router.route("solve these equations")


def test_code_and_science_together():
    result = router.route("debug my python script that simulates gravity")
    assert result == ["code", "general", "science"]


def test_class_is_not_mangled_by_plural_stripping():
    assert "code" in router.route("explain this class")


def test_custom_keywords():
    custom = QueryRouter({"legal": ["contract", "is it legal"]})
    assert custom.route("review this contract") == ["general", "legal"]
    assert custom.route("is it legal to do this") == ["general", "legal"]


# ---- weighted router (the default) ----

def test_weighted_is_the_default_mode():
    assert QueryRouter().mode == "weighted"


def test_weighted_single_ambiguous_word_does_not_trigger_a_paid_call():
    for q in ["How does social class affect education?", "The prison cell door broke",
              "Her energy policy changed", "The newspaper printed an error", "How does a library card work?"]:
        assert weighted.route(q) == ["general"], q


def test_weighted_two_weak_words_together_do_trigger():
    assert "code" in weighted.route("what does this standard library function return")


def test_weighted_class_in_a_programming_context():
    assert "code" in weighted.route("explain this Python class")


def test_weighted_percentage_and_arithmetic_patterns_route_to_math():
    assert "math" in weighted.route("What is 15% of 240?")
    assert "math" in weighted.route("what is 17 * 23")
    assert "math" in weighted.route("simplify x^2 - 9")


def test_weighted_margin_of_error_is_statistics_not_code():
    result = weighted.route("What is the margin of error in a poll?")
    assert "math" in result and "code" not in result


def test_weighted_domains_can_combine():
    assert weighted.route("write python code to calculate the kinetic energy") == ["code", "general", "math", "science"]


def test_weighted_scores_are_inspectable():
    from core.router import score_domains
    scores = score_domains("debug my python script")
    assert scores["code"] >= 2 and scores["math"] == 0


def test_invalid_mode_raises():
    import pytest
    with pytest.raises(ValueError):
        QueryRouter(mode="nonsense")
