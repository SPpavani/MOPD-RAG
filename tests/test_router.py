from core.router import QueryRouter

router = QueryRouter()


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
