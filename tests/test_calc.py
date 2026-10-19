import pytest

from core.calc import CalcError, calculate, evaluate


@pytest.mark.parametrize("expr,expected", [
    ("1234 * 5678", "7006652"), ("2^20", "1048576"), ("sqrt(1764)", "42"), ("15/100*240", "36"),
    ("factorial(10)", "3628800"), ("10 // 3", "3"), ("max(1,200)", "200"), ("comb(5,2)", "10"), ("-(2+3)*4", "-20"),
    ("comb(5, 2)", "10"), ("max(3, 9, 4)", "9"), ("2 \u00d7 3", "6"),
])
def test_arithmetic_is_exact(expr, expected):
    assert calculate(expr) == expected


def test_thousands_separators_are_rejected_not_guessed():
    # "1,234" is ambiguous (one number or two arguments); a silent wrong answer is worse than an error
    assert calculate("1,234 + 1").startswith("Error:")


def test_huge_exact_integers_cannot_crash_the_tool():
    assert calculate("(10**300)**1000").startswith("Error:")
    assert calculate("factorial(500)*factorial(500)").startswith("Error:")


def test_floats_are_trimmed():
    assert calculate("1/3") == "0.333333333333"
    assert calculate("0.1+0.2") == "0.3"


@pytest.mark.parametrize("attack", [
    "__import__('os').system('echo hi')", "open('/etc/passwd')", "().__class__", "[x for x in range(9)]",
    "lambda: 1", "exec('1')", "a + 1", "'abc' * 3", "True + 1",
])
def test_code_injection_is_rejected(attack):
    assert calculate(attack).startswith("Error:")


@pytest.mark.parametrize("bomb", ["9**9**9", "2**1001", "factorial(10**7)", "comb(10**9, 5)", "10**308 * 10"])
def test_denial_of_service_inputs_are_rejected(bomb):
    assert calculate(bomb).startswith("Error:")


def test_division_by_zero_and_empty_and_long_input():
    assert "division by zero" in calculate("1/0")
    assert calculate("").startswith("Error:")
    assert calculate("1+" * 150 + "1").startswith("Error:")


def test_evaluate_raises_calc_error():
    with pytest.raises(CalcError):
        evaluate("(1+2")
