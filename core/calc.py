"""A safe calculator tool for the math teacher.

Language models are unreliable at multi-digit arithmetic; a tool call is exact. The expression is parsed with
`ast` and only numbers, + - * / // % **, parentheses and a short list of math functions are allowed. There is
no eval(), no attribute access and no names beyond the whitelist. Size limits stop denial-of-service inputs
such as 9**9**9 or factorial(10**7).
"""
import ast
import math
import operator
from typing import Any, Callable, Dict

MAX_EXPRESSION_CHARS = 200
MAX_EXPONENT = 1000
MAX_FACTORIAL = 500
MAX_COMB = 5000
MAX_ABS_RESULT = 1e300

_BINARY: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos,
    "atan": math.atan, "log": math.log, "ln": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp,
    "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial,
    "comb": math.comb, "perm": math.perm, "min": min, "max": max,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


class CalcError(ValueError):
    pass


def _check_magnitude(value: Any) -> None:
    """Every intermediate result is bounded, so exact integers cannot grow without limit."""
    if isinstance(value, (int, float)) and abs(value) > MAX_ABS_RESULT:
        raise CalcError("result out of range")


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise CalcError(f"exponent larger than {MAX_EXPONENT} is not allowed")
            if right > 0 and abs(left) > 1 and right * math.log10(abs(left)) > math.log10(MAX_ABS_RESULT):
                raise CalcError("result out of range")
        result = _BINARY[type(node.op)](left, right)
        _check_magnitude(result)
        return result
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and not node.keywords:
        args = [_eval(a) for a in node.args]
        name = node.func.id
        if name == "factorial" and args and args[0] > MAX_FACTORIAL:
            raise CalcError(f"factorial argument larger than {MAX_FACTORIAL} is not allowed")
        if name in ("comb", "perm") and args and args[0] > MAX_COMB:
            raise CalcError(f"{name} argument larger than {MAX_COMB} is not allowed")
        result = _FUNCTIONS[name](*args)
        _check_magnitude(result)
        return result
    raise CalcError("unsupported syntax: only numbers, arithmetic operators and basic math functions are allowed")


def evaluate(expression: str) -> Any:
    """Return the numeric value of the expression, or raise CalcError."""
    # Commas are NOT stripped: "1,234" could be a thousands separator or two arguments, and guessing wrong would
    # give a silently wrong answer. Such input is rejected and the model retries without separators.
    text = (expression or "").strip().replace("^", "**").replace("\u00d7", "*").replace("\u00f7", "/")
    if not text:
        raise CalcError("empty expression")
    if len(text) > MAX_EXPRESSION_CHARS:
        raise CalcError(f"expression longer than {MAX_EXPRESSION_CHARS} characters")
    try:
        value = _eval(ast.parse(text, mode="eval"))
    except CalcError:
        raise
    except ZeroDivisionError:
        raise CalcError("division by zero") from None
    except (SyntaxError, ValueError, TypeError, OverflowError) as exc:
        raise CalcError(f"cannot evaluate: {exc}") from None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value) or abs(value) > MAX_ABS_RESULT):
        raise CalcError("result out of range")
    return value


def format_number(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def calculate(expression: str) -> str:
    """Tool entry point: always returns a string (the answer or 'Error: ...'), never raises."""
    try:
        return format_number(evaluate(expression))
    except CalcError as exc:
        return f"Error: {exc}"


CALCULATOR_TOOL = {
    "name": "calculator",
    "description": ("Evaluate an arithmetic expression exactly. Supports + - * / // % ** (or ^), parentheses, "
                    "sqrt, sin, cos, tan, log, ln, exp, abs, round, floor, ceil, factorial, comb, perm, min, max, "
                    "and the constants pi and e. Write numbers WITHOUT thousands separators (1234567, not 1,234,567); commas "
                    "only separate function arguments. Use it for EVERY numeric calculation instead of computing in your head."),
    "input_schema": {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "e.g. '1234 * 5678' or 'sqrt(1764)'"}},
        "required": ["expression"],
    },
}
