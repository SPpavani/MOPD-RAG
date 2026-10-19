from core.calc import CALCULATOR_TOOL, calculate
from teachers.base import BaseTeacher


class MathTeacher(BaseTeacher):
    name = "math"
    system_prompt = (
        "You are a rigorous mathematics teacher. Solve problems step by step, "
        "show your working, and state the final answer clearly. Use the calculator tool for every "
        "arithmetic step instead of computing in your head, and report the tool's result exactly. "
        "If the problem is ambiguous, state the assumption you are making."
    )
    tools = [CALCULATOR_TOOL]
    tool_handlers = {"calculator": lambda expression="": calculate(expression)}
