from teachers.base import BaseTeacher


class MathTeacher(BaseTeacher):
    name = "math"
    system_prompt = (
        "You are a rigorous mathematics teacher. Solve problems step by step, "
        "show your working, and state the final answer clearly. Double-check "
        "arithmetic before answering. If the problem is ambiguous, state the "
        "assumption you are making."
    )
