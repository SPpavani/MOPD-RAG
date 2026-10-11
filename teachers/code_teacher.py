from teachers.base import BaseTeacher


class CodeTeacher(BaseTeacher):
    name = "code"
    system_prompt = (
        "You are a senior software engineer. Give correct, idiomatic code in "
        "fenced code blocks, explain it briefly, and mention edge cases and "
        "time complexity where relevant. Prefer simple solutions over clever ones."
    )
