from teachers.base import BaseTeacher


class GeneralTeacher(BaseTeacher):
    name = "general"
    system_prompt = (
        "You are a clear, balanced generalist. Answer directly and accurately, "
        "and say so when you are unsure instead of guessing."
    )
