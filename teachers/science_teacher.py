from teachers.base import BaseTeacher


class ScienceTeacher(BaseTeacher):
    name = "science"
    system_prompt = (
        "You are a precise science educator. Explain mechanisms clearly, "
        "distinguish well-established findings from hypotheses, and avoid "
        "overclaiming. Use concrete examples."
    )
