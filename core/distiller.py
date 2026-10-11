"""
core/distiller.py
The "student" model that reads all teacher answers and distils one final response.
Inspired by Nemotron-3-Ultra's MOPD: feedback from N teachers -> 1 synthesised answer.

If only one teacher answered (e.g. a general question), its answer is returned
as-is: an extra synthesis call would add cost and latency but no value.
"""
from typing import Dict, Optional

import config
from teachers.base import TeacherResponse

DISTILLER_SYSTEM = """You are a master synthesizer. You receive answers from multiple specialist AI teachers,
each with different expertise. Your job is to:

1. Identify the strongest, most accurate insights from each teacher
2. Resolve any contradictions by using your judgment
3. Combine them into a single, clear, comprehensive answer
4. Cite which teacher(s) contributed each key point using [Teacher: name]
5. Add any important nuance the teachers missed

Be concise but complete. Prefer accuracy over verbosity."""

NO_ANSWERS = "No teacher responses received."


class StudentDistiller:
    def __init__(self, client=None, model: Optional[str] = None, max_tokens: Optional[int] = None):
        self._client = client
        self.model = model or config.MODEL
        self.max_tokens = max_tokens or config.MAX_TOKENS_DISTILLER

    def get_client(self):
        if self._client is None:
            import anthropic  # lazy: keeps unit tests light

            self._client = anthropic.Anthropic(
                api_key=config.ANTHROPIC_API_KEY or None,
                timeout=config.REQUEST_TIMEOUT,
                max_retries=config.MAX_RETRIES,
            )
        return self._client

    @staticmethod
    def build_prompt(query: str, teacher_outputs: Dict[str, str]) -> str:
        teacher_block = ""
        for name, answer in teacher_outputs.items():
            teacher_block += f"\n\n=== {name.upper()} TEACHER ===\n{answer}"
        return (
            f"User query: {query}\n\n"
            f"Teacher answers to synthesize:{teacher_block}\n\n"
            f"Now produce the final synthesized answer:"
        )

    def distill_response(self, query: str, teacher_outputs: Dict[str, str]) -> TeacherResponse:
        """Like distill(), but also returns token usage and latency."""
        import time

        if not teacher_outputs:
            return TeacherResponse(text=NO_ANSWERS)
        if len(teacher_outputs) == 1:
            return TeacherResponse(text=next(iter(teacher_outputs.values())))

        start = time.perf_counter()
        response = self.get_client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=DISTILLER_SYSTEM,
            messages=[{"role": "user", "content": self.build_prompt(query, teacher_outputs)}],
        )
        usage = getattr(response, "usage", None)
        return TeacherResponse(
            text=response.content[0].text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            latency_s=round(time.perf_counter() - start, 3),
        )

    def distill(self, query: str, teacher_outputs: Dict[str, str]) -> str:
        return self.distill_response(query, teacher_outputs).text
