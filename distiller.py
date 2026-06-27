"""
core/distiller.py
The "student" model that reads all teacher answers and distills the best final response.
Mirrors Nemotron-3-Ultra's MOPD: dense feedback from N teachers → 1 synthesized answer.
"""

import anthropic
from config import MODEL, MAX_TOKENS_DISTILLER, ANTHROPIC_API_KEY


DISTILLER_SYSTEM = """You are a master synthesizer. You receive answers from multiple specialist AI teachers,
each with different expertise. Your job is to:

1. Identify the strongest, most accurate insights from each teacher
2. Resolve any contradictions by using your judgment
3. Combine them into a single, clear, comprehensive answer
4. Cite which teacher(s) contributed each key point using [Teacher: name]
5. Add any important nuance the teachers missed

Be concise but complete. Prefer accuracy over verbosity."""


class StudentDistiller:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def distill(self, query: str, teacher_outputs: dict[str, str]) -> str:
        """
        teacher_outputs: {teacher_name: answer_text}
        Returns the synthesized final answer.
        """
        if not teacher_outputs:
            return "No teacher responses received."

        # Build the synthesis prompt
        teacher_block = ""
        for name, answer in teacher_outputs.items():
            teacher_block += f"\n\n=== {name.upper()} TEACHER ===\n{answer}"

        user_prompt = (
            f"User query: {query}\n\n"
            f"Teacher answers to synthesize:{teacher_block}\n\n"
            f"Now produce the final synthesized answer:"
        )

        print("[Distiller] Synthesizing teacher answers...")
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS_DISTILLER,
            system=DISTILLER_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
