"""
teachers/base.py
Shared behaviour for every teacher: build the prompt, call the model, track tokens and latency.
A teacher is a model call with a specialist system prompt.
"""
import time
from dataclasses import dataclass
from typing import Optional

import config


@dataclass
class TeacherResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class BaseTeacher:
    name = "base"
    system_prompt = "You are a helpful assistant."

    def __init__(self, client=None, model: Optional[str] = None, max_tokens: Optional[int] = None):
        self._client = client
        self.model = model or config.MODEL
        self.max_tokens = max_tokens or config.MAX_TOKENS_PER_TEACHER

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy: keeps unit tests light

            self._client = anthropic.Anthropic(
                api_key=config.ANTHROPIC_API_KEY or None,
                timeout=config.REQUEST_TIMEOUT,
                max_retries=config.MAX_RETRIES,
            )
        return self._client

    def build_user_prompt(self, query: str, context: str = "") -> str:
        if context:
            return (
                "Use the context below if it is relevant. If it is not relevant, "
                "answer from your own knowledge and say so.\n\n"
                f"<context>\n{context}\n</context>\n\n"
                f"Question: {query}"
            )
        return f"Question: {query}"

    def ask(self, query: str, context: str = "") -> TeacherResponse:
        start = time.perf_counter()
        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": self.build_user_prompt(query, context)}],
        )
        usage = getattr(response, "usage", None)
        return TeacherResponse(
            text=response.content[0].text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            latency_s=round(time.perf_counter() - start, 3),
        )

    def answer(self, query: str, context: str = "") -> str:
        return self.ask(query, context).text
