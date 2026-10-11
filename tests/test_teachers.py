from types import SimpleNamespace

from teachers import build_teachers
from teachers.math_teacher import MathTeacher


class FakeClient:
    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text="hi")],
            usage=SimpleNamespace(input_tokens=5, output_tokens=7),
        )


def test_ask_returns_text_tokens_and_latency():
    client = FakeClient()
    resp = MathTeacher(client=client).ask("2+2?", "some context")
    assert resp.text == "hi"
    assert (resp.input_tokens, resp.output_tokens) == (5, 7)
    assert resp.latency_s >= 0


def test_request_uses_specialist_system_prompt_and_context():
    client = FakeClient()
    MathTeacher(client=client).ask("2+2?", "some context")
    call = client.calls[0]
    assert "mathematics" in call["system"].lower()
    content = call["messages"][0]["content"]
    assert "<context>" in content and "some context" in content and "2+2?" in content


def test_no_context_block_when_context_is_empty():
    client = FakeClient()
    MathTeacher(client=client).ask("2+2?", "")
    assert "<context>" not in client.calls[0]["messages"][0]["content"]


def test_answer_returns_plain_text():
    assert MathTeacher(client=FakeClient()).answer("q") == "hi"


def test_build_teachers_registry_and_shared_client():
    client = FakeClient()
    teachers = build_teachers(client=client)
    assert set(teachers) == {"math", "code", "science", "general"}
    assert all(t._client is client for t in teachers.values())
