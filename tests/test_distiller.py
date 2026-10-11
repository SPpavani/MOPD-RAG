from types import SimpleNamespace

from core.distiller import NO_ANSWERS, StudentDistiller


class FakeClient:
    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text="synthesised")],
            usage=SimpleNamespace(input_tokens=30, output_tokens=12),
        )


def test_no_outputs():
    assert StudentDistiller(client=FakeClient()).distill("q", {}) == NO_ANSWERS


def test_single_teacher_is_passed_through_without_an_api_call():
    client = FakeClient()
    result = StudentDistiller(client=client).distill("q", {"general": "only answer"})
    assert result == "only answer"
    assert client.calls == []


def test_multiple_teachers_are_synthesised_in_one_call():
    client = FakeClient()
    resp = StudentDistiller(client=client).distill_response(
        "What is 2+2?", {"math": "4", "general": "four"}
    )
    assert resp.text == "synthesised"
    assert (resp.input_tokens, resp.output_tokens) == (30, 12)
    assert len(client.calls) == 1
    prompt = client.calls[0]["messages"][0]["content"]
    assert "MATH TEACHER" in prompt and "GENERAL TEACHER" in prompt and "What is 2+2?" in prompt
