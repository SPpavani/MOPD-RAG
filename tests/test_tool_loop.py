"""The teacher's tool-use loop, tested with a scripted client shaped like the Anthropic API response."""
from types import SimpleNamespace as NS

from teachers.base import BaseTeacher
from teachers.math_teacher import MathTeacher


def _text(t):
    return NS(type="text", text=t)


def _tool(id_, expression):
    return NS(type="tool_use", id=id_, name="calculator", input={"expression": expression})


def _resp(blocks, stop, i=10, o=5):
    return NS(content=blocks, stop_reason=stop, usage=NS(input_tokens=i, output_tokens=o))


class ScriptedClient:
    def __init__(self, *responses):
        self.responses, self.calls, self.messages = list(responses), [], self

    def create(self, **kw):
        import copy
        self.calls.append(copy.copy(kw) | {"messages": list(kw["messages"])})
        return self.responses.pop(0)


def test_calculator_result_is_fed_back_and_tokens_are_summed():
    client = ScriptedClient(
        _resp([_text("I'll compute it."), _tool("t1", "1234 * 5678")], "tool_use", 100, 20),
        _resp([_text("The answer is 7006652.")], "end_turn", 150, 10),
    )
    r = MathTeacher(client=client).ask("What is 1234 * 5678?")
    assert r.text == "The answer is 7006652."
    assert r.tool_calls == 1
    assert (r.input_tokens, r.output_tokens) == (250, 30)
    second_call_messages = client.calls[1]["messages"]
    assert second_call_messages[-1]["role"] == "user"
    result_block = second_call_messages[-1]["content"][0]
    assert result_block == {"type": "tool_result", "tool_use_id": "t1", "content": "7006652"}


def test_tools_are_only_sent_to_teachers_that_have_them():
    client = ScriptedClient(_resp([_text("ok")], "end_turn"))
    MathTeacher(client=client).ask("q")
    assert client.calls[0]["tools"][0]["name"] == "calculator"

    client2 = ScriptedClient(_resp([_text("ok")], "end_turn"))
    BaseTeacher(client=client2).ask("q")
    assert "tools" not in client2.calls[0]


def test_a_malicious_tool_input_is_answered_with_an_error_not_executed():
    client = ScriptedClient(
        _resp([_tool("t1", "__import__('os').system('echo pwned')")], "tool_use"),
        _resp([_text("I could not compute that.")], "end_turn"),
    )
    r = MathTeacher(client=client).ask("q")
    assert client.calls[1]["messages"][-1]["content"][0]["content"].startswith("Error:")
    assert r.text == "I could not compute that."


def test_unknown_tool_name_does_not_crash():
    bad = NS(type="tool_use", id="t9", name="shell", input={"cmd": "ls"})
    client = ScriptedClient(_resp([bad], "tool_use"), _resp([_text("done")], "end_turn"))
    assert MathTeacher(client=client).ask("q").text == "done"
    assert "unknown tool" in client.calls[1]["messages"][-1]["content"][0]["content"]


def test_runaway_tool_loop_is_bounded():
    endless = [_resp([_tool(f"t{i}", "1+1")], "tool_use") for i in range(20)]
    client = ScriptedClient(*endless)
    r = MathTeacher(client=client).ask("q")
    assert len(client.calls) == 5                  # 1 + MAX_TOOL_ROUNDS
    assert r.text == "(the model produced no final answer)"
