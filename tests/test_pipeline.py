from core.pipeline import run_query
from teachers.base import TeacherResponse


class FakeRouter:
    def __init__(self, names):
        self.names = names

    def route(self, query):
        return self.names


class FakeRAG:
    def __init__(self, context="ctx"):
        self.context = context
        self.calls = 0

    def retrieve(self, query):
        self.calls += 1
        return self.context


class FakeTeacher:
    def __init__(self, text=None, error=None):
        self.text, self.error = text, error

    def ask(self, query, context=""):
        if self.error:
            raise self.error
        return TeacherResponse(text=self.text, input_tokens=10, output_tokens=5, latency_s=0.1)


class FakeDistiller:
    def __init__(self, error=None):
        self.error = error
        self.received = None

    def distill_response(self, query, outputs):
        if self.error:
            raise self.error
        self.received = dict(outputs)
        return TeacherResponse(text="FINAL", input_tokens=20, output_tokens=8)


def _run(names, teachers, distiller=None, rag=None):
    distiller = distiller or FakeDistiller()
    rag = rag or FakeRAG()
    result = run_query("question", rag, FakeRouter(names), teachers, distiller)
    return result, distiller, rag


def test_happy_path_aggregates_outputs_and_tokens():
    teachers = {"math": FakeTeacher("m"), "general": FakeTeacher("g")}
    result, distiller, _ = _run(["general", "math"], teachers)
    assert result.final_answer == "FINAL"
    assert distiller.received == {"general": "g", "math": "m"}
    assert result.errors == {}
    assert result.stats["input_tokens"] == 10 + 10 + 20
    assert result.stats["output_tokens"] == 5 + 5 + 8
    assert result.context_used is True


def test_retrieval_happens_once_not_once_per_teacher():
    teachers = {"math": FakeTeacher("m"), "code": FakeTeacher("c"), "general": FakeTeacher("g")}
    _, _, rag = _run(["code", "general", "math"], teachers)
    assert rag.calls == 1


def test_failing_teacher_is_recorded_and_distiller_gets_the_rest():
    teachers = {"math": FakeTeacher(error=RuntimeError("api down")), "general": FakeTeacher("g")}
    result, distiller, _ = _run(["general", "math"], teachers)
    assert "math" in result.errors and "api down" in result.errors["math"]
    assert distiller.received == {"general": "g"}
    assert result.final_answer == "FINAL"


def test_all_teachers_failing_skips_the_distiller():
    teachers = {"general": FakeTeacher(error=RuntimeError("boom"))}
    result, distiller, _ = _run(["general"], teachers)
    assert result.final_answer.startswith("All teachers failed")
    assert distiller.received is None


def test_distiller_failure_falls_back_to_general_answer():
    teachers = {"math": FakeTeacher("m"), "general": FakeTeacher("g")}
    result, _, _ = _run(["general", "math"], teachers, distiller=FakeDistiller(error=RuntimeError("x")))
    assert result.final_answer == "g"
    assert "distiller" in result.errors


def test_unknown_teacher_names_from_router_are_ignored():
    teachers = {"general": FakeTeacher("g")}
    result, _, _ = _run(["general", "legal"], teachers)
    assert result.teachers_activated == ["general"]


def test_no_context_is_reported():
    teachers = {"general": FakeTeacher("g")}
    result, _, _ = _run(["general"], teachers, rag=FakeRAG(context=""))
    assert result.context_used is False
