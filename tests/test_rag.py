import pytest

from core.rag import RAGEngine


def _write(tmp_path, name, text):
    (tmp_path / name).write_text(text, encoding="utf-8")


def _engine(tmp_path, **kwargs):
    params = dict(docs_dir=str(tmp_path), chunk_size=100, chunk_overlap=20, top_k=1)
    params.update(kwargs)
    return RAGEngine(**params)


def test_retrieves_the_relevant_document(tmp_path):
    _write(tmp_path, "a.md", "The capital of France is Paris. Paris is known for the Eiffel Tower.")
    _write(tmp_path, "b.md", "Photosynthesis converts light into chemical energy in plants.")
    rag = _engine(tmp_path)
    rag.load_documents()
    context = rag.retrieve("Eiffel Tower in Paris")
    assert "a.md" in context
    assert "b.md" not in context


def test_irrelevant_query_returns_empty_context(tmp_path):
    _write(tmp_path, "a.md", "The capital of France is Paris.")
    rag = _engine(tmp_path)
    rag.load_documents()
    assert rag.retrieve("zzzz qqqq") == ""


def test_empty_folder_returns_empty_context(tmp_path):
    rag = _engine(tmp_path)
    rag.load_documents()
    assert rag.retrieve("anything") == ""


def test_loading_twice_does_not_duplicate_chunks(tmp_path):
    _write(tmp_path, "a.md", "alpha beta gamma delta " * 20)
    rag = _engine(tmp_path)
    rag.load_documents()
    first = len(rag.chunks)
    rag.load_documents()
    assert len(rag.chunks) == first


def test_ignores_other_file_types(tmp_path):
    _write(tmp_path, "notes.txt", "quantum entanglement experiment")
    _write(tmp_path, "image.png", "quantum entanglement experiment")
    rag = _engine(tmp_path)
    rag.load_documents()
    assert {c["source"] for c in rag.chunks} == {"notes.txt"}


def test_chunks_overlap(tmp_path):
    _write(tmp_path, "a.md", "x" * 250)
    rag = _engine(tmp_path, chunk_size=100, chunk_overlap=20)
    rag.load_documents()
    # step is 80: chunks start at 0, 80, 160, 240
    assert len(rag.chunks) == 4


def test_invalid_overlap_raises(tmp_path):
    with pytest.raises(ValueError):
        RAGEngine(docs_dir=str(tmp_path), chunk_size=50, chunk_overlap=50)
