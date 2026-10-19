from core.text import STOPWORDS, stem, terms


def test_stopwords_are_removed_and_plurals_stripped():
    assert terms("What are the Equations of motion?") == ["equation", "motion"]


def test_class_and_short_words_are_not_mangled():
    assert stem("class") == "class" and stem("gas") == "gas"


def test_contraction_fragments_are_not_content_terms():
    assert terms("Why doesn't adding threads help?") == ["adding", "thread", "help"]


def test_question_words_are_stopwords():
    assert {"what", "who", "how", "why", "the", "is"} <= STOPWORDS
