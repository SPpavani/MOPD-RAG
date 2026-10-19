"""Shared text helpers: tokenising, light stemming and stopwords.

Stopwords matter for retrieval: without them a question like "Who is the president of the United States?"
matches ANY document containing "is", "the" or "of", and irrelevant text gets injected into the prompt.
"""
import re
from typing import List

STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how
i if in into is it its itself just me more most my no nor not now of off on once only or other our out over own same
she should so some such than that the their them then there these they this those through to too under until up us
very was we were what when where which while who whom why will with would you your yours
tell give show explain describe please need want like use make get let know
doesn don isn wasn aren didn won couldn wouldn shouldn t s d m re ve ll
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    """Very light plural stripping: 'equations' -> 'equation' ('class' stays 'class')."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def raw_tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def terms(text: str) -> List[str]:
    """Content terms: lower-cased, stopwords removed, plurals stripped."""
    return [stem(t) for t in raw_tokens(text) if t not in STOPWORDS]
