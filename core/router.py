"""
core/router.py
Detects the domain of a query and returns the relevant teacher names.
Inspired by MoE routing: only activate the experts you need.

Fixes over the first version:
  * multi-word keywords (e.g. "how many") are matched as phrases; before, they
    could never match because the query was split into single words
  * simple plural handling ("equations" matches "equation")
  * no printing inside the router; the pipeline logs the decision
"""
import re
from typing import Dict, List, Optional, Set

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "math": [
        "calculate", "solve", "equation", "integral", "derivative", "probability",
        "algebra", "geometry", "statistics", "proof", "formula", "sum", "matrix",
        "how many", "compute", "percentage", "ratio", "theorem",
    ],
    "code": [
        "code", "program", "function", "bug", "error", "python", "javascript",
        "algorithm", "implement", "class", "loop", "debug", "script", "api",
        "database", "sql", "git", "terminal", "bash", "compile", "library",
    ],
    "science": [
        "physics", "chemistry", "biology", "neuron", "molecule", "experiment",
        "evolution", "gravity", "quantum", "atom", "dna", "species", "cell",
        "energy", "force", "reaction", "hypothesis", "scientific", "research",
    ],
}


def _stem(word: str) -> str:
    """Very light plural stripping: 'equations' -> 'equation' ('class' stays 'class')."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: str) -> Set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())}


class QueryRouter:
    def __init__(self, keywords: Optional[Dict[str, List[str]]] = None):
        self.keywords = keywords or DOMAIN_KEYWORDS

    def route(self, query: str) -> List[str]:
        """
        Returns a sorted list of teacher names to activate for this query.
        Always includes 'general'. Adds specialists based on keyword matches.
        """
        q_lower = query.lower()
        tokens = _tokens(q_lower)

        active = {"general"}   # general teacher always participates
        for domain, keywords in self.keywords.items():
            for kw in keywords:
                if " " in kw:
                    matched = kw in q_lower
                else:
                    matched = _stem(kw) in tokens
                if matched:
                    active.add(domain)
                    break
        return sorted(active)
