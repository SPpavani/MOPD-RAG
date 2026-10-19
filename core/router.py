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

from core.text import stem as _text_stem

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
    """mode="weighted" (default) uses the weighted lexicon below; mode="keyword_legacy" is the original
    any-keyword-matches router. Passing your own keyword dict always uses the legacy matching (every word strong)."""

    def __init__(self, keywords: Optional[Dict[str, List[str]]] = None, mode: str = "weighted"):
        if mode not in ("weighted", "keyword_legacy"):
            raise ValueError("mode must be 'weighted' or 'keyword_legacy'")
        self.custom = keywords is not None
        self.mode = "keyword_legacy" if keywords else mode
        self.keywords = keywords or DOMAIN_KEYWORDS

    def route(self, query: str) -> List[str]:
        """
        Returns a sorted list of teacher names to activate for this query.
        Always includes 'general'. Adds specialists based on the router mode.
        """
        if self.mode == "weighted":
            return route_weighted(query)
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


# ──────────────────────────────────────────────────────────────────────────────────────────────
#  Weighted router
#
#  The legacy router activates a specialist when ANY listed word appears, so ordinary sentences
#  trigger paid model calls: "social class" -> code, "prison cell" -> science, "margin of error" -> code.
#  The weighted router fixes that without a model call:
#    * STRONG words/phrases/patterns (weight 1.0) identify a domain on their own
#    * WEAK words (weight 0.5) are ambiguous in plain English ("class", "cell", "error", "energy",
#      "function", "library", "research") and only count when a second cue supports them
#    * a specialist activates at a total weight of at least 1.0
# ──────────────────────────────────────────────────────────────────────────────────────────────

THRESHOLD = 1.0

STRONG: Dict[str, Set[str]] = {
    "math": set("""calculate solve equation integral derivative probability algebra geometry statistics proof prove
        theorem matrix compute percentage percent ratio average mean median variance deviation quadratic polynomial
        simplify derive logarithm trigonometry fraction perimeter hexagon triangle circle series sequence permutation
        combination combinatorics arithmetic calculus sqrt factorial binomial expected-value""".split()),
    "code": set("""code python javascript java typescript golang html css sql json yaml regex api bash git github terminal
        compile debug bug algorithm implement loop recursion closure thread mutex async endpoint flask django react node
        docker pytest chmod grep segfault exception variable programming database postgresql mysql sqlite keyerror
        complexity quicksort sorting stack-trace refactor repository""".split()),
    "science": set("""physics chemistry biology neuron molecule experiment evolution gravity quantum atom dna rna species
        photosynthesis crispr vaccine immune enzyme protein gene genome chemical reaction planet orbit galaxy cosmology
        relativity newton electron proton magnetic wavelength thermodynamics kinetic chlorophyll ecosystem seasons
        hypothesis scientific biologist physicist chemist""".split()),
}

WEAK: Dict[str, Set[str]] = {
    "math": set("sum formula rate speed convert total square number estimate random value pi".split()),
    "code": set("""class error function library program script query index join parse swap integer string branch merge
        commit test standard rust""".split()),
    "science": set("energy force cell research light heat water temperature particle wave mass earth sky leaf ice float".split()),
}

STRONG_PHRASES: Dict[str, List[str]] = {
    "math": ["margin of error", "how many ways", "how many different", "how many possible", "standard deviation",
             "expected value", "square root", "how long to", "average speed", "monte carlo", "gradient descent"],
    "code": ["unit test", "null pointer", "segmentation fault", "stack trace", "time complexity",
             "command line", "pull request"],
    "science": ["black hole", "immune system", "newton's", "periodic table", "natural selection"],
}

STRONG_PATTERNS: Dict[str, List[re.Pattern]] = {
    "math": [
        re.compile(r"\d\s*%"),                         # 15%
        re.compile(r"\d\s*[-+*/^\u00d7\u00f7]\s*\d"),    # 3 + 4, 2^8, 10/4
        re.compile(r"[a-z0-9)]\s*\^\s*[0-9(]"),           # x^2, e^(2x), mc^2
        re.compile(r"\b\d[\d,.]*\s*[a-z]+\s+per\s+[a-z]+"),  # 12 litres per minute
        re.compile(r"\bf\(x\)|\bx\s*[-+=]\s*\d|\d\s*x\b"),  # f(x), x = 3, 3x
        re.compile(r"\bkm/h\b|\bm/s\b"),
    ],
    "code": [
        re.compile(r"`[^`]+`"),                          # inline code
        re.compile(r"\bdef\s+\w+\(|\bimport\s+\w+|\w+\(\)"),
        re.compile(r"\bc\+\+|\bc#|\.(py|js|ts|java|cpp|rs|go|sh|sql)\b"),
    ],
    "science": [],
}


class _Lexicon:
    """Pre-compiled matching for one weighted vocabulary."""

    def __init__(self):
        self.strong = {d: {_text_stem(w) for w in ws} for d, ws in STRONG.items()}
        self.weak = {d: {_text_stem(w) for w in ws} for d, ws in WEAK.items()}


_LEX = _Lexicon()


def score_domains(query: str) -> Dict[str, float]:
    """Total evidence per domain for a query (exposed so routing decisions can be logged and tested)."""
    q = query.lower()
    words = set(re.findall(r"[a-z0-9]+", q))
    stems = {_text_stem(w) for w in words}
    out: Dict[str, float] = {}
    for domain in STRONG:
        total = 0.0
        total += sum(1.0 for w in _LEX.strong[domain] if w in stems)
        total += sum(0.5 for w in _LEX.weak[domain] if w in stems)
        total += sum(1.0 for ph in STRONG_PHRASES.get(domain, []) if ph in q)
        total += sum(1.0 for pat in STRONG_PATTERNS.get(domain, []) if pat.search(query.lower()))
        out[domain] = total
    return out


def route_weighted(query: str, threshold: float = THRESHOLD) -> List[str]:
    scores = score_domains(query)
    active = {"general"} | {d for d, v in scores.items() if v >= threshold}
    return sorted(active)
