"""
core/router.py
Detects the domain of a query and returns the relevant teacher names.
Inspired by MoE routing — only activate the experts you need.
"""

import re


DOMAIN_KEYWORDS = {
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


class QueryRouter:
    def route(self, query: str) -> list[str]:
        """
        Returns a list of teacher names to activate for this query.
        Always includes 'general'. Adds specialists based on keyword matches.
        """
        q_lower = query.lower()
        words = re.findall(r"[a-z0-9]+", q_lower)
        word_set = set(words)

        active = set()
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if word_set & set(keywords):
                active.add(domain)

        active.add("general")   # general teacher always participates
        teachers = sorted(active)
        print(f"[Router] Activated teachers: {teachers}")
        return teachers
