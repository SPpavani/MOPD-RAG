"""
core/rag.py
Simple RAG: load docs -> chunk -> TF-IDF retrieval (no external vector DB needed).

Improvements over the first version:
  * token counts are computed once at index time, not on every query
  * chunks with zero relevance are dropped instead of being sent as "context"
  * loading twice never duplicates chunks; file order is deterministic
"""
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import config
from core.security import neutralize_context, scan_injection
from core.text import terms

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class RAGEngine:
    def __init__(
        self,
        docs_dir: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        top_k: Optional[int] = None,
        scorer: Optional[str] = None,
        min_term_coverage: Optional[float] = None,
    ):
        self.docs_dir = docs_dir or config.DOCS_DIR
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.top_k = top_k or config.TOP_K_CHUNKS
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.scorer = scorer or config.RETRIEVAL_SCORER
        if self.scorer not in ("bm25", "tfidf_legacy"):
            raise ValueError("scorer must be 'bm25' or 'tfidf_legacy'")
        self.min_term_coverage = config.MIN_TERM_COVERAGE if min_term_coverage is None else min_term_coverage
        self.k1, self.b = 1.5, 0.75   # standard BM25 constants
        self._bm25_idf: Dict[str, float] = {}
        self._avg_len = 1.0
        self.last_warnings: List[str] = []   # security findings from the most recent retrieve()

        self.chunks: List[Dict] = []   # [{text, source, tf, length}]
        self._idf: Dict[str, float] = {}
        self._loaded = False

    # ── Loading ────────────────────────────────────────────────────

    def load_documents(self) -> None:
        """Read all .txt / .md files from the docs folder and build the index."""
        docs_path = Path(self.docs_dir)
        docs_path.mkdir(parents=True, exist_ok=True)
        files = sorted(
            p for p in docs_path.iterdir() if p.suffix.lower() in (".txt", ".md")
        )

        self.chunks = []
        for path in files:
            text = path.read_text(encoding="utf-8")
            self.chunks.extend(self._chunk(text, path.name))

        self._index()
        self._loaded = True

        if not files:
            logger.warning(
                "No documents found in '%s'. Teachers will answer from general knowledge.",
                self.docs_dir,
            )
        else:
            logger.info("Loaded %d doc(s) -> %d chunks.", len(files), len(self.chunks))

    def _chunk(self, text: str, source: str) -> List[Dict]:
        step = self.chunk_size - self.chunk_overlap
        chunks = []
        for start in range(0, len(text), step):
            piece = text[start : start + self.chunk_size]
            if piece.strip():
                chunks.append({"text": piece, "source": source})
        return chunks

    # ── TF-IDF index ───────────────────────────────────────────────

    def _index(self) -> None:
        doc_freq: Counter = Counter()
        for chunk in self.chunks:
            tokens = tokenize(chunk["text"])
            chunk["tf"] = Counter(tokens)
            chunk["length"] = len(tokens)
            doc_freq.update(set(tokens))
        n = len(self.chunks)
        self._idf = {t: math.log((n + 1) / (df + 1)) + 1 for t, df in doc_freq.items()}

        # BM25 works on content terms (stopwords removed, plurals stripped)
        term_df: Counter = Counter()
        total_len = 0
        for chunk in self.chunks:
            chunk_terms = terms(chunk["text"])
            chunk["tf_terms"] = Counter(chunk_terms)
            chunk["terms_len"] = len(chunk_terms)
            total_len += len(chunk_terms)
            term_df.update(set(chunk_terms))
        self._avg_len = (total_len / n) if n else 1.0
        self._bm25_idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in term_df.items()}

    # ── Retrieval ────────────────────────────────────────────────

    def retrieve_chunks(self, query: str) -> List[Dict]:
        """Top-K relevant chunks as dicts: {text, source, score}. Irrelevant chunks are dropped."""
        if not self._loaded:
            self.load_documents()
        scored = self._score_legacy(query) if self.scorer == "tfidf_legacy" else self._score_bm25(query)
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {"text": c["text"], "source": c["source"], "score": s}
            for s, c in scored[: self.top_k]
        ]

    def _score_legacy(self, query: str):
        """The original scorer: raw TF-IDF over every token, stopwords included."""
        q_tokens = tokenize(query)
        scored = []
        for chunk in self.chunks:
            total = chunk["length"] or 1
            score = sum((chunk["tf"][t] / total) * self._idf.get(t, 0) for t in q_tokens)
            if score > 0:
                scored.append((score, chunk))
        return scored

    def _score_bm25(self, query: str):
        """BM25 over content terms, with a coverage gate.

        A chunk must contain at least `min_term_coverage` of the question's distinct content terms, so a single
        incidental word ("states", "capital") cannot drag an unrelated chunk into the prompt.
        """
        q_terms = list(dict.fromkeys(terms(query)))
        if not q_terms:
            return []
        scored = []
        for chunk in self.chunks:
            tf = chunk["tf_terms"]
            matched = [t for t in q_terms if tf.get(t)]
            if not matched or len(matched) / len(q_terms) < self.min_term_coverage:
                continue
            norm = 1 - self.b + self.b * (chunk["terms_len"] / self._avg_len)
            score = sum(
                self._bm25_idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * norm) for t in matched
            )
            scored.append((score, chunk))
        return scored

    def retrieve(self, query: str) -> str:
        """Return top-K chunks concatenated as a single context string ('' if none relevant).

        Chunks that look like prompt injection are quarantined: they are NOT returned, and a warning is
        recorded in `last_warnings` so the caller can surface it.
        """
        self.last_warnings = []
        parts = []
        for c in self.retrieve_chunks(query):
            signals = scan_injection(c["text"])
            if signals:
                self.last_warnings.append(f"quarantined a chunk from {c['source']}: {', '.join(signals)}")
                continue
            parts.append(f"[Source: {c['source']} | relevance: {c['score']:.3f}]\n{neutralize_context(c['text'])}")
        return "\n\n---\n\n".join(parts)
