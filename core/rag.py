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
    ):
        self.docs_dir = docs_dir or config.DOCS_DIR
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.top_k = top_k or config.TOP_K_CHUNKS
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunks: List[Dict] = []   # [{text, source, tf, length}]
        self._idf: Dict[str, float] = {}
        self._loaded = False

    # ── Loading ──────────────────────────────────────────────

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

    # ── TF-IDF index ─────────────────────────────────────────

    def _index(self) -> None:
        doc_freq: Counter = Counter()
        for chunk in self.chunks:
            tokens = tokenize(chunk["text"])
            chunk["tf"] = Counter(tokens)
            chunk["length"] = len(tokens)
            doc_freq.update(set(tokens))
        n = len(self.chunks)
        self._idf = {t: math.log((n + 1) / (df + 1)) + 1 for t, df in doc_freq.items()}

    # ── Retrieval ────────────────────────────────────────────

    def retrieve_chunks(self, query: str) -> List[Dict]:
        """Top-K relevant chunks as dicts: {text, source, score}. Zero-score chunks are dropped."""
        if not self._loaded:
            self.load_documents()

        q_tokens = tokenize(query)
        scored = []
        for chunk in self.chunks:
            total = chunk["length"] or 1
            score = sum((chunk["tf"][t] / total) * self._idf.get(t, 0) for t in q_tokens)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {"text": c["text"], "source": c["source"], "score": s}
            for s, c in scored[: self.top_k]
        ]

    def retrieve(self, query: str) -> str:
        """Return top-K chunks concatenated as a single context string ('' if none relevant)."""
        parts = [
            f"[Source: {c['source']} | relevance: {c['score']:.3f}]\n{c['text']}"
            for c in self.retrieve_chunks(query)
        ]
        return "\n\n---\n\n".join(parts)
