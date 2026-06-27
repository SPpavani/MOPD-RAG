"""
core/rag.py
Simple RAG: load docs → chunk → TF-IDF retrieval (no external vector DB needed).
"""

import os
import math
import re
from collections import Counter
from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_CHUNKS


class RAGEngine:
    def __init__(self):
        self.chunks: list[dict] = []   # [{text, source}]
        self._idf: dict[str, float] = {}
        self._loaded = False

    # ── Loading ──────────────────────────────────────────────

    def load_documents(self):
        """Read all .txt / .md files from DOCS_DIR and build the index."""
        os.makedirs(DOCS_DIR, exist_ok=True)
        raw_docs = []

        for fname in os.listdir(DOCS_DIR):
            if fname.endswith((".txt", ".md")):
                path = os.path.join(DOCS_DIR, fname)
                with open(path, "r", encoding="utf-8") as f:
                    raw_docs.append({"text": f.read(), "source": fname})

        if not raw_docs:
            print(f"[RAG] No documents found in '{DOCS_DIR}'. "
                  "Teachers will answer from general knowledge.")
            self._loaded = True
            return

        for doc in raw_docs:
            self.chunks.extend(self._chunk(doc["text"], doc["source"]))

        self._build_idf()
        self._loaded = True
        print(f"[RAG] Loaded {len(raw_docs)} doc(s) → {len(self.chunks)} chunks.")

    def _chunk(self, text: str, source: str) -> list[dict]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append({"text": text[start:end], "source": source})
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    # ── TF-IDF retrieval ─────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_idf(self):
        N = len(self.chunks)
        df: Counter = Counter()
        for chunk in self.chunks:
            tokens = set(self._tokenize(chunk["text"]))
            df.update(tokens)
        self._idf = {t: math.log((N + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}

    def _score(self, query: str, chunk_text: str) -> float:
        q_tokens = self._tokenize(query)
        c_tokens = self._tokenize(chunk_text)
        tf = Counter(c_tokens)
        total = len(c_tokens) or 1
        return sum((tf[t] / total) * self._idf.get(t, 0) for t in q_tokens)

    def retrieve(self, query: str) -> str:
        """Return top-K chunks concatenated as a single context string."""
        if not self._loaded:
            self.load_documents()
        if not self.chunks:
            return ""

        scored = [(self._score(query, c["text"]), c) for c in self.chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:TOP_K_CHUNKS]

        parts = []
        for score, chunk in top:
            parts.append(f"[Source: {chunk['source']} | relevance: {score:.3f}]\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)
