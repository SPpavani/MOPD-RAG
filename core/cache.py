"""A small LRU cache for finished pipeline answers.

Identical questions asked over identical context get the identical stored answer: no model calls, no tokens.
The key covers the question, the retrieved context, the model and which teachers ran, so changing any of them
is a miss. Only clean results are stored (no teacher or distiller errors). Note the stored answer is
whatever the model said the first time; use `cache=None` when fresh sampling matters.
"""
import hashlib
from collections import OrderedDict


class ResponseCache:
    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self._data: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0

    @staticmethod
    def key(query: str, context: str, model: str, teachers) -> str:
        normalised = " ".join(query.lower().split())
        raw = "\x1f".join([normalised, context, model, ",".join(sorted(teachers))])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str):
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return None

    def put(self, key: str, value) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(self.hits / total, 3) if total else 0.0,
                "tokens_saved": self.tokens_saved}
