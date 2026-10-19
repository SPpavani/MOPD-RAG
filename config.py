import os

# ─────────────────────────────────────────────
#  API
# ─────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# fast & cheap for multi-teacher calls; override with the MOPD_MODEL env var
MODEL = os.environ.get("MOPD_MODEL", "claude-haiku-4-5-20251001")
REQUEST_TIMEOUT = 60      # seconds per API call
MAX_RETRIES = 3           # automatic retries on transient API errors
MAX_WORKERS = 4           # teachers are called in parallel

# ─────────────────────────────────────────────
#  RAG settings
# ─────────────────────────────────────────────
DOCS_DIR        = "data/documents"
CHUNK_SIZE      = 400          # characters per chunk
CHUNK_OVERLAP   = 80
TOP_K_CHUNKS    = 3            # retrieved chunks per query
# "bm25" (default) or "tfidf_legacy" (the original scorer, kept so results stay comparable)
RETRIEVAL_SCORER = os.environ.get("MOPD_SCORER", "bm25")
# a chunk is only used if it contains at least this share of the question's content words
MIN_TERM_COVERAGE = 0.34

# ─────────────────────────────────────────────
#  Teacher settings
# ─────────────────────────────────────────────
MAX_TOKENS_PER_TEACHER = 512
MAX_TOKENS_DISTILLER   = 1024

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
LOG_FILE = "outputs/session_log.jsonl"
