import os

# ─────────────────────────────────────────────
#  API
# ─────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-haiku-4-5-20251001"   # fast & cheap for multi-teacher calls

# ─────────────────────────────────────────────
#  RAG settings
# ─────────────────────────────────────────────
DOCS_DIR        = "data/documents"
CHUNK_SIZE      = 400          # characters per chunk
CHUNK_OVERLAP   = 80
TOP_K_CHUNKS    = 3            # retrieved chunks per teacher

# ─────────────────────────────────────────────
#  Teacher settings
# ─────────────────────────────────────────────
MAX_TOKENS_PER_TEACHER = 512
MAX_TOKENS_DISTILLER   = 1024

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
LOG_FILE = "outputs/session_log.jsonl"
