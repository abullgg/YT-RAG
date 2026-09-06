"""
Application settings loaded from environment variables or .env file.
All tuneable values live here — override via .env before launching the server.

Qwen3 4B context window: 32,768 tokens (~131,072 chars at 4 chars/token).
Key tuning changes vs the previous Gemma 3 4B setup (8K window):
  - MAX_CONTEXT_CHARS raised from 3,000 → 12,000  (3K tokens, well within budget)
  - CHUNK_SIZE_PRESETS raised proportionally — larger chunks = less fragmentation
  - RERANKER_TOP_N raised from 10 → 15 — more candidates to fill the larger budget
  - RetrievalConfig.top_k_candidates raised from 5 → 8
  - think=False is passed directly to the Ollama API (see src/generation/llm.py)
"""

from typing import Dict, Literal
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Chunk-size presets (characters)
# ---------------------------------------------------------------------------

# Qwen3 4B has a 32K token window so we can afford larger chunks without
# overflowing the context budget. These presets feed into the ingestion
# pipeline when a doc_type is supplied at upload time.
# After any change here, delete ./data/ and re-upload documents to rebuild
# the FAISS index and BM25 index with the new chunk sizes.
CHUNK_SIZE_PRESETS: Dict[str, int] = {
    "technical_spec": 1500,  # Dense specs / manuals — slightly larger than before
                              # (was 1200 for Gemma 8K window)
    "narrative": 2500,        # Flowing prose — larger = sentences stay together
                              # (was 1800 for Gemma 8K window)
    "data_heavy": 800,        # Table/list-heavy — keep small; tables are atomic
    "default": 2000,          # Balanced default (was 1500 for Gemma 8K window)
}

DocType = Literal["technical_spec", "narrative", "data_heavy", "default"]


# ---------------------------------------------------------------------------
# Retrieval configuration model
# ---------------------------------------------------------------------------

class RetrievalConfig(BaseModel):
    """
    Controls the retrieval funnel for each /ask request.

    Qwen3 4B specifics
    ------------------
    Context window: 32,768 tokens (~131K chars). We reserve a generous
    12,000 chars (~3,000 tokens) for retrieved context, leaving ~29K
    tokens free for the system prompt (~500 tokens) and generation.
    This is a 4× increase over the Gemma 3 4B budget (3,000 chars / ~750
    tokens against its 8K window), allowing significantly more evidence
    to be passed to the model without any risk of truncation.

    Attributes:
        max_tokens_for_context: Soft character ceiling for combined context
            fed to the LLM. Trimming happens from the lowest-confidence end.
        top_k_candidates: Final number of chunks passed to the LLM after
            reranking. Raised from 5 → 8 to use the larger context budget.
        reranker_top_n: Wider candidate pool fetched from hybrid search before
            the cross-encoder sees them. Raised from 10 → 15 to give the
            cross-encoder more candidates to pick the best 8 from.
    """

    max_tokens_for_context: int = 12000  # 4× Gemma budget; safe for Qwen3 32K window
    top_k_candidates: int = 8            # was 5 — more chunks fit in the larger budget
    reranker_top_n: int = 15             # was 10 — broader candidate pool for reranker


# ---------------------------------------------------------------------------
# Main settings class
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Application-wide settings, loaded from environment / .env file."""

    PROJECT_NAME: str = "RAG Document Assistant"

    # ── LLM ─────────────────────────────────────────────────────────────── #
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "qwen3:4b"

    # ── Embeddings ───────────────────────────────────────────────────────── #
    # Available BGE variants (EMBEDDING_DIMENSION must match the chosen model):
    #   BAAI/bge-large-en-v1.5 → 1024-dim  (best quality, slower)
    #   BAAI/bge-base-en-v1.5  → 768-dim   (recommended)
    #   BAAI/bge-small-en-v1.5 → 384-dim   (fastest)
    # After switching models, delete ./data/ to clear the stale index.
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768  # Must match the chosen model above

    # ── Vector Store ─────────────────────────────────────────────────────── #
    FAISS_INDEX_PATH: str = "./data/faiss_index"

    # ── Ingestion ────────────────────────────────────────────────────────── #
    # CHUNK_SIZE / CHUNK_OVERLAP are the fallback values when no doc_type
    # preset is specified. Prefer CHUNK_SIZE_PRESETS for new uploads.
    #
    # Qwen3 4B note: the 32K token window lets us use larger chunks without
    # overflowing context. 2000 chars ≈ 500 tokens — well within budget.
    # Gemma 3 4B used 1500 chars to stay within its 8K window.
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 200

    # Header detection
    HEADER_DETECTION_ENABLED: bool = True
    HEADER_CONFIDENCE_THRESHOLD: float = 0.75

    # ── Cross-Encoder Reranker ───────────────────────────────────────────── #
    # Reranking runs AFTER hybrid retrieval. Stage 1 retrieves RERANKER_TOP_N
    # candidates quickly (bi-encoder), then Stage 2 scores every (query, chunk)
    # pair with the cross-encoder and keeps only the best top_k for the LLM.
    #
    # Set RERANKER_ENABLED=false to skip reranking (faster, lower quality).
    #
    # Available models:
    #   BAAI/bge-reranker-base   — ~280 MB  (recommended)
    #   BAAI/bge-reranker-large  — ~1.3 GB  (highest accuracy, slower)
    #   cross-encoder/ms-marco-MiniLM-L-6-v2 — ~80 MB (fast alternative)
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    # Number of candidates fetched from hybrid search before reranking.
    # Must be ≥ top_k in each /ask request. Raised from 10 → 15 to give
    # the cross-encoder a broader pool to pick from, now that the larger
    # Qwen3 context budget can absorb more top-k chunks.
    RERANKER_TOP_N: int = 15

    # ── Context budget ───────────────────────────────────────────────────── #
    # Qwen3 4B: 32K token window. 12,000 chars ≈ 3,000 tokens — a safe,
    # generous budget that leaves >28K tokens free for system prompt +
    # generation. Previous Gemma 3 4B budget was 3,000 chars (≈750 tokens)
    # against its 8K window, which was intentionally conservative.
    # Override per-request via the max_context_chars field in AskRequest.
    MAX_CONTEXT_CHARS: int = 12000

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")


settings = Settings()
