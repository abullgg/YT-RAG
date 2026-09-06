"""
RAG Document Assistant — FastAPI server.

Initialises logging, creates the app, mounts all routers, and manages
global state (FAISS index, document registry, job tracker, hybrid retriever).

Shared ML singletons (embedding_service, reranker) are initialised in the
startup event so the BGE models are loaded exactly ONCE and shared across
all route modules via `import src.main as state`.

Run with: uvicorn src.main:app --reload
"""

import logging
from typing import Dict, Optional

import faiss
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.embeddings.embedding import EmbeddingService
from src.persistence.storage import PersistentStorage
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.tasks.job_tracker import UploadJobTracker
from src.utils.logger import setup_logging
from src.vector_store.faiss_index import RetrievalService

storage = PersistentStorage()

# ------------------------------------------------------------------ #
#  Bootstrap
# ------------------------------------------------------------------ #

setup_logging()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Global State  (shared across route modules via import)
# ------------------------------------------------------------------ #

# {doc_id: metadata} for all indexed documents
indexed_documents: Dict[str, dict] = {}

# Shared FAISS index — created on first upload
faiss_index: Optional[faiss.IndexFlatIP] = None

# Lightweight service singletons (no heavy models, safe to create at module load)
retrieval_service = RetrievalService()
job_tracker = UploadJobTracker()
hybrid_retriever = HybridRetriever()

# Heavy ML model singletons — declared here so all route modules can access
# them via `import src.main as state`. Initialised in the startup event
# (NOT at module level) to ensure only ONE copy of each model is ever loaded.
embedding_service: Optional[EmbeddingService] = None
reranker: Optional[CrossEncoderReranker] = None

# Load state from disk
logger.info("Loading persisted state...")
faiss_index, indexed_documents, chunk_storage = storage.load_state()
retrieval_service._chunk_store = chunk_storage
hybrid_retriever.index_chunks(retrieval_service.get_all_chunks())

# ------------------------------------------------------------------ #
#  FastAPI Application
# ------------------------------------------------------------------ #

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Upload PDF / TXT documents, index them with FAISS, "
        "and ask questions answered by Qwen3 via Ollama."
    ),
    version="1.0.0",
)

# ── CORS ── allow the Next.js frontend (dev: 3000, prod: any configured origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers — imported here (after globals) to avoid circular-import issues
from src.api.routes import upload, ask, health, status  # noqa: E402

app.include_router(health.router, tags=["monitoring"])
app.include_router(upload.router, tags=["upload"])
app.include_router(ask.router, tags=["query"])
app.include_router(status.router, tags=["status"])


@app.on_event("startup")
async def _startup_event() -> None:
    """
    Load the two heavy ML models exactly once, at server startup.

    Both are stored as module-level globals so every route module that does
    ``import src.main as state`` can reach them via
    ``state.embedding_service`` / ``state.reranker``.

    Loading happens here (not at module-import time) so that:
    - The models are available before the first request is served.
    - They are never duplicated across route modules.
    """
    global embedding_service, reranker

    logger.info("=" * 50)
    logger.info("Initializing global ML singletons...")

    embedding_service = EmbeddingService()
    logger.info("✓ EmbeddingService loaded (BGE bi-encoder, dim=%d)",
                embedding_service.dimension)

    if settings.RERANKER_ENABLED:
        reranker = CrossEncoderReranker()
        logger.info("✓ CrossEncoderReranker loaded (model='%s')",
                    settings.RERANKER_MODEL)
    else:
        logger.info("– CrossEncoderReranker skipped (RERANKER_ENABLED=false)")

    logger.info("Global singletons initialized successfully")
    logger.info("=" * 50)


@app.get("/")
async def root():
    """Root endpoint — API welcome message."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API.",
        "documentation": "/docs",
    }


# ------------------------------------------------------------------ #
#  Entrypoint
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
