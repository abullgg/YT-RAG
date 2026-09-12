"""
Pydantic Schemas
================
Request and response models for every API endpoint, plus shared
base models for documents and chunks.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
#  Shared / Base Models
# ------------------------------------------------------------------ #

class DocumentBase(BaseModel):
    """Base schema for a document reference."""
    filename: str
    metadata: Optional[Dict] = None


class DocumentCreate(DocumentBase):
    """Schema used internally when creating a new document record."""
    content: str


class ChunkResponse(BaseModel):
    """A single retrieved chunk with its similarity score."""
    text: str
    score: float
    metadata: Dict


# ------------------------------------------------------------------ #
#  Rich chunk model (Fix 3)
# ------------------------------------------------------------------ #

class RetrievedChunk(BaseModel):
    """
    Full metadata for a single retrieved passage.
    Returned by the updated /ask endpoint so the frontend can display
    section breadcrumbs, relevance scores, and block-type badges.
    """
    position: int = Field(default=0, description="FAISS vector position")
    doc_id: str
    text: str
    headers: Dict[str, Optional[str]] = Field(
        default_factory=lambda: {"H1": None, "H2": None, "H3": None}
    )
    page_num: Optional[int] = None
    chunk_index: int = 0
    block_type: str = "text"          # "text" | "atomic_block"
    block_metadata: Optional[Dict[str, Any]] = None
    relevance_score: float = Field(
        default=0.0,
        description=(
            "Retrieval/reranker relevance signal, not a probability that the "
            "generated answer is correct."
        ),
    )
    # Temporary compatibility alias for clients using the former name.
    confidence_score: float = Field(
        default=0.0,
        description="Deprecated alias for relevance_score; not answer confidence.",
        deprecated=True,
    )
    source_label: str = "Document Root"


# ------------------------------------------------------------------ #
#  /health
# ------------------------------------------------------------------ #

class HealthResponse(BaseModel):
    """Response schema for GET /health."""
    status: str
    documents_indexed: int


# ------------------------------------------------------------------ #
#  /upload
# ------------------------------------------------------------------ #

class UploadResponse(BaseModel):
    """Response schema for POST /upload."""
    document_id: str
    filename: str
    chunks_created: Optional[int] = 0
    status: str
    message: str
    job_id: Optional[str] = None


# ------------------------------------------------------------------ #
#  /ask
# ------------------------------------------------------------------ #

class AskRequest(BaseModel):
    """Body schema for POST /ask."""
    question: str = Field(..., min_length=1, description="The question to answer")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of chunks to retrieve")
    document_id: Optional[str] = Field(
        default=None,
        description="Optional document ID to restrict retrieval to a specific document.",
    )
    max_context_chars: Optional[int] = Field(
        default=None,
        description=(
            "Soft character ceiling for combined context sent to the LLM. "
            "Overrides the server default (MAX_CONTEXT_CHARS). "
            "Lowest-relevance chunks are trimmed if the budget is exceeded."
        ),
    )


class AskResponse(BaseModel):
    """Response schema for POST /ask; scores are relevance signals, not confidence probabilities."""
    answer: str
    sources: List[str]                        # Plain text list (backward compat)
    source_chunks: List[RetrievedChunk] = []  # Rich chunks with metadata
    relevance_score: float = Field(
        default=0.0,
        description="Best retrieved relevance signal; not answer correctness probability.",
    )
    # Temporary compatibility alias. Clients should migrate to relevance_score.
    confidence: float = Field(
        default=0.0,
        description="Deprecated alias for relevance_score; not answer confidence.",
        deprecated=True,
    )
    context_chars_used: int = 0               # Diagnostic: total chars in context
    context_budget_remaining: int = 0         # Diagnostic: budget left over
