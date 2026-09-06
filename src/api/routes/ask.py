"""
POST /ask — Question-answering endpoint.

Embeds the question, runs hybrid retrieval (FAISS + BM25),
and sends retrieved passages to the local Ollama LLM for a grounded answer.

Pipeline
--------
1. embed_query()            -- BGE query prefix applied
2. hybrid_search_with_metadata() -- FAISS cosine + BM25 fused; returns rich metadata
3. CrossEncoderReranker     -- scores (query, chunk) pairs; keeps top_k best
4. Context budget trimming  -- remove lowest-confidence chunks if over char limit
5. Build context            -- join top_k chunks with source labels
6. LLMService               -- grounded answer generation via Ollama
7. Return AskResponse       -- includes rich source_chunks + diagnostics

Step 3 is skipped when RERANKER_ENABLED=false in config / .env.
Step 4 respects the per-request max_context_chars override or the server default.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException

from src.core.config import settings, RetrievalConfig
from src.generation.llm import LLMService
from src.models.schemas import AskRequest, AskResponse, RetrievedChunk
from src.utils.errors import EmbeddingError, RetrievalError, LLMServiceError
import src.main as state

logger = logging.getLogger(__name__)

router = APIRouter()

# ML model singletons (embedding_service, reranker) live in src.main and are
# initialized ONCE in the startup event. Only the LLM client stays local.
_llm_service: Optional[LLMService] = None


def _get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


# ---------------------------------------------------------------------------
# Context budget helpers
# ---------------------------------------------------------------------------

def _trim_to_budget(
    rich_chunks: List[Dict[str, Any]],
    max_chars: int,
) -> List[Dict[str, Any]]:
    """
    Remove lowest-confidence chunks (from the tail) until the combined text
    fits within *max_chars*. Returns the trimmed list.
    """
    total = sum(len(c.get("text", "")) for c in rich_chunks)
    if total <= max_chars:
        return rich_chunks

    # Chunks are already sorted best→worst; trim from the end
    while rich_chunks and total > max_chars:
        removed = rich_chunks.pop()
        total -= len(removed.get("text", ""))
        logger.debug(
            "Context budget: removed chunk (score=%.4f, %d chars) — budget remaining=%d",
            removed.get("score", 0.0),
            len(removed.get("text", "")),
            max_chars - total,
        )

    logger.info(
        "Context budget trim: %d chunk(s) kept, %d chars used / %d budget",
        len(rich_chunks), total, max_chars,
    )
    return rich_chunks


def _to_retrieved_chunk(entry: Dict[str, Any], score: float) -> RetrievedChunk:
    """Convert a raw chunk-store dict to a validated RetrievedChunk model."""
    from src.vector_store.faiss_index import format_source_label
    headers = entry.get("headers", {"H1": None, "H2": None, "H3": None})
    return RetrievedChunk(
        position=entry.get("position", 0),
        doc_id=entry.get("doc_id", "unknown"),
        text=entry.get("text", ""),
        headers=headers,
        page_num=entry.get("page_num"),
        chunk_index=entry.get("chunk_index", 0),
        block_type=entry.get("block_type", "text"),
        block_metadata=entry.get("block_metadata"),
        confidence_score=round(score, 4),
        source_label=entry.get("source_label") or format_source_label(headers),
    )


# ---------------------------------------------------------------------------
# /ask endpoint
# ---------------------------------------------------------------------------

@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Ask a question against indexed documents.

    Retrieves the most relevant passages from the FAISS + BM25 index,
    optionally reranks them with a cross-encoder, trims to the context budget,
    and returns an answer grounded strictly in those passages.
    """
    logger.info(
        "===== Ask request: '%s' (top_k=%d, reranker=%s) =====",
        request.question, request.top_k, settings.RERANKER_ENABLED,
    )

    # ── Pre-flight checks ───────────────────────────────────────────────────
    if not state.indexed_documents:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Upload a document first.",
        )

    if state.faiss_index is None or state.faiss_index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="The FAISS index is empty. Upload a document first.",
        )

    if state.embedding_service is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service not ready. Server may still be starting up.",
        )

    # ── Resolve context budget ──────────────────────────────────────────────
    max_context_chars: int = (
        request.max_context_chars
        if request.max_context_chars is not None
        else settings.MAX_CONTEXT_CHARS
    )

    try:
        # ── Step 1: Embed the query ─────────────────────────────────────────
        query_embedding: np.ndarray = state.embedding_service.embed_query(request.question)

        # ── Step 2: Hybrid retrieval (Stage 1) ──────────────────────────────
        retrieve_k: int = settings.RERANKER_TOP_N if state.reranker else request.top_k

        # Use the metadata-rich search path
        rich_candidates: List[Dict[str, Any]] = state.retrieval_service.hybrid_search_with_metadata(
            query_embedding=query_embedding,
            query_text=request.question,
            faiss_index=state.faiss_index,
            hybrid_retriever=state.hybrid_retriever,
            top_k=retrieve_k,
            filter_doc_id=request.document_id,
        )

        if not rich_candidates:
            return AskResponse(
                answer="No relevant information was found in the indexed documents.",
                sources=[],
                source_chunks=[],
                confidence=0.0,
                context_chars_used=0,
                context_budget_remaining=max_context_chars,
            )

        # ── Step 3: Cross-encoder reranking (Stage 2) ───────────────────────
        candidate_texts = [c.get("text", "") for c in rich_candidates]

        if state.reranker:
            ranked_pairs: List[Tuple[str, float]] = await asyncio.to_thread(
                state.reranker.rerank,
                request.question,
                candidate_texts,
                request.top_k,
            )
            # Re-order rich_candidates to match reranker output
            text_to_entry: Dict[str, Dict[str, Any]] = {
                c.get("text", ""): c for c in rich_candidates
            }
            rich_reranked: List[Dict[str, Any]] = []
            reranked_scores: List[float] = []
            for text, score in ranked_pairs:
                entry = text_to_entry.get(text, {"text": text})
                entry["score"] = score
                rich_reranked.append(entry)
                reranked_scores.append(score)

            rich_candidates = rich_reranked
            logger.info(
                "Reranking complete — %d candidates → %d chunks (top score=%.4f)",
                retrieve_k, len(rich_candidates),
                reranked_scores[0] if reranked_scores else 0.0,
            )
        else:
            # No reranker — use hybrid scores as-is, truncate to top_k
            rich_candidates = rich_candidates[: request.top_k]

        # ── Step 4: Context budget trimming ─────────────────────────────────
        rich_candidates = _trim_to_budget(rich_candidates, max_context_chars)

        if not rich_candidates:
            return AskResponse(
                answer="Context budget too small to fit any retrieved passage. "
                       "Try increasing max_context_chars.",
                sources=[],
                source_chunks=[],
                confidence=0.0,
                context_chars_used=0,
                context_budget_remaining=max_context_chars,
            )

        # ── Step 5: Build context ────────────────────────────────────────────
        source_chunks: List[RetrievedChunk] = []
        for entry in rich_candidates:
            score = float(entry.get("score", 0.0))
            source_chunks.append(_to_retrieved_chunk(entry, score))

        context: str = "\n\n---\n\n".join(
            f"[Source {i + 1} — {sc.source_label}]:\n{sc.text}"
            for i, sc in enumerate(source_chunks)
        )

        context_chars_used = sum(len(sc.text) for sc in source_chunks)
        context_budget_remaining = max(0, max_context_chars - context_chars_used)

        # ── Step 6: Generate answer ──────────────────────────────────────────
        llm = _get_llm_service()
        answer: str = llm.generate_answer(
            question=request.question,
            context=context,
        )

        max_confidence: float = round(
            max((sc.confidence_score for sc in source_chunks), default=0.0), 4
        )

        # ── Step 7: Diagnostics ──────────────────────────────────────────────
        logger.info(
            "Retrieval complete — top_k=%d, context_chars=%d, budget_remaining=%d, "
            "max_confidence=%.4f",
            len(source_chunks), context_chars_used, context_budget_remaining, max_confidence,
        )

        return AskResponse(
            answer=answer,
            sources=[sc.text for sc in source_chunks],   # backward-compat plain list
            source_chunks=source_chunks,
            confidence=max_confidence,
            context_chars_used=context_chars_used,
            context_budget_remaining=context_budget_remaining,
        )

    except HTTPException:
        raise
    except LLMServiceError as exc:
        logger.error("LLM error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (EmbeddingError, RetrievalError) as exc:
        logger.error("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error answering question")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error while processing your question: {exc}",
        ) from exc
