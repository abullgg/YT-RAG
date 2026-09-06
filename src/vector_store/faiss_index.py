"""
FAISS Retrieval Service
-----------------------
Manages a FAISS IndexFlatIP (inner product) index for dense-vector similarity search.

Embeddings are L2-normalised at encode time, so inner product equals cosine
similarity. Scores fall in the range (-1, 1] where 1.0 is a perfect match.

Chunk Store Schema (Fix 3)
~~~~~~~~~~~~~~~~~~~~~~~~~~
Each FAISS vector position maps to a rich metadata dict:

    _chunk_store: Dict[int, Dict[str, Any]] = {
        position: {
            "doc_id":             str,
            "text":               str,
            "headers":            {"H1": str|None, "H2": str|None, "H3": str|None},
            "page_num":           int|None,   # PDF pages only (future)
            "chunk_index":        int,         # position within document
            "block_type":         str,         # "text" | "atomic_block"
            "block_metadata":     dict|None,   # e.g. {type:"table", format:"markdown"}
            "original_char_count": int,
            "char_position_in_doc": int,
        }
    }

Backward compatibility
~~~~~~~~~~~~~~~~~~~~~~
Old pickled chunk stores contain (doc_id, text) tuples. The
``_migrate_legacy_entry()`` method detects and upgrades them transparently so
existing ``./data/chunks.pkl`` files continue to work without re-indexing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np

from src.core.config import settings
from src.utils.errors import RetrievalError

logger = logging.getLogger(__name__)

# Type alias for the rich metadata dict stored per chunk position
ChunkMetadata = Dict[str, Any]


def _migrate_legacy_entry(entry: Any) -> ChunkMetadata:
    """
    Convert a legacy ``(doc_id, chunk_text)`` tuple to the new dict schema.
    Called lazily whenever an old-format entry is encountered in the chunk store.
    """
    if isinstance(entry, dict):
        return entry  # already new format

    if isinstance(entry, tuple) and len(entry) == 2:
        doc_id, text = entry
        logger.debug("Migrating legacy chunk entry for doc_id=%s", doc_id)
        return {
            "doc_id": doc_id,
            "text": text,
            "headers": {"H1": None, "H2": None, "H3": None},
            "page_num": None,
            "chunk_index": 0,
            "block_type": "text",
            "block_metadata": None,
            "original_char_count": len(text),
            "char_position_in_doc": 0,
        }

    # Unknown format — return a safe sentinel
    logger.warning("Unknown chunk store entry format: %r — skipping", type(entry))
    return {
        "doc_id": "unknown",
        "text": "",
        "headers": {"H1": None, "H2": None, "H3": None},
        "page_num": None,
        "chunk_index": 0,
        "block_type": "text",
        "block_metadata": None,
        "original_char_count": 0,
        "char_position_in_doc": 0,
    }


def format_source_label(headers: Dict[str, Optional[str]]) -> str:
    """
    Build a human-readable breadcrumb from header metadata.

    Examples:
        {"H1": "Introduction", "H2": "Background", "H3": None}
            → "Introduction > Background"
        {"H1": None, "H2": None, "H3": None}
            → "Document Root"

    Args:
        headers: Dict with keys H1, H2, H3 (values may be None).

    Returns:
        Formatted string suitable for display in the frontend.
    """
    parts = [h for h in [headers.get("H1"), headers.get("H2"), headers.get("H3")] if h]
    return " > ".join(parts) if parts else "Document Root"


class RetrievalService:
    """
    Wraps FAISS and pairs every indexed vector with its source metadata.

    Args:
        dimension: Embedding vector size. Must match the active embedding model.
    """

    def __init__(self, dimension: int = settings.EMBEDDING_DIMENSION) -> None:
        self.dimension: int = dimension
        # Maps FAISS vector position → rich ChunkMetadata dict
        self._chunk_store: Dict[int, Any] = {}

    # ------------------------------------------------------------------ #
    #  Indexing
    # ------------------------------------------------------------------ #

    def add_to_index(
        self,
        embeddings: np.ndarray,
        doc_id: str,
        chunks: List[Any],           # List[str] or List[ChunkResult]
        existing_index: Optional[faiss.IndexFlatIP] = None,
    ) -> faiss.IndexFlatIP:
        """
        Add embeddings to the FAISS index and record their source metadata.

        Accepts either a list of plain strings (backward compat) or a list of
        :class:`ChunkResult` objects from the new processor pipeline.

        Creates a new IndexFlatIP if no existing index is provided.

        Args:
            embeddings:     2-D float32 array of shape (n, dimension).
            doc_id:         ID of the document these chunks belong to.
            chunks:         Original text chunks or ChunkResult objects.
            existing_index: Existing FAISS index to append to. Creates new if None.

        Returns:
            The updated (or newly created) FAISS index.

        Raises:
            RetrievalError: If the add operation fails.
        """
        try:
            if existing_index is None:
                logger.info("Creating new FAISS IndexFlatIP (dim=%d)", self.dimension)
                index = faiss.IndexFlatIP(self.dimension)
            else:
                index = existing_index

            start_pos: int = index.ntotal
            index.add(embeddings)

            for i, chunk in enumerate(chunks):
                pos = start_pos + i
                # Support both legacy plain strings and new ChunkResult objects
                if isinstance(chunk, str):
                    self._chunk_store[pos] = {
                        "doc_id": doc_id,
                        "text": chunk,
                        "headers": {"H1": None, "H2": None, "H3": None},
                        "page_num": None,
                        "chunk_index": i,
                        "block_type": "text",
                        "block_metadata": None,
                        "original_char_count": len(chunk),
                        "char_position_in_doc": 0,
                    }
                else:
                    # ChunkResult object
                    self._chunk_store[pos] = {
                        "doc_id": doc_id,
                        "text": chunk.text,
                        "headers": chunk.headers,
                        "page_num": getattr(chunk, "page_num", None),
                        "chunk_index": chunk.chunk_index,
                        "block_type": chunk.block_type,
                        "block_metadata": chunk.block_metadata,
                        "original_char_count": chunk.original_char_count,
                        "char_position_in_doc": chunk.char_position_in_doc,
                    }

            logger.info(
                "Added %d vectors for doc '%s' — index now contains %d vectors",
                len(chunks), doc_id, index.ntotal,
            )
            return index

        except Exception as exc:
            raise RetrievalError(
                f"Failed to add {len(chunks)} vectors to index: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    #  Search
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_embedding: np.ndarray,
        faiss_index: faiss.IndexFlatIP,
        top_k: int = 3,
        filter_doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find the top-k most similar chunks to a query embedding.

        Args:
            query_embedding: 1-D or 2-D float32 query vector from embed_query().
            faiss_index:     The FAISS index to search.
            top_k:           Number of results to return.
            filter_doc_id:   If set, only return chunks from this document.

        Returns:
            List of dicts with keys matching the chunk store schema plus
            ``score`` and ``source_label``.

        Raises:
            RetrievalError: If the FAISS search fails.
        """
        try:
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            effective_k = faiss_index.ntotal if filter_doc_id else min(top_k, faiss_index.ntotal)
            if effective_k == 0:
                return []

            logger.info(
                "Searching FAISS index (%d vectors) for top-%d (filter_doc_id=%s)",
                faiss_index.ntotal, effective_k, filter_doc_id,
            )

            scores_raw, indices = faiss_index.search(query_embedding, effective_k)

            results: List[Dict[str, Any]] = []
            for rank, (score, idx) in enumerate(zip(scores_raw[0], indices[0])):
                if idx == -1:
                    continue

                raw_entry = self._chunk_store.get(int(idx))
                if raw_entry is None:
                    continue

                entry = _migrate_legacy_entry(raw_entry)

                if filter_doc_id and entry["doc_id"] != filter_doc_id:
                    continue

                result = {
                    **entry,
                    "position": int(idx),
                    "score": round(float(score), 4),
                    "source_label": format_source_label(entry.get("headers", {})),
                }
                results.append(result)
                logger.debug(
                    "  Rank %d: doc=%s, score=%.4f, preview='%s…'",
                    rank + 1, entry["doc_id"], score, entry["text"][:80],
                )

                if len(results) >= top_k:
                    break

            logger.info("Returning %d search results", len(results))
            return results

        except Exception as exc:
            raise RetrievalError(f"FAISS search failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  BM25 support
    # ------------------------------------------------------------------ #

    def get_all_chunks(self) -> List[Tuple[str, str]]:
        """
        Return all (doc_id, text) pairs ordered by FAISS index position.
        Used by HybridRetriever to build the BM25 corpus.

        Handles both legacy tuple entries and new dict entries.
        """
        result = []
        for i in range(len(self._chunk_store)):
            raw = self._chunk_store.get(i)
            if raw is None:
                continue
            entry = _migrate_legacy_entry(raw)
            result.append((entry["doc_id"], entry["text"]))
        return result

    # ------------------------------------------------------------------ #
    #  Hybrid search
    # ------------------------------------------------------------------ #

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        faiss_index: faiss.IndexFlatIP,
        hybrid_retriever,
        top_k: int = 3,
        filter_doc_id: Optional[str] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Run hybrid search (FAISS + BM25) via HybridRetriever.
        Falls back to pure semantic search if BM25 is unavailable.

        Returns:
            Tuple of (chunk_texts, scores).
        """
        try:
            chunks, scores = hybrid_retriever.hybrid_search(
                query_embedding=query_embedding,
                query_text=query_text,
                faiss_index=faiss_index,
                top_k=top_k,
                filter_doc_id=filter_doc_id,
            )
            return chunks, scores
        except Exception as exc:
            logger.error("Hybrid search failed: %s — falling back to semantic", exc)
            results = self.search(query_embedding, faiss_index, top_k, filter_doc_id=filter_doc_id)
            return [r["text"] for r in results], [r["score"] for r in results]

    # ------------------------------------------------------------------ #
    #  Rich hybrid search (returns full metadata)
    # ------------------------------------------------------------------ #

    def hybrid_search_with_metadata(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        faiss_index: faiss.IndexFlatIP,
        hybrid_retriever,
        top_k: int = 3,
        filter_doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Like hybrid_search() but returns the full metadata dict per chunk
        instead of just (text, score) tuples. Used by the /ask endpoint when
        the RetrievedChunk response schema is active.
        """
        try:
            chunks, scores = hybrid_retriever.hybrid_search(
                query_embedding=query_embedding,
                query_text=query_text,
                faiss_index=faiss_index,
                top_k=top_k,
                filter_doc_id=filter_doc_id,
            )
        except Exception as exc:
            logger.error("Hybrid search failed: %s — falling back to semantic", exc)
            return self.search(query_embedding, faiss_index, top_k, filter_doc_id=filter_doc_id)

        # Enrich each text result with its stored metadata
        rich_results: List[Dict[str, Any]] = []
        for chunk_text, score in zip(chunks, scores):
            # Look up the chunk in the store by text match (BM25 returns plain text)
            meta = self._find_metadata_by_text(chunk_text, filter_doc_id)
            rich_results.append({
                **(meta or {}),
                "text": chunk_text,
                "score": round(float(score), 4),
                "source_label": format_source_label((meta or {}).get("headers", {})),
            })
        return rich_results

    def _find_metadata_by_text(
        self, text: str, filter_doc_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Reverse-lookup chunk store by text content (used for BM25 results)."""
        for entry in self._chunk_store.values():
            e = _migrate_legacy_entry(entry)
            if e["text"] == text:
                if filter_doc_id is None or e["doc_id"] == filter_doc_id:
                    return e
        return None

    # ------------------------------------------------------------------ #
    #  Stats
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_index_stats(faiss_index: Optional[faiss.IndexFlatIP]) -> Dict[str, Any]:
        """Return total vector count and dimension of the FAISS index."""
        if faiss_index is None:
            return {"total_vectors": 0, "dimension": 0}
        return {"total_vectors": faiss_index.ntotal, "dimension": faiss_index.d}
