"""
FAISS Retrieval Service
-----------------------
Manages a FAISS IndexFlatIP (inner product) index for dense-vector similarity search.

Embeddings are L2-normalised at encode time, so inner product equals cosine
similarity. Scores fall in the range (-1, 1] where 1.0 is a perfect match.

Chunk Store Schema
~~~~~~~~~~~~~~~~~~
Each FAISS vector position maps to a rich metadata dict:

    _chunk_store: Dict[int, Dict[str, Any]] = {
        position: {
            "doc_id":             str,
            "text":               str,
            "headers":            {"H1": str|None, "H2": str|None, "H3": str|None},
            "page_num":           int|None,
            "chunk_index":        int,
            "block_type":         str,         # "text" | "atomic_block" | "transcript"
            "block_metadata":     dict|None,
            "source_type":        str,         # "document" | "youtube"
            "video_id":           str|None,
            "video_url":          str|None,
            "video_title":        str|None,
            "channel":            str|None,
            "start_seconds":      float|None,
            "end_seconds":        float|None,
            "timestamp_url":      str|None,
            "transcript_origin":  str|None,
            "language":           str|None,
            "source_label":       str|None,
            "original_char_count": int,
            "char_position_in_doc": int,
        }
    }
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
    Convert a legacy entry or ensure modern fields exist in a chunk dict.
    """
    if isinstance(entry, dict):
        # Ensure all modern keys are present
        entry.setdefault("headers", {"H1": None, "H2": None, "H3": None})
        entry.setdefault("page_num", None)
        entry.setdefault("chunk_index", 0)
        entry.setdefault("block_type", "text")
        entry.setdefault("block_metadata", None)
        entry.setdefault("source_type", "document")
        entry.setdefault("video_id", None)
        entry.setdefault("video_url", None)
        entry.setdefault("video_title", None)
        entry.setdefault("channel", None)
        entry.setdefault("start_seconds", None)
        entry.setdefault("end_seconds", None)
        entry.setdefault("timestamp_url", None)
        entry.setdefault("transcript_origin", None)
        entry.setdefault("language", None)
        entry.setdefault("source_label", None)
        entry.setdefault("original_char_count", len(entry.get("text", "")))
        entry.setdefault("char_position_in_doc", 0)
        return entry

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
            "source_type": "document",
            "video_id": None,
            "video_url": None,
            "video_title": None,
            "channel": None,
            "start_seconds": None,
            "end_seconds": None,
            "timestamp_url": None,
            "transcript_origin": None,
            "language": None,
            "source_label": None,
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
        "source_type": "document",
        "video_id": None,
        "video_url": None,
        "video_title": None,
        "channel": None,
        "start_seconds": None,
        "end_seconds": None,
        "timestamp_url": None,
        "transcript_origin": None,
        "language": None,
        "source_label": None,
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
    """
    parts = [h for h in [headers.get("H1"), headers.get("H2"), headers.get("H3")] if h]
    return " > ".join(parts) if parts else "Document Root"


class RetrievalService:
    """
    Wraps FAISS and pairs every indexed vector with its source metadata.
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
        chunks: List[Any],
        existing_index: Optional[faiss.IndexFlatIP] = None,
    ) -> faiss.IndexFlatIP:
        """
        Add embeddings to the FAISS index and record their source metadata.
        Accepts strings, ChunkResult objects, or transcript chunk dicts.
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
                if isinstance(chunk, str):
                    self._chunk_store[pos] = {
                        "doc_id": doc_id,
                        "text": chunk,
                        "headers": {"H1": None, "H2": None, "H3": None},
                        "page_num": None,
                        "chunk_index": i,
                        "block_type": "text",
                        "block_metadata": None,
                        "source_type": "document",
                        "video_id": None,
                        "video_url": None,
                        "video_title": None,
                        "channel": None,
                        "start_seconds": None,
                        "end_seconds": None,
                        "timestamp_url": None,
                        "transcript_origin": None,
                        "language": None,
                        "source_label": None,
                        "original_char_count": len(chunk),
                        "char_position_in_doc": 0,
                    }
                elif isinstance(chunk, dict):
                    # Dict chunk (e.g. from transcript processor)
                    self._chunk_store[pos] = {
                        "doc_id": doc_id,
                        "text": chunk.get("text", ""),
                        "headers": chunk.get("headers", {"H1": None, "H2": None, "H3": None}),
                        "page_num": chunk.get("page_num"),
                        "chunk_index": chunk.get("chunk_index", i),
                        "block_type": chunk.get("block_type", "transcript"),
                        "block_metadata": chunk.get("block_metadata"),
                        "source_type": chunk.get("source_type", "youtube"),
                        "video_id": chunk.get("video_id"),
                        "video_url": chunk.get("video_url"),
                        "video_title": chunk.get("video_title"),
                        "channel": chunk.get("channel"),
                        "start_seconds": chunk.get("start_seconds"),
                        "end_seconds": chunk.get("end_seconds"),
                        "timestamp_url": chunk.get("timestamp_url"),
                        "transcript_origin": chunk.get("transcript_origin"),
                        "language": chunk.get("language"),
                        "source_label": chunk.get("source_label"),
                        "original_char_count": chunk.get("original_char_count", len(chunk.get("text", ""))),
                        "char_position_in_doc": chunk.get("char_position_in_doc", 0),
                    }
                else:
                    # ChunkResult object
                    self._chunk_store[pos] = {
                        "doc_id": doc_id,
                        "text": chunk.text,
                        "headers": getattr(chunk, "headers", {"H1": None, "H2": None, "H3": None}),
                        "page_num": getattr(chunk, "page_num", None),
                        "chunk_index": getattr(chunk, "chunk_index", i),
                        "block_type": getattr(chunk, "block_type", "text"),
                        "block_metadata": getattr(chunk, "block_metadata", None),
                        "source_type": getattr(chunk, "source_type", "document"),
                        "video_id": getattr(chunk, "video_id", None),
                        "video_url": getattr(chunk, "video_url", None),
                        "video_title": getattr(chunk, "video_title", None),
                        "channel": getattr(chunk, "channel", None),
                        "start_seconds": getattr(chunk, "start_seconds", None),
                        "end_seconds": getattr(chunk, "end_seconds", None),
                        "timestamp_url": getattr(chunk, "timestamp_url", None),
                        "transcript_origin": getattr(chunk, "transcript_origin", None),
                        "language": getattr(chunk, "language", None),
                        "source_label": getattr(chunk, "source_label", None),
                        "original_char_count": getattr(chunk, "original_char_count", len(chunk.text)),
                        "char_position_in_doc": getattr(chunk, "char_position_in_doc", 0),
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

                source_label = entry.get("source_label") or format_source_label(entry.get("headers", {}))
                result = {
                    **entry,
                    "position": int(idx),
                    "score": round(float(score), 4),
                    "source_label": source_label,
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
    #  BM25 corpus helper
    # ------------------------------------------------------------------ #

    def get_all_chunks(self) -> List[Tuple[str, str]]:
        """
        Return all (doc_id, text) pairs ordered by FAISS index position.
        Used by HybridRetriever to build the BM25 corpus.
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
        Returns:
            Tuple of (chunk_texts, scores).
        """
        try:
            pos_scores = hybrid_retriever.hybrid_search(
                query_embedding=query_embedding,
                query_text=query_text,
                faiss_index=faiss_index,
                top_k=top_k,
                filter_doc_id=filter_doc_id,
            )
            chunks = [
                self._chunk_store[pos]["text"]
                for pos, _ in pos_scores
                if pos in self._chunk_store
            ]
            scores = [score for _, score in pos_scores]
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
        Like hybrid_search() but returns the full metadata dict per chunk.
        Uses stable vector positions throughout to avoid text collisions.
        """
        try:
            pos_scores = hybrid_retriever.hybrid_search(
                query_embedding=query_embedding,
                query_text=query_text,
                faiss_index=faiss_index,
                top_k=top_k,
                filter_doc_id=filter_doc_id,
            )
        except Exception as exc:
            logger.error("Hybrid search failed: %s — falling back to semantic", exc)
            return self.search(query_embedding, faiss_index, top_k, filter_doc_id=filter_doc_id)

        rich_results: List[Dict[str, Any]] = []
        for pos, score in pos_scores:
            raw_entry = self._chunk_store.get(pos)
            if raw_entry is None:
                continue
            meta = _migrate_legacy_entry(raw_entry)
            source_label = meta.get("source_label") or format_source_label(meta.get("headers", {}))
            rich_results.append({
                **meta,
                "position": pos,
                "score": round(float(score), 4),
                "source_label": source_label,
            })
        return rich_results

    def _find_metadata_by_text(
        self, text: str, filter_doc_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Reverse-lookup chunk store by text content (kept as helper)."""
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
