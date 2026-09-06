"""
Hybrid Retriever
----------------
Combines FAISS semantic search with BM25 keyword search.
Final score = 70% semantic + 30% keyword (both normalised to 0-1 before blending).
"""

import logging
from typing import List, Tuple
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Blends FAISS cosine similarity scores with BM25 keyword match scores."""

    def __init__(self):
        self._bm25: BM25Okapi = None
        self._chunks: List[str] = []
        self._doc_ids: List[str] = []

    def index_chunks(self, chunk_data: List[Tuple[str, str]]) -> None:
        """
        Build the BM25 index from a list of (doc_id, chunk_text) tuples.
        Called after every upload and on server startup when restoring state.
        """
        if not chunk_data:
            logger.warning("Empty chunk list — BM25 index not built.")
            self._bm25 = None
            self._chunks = []
            self._doc_ids = []
            return

        self._chunks = [c[1] for c in chunk_data]
        self._doc_ids = [c[0] for c in chunk_data]
        tokenized_corpus = [doc.lower().split() for doc in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Built BM25 index over %d chunks.", len(self._chunks))

    def search_semantic(self, query_embedding, faiss_index, top_k=5, filter_doc_id=None) -> Tuple[List[str], List[float]]:
        """Run FAISS vector search and return (chunks, scores)."""
        import numpy as np
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        effective_k = faiss_index.ntotal if filter_doc_id else min(top_k, faiss_index.ntotal)
        if effective_k == 0:
            return [], []

        distances, indices = faiss_index.search(query_embedding, effective_k)

        chunks = []
        scores = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self._chunks):
                continue
            if filter_doc_id and self._doc_ids[idx] != filter_doc_id:
                continue
            chunks.append(self._chunks[idx])
            scores.append(1.0 / (1.0 + float(dist)))
            if len(chunks) >= top_k:
                break

        return chunks, scores

    def search_keyword(self, query_text: str, top_k=5, filter_doc_id=None) -> Tuple[List[str], List[float]]:
        """Run BM25 keyword search and return (chunks, scores)."""
        if not self._bm25 or not query_text:
            return [], []

        tokenized_query = query_text.lower().split()
        doc_scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)

        chunks = []
        scores = []
        for idx in top_indices:
            if doc_scores[idx] > 0:
                if filter_doc_id and self._doc_ids[idx] != filter_doc_id:
                    continue
                chunks.append(self._chunks[idx])
                scores.append(float(doc_scores[idx]))
                if len(chunks) >= top_k:
                    break

        return chunks, scores

    def hybrid_search(self, query_embedding, query_text: str, faiss_index, top_k=3, filter_doc_id=None) -> Tuple[List[str], List[float]]:
        """
        Run semantic + keyword search and return top-k results by blended score.

        Falls back to pure semantic search if BM25 index is not available.
        """
        if not self._bm25 or not self._chunks:
            logger.warning("BM25 index missing — falling back to semantic search.")
            return self.search_semantic(query_embedding, faiss_index, top_k=top_k, filter_doc_id=filter_doc_id)

        sem_chunks, sem_scores = self.search_semantic(query_embedding, faiss_index, top_k=5, filter_doc_id=filter_doc_id)
        kw_chunks, kw_scores = self.search_keyword(query_text, top_k=5, filter_doc_id=filter_doc_id)

        # Normalise both score sets to 0-1
        max_sem = max(sem_scores) if sem_scores else 1.0
        norm_sem = {c: s / max_sem for c, s in zip(sem_chunks, sem_scores)}

        max_kw = max(kw_scores) if kw_scores else 1.0
        norm_kw = {c: s / max_kw for c, s in zip(kw_chunks, kw_scores)}

        all_chunks = set(sem_chunks) | set(kw_chunks)

        # Blend: 70% semantic + 30% keyword
        final_scores = {
            chunk: (norm_sem.get(chunk, 0.0) * 0.7) + (norm_kw.get(chunk, 0.0) * 0.3)
            for chunk in all_chunks
        }

        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [item[0] for item in sorted_items], [item[1] for item in sorted_items]
