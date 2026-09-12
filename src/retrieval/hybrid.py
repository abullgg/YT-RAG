"""
Hybrid Retriever
----------------
Combines FAISS semantic search with BM25 keyword search.

The final value is a query-relative *ranking score*, not a probability or a
confidence estimate for a generated answer.  FAISS and BM25 have different
score scales, so their candidate scores are min-max normalised separately
before applying the configured ranking weights.
"""

import logging
from typing import List, Tuple
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# Ranking weights only. They do not express answer correctness probability.
SEMANTIC_RANKING_WEIGHT = 0.7
KEYWORD_RANKING_WEIGHT = 0.3


def _min_max_normalize(scores: List[float]) -> List[float]:
    """Normalise one retrieval channel's candidate scores to ``[0, 1]``.

    FAISS cosine similarity may legitimately be zero or negative, therefore
    dividing by the maximum score is not valid.  Per-query min-max
    normalisation is monotonic: it preserves the ordering of candidates while
    putting cosine similarity and BM25 on a common scale for rank fusion.

    When only one candidate is present, or all candidates tie, every candidate
    receives ``1.0``.  That represents an unresolved tie within this channel,
    not an absolute relevance probability.
    """
    if not scores:
        return []

    low = min(scores)
    high = max(scores)
    if high == low:
        return [1.0] * len(scores)

    span = high - low
    return [(score - low) / span for score in scores]


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

        similarities, indices = faiss_index.search(query_embedding, effective_k)

        chunks = []
        scores = []
        for score, idx in zip(similarities[0], indices[0]):
            if idx == -1 or idx >= len(self._chunks):
                continue
            if filter_doc_id and self._doc_ids[idx] != filter_doc_id:
                continue
            chunks.append(self._chunks[idx])
            # IndexFlatIP over L2-normalised embeddings returns cosine
            # similarity.  Preserve that raw semantic relevance signal.
            scores.append(float(score))
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

        # The channels use incompatible raw scales (cosine and BM25).  Apply
        # monotonic, per-query normalisation for rank fusion; these values are
        # not calibrated probabilities.
        norm_sem = dict(zip(sem_chunks, _min_max_normalize(sem_scores)))
        norm_kw = dict(zip(kw_chunks, _min_max_normalize(kw_scores)))

        all_chunks = set(sem_chunks) | set(kw_chunks)

        # Blend ranking signals only; this is not answer confidence.
        final_scores = {
            chunk: (
                norm_sem.get(chunk, 0.0) * SEMANTIC_RANKING_WEIGHT
                + norm_kw.get(chunk, 0.0) * KEYWORD_RANKING_WEIGHT
            )
            for chunk in all_chunks
        }

        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [item[0] for item in sorted_items], [item[1] for item in sorted_items]
