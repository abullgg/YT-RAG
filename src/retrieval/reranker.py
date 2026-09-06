"""
Cross-Encoder Reranker
----------------------
Reranks a set of candidate chunks retrieved by the hybrid bi-encoder search
using a cross-encoder model that processes (query, passage) pairs jointly.

Why cross-encoder reranking?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Bi-encoder retrieval (BGE + FAISS) is fast because query and passages are
embedded independently. However, independent embeddings cannot model the
precise interaction between a specific query and a specific passage.

A cross-encoder fixes this by concatenating the query and passage and
running full self-attention across both at inference time. This is ~100x
slower per pair but produces significantly more accurate relevance scores.

Two-stage strategy
~~~~~~~~~~~~~~~~~~
    Stage 1 — Retrieval (bi-encoder, fast):
        Fetch RERANKER_TOP_N (~10) candidate chunks from FAISS + BM25.

    Stage 2 — Reranking (cross-encoder, accurate):
        Score every (query, chunk) pair. Keep the best top_k for the LLM.

This gives the precision of a cross-encoder without the latency of running
it across the entire corpus.

Model
~~~~~
Default: ``BAAI/bge-reranker-base``

The BGE reranker family is consistent with our BGE bi-encoder embeddings:
  BAAI/bge-reranker-base   — ~280 MB, recommended balance of speed + quality
  BAAI/bge-reranker-large  — ~1.3 GB, highest accuracy, noticeably slower
  cross-encoder/ms-marco-MiniLM-L-6-v2  — ~80 MB, very fast alternative

Scores
~~~~~~
The cross-encoder outputs raw logit scores (unbounded floats). Higher means
more relevant. We apply a sigmoid transform to normalise scores to (0, 1]
before returning them, making them interpretable as relevance probabilities.
"""

import logging
import math
from typing import List, Tuple

from sentence_transformers import CrossEncoder

from src.core.config import settings
from src.utils.errors import RetrievalError

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder relevance model.

    Usage::

        reranker = CrossEncoderReranker()
        ranked = reranker.rerank(query, candidates, top_k=3)
        chunks = [text for text, score in ranked]

    Args:
        model_name: HuggingFace model identifier. Defaults to
            ``settings.RERANKER_MODEL`` (``BAAI/bge-reranker-base``).
    """

    def __init__(self, model_name: str = settings.RERANKER_MODEL) -> None:
        self.model_name = model_name
        logger.info("Loading cross-encoder reranker '%s' …", self.model_name)
        try:
            # max_length=512 is the BGE reranker's context window.
            # Passages longer than this are truncated automatically.
            self.model = CrossEncoder(self.model_name, max_length=512)
        except Exception as exc:
            raise RetrievalError(
                f"Failed to load cross-encoder reranker '{self.model_name}': {exc}"
            ) from exc
        logger.info("Reranker loaded (model='%s')", self.model_name)

    def rerank(
        self,
        query: str,
        chunks: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Score and rerank candidate chunks for a given query.

        Each (query, chunk) pair is scored jointly by the cross-encoder.
        Scores are normalised with sigmoid to the range (0, 1].

        Args:
            query:   The user's question (raw text, no BGE prefix needed).
            chunks:  Candidate passages from Stage-1 retrieval.
            top_k:   Number of top-ranked chunks to return.

        Returns:
            List of ``(chunk_text, normalised_score)`` tuples, sorted by
            score descending, truncated to ``top_k``.

        Raises:
            RetrievalError: If the cross-encoder prediction step fails.
        """
        if not chunks:
            return []

        top_k = min(top_k, len(chunks))

        # Build (query, passage) input pairs for the cross-encoder
        pairs = [[query, chunk] for chunk in chunks]

        logger.info(
            "Reranking %d candidates → top %d (model='%s')",
            len(chunks), top_k, self.model_name,
        )

        try:
            raw_scores = self.model.predict(pairs, show_progress_bar=False)
        except Exception as exc:
            raise RetrievalError(
                f"Cross-encoder prediction failed: {exc}"
            ) from exc

        # Normalise raw logits → (0, 1] via sigmoid so scores are interpretable
        # Apply temperature scaling and bias shift to boost non-prose document scores
        def _sigmoid(x: float) -> float:
            temperature_scaled = x * 1.5  # Spread the scores out
            bias_shift = 1.0              # Push neutral 0.0 logs to positive
            return 1.0 / (1.0 + math.exp(-(temperature_scaled + bias_shift)))

        scored: List[Tuple[str, float]] = [
            (chunk, round(_sigmoid(float(score)), 4))
            for chunk, score in zip(chunks, raw_scores)
        ]

        # Sort by normalised score descending
        scored.sort(key=lambda pair: pair[1], reverse=True)

        for rank, (chunk, score) in enumerate(scored[:top_k]):
            logger.debug(
                "  Rerank %d: score=%.4f, preview='%s…'",
                rank + 1, score, chunk[:80],
            )

        return scored[:top_k]
