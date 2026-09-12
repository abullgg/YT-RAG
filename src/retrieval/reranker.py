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
The cross-encoder outputs raw logits (unbounded floats). Higher means more
relevant. We apply an unbiased sigmoid solely to provide a bounded relevance
signal for display and diagnostics. It is not a calibrated probability that a
retrieved passage—or a generated answer—is correct.
"""

import logging
import math
from typing import List, Tuple

from sentence_transformers import CrossEncoder

from src.core.config import settings
from src.utils.errors import RetrievalError

logger = logging.getLogger(__name__)


def logit_to_relevance_score(logit: float) -> float:
    """Convert a reranker logit to a bounded, uncalibrated relevance signal.

    This intentionally applies neither a bias nor temperature scaling: a
    neutral logit maps to 0.5, negative logits below 0.5, and positive logits
    above 0.5.  The result preserves ranking order but is not a probability of
    answer correctness.
    """
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


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
        Scores are mapped with an unbiased sigmoid to a bounded relevance
        signal. They are not calibrated probabilities.

        Args:
            query:   The user's question (raw text, no BGE prefix needed).
            chunks:  Candidate passages from Stage-1 retrieval.
            top_k:   Number of top-ranked chunks to return.

        Returns:
            List of ``(chunk_text, relevance_score)`` tuples, sorted by
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

        scored: List[Tuple[str, float]] = [
            (chunk, round(logit_to_relevance_score(float(score)), 4))
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
