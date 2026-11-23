"""Deterministic regression tests for retrieval and reranker score handling.

The tests exercise the project's scoring functions while replacing optional ML
dependencies with tiny fakes.  They require only the Python standard library.
"""

import sys
import types
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _UnusedBM25:
    def __init__(self, *_args, **_kwargs):
        pass


class _FakeModel:
    def __init__(self, logits):
        self.logits = logits

    def predict(self, _pairs, show_progress_bar=False):
        return self.logits


# The production modules import these optional dependencies.  Stubbing them
# lets this regression suite run without downloading/loading ML models.
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
rank_bm25 = types.ModuleType("rank_bm25")
rank_bm25.BM25Okapi = _UnusedBM25
sys.modules.setdefault("rank_bm25", rank_bm25)

sentence_transformers = types.ModuleType("sentence_transformers")
sentence_transformers.CrossEncoder = object
sys.modules.setdefault("sentence_transformers", sentence_transformers)

import src.utils.errors
import src.core.config as config
config.settings.RERANKER_MODEL = "test-reranker"

from src.retrieval.hybrid import HybridRetriever, _min_max_normalize
from src.retrieval.reranker import CrossEncoderReranker, logit_to_relevance_score


class _FakeFaissIndex:
    ntotal = 3

    def search(self, _query, _count):
        # IndexFlatIP-style similarities: higher is better, including negative.
        return [[0.80, 0.10, -0.20]], [[0, 1, 2]]


class _Query:
    ndim = 2


class ScoringTests(unittest.TestCase):
    def test_semantic_search_preserves_raw_inner_product_scores_and_positions(self):
        retriever = HybridRetriever()
        retriever._chunks = ["best", "middle", "negative"]
        retriever._doc_ids = ["doc"] * 3

        results = retriever.search_semantic(_Query(), _FakeFaissIndex(), top_k=3)

        self.assertEqual(results, [(0, 0.80), (1, 0.10), (2, -0.20)])

    def test_min_max_normalisation_handles_negative_cosines_and_preserves_order(self):
        normalised = _min_max_normalize([-0.20, -0.40, -0.60])

        self.assertEqual(normalised[0], 1.0)
        self.assertAlmostEqual(normalised[1], 0.5)
        self.assertEqual(normalised[2], 0.0)
        self.assertGreater(normalised[0], normalised[1])
        self.assertGreater(normalised[1], normalised[2])
        self.assertEqual(_min_max_normalize([0.25, 0.25]), [1.0, 1.0])

    def test_hybrid_weights_are_ranking_weights_not_an_automatic_high_score(self):
        retriever = HybridRetriever()
        retriever._bm25 = object()
        retriever._chunks = ["semantic-only", "keyword-only", "weak"]
        retriever.search_semantic = lambda *_args, **_kwargs: [
            (0, 0.9), (2, 0.1)
        ]
        retriever.search_keyword = lambda *_args, **_kwargs: [
            (1, 12.0), (2, 2.0)
        ]

        pos_scores = retriever.hybrid_search(_Query(), "query", _FakeFaissIndex(), top_k=3)
        result = dict(pos_scores)

        self.assertEqual(result[0], 0.7)
        self.assertEqual(result[1], 0.3)
        self.assertEqual(result[2], 0.0)

    def test_repeated_text_chunks_retain_distinct_positions_in_hybrid_search(self):
        retriever = HybridRetriever()
        retriever._bm25 = object()
        # Chunks at pos 0 and pos 1 have IDENTICAL wording
        retriever._chunks = ["identical text", "identical text"]
        retriever._doc_ids = ["video_1", "video_1"]
        retriever.search_semantic = lambda *_args, **_kwargs: [(0, 0.85), (1, 0.85)]
        retriever.search_keyword = lambda *_args, **_kwargs: [(0, 5.0), (1, 5.0)]

        pos_scores = retriever.hybrid_search(_Query(), "query", _FakeFaissIndex(), top_k=2)

        # Both positions 0 and 1 must be present; neither dropped due to text collision
        positions = [pos for pos, _ in pos_scores]
        self.assertIn(0, positions)
        self.assertIn(1, positions)
        self.assertEqual(len(pos_scores), 2)

    def test_unbiased_reranker_signal_is_centered_at_neutral_logit(self):
        self.assertAlmostEqual(logit_to_relevance_score(0.0), 0.5)
        self.assertLess(logit_to_relevance_score(-1.0), 0.5)
        self.assertGreater(logit_to_relevance_score(1.0), 0.5)

    def test_reranker_preserves_logit_ranking_without_a_bias(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "test-reranker"
        reranker.model = _FakeModel([-1.0, 0.0, 1.0])

        ranked = reranker.rerank("question", ["negative", "neutral", "positive"], top_k=3)

        self.assertEqual([chunk for chunk, _ in ranked], ["positive", "neutral", "negative"])
        self.assertEqual([score for _, score in ranked], [0.7311, 0.5, 0.2689])

    def test_rerank_candidates_preserves_metadata_and_positions_without_collisions(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "test-reranker"
        reranker.model = _FakeModel([0.0, 2.0])  # cand 1 lower logit, cand 2 higher logit

        candidates = [
            {"position": 0, "text": "Same repeated phrase", "start_seconds": 15.0},
            {"position": 1, "text": "Same repeated phrase", "start_seconds": 90.0},
        ]

        reranked = reranker.rerank_candidates("question", candidates, top_k=2)

        # Higher logit (index 1) should come first, retaining start_seconds=90.0
        self.assertEqual(reranked[0]["position"], 1)
        self.assertEqual(reranked[0]["start_seconds"], 90.0)
        self.assertEqual(reranked[1]["position"], 0)
        self.assertEqual(reranked[1]["start_seconds"], 15.0)


if __name__ == "__main__":
    unittest.main()
