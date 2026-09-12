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

config = sys.modules.setdefault("src.core.config", types.ModuleType("src.core.config"))
config.settings = types.SimpleNamespace(RERANKER_MODEL="test-reranker")

errors = types.ModuleType("src.utils.errors")
errors.RetrievalError = RuntimeError
sys.modules.setdefault("src.utils.errors", errors)

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
    def test_semantic_search_preserves_raw_inner_product_scores(self):
        retriever = HybridRetriever()
        retriever._chunks = ["best", "middle", "negative"]
        retriever._doc_ids = ["doc"] * 3

        chunks, scores = retriever.search_semantic(_Query(), _FakeFaissIndex(), top_k=3)

        self.assertEqual(chunks, ["best", "middle", "negative"])
        self.assertEqual(scores, [0.80, 0.10, -0.20])

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
        retriever.search_semantic = lambda *_args, **_kwargs: (
            ["semantic-only", "weak"], [0.9, 0.1]
        )
        retriever.search_keyword = lambda *_args, **_kwargs: (
            ["keyword-only", "weak"], [12.0, 2.0]
        )

        chunks, scores = retriever.hybrid_search(_Query(), "query", _FakeFaissIndex(), top_k=3)
        result = dict(zip(chunks, scores))

        self.assertEqual(result["semantic-only"], 0.7)
        self.assertEqual(result["keyword-only"], 0.3)
        self.assertEqual(result["weak"], 0.0)

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


if __name__ == "__main__":
    unittest.main()
