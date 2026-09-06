"""
Embedding Service
-----------------
Wraps sentence-transformers to produce dense vector embeddings
using the configured BGE model (default: BAAI/bge-base-en-v1.5, 768-dim).

BGE requires a specific prefix when encoding user queries for retrieval.
Use embed_query() for questions and embed_chunks() / embed_text() for documents.
Changing the model requires clearing ./data/ and re-uploading all documents.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.config import settings
from src.utils.errors import EmbeddingError

logger = logging.getLogger(__name__)

# BGE retrieval prefix — applied automatically by embed_query().
# Do not add it manually.
_BGE_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "


class EmbeddingService:
    """
    Generates dense vector embeddings using a BGE sentence-transformer model.

    Two encoding methods are provided:
    - embed_query()  — for user questions (adds the BGE query prefix).
    - embed_chunks() — for document passages (no prefix).

    The model is loaded once at startup and reused for all calls.
    """

    def __init__(self) -> None:
        self.model_name: str = settings.EMBEDDING_MODEL
        self.dimension: int = settings.EMBEDDING_DIMENSION

        logger.info(
            "Loading embedding model '%s' (dim=%d) …",
            self.model_name,
            self.dimension,
        )
        try:
            self.model: SentenceTransformer = SentenceTransformer(self.model_name)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{self.model_name}': {exc}"
            ) from exc

        logger.info(
            "Embedding model loaded (model='%s', dimension=%d)",
            self.model_name,
            self.dimension,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a user question for retrieval.

        Applies the BGE query prefix before encoding.
        Always use this method for questions — never embed_text() or embed_chunks().

        Returns a 1-D float32 array of shape (dimension,).
        """
        prefixed = _BGE_QUERY_PREFIX + query
        logger.debug("Embedding query (%d chars, with BGE prefix)", len(prefixed))
        try:
            embedding: np.ndarray = self.model.encode(
                prefixed,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return embedding.astype(np.float32)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed query: {exc}") from exc

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single document passage (no query prefix).

        Returns a 1-D float32 array of shape (dimension,).
        """
        logger.debug("Embedding passage (%d chars)", len(text))
        try:
            embedding: np.ndarray = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return embedding.astype(np.float32)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed text: {exc}") from exc

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Embed a batch of document passages (no query prefix).

        Returns a 2-D float32 array of shape (len(chunks), dimension).
        """
        logger.info("Embedding %d passage chunks …", len(chunks))
        try:
            embeddings: np.ndarray = self.model.encode(
                chunks,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=32,
                normalize_embeddings=True,
            )
            embeddings = embeddings.astype(np.float32)
            logger.info(
                "Batch embedding complete — shape %s, dtype %s",
                embeddings.shape,
                embeddings.dtype,
            )
            return embeddings
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to embed {len(chunks)} chunks: {exc}"
            ) from exc

    def get_embedding_dimension(self) -> int:
        """Return the vector dimension of the loaded embedding model."""
        return self.dimension
