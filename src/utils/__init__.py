"""Utility package — logging, custom exceptions, helpers."""
from src.utils.logger import setup_logging
from src.utils.errors import (
    DocumentProcessingError,
    EmbeddingError,
    LLMServiceError,
    RetrievalError,
)

__all__ = [
    "setup_logging",
    "DocumentProcessingError",
    "EmbeddingError",
    "LLMServiceError",
    "RetrievalError",
]
