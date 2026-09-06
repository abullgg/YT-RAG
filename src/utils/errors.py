"""
Custom exceptions for the RAG Document Assistant.
Each maps to a specific failure domain so routes can return the correct HTTP status code.
"""


class DocumentProcessingError(Exception):
    """Raised when document ingestion fails (e.g. corrupt PDF, unsupported type, empty file)."""

    def __init__(self, message: str = "Document processing failed.") -> None:
        self.message = message
        super().__init__(self.message)


class EmbeddingError(Exception):
    """Raised when the embedding model fails to encode text."""

    def __init__(self, message: str = "Embedding generation failed.") -> None:
        self.message = message
        super().__init__(self.message)


class RetrievalError(Exception):
    """Raised when a FAISS index operation fails (e.g. empty index, dimension mismatch)."""

    def __init__(self, message: str = "Retrieval operation failed.") -> None:
        self.message = message
        super().__init__(self.message)


class LLMServiceError(Exception):
    """Raised when the Ollama API call fails (e.g. network timeout, malformed response)."""

    def __init__(self, message: str = "LLM service error.") -> None:
        self.message = message
        super().__init__(self.message)
