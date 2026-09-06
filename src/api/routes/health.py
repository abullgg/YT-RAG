"""
GET /health — System Health Check
"""

import logging

from fastapi import APIRouter

from src.models.schemas import HealthResponse
import src.main as state

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Returns the server status and the number of documents currently indexed.
    """
    doc_count = len(state.indexed_documents)
    logger.info("Health check — %d documents indexed", doc_count)
    return HealthResponse(
        status="healthy",
        documents_indexed=doc_count,
    )
