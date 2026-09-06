"""
POST /upload — Document Ingestion Endpoint
"""

import logging
import uuid
from typing import List

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, BackgroundTasks

from src.core.config import settings
from src.ingestion.processor import DocumentProcessor
from src.models.schemas import UploadResponse
from src.utils.errors import DocumentProcessingError, EmbeddingError, RetrievalError
from src.tasks.uploader import process_upload_task
import src.main as state

logger = logging.getLogger(__name__)

router = APIRouter()

# DocumentProcessor holds no ML models — safe to create once at module load.
_processor = DocumentProcessor()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> UploadResponse:
    """
    Upload & Index a Document.

    Accepts a PDF or TXT file. Files < 1MB process instantly.
    Files > 1MB automatically queue into the asynchronous background queue.
    """
    filename: str = file.filename or "unknown"
    content_type: str = file.content_type or ""
    logger.info("===== Upload request: '%s' =====", filename)

    # Validate file type
    allowed_extensions = (".pdf", ".txt")
    if not filename.lower().endswith(allowed_extensions):
        logger.warning("Rejected file '%s': unsupported type", filename)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{filename}'. Allowed types: {', '.join(allowed_extensions)}"
        )

    try:
        # Buffer entire file into memory immediately so FastAPI doesn't delete the temporary 
        # buffer while the background queue is processing it.
        raw_bytes: bytes = await file.read()
        
        # Determine background threshold (1MB = 1024 * 1024 bytes)
        file_size = len(raw_bytes)
        is_large = file_size > (1024 * 1024)
        
        # Create Job Tracking Token
        job_id = str(uuid.uuid4())
        state.job_tracker.create_job(job_id, filename)

        if is_large:
            logger.info(f"File Size ({file_size}b) > 1MB. Queuing to background...")
            background_tasks.add_task(
                process_upload_task,
                job_id=job_id,
                raw_bytes=raw_bytes,
                filename=filename,
                content_type=content_type,
                processor=_processor,
                embedding_service=state.embedding_service  # shared singleton
            )
            return UploadResponse(
                document_id=job_id,
                filename=filename,
                job_id=job_id,
                status="queued",
                message="File is large and currently queuing for background inference."
            )
        else:
            logger.info(f"File Size ({file_size}b) < 1MB. Running instantaneously...")
            await process_upload_task(
                job_id=job_id,
                raw_bytes=raw_bytes,
                filename=filename,
                content_type=content_type,
                processor=_processor,
                embedding_service=state.embedding_service  # shared singleton
            )
            
            job = state.job_tracker.get_job(job_id)
            if not job or job.get("status") == "failed":
                raise HTTPException(status_code=500, detail=job.get("error", "Unknown ingestion error"))
                
            return UploadResponse(
                document_id=job_id,
                filename=filename,
                chunks_created=job.get("chunks_created", 0),
                status="completed",
                message="Document uploaded and indexed successfully."
            )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error pushing '%s' into workflow", filename)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {exc}",
        ) from exc
