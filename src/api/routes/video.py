"""
POST /videos/ingest — YouTube Transcript Ingestion Endpoint
"""

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.ingestion.youtube import extract_video_id
from src.models.schemas import VideoIngestRequest, VideoIngestResponse
from src.tasks.video_ingest import process_video_ingest_task
import src.main as state

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/videos/ingest", response_model=VideoIngestResponse)
async def ingest_video(
    request: VideoIngestRequest,
    background_tasks: BackgroundTasks,
) -> VideoIngestResponse:
    """
    Ingest a YouTube video by URL or ID.
    Fetches captions, chunks by speech boundaries, embeds, and indexes into FAISS + BM25.
    Idempotent: Re-ingesting an already-indexed video returns existing indexed details
    without duplicating chunks, unless reindex=True.
    """
    # ── 1. Validate and extract video ID ──────────────────────────────────
    try:
        video_id = extract_video_id(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if state.embedding_service is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service not ready. Server may still be starting up.",
        )

    doc_id = f"yt_{video_id}"
    job_id = str(uuid.uuid4())

    # ── 2. Idempotency Shortcut ───────────────────────────────────────────
    if not request.reindex and doc_id in state.indexed_documents:
        existing = state.indexed_documents[doc_id]
        logger.info("Video '%s' is already indexed; returning existing record.", video_id)
        state.job_tracker.create_job(
            job_id=job_id,
            filename=existing.get("title", f"YouTube Video ({video_id})"),
            source_type="youtube",
            video_id=video_id,
        )
        state.job_tracker.mark_complete(
            job_id=job_id,
            chunks=existing.get("chunks_created", 0),
            extra_info={
                "video_title": existing.get("title"),
                "channel": existing.get("channel"),
                "estimated_duration": existing.get("estimated_duration"),
                "transcript_origin": existing.get("transcript_origin"),
            },
        )
        return VideoIngestResponse(
            document_id=doc_id,
            video_id=video_id,
            video_title=existing.get("title", f"YouTube Video ({video_id})"),
            channel=existing.get("channel"),
            estimated_duration=existing.get("estimated_duration"),
            status="completed",
            stage="completed",
            message="Video is already indexed and ready for querying.",
            transcript_origin=existing.get("transcript_origin"),
            chunks_created=existing.get("chunks_created", 0),
            job_id=job_id,
        )

    # ── 3. Dispatch Background Task ───────────────────────────────────────
    state.job_tracker.create_job(
        job_id=job_id,
        filename=f"YouTube Video ({video_id})",
        source_type="youtube",
        video_id=video_id,
    )

    background_tasks.add_task(
        process_video_ingest_task,
        job_id=job_id,
        video_id=video_id,
        embedding_service=state.embedding_service,
        reindex=request.reindex,
    )

    return VideoIngestResponse(
        document_id=doc_id,
        video_id=video_id,
        video_title=f"YouTube Video ({video_id})",
        status="processing",
        stage="fetching_captions",
        message="Video ingestion queued for processing.",
        chunks_created=0,
        job_id=job_id,
    )

