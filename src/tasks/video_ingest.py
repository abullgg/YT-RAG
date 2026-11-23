"""
Video Ingestion Task
====================
Asynchronous background task for ingesting YouTube videos:
fetches captions, chunks transcripts by speech boundaries, embeds with BGE,
indexes into FAISS and BM25, and persists state to disk with stage progress telemetry.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import numpy as np

import src.main as state
from src.ingestion.youtube import (
    extract_video_id,
    fetch_video_metadata,
    fetch_transcript,
    chunk_transcript,
)
from src.embeddings.embedding import EmbeddingService

logger = logging.getLogger(__name__)


async def process_video_ingest_task(
    job_id: str,
    video_id: str,
    embedding_service: EmbeddingService,
    reindex: bool = False,
) -> None:
    """
    Executes the YouTube transcript ingestion pipeline:
      fetching metadata -> captions -> chunking -> embedding -> indexing -> persistence.
    """
    tracker = state.job_tracker
    doc_id = f"yt_{video_id}"

    try:
        # ── Step 0: Idempotency Check ───────────────────────────────────────
        if not reindex and doc_id in state.indexed_documents:
            logger.info("Video '%s' is already indexed (idempotent shortcut).", video_id)
            existing_doc = state.indexed_documents[doc_id]
            tracker.mark_complete(
                job_id,
                existing_doc.get("chunks_created", 0),
                extra_info={
                    "video_title": existing_doc.get("title"),
                    "channel": existing_doc.get("channel"),
                    "estimated_duration": existing_doc.get("estimated_duration"),
                    "transcript_origin": existing_doc.get("transcript_origin"),
                },
            )
            return

        # ── Step 1: Fetch Video Metadata & Captions ────────────────────────
        tracker.update_stage(
            job_id,
            stage="fetching_captions",
            stage_message="Fetching video metadata and captions from YouTube...",
            progress=15,
        )

        # Offload network I/O to thread
        meta = await asyncio.to_thread(fetch_video_metadata, video_id)
        video_title = meta["title"]
        channel = meta["channel"]
        video_url = meta["video_url"]

        raw_segments, origin, language = await asyncio.to_thread(
            fetch_transcript, video_id
        )

        estimated_duration: Optional[int] = None
        if raw_segments:
            def _get_end_sec(s: Any) -> float:
                st = s.get("start", 0.0) if isinstance(s, dict) else getattr(s, "start", 0.0)
                du = s.get("duration", 0.0) if isinstance(s, dict) else getattr(s, "duration", 0.0)
                return float(st or 0.0) + float(du or 0.0)

            estimated_duration = int(
                round(max(_get_end_sec(seg) for seg in raw_segments))
            )

        tracker.update_stage(
            job_id,
            stage="chunking",
            stage_message=f"Chunking {origin} captions into speech passages...",
            progress=30,
            extra_info={
                "video_title": video_title,
                "channel": channel,
                "estimated_duration": estimated_duration,
                "transcript_origin": origin,
            },
        )

        # ── Step 2: Speech-Aware Chunking ───────────────────────────────────
        chunks = chunk_transcript(
            raw_segments=raw_segments,
            video_id=video_id,
            video_title=video_title,
            video_url=video_url,
            channel=channel,
            transcript_origin=origin,
            language=language,
        )

        if not chunks:
            raise ValueError(f"No usable transcript chunks produced for video '{video_id}'.")

        # ── Step 3: Embed Chunks with Batch Telemetry ───────────────────────
        tracker.update_stage(
            job_id,
            stage="embedding",
            stage_message="Generating dense vector embeddings...",
            progress=40,
        )

        chunk_texts = [c["text"] for c in chunks]
        all_embeddings: List[np.ndarray] = []
        batch_size = max(1, len(chunk_texts) // 10)

        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i : i + batch_size]
            emb = await asyncio.to_thread(embedding_service.embed_chunks, batch)
            all_embeddings.append(emb)

            ratio_done = (i + len(batch)) / len(chunk_texts)
            curr_prog = min(int(40 + (ratio_done * 45)), 85)
            tracker.update_stage(
                job_id,
                stage="embedding",
                stage_message=f"Embedding chunks ({int(ratio_done * 100)}%)...",
                progress=curr_prog,
            )

        embeddings = np.vstack(all_embeddings)

        # ── Step 4: Indexing in FAISS & BM25 ─────────────────────────────────
        tracker.update_stage(
            job_id,
            stage="indexing",
            stage_message="Indexing vectors into FAISS and BM25...",
            progress=90,
        )

        state.faiss_index = state.retrieval_service.add_to_index(
            embeddings=embeddings,
            doc_id=doc_id,
            chunks=chunks,
            existing_index=state.faiss_index,
        )

        # Update global indexed documents registry
        state.indexed_documents[doc_id] = {
            "filename": f"{video_title} (YouTube)",
            "video_id": video_id,
            "video_url": video_url,
            "title": video_title,
            "channel": channel,
            "estimated_duration": estimated_duration,
            "transcript_origin": origin,
            "source_type": "youtube",
            "chunks_created": len(chunks),
            "text_length": sum(len(c["text"]) for c in chunks),
        }

        # ── Step 5: Save State & BM25 Sync ──────────────────────────────────
        state.storage.save_state(
            state.faiss_index,
            state.indexed_documents,
            state.retrieval_service._chunk_store,
        )
        state.hybrid_retriever.index_chunks(state.retrieval_service.get_all_chunks())

        tracker.mark_complete(
            job_id,
            len(chunks),
            extra_info={
                "video_title": video_title,
                "channel": channel,
                "estimated_duration": estimated_duration,
                "transcript_origin": origin,
            },
        )
        logger.info("Successfully ingested YouTube video '%s' (%d chunks).", video_id, len(chunks))

    except Exception as exc:
        logger.exception("Failed to ingest YouTube video '%s'", video_id)
        tracker.mark_failed(job_id, str(exc))

