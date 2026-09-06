"""
Uploader Task
=============
Asynchronous task logic for background processing large document uploads.
Uses a batched embedding approach to provide granular progress tracking.

The chunk_text() method now returns List[ChunkResult] (not List[str]).
We extract the plain text strings for embedding, but pass the full ChunkResult
objects to add_to_index() so the rich metadata is preserved in the chunk store.
"""

import logging
import asyncio
from typing import List
import numpy as np

import src.main as state
from src.ingestion.processor import DocumentProcessor, ChunkResult
from src.embeddings.embedding import EmbeddingService

logger = logging.getLogger(__name__)

async def process_upload_task(
    job_id: str,
    raw_bytes: bytes,
    filename: str,
    content_type: str,
    processor: DocumentProcessor,
    embedding_service: EmbeddingService
) -> None:
    """
    Executes the ingestion → embedding → FAISS indexing steps.
    Regularly updates the global job_tracker with current percentage.
    """
    tracker = state.job_tracker
    tracker.update_status(job_id, status="processing", progress=10)

    try:
        # 1. Extract Text
        text = await processor.extract_text(raw_bytes, filename, content_type)
        if not text.strip():
            raise ValueError(f"No readable text found in '{filename}'.")

        tracker.update_status(job_id, status="processing", progress=20)

        # 2. Chunk text — returns List[ChunkResult] (rich metadata objects)
        chunk_results: List[ChunkResult] = processor.chunk_text(text)
        if not chunk_results:
            raise ValueError(f"Text from '{filename}' produced no usable chunks.")

        tracker.update_status(job_id, status="processing", progress=25)

        # 3. Extract plain text strings for embedding
        chunk_texts: List[str] = [cr.text for cr in chunk_results]

        # 4. Embed chunks in batches for progress tracking (25% → 85%)
        all_embeddings = []
        batch_size = max(1, len(chunk_texts) // 10)

        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i : i + batch_size]

            # Offload CPU-bound encoding to thread pool
            emb = await asyncio.to_thread(embedding_service.embed_chunks, batch)
            all_embeddings.append(emb)

            ratio_done = (i + len(batch)) / len(chunk_texts)
            curr_prog = min(int(25 + (ratio_done * 60)), 85)  # 25 → 85
            tracker.update_status(job_id, status="processing", progress=curr_prog)

        embeddings = np.vstack(all_embeddings)
        tracker.update_status(job_id, status="processing", progress=90)

        # 5. Document ID = Job ID (for easy tracking)
        doc_id = job_id

        # 6. Add to FAISS Index — pass ChunkResult objects so rich metadata is stored
        state.faiss_index = state.retrieval_service.add_to_index(
            embeddings=embeddings,
            doc_id=doc_id,
            chunks=chunk_results,        # List[ChunkResult], not List[str]
            existing_index=state.faiss_index,
        )

        # 7. Global Registry
        state.indexed_documents[doc_id] = {
            "filename": filename,
            "chunks_created": len(chunk_results),
            "text_length": len(text),
        }

        tracker.update_status(job_id, status="processing", progress=95)

        # 8. Persistent Storage sync
        state.storage.save_state(
            state.faiss_index,
            state.indexed_documents,
            state.retrieval_service._chunk_store,
        )

        # 9. Sync BM25 Keyword Search Index
        state.hybrid_retriever.index_chunks(state.retrieval_service.get_all_chunks())

        tracker.mark_complete(job_id, len(chunk_results))

    except Exception as exc:
        logger.error("Background Task %s failed: %s", job_id, exc)
        tracker.mark_failed(job_id, str(exc))
