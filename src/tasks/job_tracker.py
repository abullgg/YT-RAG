"""
Job Tracker
-----------
In-memory tracker for async upload and video ingestion jobs.
Maintains simple high-level status ('queued' | 'processing' | 'completed' | 'failed')
alongside granular ingestion stages ('fetching_captions', 'chunking', etc.)
for smooth frontend polling without breaking existing contracts.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class UploadJobTracker:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(
        self,
        job_id: str,
        filename: str,
        source_type: str = "document",
        video_id: Optional[str] = None,
    ):
        self.jobs[job_id] = {
            "filename": filename,
            "source_type": source_type,
            "status": "queued",
            "stage": "queued",
            "stage_message": "Job queued for processing",
            "progress": 0,
            "chunks_created": 0,
            "video_id": video_id,
            "video_title": None,
            "channel": None,
            "estimated_duration": None,
            "transcript_origin": None,
            "error": None,
        }
        logger.info(f"Created Job: {job_id} (source={source_type})")

    def update_status(self, job_id: str, status: str, progress: int = 0):
        """Backward-compatible update method."""
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
            self.jobs[job_id]["progress"] = progress

    def update_stage(
        self,
        job_id: str,
        stage: str,
        stage_message: str = "",
        progress: Optional[int] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ):
        """Granular stage telemetry while maintaining standard processing status."""
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "processing"
            self.jobs[job_id]["stage"] = stage
            if stage_message:
                self.jobs[job_id]["stage_message"] = stage_message
            if progress is not None:
                self.jobs[job_id]["progress"] = progress
            if extra_info:
                self.jobs[job_id].update(extra_info)
            logger.debug(f"Job {job_id} -> stage={stage} ({progress}%)")

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def mark_complete(
        self,
        job_id: str,
        chunks: int,
        extra_info: Optional[Dict[str, Any]] = None,
    ):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["stage"] = "completed"
            self.jobs[job_id]["stage_message"] = "Processing complete"
            self.jobs[job_id]["progress"] = 100
            self.jobs[job_id]["chunks_created"] = chunks
            if extra_info:
                self.jobs[job_id].update(extra_info)
        logger.info(f"Job {job_id} Completed: {chunks} chunks processed.")

    def mark_failed(self, job_id: str, error: str, stage: str = "failed"):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["stage"] = stage
            self.jobs[job_id]["stage_message"] = f"Failed: {error}"
            self.jobs[job_id]["error"] = error
        logger.error(f"Job {job_id} Failed at stage {stage}: {error}")
