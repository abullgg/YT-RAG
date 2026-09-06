"""
Job Tracker
-----------
In-memory tracker for async upload jobs. Stores status, progress, and error
for each job so the frontend can poll for completion.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class UploadJobTracker:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        
    def create_job(self, job_id: str, filename: str):
        self.jobs[job_id] = {
            "filename": filename,
            "status": "queued",
            "progress": 0,
            "chunks_created": 0,
            "error": None
        }
        logger.info(f"Created Job: {job_id}")
        
    def update_status(self, job_id: str, status: str, progress: int = 0):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
            self.jobs[job_id]["progress"] = progress
            
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)
        
    def mark_complete(self, job_id: str, chunks: int):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 100
            self.jobs[job_id]["chunks_created"] = chunks
        logger.info(f"Job {job_id} Completed: {chunks} chunks processed.")
            
    def mark_failed(self, job_id: str, error: str):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = error
        logger.error(f"Job {job_id} Failed: {error}")
