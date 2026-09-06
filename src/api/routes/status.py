"""
GET /status/{job_id} — Polling endpoint for Upload Jobs.
"""
from fastapi import APIRouter, HTTPException
import src.main as state

router = APIRouter()

@router.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Returns the current progression dictionary of a dispatched upload task.
    """
    job = state.job_tracker.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}
