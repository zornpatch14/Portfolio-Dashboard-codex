from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from ..schemas import JobStatusResponse, OptimizerJobRequest, OptimizerJobResponse
from ..services.riskfolio_optimizer import riskfolio_jobs

router = APIRouter(prefix="/api/v1", tags=["optimizer"])


@router.post("/optimizer/riskfolio", response_model=OptimizerJobResponse)
async def riskfolio(request: OptimizerJobRequest) -> OptimizerJobResponse:
    request.objective = "riskfolio"
    return riskfolio_jobs.submit_mean_risk(request)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str) -> JobStatusResponse:
    try:
        if not riskfolio_jobs.owns_job(job_id):
            raise KeyError(job_id)
        return riskfolio_jobs.job_status(job_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str):
    try:
        if not riskfolio_jobs.owns_job(job_id):
            raise KeyError(job_id)
        event_stream = riskfolio_jobs.job_events(job_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    return StreamingResponse(event_stream, media_type="text/event-stream")
