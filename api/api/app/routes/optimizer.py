from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from ..schemas import JobStatusResponse, OptimizerJobRequest, OptimizerJobResponse
from ..services.stub_store import store

router = APIRouter(prefix="/api/v1", tags=["optimizer"])


@router.post("/optimizer/allocator", response_model=OptimizerJobResponse)
async def allocator(request: OptimizerJobRequest) -> OptimizerJobResponse:
    request.objective = "allocator"
    return store.optimizer(request)


@router.post("/optimizer/riskfolio", response_model=OptimizerJobResponse)
async def riskfolio(request: OptimizerJobRequest) -> OptimizerJobResponse:
    request.objective = "riskfolio"
    return store.optimizer(request)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str) -> JobStatusResponse:
    try:
        return store.job_status(job_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str):
    try:
        event_stream = store.job_events(job_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    return StreamingResponse(event_stream, media_type="text/event-stream")
