from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import SelectionQueryParams
from ..schemas import MetricsResponse
from ..services.stub_store import store

router = APIRouter(prefix="/api/v1", tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(selection=Depends(SelectionQueryParams)) -> MetricsResponse:
    return store.metrics(selection)
