from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_selection
from ..schemas import MetricsResponse
from ..services.data_store import store

router = APIRouter(prefix="/api/v1", tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(selection=Depends(get_selection)) -> MetricsResponse:
    return store.metrics(selection)
