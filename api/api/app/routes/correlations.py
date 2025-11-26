from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_selection
from ..schemas import CorrelationResponse
from ..services.stub_store import store

router = APIRouter(prefix="/api/v1", tags=["correlations"])


@router.get("/correlations", response_model=CorrelationResponse)
async def correlations(
    selection=Depends(get_selection),
    mode: str = Query(
        default="returns",
        description="Correlation mode (returns, drawdown_pct, pl, slope)",
    ),
) -> CorrelationResponse:
    return store.correlations(selection, mode)
