from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..dependencies import get_selection
from ..schemas import CorrelationResponse

router = APIRouter(prefix="/api/v1", tags=["correlations"])


@router.get("/correlations", response_model=CorrelationResponse)
async def correlations(
    selection=Depends(get_selection),
    mode: str = Query(
        default="returns",
        description="Correlation mode (returns, drawdown_pct, pl, slope)",
    ),
) -> CorrelationResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Correlations endpoint coming soon.",
    )
