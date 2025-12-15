from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_selection

router = APIRouter(prefix="/api/v1/export", tags=["exports"])


@router.get("/trades")
async def export_trades(selection=Depends(get_selection)):
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Trade export coming soon.",
    )


@router.get("/metrics")
async def export_metrics(selection=Depends(get_selection)):
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Metrics export coming soon.",
    )
