from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_selection
from ..schemas import CTAResponse

router = APIRouter(prefix="/api/v1", tags=["cta"])


@router.get("/cta", response_model=CTAResponse)
async def cta(selection=Depends(get_selection)) -> CTAResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="CTA endpoint coming soon.",
    )
