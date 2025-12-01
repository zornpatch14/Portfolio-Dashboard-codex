from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_selection
from ..schemas import CTAResponse
from ..services.stub_store import store

router = APIRouter(prefix="/api/v1", tags=["cta"])


@router.get("/cta", response_model=CTAResponse)
async def cta(selection=Depends(get_selection)) -> CTAResponse:
    return store.cta(selection)
