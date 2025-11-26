from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..dependencies import get_selection
from ..services.stub_store import store

router = APIRouter(prefix="/api/v1/export", tags=["exports"])


@router.get("/trades")
async def export_trades(selection=Depends(get_selection)):
    filename, content = store.export_rows(selection, kind="trades")
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/metrics")
async def export_metrics(selection=Depends(get_selection)):
    filename, content = store.export_rows(selection, kind="metrics")
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
