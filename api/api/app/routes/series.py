from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends

from ..dependencies import DownsampleFlag, SelectionQueryParams
from ..schemas import HistogramResponse, SeriesResponse
from ..services.data_store import store

router = APIRouter(prefix="/api/v1/series", tags=["series"])


def _series_endpoint(name: str) -> Callable:
    async def handler(
        selection=Depends(SelectionQueryParams),
        downsample: bool = Depends(DownsampleFlag),
    ) -> SeriesResponse:
        return store.series(name, selection, downsample)

    return handler


router.add_api_route("/equity", _series_endpoint("equity"), response_model=SeriesResponse, methods=["GET"])
router.add_api_route(
    "/equity-percent", _series_endpoint("equity_percent"), response_model=SeriesResponse, methods=["GET"]
)
router.add_api_route("/drawdown", _series_endpoint("drawdown"), response_model=SeriesResponse, methods=["GET"])
router.add_api_route("/intraday-dd", _series_endpoint("intraday_drawdown"), response_model=SeriesResponse, methods=["GET"])
router.add_api_route("/netpos", _series_endpoint("netpos"), response_model=SeriesResponse, methods=["GET"])
router.add_api_route("/margin", _series_endpoint("margin"), response_model=SeriesResponse, methods=["GET"])
@router.get("/histogram", response_model=HistogramResponse)
async def histogram(selection=Depends(SelectionQueryParams)) -> HistogramResponse:  # type: ignore[override]
    return store.histogram(selection)
