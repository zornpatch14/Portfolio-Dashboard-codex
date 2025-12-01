from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends

from ..dependencies import DownsampleFlag, get_selection
from ..schemas import HistogramResponse, SeriesResponse
from ..services.data_store import store

router = APIRouter(prefix="/api/v1/series", tags=["series"])


def _series_endpoint(name: str) -> Callable:
    async def handler(
        selection=Depends(get_selection),
        downsample_flag: DownsampleFlag = Depends(DownsampleFlag),
    ) -> SeriesResponse:
        downsample = downsample_flag()
        return store.series(name, selection, downsample)

    return handler


router.add_api_route("/equity", _series_endpoint("equity"), response_model=SeriesResponse, methods=["GET"])
router.add_api_route("/equity-percent", _series_endpoint("equity_percent"), response_model=SeriesResponse, methods=["GET"])
router.add_api_route("/drawdown", _series_endpoint("drawdown"), response_model=SeriesResponse, methods=["GET"])
router.add_api_route("/intraday-dd", _series_endpoint("intraday_drawdown"), response_model=SeriesResponse, methods=["GET"])
router.add_api_route("/netpos", _series_endpoint("netpos"), response_model=SeriesResponse, methods=["GET"])
router.add_api_route("/margin", _series_endpoint("margin"), response_model=SeriesResponse, methods=["GET"])
@router.get("/histogram", response_model=HistogramResponse)
async def histogram(selection=Depends(get_selection)) -> HistogramResponse:  # type: ignore[override]
    return store.histogram(selection)
