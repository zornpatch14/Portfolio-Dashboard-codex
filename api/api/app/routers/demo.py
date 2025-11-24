from __future__ import annotations

from pathlib import Path
from pathlib import Path
from typing import List

from fastapi import APIRouter, Query

from api.app.compute.caches import PerFileCache
from api.app.compute.downsampling import downsample_timeseries
from api.app.compute.portfolio import PortfolioAggregator
from api.app.data.loader import list_sample_trade_files

router = APIRouter(prefix="/api/v1", tags=["demo"])


@router.get("/series/equity")
def get_equity(downsample: bool = Query(True), target_points: int = Query(500)) -> dict:
    cache = PerFileCache()
    aggregator = PortfolioAggregator(cache)
    files: List[Path] = list_sample_trade_files()
    view = aggregator.aggregate(files)
    equity_df = view.equity
    if downsample:
        result = downsample_timeseries(equity_df, "timestamp", "equity", target_points=target_points)
        equity_df = result.downsampled
    return {"equity": equity_df.to_dicts(), "contributors": [p.name for p in files]}


@router.get("/series/netpos")
def get_netpos() -> dict:
    cache = PerFileCache()
    files: List[Path] = list_sample_trade_files()
    aggregator = PortfolioAggregator(cache)
    view = aggregator.aggregate(files)
    return {"net_position": view.net_position.to_dicts()}


@router.get("/series/margin")
def get_margin() -> dict:
    cache = PerFileCache()
    files: List[Path] = list_sample_trade_files()
    aggregator = PortfolioAggregator(cache)
    view = aggregator.aggregate(files)
    return {"margin": view.margin.to_dicts()}


@router.get("/series/daily-returns")
def get_daily_returns() -> dict:
    cache = PerFileCache()
    files: List[Path] = list_sample_trade_files()
    aggregator = PortfolioAggregator(cache)
    view = aggregator.aggregate(files)
    return {"daily_returns": view.daily_returns.to_dicts()}

