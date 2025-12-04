from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, Query
import polars as pl

from api.app.compute.caches import PerFileCache
from api.app.compute.downsampling import DownsampleResult, downsample_timeseries
from api.app.compute.portfolio import ContributorSeries, PortfolioAggregator
from api.app.data.loader import list_sample_trade_files
from api.app.schemas import Selection, SeriesContributor, SeriesPoint, SeriesResponse

router = APIRouter(prefix="/api/v1", tags=["demo"])


def _downsample_frame(frame, timestamp_col: str, value_col: str, downsample: bool, target_points: int) -> tuple:
    raw = len(frame)
    sampled = raw
    trimmed = frame
    if downsample and raw:
        result: DownsampleResult = downsample_timeseries(frame, timestamp_col, value_col, target_points=target_points)
        trimmed = result.downsampled
        raw = result.raw_count
        sampled = result.downsampled_count
    return trimmed, raw, sampled


def _to_points(frame, timestamp_col: str, value_col: str) -> list[SeriesPoint]:
    points: list[SeriesPoint] = []
    for row in frame.iter_rows(named=True):
        points.append(SeriesPoint(timestamp=row[timestamp_col], value=float(row[value_col])))
    return points


def _drawdown_from_equity(frame):
    values = []
    running_max = float("-inf")
    for ts, eq in zip(frame["timestamp"], frame["equity"]):
        eq_val = float(eq)
        running_max = max(running_max, eq_val)
        values.append((ts, eq_val - running_max))
    return values


def _portfolio_frame(series_name: str, view) -> tuple:
    if series_name == "equity":
        return view.equity, "equity", "equity"

    if series_name in {"equity_percent", "equity-percent"}:
        return view.percent_equity, "percent_equity", "equity_percent"

    if series_name in {"drawdown", "intraday_drawdown"}:
        dd_values = _drawdown_from_equity(view.equity)
        frame = pl.DataFrame(dd_values, schema=["timestamp", "drawdown"]) if dd_values else pl.DataFrame(
            {"timestamp": [], "drawdown": []}
        )
        return frame, "drawdown", "drawdown"

    if series_name == "netpos":
        return view.net_position, "net_position", "netpos"

    if series_name == "margin":
        return view.margin, "margin_used", "margin"

    if series_name == "daily_returns":
        returns = view.daily_returns.select(pl.col("date").alias("timestamp"), pl.col("daily_return"))
        return returns, "daily_return", "daily_returns"

    raise ValueError(f"Unsupported series {series_name}")


def _frame_from_contributor(series_name: str, contributor: ContributorSeries):
    bundle = contributor.bundle
    if series_name == "equity":
        return bundle.equity, "equity"
    if series_name in {"equity_percent", "equity-percent"}:
        return bundle.percent_equity, "percent_equity"
    if series_name in {"drawdown", "intraday_drawdown"}:
        equity = bundle.equity
        dd_values = _drawdown_from_equity(equity)
        frame = pl.DataFrame(dd_values, schema=["timestamp", "drawdown"]) if dd_values else pl.DataFrame({"timestamp": [], "drawdown": []})
        return frame, "drawdown"
    if series_name == "netpos":
        return bundle.net_position, "net_position"
    if series_name == "margin":
        margin = bundle.margin.select(pl.col("timestamp"), pl.col("margin_used"))
        return margin, "margin_used"
    if series_name == "daily_returns":
        returns = bundle.daily_returns.select(pl.col("date").alias("timestamp"), pl.col("daily_return"))
        return returns, "daily_return"
    return None


def _build_per_file(series_name: str, view, downsample: bool, target_points: int) -> list[SeriesContributor]:
    lines: list[SeriesContributor] = []
    for contributor in view.contributors:
        frame_info = _frame_from_contributor(series_name, contributor)
        if frame_info is None:
            continue
        frame, value_col = frame_info
        trimmed, _, _ = _downsample_frame(frame, "timestamp", value_col, downsample, target_points)
        points = _to_points(trimmed, "timestamp", value_col)
        lines.append(
            SeriesContributor(
                contributor_id=contributor.file_id,
                label=contributor.label or contributor.file_id,
                symbol=contributor.symbol,
                interval=contributor.interval,
                strategy=contributor.strategy,
                points=points,
            )
        )
    return lines


def _build_series_response(series_name: str, view, selection: Selection, downsample: bool, target_points: int) -> SeriesResponse:
    frame, value_col, label = _portfolio_frame(series_name, view)
    trimmed, raw, sampled = _downsample_frame(frame, "timestamp", value_col, downsample, target_points)
    portfolio_points = _to_points(trimmed, "timestamp", value_col)
    per_file = _build_per_file(series_name, view, downsample, target_points)
    return SeriesResponse(
        series=label,
        selection=selection,
        downsampled=downsample,
        raw_count=raw,
        downsampled_count=sampled,
        portfolio=portfolio_points,
        per_file=per_file,
    )


@router.get("/series/equity", response_model=SeriesResponse)
def get_equity(downsample: bool = Query(True), target_points: int = Query(500)) -> SeriesResponse:
    cache = PerFileCache()
    aggregator = PortfolioAggregator(cache)
    files: List[Path] = list_sample_trade_files()
    view = aggregator.aggregate(files)
    selection = Selection(files=[p.stem for p in files])
    return _build_series_response("equity", view, selection, downsample, target_points)


@router.get("/series/netpos", response_model=SeriesResponse)
def get_netpos(downsample: bool = Query(True), target_points: int = Query(500)) -> SeriesResponse:
    cache = PerFileCache()
    files: List[Path] = list_sample_trade_files()
    aggregator = PortfolioAggregator(cache)
    view = aggregator.aggregate(files)
    selection = Selection(files=[p.stem for p in files])
    return _build_series_response("netpos", view, selection, downsample, target_points)


@router.get("/series/margin", response_model=SeriesResponse)
def get_margin(downsample: bool = Query(True), target_points: int = Query(500)) -> SeriesResponse:
    cache = PerFileCache()
    files: List[Path] = list_sample_trade_files()
    aggregator = PortfolioAggregator(cache)
    view = aggregator.aggregate(files)
    selection = Selection(files=[p.stem for p in files])
    return _build_series_response("margin", view, selection, downsample, target_points)


@router.get("/series/daily-returns", response_model=SeriesResponse)
def get_daily_returns(downsample: bool = Query(True), target_points: int = Query(500)) -> SeriesResponse:
    cache = PerFileCache()
    files: List[Path] = list_sample_trade_files()
    aggregator = PortfolioAggregator(cache)
    view = aggregator.aggregate(files)
    selection = Selection(files=[p.stem for p in files])
    return _build_series_response("daily_returns", view, selection, downsample, target_points)
