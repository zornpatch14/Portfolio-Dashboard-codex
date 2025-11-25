from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from api.app.compute.caches import PerFileCache
from api.app.compute.downsampling import DownsampleResult, downsample_timeseries


@dataclass
class PortfolioView:
    equity: pl.DataFrame
    daily_returns: pl.DataFrame
    net_position: pl.DataFrame
    margin: pl.DataFrame
    contributors: list[Path]


class PortfolioAggregator:
    def __init__(self, cache: PerFileCache) -> None:
        self.cache = cache

    def aggregate(self, files: Iterable[Path]) -> PortfolioView:
        files = list(files)
        if not files:
            empty = pl.DataFrame({"timestamp": [], "value": []})
            return PortfolioView(
                equity=empty,
                daily_returns=empty,
                net_position=empty,
                margin=empty,
                contributors=[],
            )

        daily_frames = [self.cache.daily_returns(path) for path in files]
        combined_daily = pl.concat(daily_frames).group_by("date").agg(
            pnl=pl.col("pnl").sum(),
            capital=pl.col("capital").sum(),
        )
        combined_daily = combined_daily.with_columns(
            pl.when(pl.col("capital") > 0)
            .then(pl.col("pnl") / pl.col("capital"))
            .otherwise(0)
            .alias("daily_return")
        ).sort("date")

        starting_equity = self.cache.starting_equity
        equity_curve = combined_daily.select(
            pl.col("date").alias("timestamp"),
            (pl.col("pnl").cum_sum() + starting_equity).alias("equity"),
        )

        netpos_frames = [self.cache.net_position(path) for path in files]
        netpos = _combine_timeseries(netpos_frames, "net_position")

        margin_frames = [self.cache.margin_usage(path) for path in files]
        margin = _combine_timeseries(margin_frames, "margin_used")

        return PortfolioView(
            equity=equity_curve,
            daily_returns=combined_daily,
            net_position=netpos,
            margin=margin,
            contributors=files,
        )

    def downsample_equity(self, view: PortfolioView, target_points: int = 2000) -> DownsampleResult:
        return downsample_timeseries(view.equity, "timestamp", "equity", target_points=target_points)


def _combine_timeseries(frames: list[pl.DataFrame], value_col: str) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame({"timestamp": [], value_col: []})

    stacked = pl.concat(frames).sort("timestamp")
    combined = stacked.group_by("timestamp").agg(pl.col(value_col).sum()).sort("timestamp")
    return combined

