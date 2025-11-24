from __future__ import annotations

from typing import Optional

from fastapi import Query

from .schemas import Selection


class SelectionQueryParams:
    def __init__(
        self,
        files: list[str] = Query(default_factory=list, description="File identifiers to include"),
        symbols: list[str] = Query(default_factory=list, description="Symbols to filter"),
        intervals: list[str] = Query(default_factory=list, description="Intervals to filter"),
        strategies: list[str] = Query(default_factory=list, description="Strategies to filter"),
        direction: Optional[str] = Query(default=None, description="Direction filter"),
        start_date: Optional[str] = Query(default=None, description="Inclusive start date (YYYY-MM-DD)"),
        end_date: Optional[str] = Query(default=None, description="Inclusive end date (YYYY-MM-DD)"),
        spike_flag: bool = Query(default=False, description="Whether to include spike filter"),
        data_version: Optional[str] = Query(default=None, description="Data version key"),
    ):
        self.selection = Selection(
            files=files,
            symbols=symbols,
            intervals=intervals,
            strategies=strategies,
            direction=direction,
            start_date=start_date,
            end_date=end_date,
            spike_flag=spike_flag,
            data_version=data_version,
        )

    def __call__(self) -> Selection:
        return self.selection


class DownsampleFlag:
    def __init__(self, downsample: bool = Query(default=True, description="Downsample series output")):
        self.downsample = downsample

    def __call__(self) -> bool:
        return self.downsample
