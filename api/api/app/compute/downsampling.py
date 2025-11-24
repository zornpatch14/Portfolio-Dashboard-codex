from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl


@dataclass
class DownsampleResult:
    downsampled: pl.DataFrame
    raw_count: int
    downsampled_count: int


def downsample_timeseries(
    frame: pl.DataFrame,
    timestamp_col: str,
    value_col: str,
    target_points: int = 2000,
    method: Literal["lttb", "mean"] = "lttb",
) -> DownsampleResult:
    """Downsample a timeseries using a simple LTTB or mean bucket strategy."""

    raw_count = len(frame)
    if raw_count <= target_points:
        return DownsampleResult(frame, raw_count=raw_count, downsampled_count=raw_count)

    frame = frame.sort(timestamp_col)
    if method == "mean":
        buckets = np.array_split(np.arange(raw_count), target_points)
        indices = [int(bucket.mean()) for bucket in buckets if len(bucket)]
    else:
        indices = _lttb_indices(frame[value_col].to_numpy(), target_points)

    sampled = frame[indices]
    return DownsampleResult(sampled, raw_count=raw_count, downsampled_count=len(sampled))


def _lttb_indices(values: np.ndarray, threshold: int) -> list[int]:
    """
    Lightly adapted LTTB algorithm for numeric arrays.
    Returns indices of points to keep.
    """

    data_length = len(values)
    if threshold >= data_length or threshold == 0:
        return list(range(data_length))

    bucket_size = (data_length - 2) / (threshold - 2)
    a = 0
    sampled = [0]

    for i in range(0, threshold - 2):
        start = int(np.floor((i + 1) * bucket_size)) + 1
        end = int(np.floor((i + 2) * bucket_size)) + 1
        bucket = values[start:end]
        if not len(bucket):
            continue
        avg_x = (i + 1.5) * bucket_size
        avg_y = bucket.mean()

        range_start = int(np.floor(i * bucket_size)) + 1
        range_end = int(np.floor((i + 1) * bucket_size)) + 1
        point_range = values[range_start:range_end]
        point_xs = np.arange(range_start, range_start + len(point_range))

        area = np.abs((a - avg_x) * (point_range - avg_y))
        if len(area) == 0:
            continue
        a = point_xs[area.argmax()]
        sampled.append(int(a))

    sampled.append(data_length - 1)
    return sampled

