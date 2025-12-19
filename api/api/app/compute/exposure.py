from __future__ import annotations

from typing import Iterable

import polars as pl


def _with_group_over(expr: pl.Expr, group_cols: list[str] | None) -> pl.Expr:
    if group_cols:
        return expr.over(group_cols)
    return expr


def build_event_deltas(
    intervals: pl.DataFrame,
    *,
    group_cols: list[str] | None = None,
    start_col: str = "start",
    value_col: str = "net_position",
) -> pl.DataFrame:
    """Convert interval states into event deltas (one delta per interval start)."""

    if intervals.is_empty():
        schema = {"timestamp": [], "delta": []}
        if group_cols:
            schema = {**{col: [] for col in group_cols}, **schema}
        return pl.DataFrame(schema)

    cols = [start_col, value_col]
    if group_cols:
        cols = [*group_cols, *cols]

    ordered = intervals.select(cols).with_row_count("_row")
    sort_cols = [start_col, "_row"]
    if group_cols:
        sort_cols = [*group_cols, *sort_cols]
    ordered = ordered.sort(sort_cols)

    prev_val = _with_group_over(pl.col(value_col).shift(1), group_cols).fill_null(0)
    ordered = ordered.with_columns((pl.col(value_col) - prev_val).alias("delta"))
    ordered = ordered.rename({start_col: "timestamp"})

    out_cols = ["timestamp", "delta"]
    if group_cols:
        out_cols = [*group_cols, *out_cols]
    return ordered.select(out_cols)


def coalesce_event_deltas(
    events: pl.DataFrame,
    *,
    group_cols: list[str] | None = None,
    timestamp_col: str = "timestamp",
    delta_col: str = "delta",
) -> pl.DataFrame:
    """Coalesce event deltas by timestamp (and optional group columns)."""

    if events.is_empty():
        schema = {timestamp_col: [], delta_col: []}
        if group_cols:
            schema = {**{col: [] for col in group_cols}, **schema}
        return pl.DataFrame(schema)

    group_keys = [timestamp_col]
    if group_cols:
        group_keys = [*group_cols, *group_keys]

    return events.group_by(group_keys).agg(pl.col(delta_col).sum().alias(delta_col)).sort(group_keys)


def rebuild_positions(
    events: pl.DataFrame,
    *,
    group_cols: list[str] | None = None,
    timestamp_col: str = "timestamp",
    delta_col: str = "delta",
    value_col: str = "net_position",
) -> pl.DataFrame:
    """Rebuild stepwise positions from coalesced deltas."""

    if events.is_empty():
        schema = {timestamp_col: [], value_col: []}
        if group_cols:
            schema = {**{col: [] for col in group_cols}, **schema}
        return pl.DataFrame(schema)

    sort_cols = [timestamp_col]
    if group_cols:
        sort_cols = [*group_cols, *sort_cols]

    ordered = events.sort(sort_cols)
    pos_expr = _with_group_over(pl.col(delta_col).cum_sum(), group_cols).alias(value_col)
    ordered = ordered.with_columns(pos_expr)

    out_cols = [timestamp_col, value_col]
    if group_cols:
        out_cols = [*group_cols, *out_cols]
    return ordered.select(out_cols)


def build_step_points_from_intervals(
    intervals: pl.DataFrame,
    *,
    group_cols: list[str] | None = None,
    start_col: str = "start",
    end_col: str = "end",
    value_col: str = "net_position",
) -> pl.DataFrame:
    """Emit stepwise points (start + end) after delta-coalescing."""

    if intervals.is_empty():
        schema = {"timestamp": [], value_col: []}
        if group_cols:
            schema = {**{col: [] for col in group_cols}, **schema}
        return pl.DataFrame(schema)

    events = build_event_deltas(intervals, group_cols=group_cols, start_col=start_col, value_col=value_col)
    events = coalesce_event_deltas(events, group_cols=group_cols, timestamp_col="timestamp", delta_col="delta")
    positions = rebuild_positions(events, group_cols=group_cols, timestamp_col="timestamp", delta_col="delta", value_col=value_col)

    if group_cols:
        ends = intervals.group_by(group_cols).agg(pl.col(end_col).max().alias("_end"))
        positions = positions.join(ends, on=group_cols, how="left")
        next_ts = pl.col("timestamp").shift(-1).over(group_cols)
    else:
        last_end = intervals[end_col].max()
        positions = positions.with_columns(pl.lit(last_end).alias("_end"))
        next_ts = pl.col("timestamp").shift(-1)

    positions = positions.with_columns(next_ts.alias("_next"))
    positions = positions.with_columns(
        pl.when(pl.col("_next").is_not_null())
        .then(pl.col("_next"))
        .otherwise(pl.col("_end"))
        .alias("_end_ts")
    )

    points = positions.with_columns(pl.concat_list([pl.col("timestamp"), pl.col("_end_ts")]).alias("timestamp"))
    points = points.explode("timestamp")

    drop_cols = ["_end", "_next", "_end_ts"]
    return points.drop(drop_cols)


__all__ = [
    "build_event_deltas",
    "coalesce_event_deltas",
    "rebuild_positions",
    "build_step_points_from_intervals",
]
