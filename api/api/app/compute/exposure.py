from __future__ import annotations

from datetime import date, datetime, timedelta

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


def build_step_points_from_positions(
    positions: pl.DataFrame,
    *,
    end_times: pl.DataFrame | datetime | None,
    group_cols: list[str] | None = None,
    timestamp_col: str = "timestamp",
    value_col: str = "net_position",
    end_col: str = "end",
) -> pl.DataFrame:
    """Emit stepwise points from positions using end timestamps."""

    if positions.is_empty():
        schema = {timestamp_col: [], value_col: []}
        if group_cols:
            schema = {**{col: [] for col in group_cols}, **schema}
        return pl.DataFrame(schema)

    if group_cols:
        if isinstance(end_times, pl.DataFrame):
            positions = positions.join(end_times, on=group_cols, how="left")
        elif end_times is not None:
            positions = positions.with_columns(pl.lit(end_times).alias(end_col))
        else:
            raise ValueError("end_times is required for grouped step points")
        next_ts = pl.col(timestamp_col).shift(-1).over(group_cols)
    else:
        if isinstance(end_times, pl.DataFrame):
            end_val = end_times[end_col].max()
        else:
            end_val = end_times
        positions = positions.with_columns(pl.lit(end_val).alias(end_col))
        next_ts = pl.col(timestamp_col).shift(-1)

    positions = positions.with_columns(next_ts.alias("_next"))
    positions = positions.with_columns(
        pl.when(pl.col("_next").is_not_null())
        .then(pl.col("_next"))
        .otherwise(pl.col(end_col))
        .alias("_end_ts")
    )

    points = positions.with_columns(pl.concat_list([pl.col(timestamp_col), pl.col("_end_ts")]).alias(timestamp_col))
    points = points.explode(timestamp_col)

    drop_cols = [end_col, "_next", "_end_ts"]
    return points.drop(drop_cols)


def positions_to_intervals(
    positions: pl.DataFrame,
    *,
    end_times: pl.DataFrame | datetime,
    group_cols: list[str] | None = None,
    timestamp_col: str = "timestamp",
    value_col: str = "net_position",
    end_col: str = "end",
) -> pl.DataFrame:
    """Convert step positions into explicit start/end intervals."""

    if positions.is_empty():
        schema = {"start": [], "end": [], value_col: []}
        if group_cols:
            schema = {**{col: [] for col in group_cols}, **schema}
        return pl.DataFrame(schema)

    if group_cols:
        if not isinstance(end_times, pl.DataFrame):
            raise ValueError("Grouped intervals require end_times DataFrame")
        positions = positions.join(end_times, on=group_cols, how="left")
        next_ts = pl.col(timestamp_col).shift(-1).over(group_cols)
    else:
        end_val = end_times[end_col].max() if isinstance(end_times, pl.DataFrame) else end_times
        positions = positions.with_columns(pl.lit(end_val).alias(end_col))
        next_ts = pl.col(timestamp_col).shift(-1)

    positions = positions.with_columns(next_ts.alias("_next"))
    positions = positions.with_columns(
        pl.when(pl.col("_next").is_not_null())
        .then(pl.col("_next"))
        .otherwise(pl.col(end_col))
        .alias("end")
    )
    positions = positions.rename({timestamp_col: "start"})

    out_cols = ["start", "end", value_col]
    if group_cols:
        out_cols = [*group_cols, *out_cols]
    return positions.select(out_cols)


def build_per_file_stepwise(
    intervals: pl.DataFrame,
    *,
    file_col: str | None = None,
    start_col: str = "start",
    end_col: str = "end",
    value_col: str = "net_position",
) -> pl.DataFrame:
    """Stepwise series per file (or all intervals when file_col is None)."""

    group_cols = [file_col] if file_col and file_col in intervals.columns else None
    return build_step_points_from_intervals(
        intervals,
        group_cols=group_cols,
        start_col=start_col,
        end_col=end_col,
        value_col=value_col,
    )


def build_symbol_positions(
    intervals: pl.DataFrame,
    *,
    file_col: str | None = None,
    symbol_col: str = "symbol",
    start_col: str = "start",
    value_col: str = "net_position",
) -> pl.DataFrame:
    """Step positions per symbol by coalescing per-file deltas."""

    group_cols: list[str] = [symbol_col]
    if file_col and file_col in intervals.columns:
        file_group = [file_col, symbol_col]
    else:
        file_group = [symbol_col]

    events = build_event_deltas(intervals, group_cols=file_group, start_col=start_col, value_col=value_col)
    if file_col and file_col in events.columns:
        events = events.drop(file_col)
    events = coalesce_event_deltas(events, group_cols=group_cols, timestamp_col="timestamp", delta_col="delta")
    return rebuild_positions(events, group_cols=group_cols, timestamp_col="timestamp", delta_col="delta", value_col=value_col)


def build_symbol_stepwise(
    intervals: pl.DataFrame,
    *,
    file_col: str | None = None,
    symbol_col: str = "symbol",
    start_col: str = "start",
    end_col: str = "end",
    value_col: str = "net_position",
) -> pl.DataFrame:
    """Stepwise series per symbol from per-file intervals."""

    positions = build_symbol_positions(
        intervals,
        file_col=file_col,
        symbol_col=symbol_col,
        start_col=start_col,
        value_col=value_col,
    )
    end_times = intervals.group_by(symbol_col).agg(pl.col(end_col).max().alias("end"))
    return build_step_points_from_positions(
        positions,
        end_times=end_times,
        group_cols=[symbol_col],
        timestamp_col="timestamp",
        value_col=value_col,
        end_col="end",
    )


def build_portfolio_positions(
    symbol_positions: pl.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    value_col: str = "net_position",
) -> pl.DataFrame:
    """Aggregate symbol positions into portfolio exposure (sum of abs)."""

    if symbol_positions.is_empty():
        return pl.DataFrame({timestamp_col: [], value_col: []})

    return (
        symbol_positions.with_columns(pl.col(value_col).abs().alias("_abs"))
        .group_by(timestamp_col)
        .agg(pl.col("_abs").sum().alias(value_col))
        .sort(timestamp_col)
    )


def build_portfolio_stepwise(
    symbol_positions: pl.DataFrame,
    *,
    end_time: datetime | None,
    value_col: str = "net_position",
) -> pl.DataFrame:
    """Stepwise portfolio series (sum of abs per symbol)."""

    positions = build_portfolio_positions(symbol_positions, timestamp_col="timestamp", value_col=value_col)
    return build_step_points_from_positions(
        positions,
        end_times=end_time,
        group_cols=None,
        timestamp_col="timestamp",
        value_col=value_col,
        end_col="end",
    )


def bucket_intervals_daily_max(
    intervals: pl.DataFrame,
    *,
    start_col: str = "start",
    end_col: str = "end",
    value_col: str = "net_position",
) -> pl.DataFrame:
    """Bucket intervals by calendar day, taking max value per day."""

    if intervals.is_empty():
        return pl.DataFrame({"date": [], value_col: []})

    daily_max: dict[date, float] = {}
    for row in intervals.iter_rows(named=True):
        start = row.get(start_col)
        end = row.get(end_col)
        if start is None or end is None or start >= end:
            continue
        value = float(row.get(value_col) or 0.0)
        end_adj = end - timedelta(microseconds=1)
        current = start.date()
        last = end_adj.date()
        while current <= last:
            prev = daily_max.get(current)
            if prev is None or value > prev:
                daily_max[current] = value
            current = current + timedelta(days=1)

    if not daily_max:
        return pl.DataFrame({"date": [], value_col: []})

    first_day = min(daily_max)
    last_day = max(daily_max)
    days = []
    values = []
    current = first_day
    while current <= last_day:
        days.append(current)
        values.append(float(daily_max.get(current, 0.0)))
        current = current + timedelta(days=1)

    return pl.DataFrame({"date": days, value_col: values})


def build_portfolio_daily_max(
    symbol_positions: pl.DataFrame,
    *,
    end_time: datetime,
    value_col: str = "net_position",
) -> pl.DataFrame:
    """Daily max portfolio exposure from symbol positions."""

    positions = build_portfolio_positions(symbol_positions, timestamp_col="timestamp", value_col=value_col)
    intervals = positions_to_intervals(
        positions,
        end_times=end_time,
        group_cols=None,
        timestamp_col="timestamp",
        value_col=value_col,
        end_col="end",
    )
    return bucket_intervals_daily_max(intervals, start_col="start", end_col="end", value_col=value_col)


__all__ = [
    "build_event_deltas",
    "coalesce_event_deltas",
    "rebuild_positions",
    "build_step_points_from_intervals",
    "build_step_points_from_positions",
    "positions_to_intervals",
    "build_per_file_stepwise",
    "build_symbol_positions",
    "build_symbol_stepwise",
    "build_portfolio_positions",
    "build_portfolio_stepwise",
    "bucket_intervals_daily_max",
    "build_portfolio_daily_max",
]
