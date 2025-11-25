from __future__ import annotations

import tempfile
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List
from uuid import uuid4

import numpy as np
import polars as pl
from fastapi import HTTPException, UploadFile, status

from api.app.compute.caches import PerFileCache
from api.app.compute.downsampling import downsample_timeseries
from api.app.compute.portfolio import PortfolioAggregator, PortfolioView
from api.app.constants import get_contract_spec
from api.app.ingest import IngestService, TradeFileMetadata
from api.app.schemas import (
    FileMetadata,
    FileUploadResponse,
    HistogramBucket,
    HistogramResponse,
    MetricsResponse,
    MetricsRow,
    Selection,
    SelectionMeta,
    SeriesPoint,
    SeriesResponse,
)

logger = logging.getLogger(__name__)


def _meta_to_schema(meta: TradeFileMetadata) -> FileMetadata:
    spec = get_contract_spec(meta.symbol)
    intervals: list[str] = []
    if meta.interval is not None:
        intervals = [str(meta.interval)]
    return FileMetadata(
        file_id=meta.file_id,
        filename=meta.filename,
        symbols=[meta.symbol] if meta.symbol else [],
        intervals=intervals,
        strategies=[meta.strategy] if meta.strategy else [],
        date_min=meta.date_min.date() if isinstance(meta.date_min, datetime) else meta.date_min,
        date_max=meta.date_max.date() if isinstance(meta.date_max, datetime) else meta.date_max,
        mtm_available=meta.mtm_rows > 0,
        margin_per_contract=spec.initial_margin if spec else None,
        big_point_value=spec.big_point_value if spec else None,
    )


def _drawdown_from_equity(equity) -> list[tuple[datetime, float]]:
    values = []
    running_max = float("-inf")
    for ts, eq in zip(equity["timestamp"], equity["equity"]):
        eq_val = float(eq)
        running_max = max(running_max, eq_val)
        values.append((ts, eq_val - running_max))
    return values


def _max_drawdown(equity) -> float:
    dd = _drawdown_from_equity(equity)
    if not dd:
        return 0.0
    return float(min(val for _, val in dd))


class DataStore:
    """File-backed store that wraps ingest + basic compute."""

    def __init__(self, storage_root: str | Path | None = None) -> None:
        root = Path(storage_root or os.getenv("DATA_ROOT", "./data"))
        self.storage_root = root
        self.ingest_service = IngestService(storage_root=root)
        self.cache = PerFileCache(storage_dir=root / ".cache")
        self.aggregator = PortfolioAggregator(self.cache)

    # ---------------- ingest & metadata ----------------
    def ingest(self, files: Iterable[UploadFile]) -> FileUploadResponse:
        if not files:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")

        uploaded: List[FileMetadata] = []
        for upload in files:
            with tempfile.NamedTemporaryFile(suffix=f"_{upload.filename}", delete=False) as tmp:
                content = upload.file.read()
                tmp.write(content)
                tmp_path = Path(tmp.name)

            try:
                meta = self.ingest_service.ingest_file(tmp_path)
            except ValueError as exc:
                logger.warning("Ingest failed for %s: %s", upload.filename, exc)
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            except Exception as exc:
                logger.error("Unexpected ingest failure for %s: %s", upload.filename, exc)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to ingest {upload.filename}: {exc}",
                ) from exc
            uploaded.append(_meta_to_schema(meta))

        job_id = str(uuid4())
        return FileUploadResponse(job_id=job_id, files=uploaded, message="ingest completed")

    def list_files(self) -> list[FileMetadata]:
        return [_meta_to_schema(m) for m in self.ingest_service.index.all()]

    def get_file(self, file_id: str) -> FileMetadata:
        meta = self.ingest_service.index.get(file_id)
        if not meta:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
        return _meta_to_schema(meta)

    def selection_meta(self) -> SelectionMeta:
        files = self.list_files()
        symbols = sorted({s for f in files for s in f.symbols})
        intervals = sorted({i for f in files for i in f.intervals})
        strategies = sorted({s for f in files for s in f.strategies})
        date_min = min((f.date_min for f in files if f.date_min), default=None)
        date_max = max((f.date_max for f in files if f.date_max), default=None)
        return SelectionMeta(
            symbols=symbols,
            intervals=intervals,
            strategies=strategies,
            date_min=date_min,
            date_max=date_max,
            files=files,
        )

    # ---------------- helpers ----------------
    def _paths_for_selection(self, selection: Selection) -> list[Path]:
        """Filter available files using selection filters and return Parquet paths."""

        filtered = self._metas_for_selection(selection)
        return [Path(m.trades_path) for m in filtered]

    def _metas_for_selection(self, selection: Selection) -> list[TradeFileMetadata]:
        """Filter available files using selection filters and return metadata."""

        all_meta = self.ingest_service.index.all()
        if selection.files:
            # explicit ids take precedence
            filtered = []
            for fid in selection.files:
                meta = self.ingest_service.index.get(fid)
                if not meta:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File id {fid} not found")
                filtered.append(meta)
        else:
            filtered = all_meta

        symbols = set(selection.symbols or [])
        intervals = set(selection.intervals or [])
        strategies = set(selection.strategies or [])
        start_date = selection.start_date
        end_date = selection.end_date

        def keep(meta: TradeFileMetadata) -> bool:
            if symbols and meta.symbol and meta.symbol not in symbols:
                return False
            if intervals and meta.interval is not None and str(meta.interval) not in intervals:
                return False
            if strategies and meta.strategy and meta.strategy not in strategies:
                return False
            if start_date and meta.date_max and meta.date_max.date() < start_date:
                return False
            if end_date and meta.date_min and meta.date_min.date() > end_date:
                return False
            return True

        filtered = [m for m in filtered if keep(m)]
        if not filtered:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No files match selection filters")

        return filtered

    def _to_points(self, rows, ts_key: str, value_key: str) -> list[SeriesPoint]:
        points: list[SeriesPoint] = []
        for row in rows:
            ts = row[ts_key]
            if isinstance(ts, datetime):
                ts_val = ts
            else:
                # Polars date/datetime come through as Python types already
                ts_val = ts
            points.append(SeriesPoint(timestamp=ts_val, value=float(row[value_key])))
        return points

    def _view_for_selection(self, selection: Selection) -> PortfolioView:
        metas = self._metas_for_selection(selection)
        paths = [Path(m.trades_path) for m in metas]
        view = self.aggregator.aggregate(
            paths,
            metas=metas,
            contract_multipliers=selection.contract_multipliers,
            margin_overrides=selection.margin_overrides,
        )
        return self._filter_view_by_date(view, selection)

    def _filter_view_by_date(self, view: PortfolioView, selection: Selection) -> PortfolioView:
        start_dt = selection.start_date
        end_dt = selection.end_date
        if not start_dt and not end_dt:
            return view

        start_ts = datetime.combine(start_dt, datetime.min.time()) if start_dt else None
        end_ts = datetime.combine(end_dt, datetime.max.time()) if end_dt else None

        def filter_frame(frame: pl.DataFrame, column: str, is_date: bool = False) -> pl.DataFrame:
            out = frame
            start_bound = start_dt if is_date else start_ts
            end_bound = end_dt if is_date else end_ts
            if start_bound:
                out = out.filter(pl.col(column) >= start_bound)
            if end_bound:
                out = out.filter(pl.col(column) <= end_bound)
            return out

        view.equity = filter_frame(view.equity, "timestamp")
        view.percent_equity = filter_frame(view.percent_equity, "timestamp")
        view.net_position = filter_frame(view.net_position, "timestamp")
        view.margin = filter_frame(view.margin, "timestamp")
        if start_dt or end_dt:
            view.daily_returns = filter_frame(view.daily_returns, "date", is_date=True)
        return view

    # ---------------- series & metrics ----------------
    def series(self, series_name: str, selection: Selection, downsample: bool) -> SeriesResponse:
        view = self._view_for_selection(selection)

        def build_timeseries(frame: pl.DataFrame, value_col: str, label: str) -> SeriesResponse:
            raw = len(frame)
            sampled = raw
            if downsample:
                result = downsample_timeseries(frame, "timestamp", value_col, target_points=2000)
                frame = result.downsampled
                raw = result.raw_count
                sampled = result.downsampled_count
            data = self._to_points(frame.iter_rows(named=True), "timestamp", value_col)
            return SeriesResponse(
                series=label,
                selection=selection,
                downsampled=downsample,
                raw_count=raw,
                downsampled_count=sampled,
                data=data,
            )

        if series_name == "equity":
            return build_timeseries(view.equity, "equity", "equity")

        if series_name == "equity_percent":
            return build_timeseries(view.percent_equity, "percent_equity", "equity_percent")

        if series_name in {"drawdown", "intraday_drawdown"}:
            dd_values = _drawdown_from_equity(view.equity)
            frame = pl.DataFrame(dd_values, schema=["timestamp", "drawdown"]) if dd_values else pl.DataFrame(
                {"timestamp": [], "drawdown": []}
            )
            return build_timeseries(frame, "drawdown", "drawdown")

        if series_name == "netpos":
            return build_timeseries(view.net_position, "net_position", "netpos")

        if series_name == "margin":
            return build_timeseries(view.margin, "margin_used", "margin")

        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Series not implemented")

    def histogram(self, selection: Selection, bins: int = 20) -> HistogramResponse:
        view = self._view_for_selection(selection)
        returns = view.daily_returns

        if returns.is_empty():
            return HistogramResponse(label="return_distribution", selection=selection, buckets=[])

        values = returns["daily_return"].to_numpy()
        hist, edges = np.histogram(values, bins=bins)

        buckets: list[HistogramBucket] = []
        for count, start, end in zip(hist, edges[:-1], edges[1:]):
            label = f"{start * 100:.2f}% to {end * 100:.2f}%"
            buckets.append(HistogramBucket(bucket=label, count=int(count)))

        return HistogramResponse(label="return_distribution", selection=selection, buckets=buckets)

    def metrics(self, selection: Selection) -> MetricsResponse:
        metas = self._metas_for_selection(selection)
        paths = [Path(m.trades_path) for m in metas]
        rows: list[MetricsRow] = []

        # Per-file metrics
        for meta in metas:
            path = Path(meta.trades_path)
            symbol = (meta.symbol or "").upper()
            cmult = selection.contract_multipliers.get(symbol)
            marg = selection.margin_overrides.get(symbol)

            equity = self.cache.equity_curve(path, contract_multiplier=cmult, margin_override=marg)
            trades_df = self.cache.load_trades(path).trades
            daily = self.cache.daily_returns(path, contract_multiplier=cmult, margin_override=marg)

            net_profit = float(equity["equity"][-1] - self.cache.starting_equity) if len(equity) else 0.0
            trades = len(trades_df)
            wins = int((trades_df["net_profit"] > 0).sum()) if trades else 0
            win_rate = (wins / trades * 100.0) if trades else 0.0
            expectancy = (net_profit / trades) if trades else 0.0
            max_dd = _max_drawdown(equity)
            avg_daily = float(daily["daily_return"].mean()) if len(daily) else 0.0

            def add(metric: str, value: float) -> None:
                rows.append(MetricsRow(file_id=path.stem, metric=metric, value=value, level="file"))

            add("net_profit", net_profit)
            add("trades", float(trades))
            add("max_drawdown", max_dd)
            add("win_rate", win_rate)
            add("expectancy", expectancy)
            add("avg_daily_return", avg_daily)

        # Portfolio metrics
        view = self._view_for_selection(selection)
        if len(view.equity):
            portfolio_net = float(view.equity["equity"][-1] - self.cache.starting_equity)
            port_dd = _max_drawdown(view.equity)
            port_daily = float(view.daily_returns["daily_return"].mean()) if len(view.daily_returns) else 0.0
            def addp(metric: str, value: float) -> None:
                rows.append(MetricsRow(file_id="portfolio", metric=metric, value=value, level="portfolio"))

            addp("net_profit", portfolio_net)
            addp("trades", float(sum(len(self.cache.load_trades(p).trades) for p in paths)))
            addp("max_drawdown", port_dd)
            addp("avg_daily_return", port_daily)

        return MetricsResponse(selection=selection, rows=rows)
store = DataStore()
