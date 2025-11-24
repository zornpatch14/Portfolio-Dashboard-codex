from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, List
from uuid import uuid4

import polars as pl
from fastapi import HTTPException, UploadFile, status

from api.app.compute.caches import PerFileCache
from api.app.compute.downsampling import downsample_timeseries
from api.app.compute.portfolio import PortfolioAggregator
from api.app.ingest import IngestService, TradeFileMetadata
from api.app.schemas import (
    FileMetadata,
    FileUploadResponse,
    MetricsResponse,
    MetricsRow,
    Selection,
    SelectionMeta,
    SeriesPoint,
    SeriesResponse,
)


def _meta_to_schema(meta: TradeFileMetadata) -> FileMetadata:
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
    )


def _drawdown_from_equity(equity) -> list[tuple[datetime, float]]:
    values = []
    running_max = float("-inf")
    for ts, eq in zip(equity["timestamp"], equity["equity"]):
        eq_val = float(eq)
        running_max = max(running_max, eq_val)
        values.append((ts, eq_val - running_max))
    return values


class DataStore:
    """File-backed store that wraps ingest + basic compute."""

    def __init__(self, storage_root: str | Path | None = None) -> None:
        self.ingest_service = IngestService(storage_root=storage_root)
        self.cache = PerFileCache(storage_dir=Path(storage_root or ".") / ".cache")
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

            meta = self.ingest_service.ingest_file(tmp_path)
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
        ids = selection.files or [m.file_id for m in self.ingest_service.index.all()]
        metas = []
        for fid in ids:
            meta = self.ingest_service.index.get(fid)
            if not meta:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File id {fid} not found")
            metas.append(meta)
        return [Path(m.trades_path) for m in metas]

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

    # ---------------- series & metrics ----------------
    def series(self, series_name: str, selection: Selection, downsample: bool) -> SeriesResponse:
        paths = self._paths_for_selection(selection)
        view = self.aggregator.aggregate(paths)

        if series_name == "equity":
            frame = view.equity
            if downsample:
                result = downsample_timeseries(frame, "timestamp", "equity", target_points=2000)
                frame = result.downsampled
                raw = result.raw_count
                sampled = result.downsampled_count
            else:
                raw = len(frame)
                sampled = len(frame)
            data = self._to_points(frame.iter_rows(named=True), "timestamp", "equity")
            return SeriesResponse(
                series="equity",
                selection=selection,
                downsampled=downsample,
                raw_count=raw,
                downsampled_count=sampled,
                data=data,
            )

        if series_name in {"drawdown", "intraday_drawdown"}:
            values = _drawdown_from_equity(view.equity)
            raw = len(values)
            sampled = raw
            if downsample and raw > 0:
                ts = [v[0] for v in values]
                dd = [v[1] for v in values]
                frame = view.equity.with_columns(
                    pl.Series(name="drawdown", values=dd)
                )
                result = downsample_timeseries(frame, "timestamp", "drawdown", target_points=2000)
                values = list(zip(result.downsampled["timestamp"], result.downsampled["drawdown"]))
                sampled = result.downsampled_count
            data = [SeriesPoint(timestamp=ts, value=float(val)) for ts, val in values]
            return SeriesResponse(
                series="drawdown",
                selection=selection,
                downsampled=downsample,
                raw_count=raw,
                downsampled_count=sampled,
                data=data,
            )

        if series_name == "netpos":
            frame = view.net_position
            data = self._to_points(frame.iter_rows(named=True), "timestamp", "net_position")
            return SeriesResponse(
                series="netpos",
                selection=selection,
                downsampled=False,
                raw_count=len(data),
                downsampled_count=len(data),
                data=data,
            )

        if series_name == "margin":
            frame = view.margin
            data = self._to_points(frame.iter_rows(named=True), "timestamp", "margin_used")
            return SeriesResponse(
                series="margin",
                selection=selection,
                downsampled=False,
                raw_count=len(data),
                downsampled_count=len(data),
                data=data,
            )

        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Series not implemented")

    def metrics(self, selection: Selection) -> MetricsResponse:
        paths = self._paths_for_selection(selection)
        rows: list[MetricsRow] = []

        # Per-file metrics
        for path in paths:
            bundle = self.cache.equity_curve(path)
            net_profit = float(bundle["equity"][-1] - self.cache.starting_equity) if len(bundle) else 0.0
            trades = len(self.cache.load_trades(path).trades)
            rows.append(
                MetricsRow(
                    file_id=path.stem,
                    metric="net_profit",
                    value=net_profit,
                    level="file",
                )
            )
            rows.append(
                MetricsRow(
                    file_id=path.stem,
                    metric="trades",
                    value=float(trades),
                    level="file",
                )
            )

        # Portfolio metric
        view = self.aggregator.aggregate(paths)
        if len(view.equity):
            portfolio_net = float(view.equity["equity"][-1] - self.cache.starting_equity)
            rows.append(
                MetricsRow(
                    file_id="portfolio",
                    metric="net_profit",
                    value=portfolio_net,
                    level="portfolio",
                )
            )

        return MetricsResponse(selection=selection, rows=rows)


store = DataStore(storage_root=Path(".") / "data")
