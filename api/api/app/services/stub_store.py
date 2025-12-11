from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple
from uuid import uuid4

from ..schemas import (
    CTARecord,
    CTAResponse,
    CorrelationResponse,
    FileMetadata,
    SeriesContributor,
    FileUploadResponse,
    JobStatusResponse,
    MetricsResponse,
    MetricsRow,
    Selection,
    SelectionMeta,
    SeriesPoint,
    SeriesResponse,
)


class InMemoryStore:
    def __init__(self) -> None:
        self.files: Dict[str, FileMetadata] = {}
        self.jobs: Dict[str, JobStatusResponse] = {}
        self._seed()

    def _seed(self) -> None:
        file_id = str(uuid4())
        meta = FileMetadata(
            file_id=file_id,
            filename="baseline_trades.parquet",
            symbols=["ES", "NQ"],
            intervals=["15", "60"],
            strategies=["trend", "mean_revert"],
            date_min=datetime.utcnow().date(),
            date_max=datetime.utcnow().date(),
            mtm_available=True,
        )
        self.files[file_id] = meta

    def ingest(self, filenames: Iterable[str]) -> FileUploadResponse:
        uploaded: List[FileMetadata] = []
        for name in filenames:
            file_id = str(uuid4())
            meta = FileMetadata(
                file_id=file_id,
                filename=name,
                symbols=["ES"],
                intervals=["30"],
                strategies=["allocator"],
                date_min=datetime.utcnow().date(),
                date_max=datetime.utcnow().date(),
                mtm_available=False,
            )
            self.files[file_id] = meta
            uploaded.append(meta)

        job_id = str(uuid4())
        self.jobs[job_id] = JobStatusResponse(job_id=job_id, status="completed", progress=100)
        return FileUploadResponse(job_id=job_id, files=uploaded)

    def list_files(self) -> List[FileMetadata]:
        return list(self.files.values())

    def get_file(self, file_id: str) -> FileMetadata:
        if file_id not in self.files:
            raise KeyError(file_id)
        return self.files[file_id]

    def selection_meta(self) -> SelectionMeta:
        files = self.list_files()
        symbols = sorted({symbol for meta in files for symbol in meta.symbols})
        intervals = sorted({interval for meta in files for interval in meta.intervals})
        strategies = sorted({strategy for meta in files for strategy in meta.strategies})
        date_min = min((meta.date_min for meta in files if meta.date_min), default=None)
        date_max = max((meta.date_max for meta in files if meta.date_max), default=None)
        return SelectionMeta(
            symbols=symbols,
            intervals=intervals,
            strategies=strategies,
            date_min=date_min,
            date_max=date_max,
            files=files,
        )

    def series(self, series_name: str, selection: Selection, downsample: bool) -> SeriesResponse:
        now = datetime.utcnow()
        raw_points = [SeriesPoint(timestamp=now - timedelta(minutes=i * 15), value=1000 + i * 10) for i in range(10)]
        data = raw_points[::2] if downsample else raw_points
        points = list(reversed(data))
        contributors = selection.files or list(self.files.keys())
        per_file: list[SeriesContributor] = []
        for idx, file_id in enumerate(contributors):
            meta = self.files.get(file_id)
            label = meta.filename if meta else file_id
            offset = 1.0 + idx * 0.05
            contributor_points = [
                SeriesPoint(timestamp=pt.timestamp, value=pt.value * offset) for pt in points
            ]
            per_file.append(
                SeriesContributor(
                    contributor_id=file_id,
                    label=label,
                    points=contributor_points,
                    symbol=None,
                    interval=None,
                    strategy=None,
                )
            )
        return SeriesResponse(
            series=series_name,
            selection=selection,
            downsampled=downsample,
            raw_count=len(raw_points),
            downsampled_count=len(points),
            portfolio=points,
            per_file=per_file,
        )

    def metrics(self, selection: Selection) -> MetricsResponse:
        rows = [
            MetricsRow(file_id=file_id, metric="net_profit", value=12345.67, level="file")
            for file_id in selection.files or self.files.keys()
        ]
        rows.append(MetricsRow(file_id="portfolio", metric="net_profit", value=23456.78, level="portfolio"))
        return MetricsResponse(selection=selection, rows=rows)

    def correlations(self, selection: Selection, mode: str) -> CorrelationResponse:
        labels = selection.files or list(self.files.keys())
        size = len(labels)
        matrix = [[1.0 if i == j else 0.25 for j in range(size)] for i in range(size)]
        return CorrelationResponse(selection=selection, mode=mode, matrix=matrix, labels=labels)

    def cta(self, selection: Selection) -> CTAResponse:
        records = [
            CTARecord(
                file_id=file_id,
                symbol="ES",
                interval="30",
                strategy="trend",
                score=0.8,
                description="Stub CTA score",
            )
            for file_id in selection.files or self.files.keys()
        ]
        return CTAResponse(selection=selection, records=records)

    def export_rows(self, selection: Selection, kind: str) -> Tuple[str, str]:
        filename = f"{kind}_export.csv"
        headers = "file_id,value"
        body = "\n".join(f"{file_id},100" for file_id in selection.files or self.files.keys())
        return filename, "\n".join([headers, body])


store = InMemoryStore()
