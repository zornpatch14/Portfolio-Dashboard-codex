from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    file_id: str
    filename: str
    symbols: List[str] = Field(default_factory=list)
    intervals: List[str] = Field(default_factory=list)
    strategies: List[str] = Field(default_factory=list)
    date_min: Optional[date] = None
    date_max: Optional[date] = None
    mtm_available: bool = False
    margin_per_contract: Optional[float] = None
    big_point_value: Optional[float] = None


class FileUploadResponse(BaseModel):
    job_id: str
    files: List[FileMetadata]
    message: str = "ingest scheduled"


class Selection(BaseModel):
    files: List[str] = Field(default_factory=list)
    symbols: List[str] = Field(default_factory=list)
    intervals: List[str] = Field(default_factory=list)
    strategies: List[str] = Field(default_factory=list)
    direction: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    contract_multipliers: Dict[str, float] = Field(default_factory=dict, description="Per-symbol contract multipliers (SYMBOL:VALUE)")
    margin_overrides: Dict[str, float] = Field(default_factory=dict, description="Per-symbol margin overrides (SYMBOL:VALUE)")
    spike_flag: bool = False
    data_version: Optional[str] = None


class SelectionMeta(BaseModel):
    symbols: List[str]
    intervals: List[str]
    strategies: List[str]
    date_min: Optional[date]
    date_max: Optional[date]
    files: List[FileMetadata]


class SeriesPoint(BaseModel):
    timestamp: datetime
    value: float


class SeriesContributor(BaseModel):
    contributor_id: str
    label: str
    symbol: Optional[str] = None
    interval: Optional[str] = None
    strategy: Optional[str] = None
    points: List[SeriesPoint] = Field(default_factory=list)


class SeriesResponse(BaseModel):
    series: str
    selection: Selection
    downsampled: bool = True
    raw_count: int = 0
    downsampled_count: int = 0
    portfolio: List[SeriesPoint] = Field(default_factory=list)
    per_file: List[SeriesContributor] = Field(default_factory=list)


class HistogramBucket(BaseModel):
    bucket: str
    count: int


class HistogramResponse(BaseModel):
    label: str
    selection: Selection
    buckets: List[HistogramBucket]


class MetricsRow(BaseModel):
    file_id: str
    metric: str
    value: float
    level: str = "file"


class MetricsResponse(BaseModel):
    selection: Selection
    rows: List[MetricsRow]


class CorrelationResponse(BaseModel):
    selection: Selection
    mode: str
    matrix: List[List[float]]
    labels: List[str]


class CTARecord(BaseModel):
    file_id: str
    symbol: str
    interval: str
    strategy: str
    score: float
    description: Optional[str] = None


class CTAResponse(BaseModel):
    selection: Selection
    records: List[CTARecord]


class OptimizerJobRequest(BaseModel):
    selection: Selection
    constraints: Dict[str, float] = Field(default_factory=dict)
    objective: str = "risk_parity"
    tags: List[str] = Field(default_factory=list)


class OptimizerJobResponse(BaseModel):
    job_id: str
    status: str = "queued"
    message: str = "optimizer job accepted"


class JobResult(BaseModel):
    weights: Dict[str, float] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    result: Optional[JobResult] = None
    error: Optional[str] = None


class JobEvent(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str


class ExportResponse(BaseModel):
    job_id: Optional[str] = None
    content_type: str
    filename: str
