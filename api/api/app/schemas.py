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
    contract_multipliers: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-file contract multipliers (FILE_ID:VALUE)",
    )
    margin_overrides: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-file margin overrides (FILE_ID:VALUE)",
    )
    spike_flag: bool = False
    data_version: Optional[str] = None
    account_equity: float | None = None




class SelectionMeta(BaseModel):
    symbols: List[str]
    intervals: List[str]
    strategies: List[str]
    date_min: Optional[date]
    date_max: Optional[date]
    files: List[FileMetadata]
    account_equity: float



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

    start_value: float

    end_value: float





class HistogramResponse(BaseModel):

    label: str

    selection: Selection

    buckets: List[HistogramBucket]





class MetricsBlock(BaseModel):
    """Container for a cohesive set of related metrics."""

    key: str
    label: str
    file_id: Optional[str] = None
    metrics: Dict[str, float | int | str | None] = Field(default_factory=dict)



class MetricsResponse(BaseModel):
    """Structured metrics payload used by /api/v1/metrics.

    - `portfolio` holds the aggregated metrics for the active selection.
    - `files` lists per-trade-file metrics using the same keys.
    """

    selection: Selection
    portfolio: MetricsBlock
    files: List[MetricsBlock]





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





class BoundsConfig(BaseModel):
    default_min: float = 0.0
    default_max: float = 1.0
    overrides: Dict[str, tuple[float | None, float | None]] = Field(default_factory=dict)


class CapConfig(BaseModel):
    name: str
    max_weight: float


class MeanRiskSettings(BaseModel):
    objective: str = "Sharpe"
    risk_measure: str = "MV"
    return_model: str = "arithmetic"
    method_mu: str = "hist"
    method_cov: str = "hist"
    solver: Optional[str] = None
    risk_free_rate: float = 0.0
    risk_aversion: float = 2.0
    alpha: float = 0.05
    budget: float = 1.0
    bounds: BoundsConfig = Field(default_factory=BoundsConfig)
    symbol_caps: List[CapConfig] = Field(default_factory=list)
    strategy_caps: List[CapConfig] = Field(default_factory=list)
    efficient_frontier_points: int = 0
    max_risk: float | None = None
    min_return: float | None = None
    turnover_limit: float | None = None


class OptimizerJobRequest(BaseModel):
    selection: Selection
    constraints: Dict[str, float] = Field(default_factory=dict)
    objective: str = "mean_risk"
    tags: List[str] = Field(default_factory=list)
    mean_risk: MeanRiskSettings | None = Field(
        default=None, description="Mean-risk optimizer configuration"
    )


class OptimizerJobResponse(BaseModel):
    job_id: str
    status: str = "queued"
    message: str = "optimizer job accepted"


class AllocationRow(BaseModel):
    asset: str
    label: str
    weight: float
    contracts: float
    margin_per_contract: float | None = None


class ContractRow(BaseModel):
    asset: str
    contracts: float
    notional: float
    margin: float | None = None


class OptimizerSummary(BaseModel):
    objective: str
    risk_measure: str
    expected_return: float
    risk: float
    sharpe: float
    max_drawdown: float
    capital: float


class FrontierPoint(BaseModel):
    expected_return: float
    risk: float
    weights: Dict[str, float]


class BacktestSeries(BaseModel):
    label: str
    points: List[SeriesPoint] = Field(default_factory=list)


class OptimizerCorrelation(BaseModel):
    mode: str = "returns"
    labels: List[str]
    matrix: List[List[float]]


class JobResult(BaseModel):
    summary: Optional[OptimizerSummary] = None
    weights: List[AllocationRow] = Field(default_factory=list)
    contracts: List[ContractRow] = Field(default_factory=list)
    frontier: List[FrontierPoint] = Field(default_factory=list)
    backtest: List[BacktestSeries] = Field(default_factory=list)
    correlation: Optional[OptimizerCorrelation] = None


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

