# Repo Map

## Directory Tree (Key Paths)
```text
.
|-- AGENTS.md
|-- REPO_MAP.md
|-- docker-compose.yml
|-- api/
|   |-- requirements.txt
|   |-- pyproject.toml
|   |-- api/app/
|   |   |-- main.py
|   |   |-- dependencies.py
|   |   |-- schemas.py
|   |   |-- routes/
|   |   |-- services/
|   |   |-- compute/
|   |   |-- ingest.py
|   |   `-- data/loader.py
|-- web/
|   |-- package.json
|   |-- next.config.mjs
|   |-- app/
|   |   |-- layout.tsx
|   |   `-- page.tsx
|   |-- lib/
|   |   |-- api.ts
|   |   |-- selections.ts
|   |   `-- types/selection.ts
|   `-- components/
|-- tests/
|   |-- test_ingest.py
|   |-- test_compute.py
|   `-- data/
|-- storage/         (runtime/cache outputs)
`-- data/            (runtime/cache outputs)
```

## Key Entry Points
- Runtime stack: `docker-compose.yml`
- API bootstrap: `api/api/app/main.py`
- Frontend bootstrap: `web/app/layout.tsx`
- Main UI page and tab logic: `web/app/page.tsx`
- Frontend API contract/client: `web/lib/api.ts`

## UI Surfaces
Defined in `web/app/page.tsx` tabs:
- `Load Trade Lists`: upload + selection + metadata preview + export options
- `Summary`
- `Equity Curves`
- `Portfolio Drawdown`
- `Margin`
- `Trade P/L Histogram`
- `Correlations`
- `Riskfolio`
- `CTA Report`
- `Metrics`

Primary chart/view components live in `web/components/`:
- `EquityMultiChart.tsx`
- `SeriesChart.tsx`
- `HistogramChart.tsx`
- `CorrelationHeatmap.tsx`
- `EfficientFrontierChart.tsx`
- `FrontierAllocationAreaChart.tsx`
- `CtaTimeSeriesChart.tsx`
- `MetricsGrid.tsx`

## API Surfaces
Registered in `api/api/app/main.py`:
- `GET /health`
- `POST /api/v1/upload` (`routes/upload.py`)
- `GET /api/v1/files`, `GET /api/v1/files/{file_id}`, `GET /api/v1/selection/meta` (`routes/files.py`)
- `GET /api/v1/series/equity`
- `GET /api/v1/series/equity-percent`
- `GET /api/v1/series/drawdown`
- `GET /api/v1/series/intraday-dd`
- `GET /api/v1/series/netpos`
- `GET /api/v1/series/margin`
- `GET /api/v1/series/histogram`
- `POST /api/v1/series/histogram/composite` (`routes/series.py`)
- `GET /api/v1/metrics` (`routes/metrics.py`)
- `GET /api/v1/cta` (`routes/cta.py`)
- `POST /api/v1/optimizer/riskfolio`, `GET /api/v1/jobs/{job_id}`, `GET /api/v1/jobs/{job_id}/events` (`routes/optimizer.py`)
- `GET /api/v1/correlations` (`routes/correlations.py`, currently returns 501)
- `GET /api/v1/export/trades`, `GET /api/v1/export/metrics` (`routes/exports.py`, currently return 501)

## Data Layer
- Ingest:
  - `api/api/app/ingest.py` parses XLSX trade lists and MTM sheets.
  - Writes to `DATA_ROOT/parquet/trades/*.parquet`, `DATA_ROOT/parquet/mtm/*.parquet`.
  - Metadata index at `DATA_ROOT/metadata/index.json`.
- Core store/orchestration:
  - `api/api/app/services/data_store.py` handles selection filtering, aggregation, metrics, histogram, CTA, and cache hydration.
- Compute/cache:
  - `api/api/app/compute/caches.py` writes per-file artifacts to `DATA_ROOT/.cache/per_file` (or `API_CACHE_DIR` override).
  - `api/api/app/compute/portfolio.py` combines per-file series into portfolio views.
  - `api/api/app/services/cache.py` uses `REDIS_URL` cache backend or in-memory fallback.
- Mirror process:
  - `api/api/app/main.py` startup worker + `api/api/app/utils/parquet_mirror.py` mirror parquet outputs to CSV under `CSV tests/`.

## Typical Wiring Paths
1. Upload path:
`web/app/page.tsx` upload action -> `web/lib/api.ts:uploadFiles` -> `POST /api/v1/upload` -> `DataStore.ingest` -> `IngestService.ingest_file` -> parquet + metadata.

2. Series chart path:
`web/app/page.tsx` query hooks -> `web/lib/api.ts:fetchSeries` -> `/api/v1/series/*` -> `DataStore.series` -> `PortfolioAggregator` + `PerFileCache` -> `SeriesResponse` -> chart components.

3. Optimizer path:
`web/app/page.tsx` run optimizer -> `submitRiskfolioJob` -> `/api/v1/optimizer/riskfolio` -> `RiskfolioJobManager` -> `MeanRiskOptimizer.optimize` -> `DataStore.returns_frame` -> job polling via `/api/v1/jobs/{job_id}`.

4. Histogram composite path (walk-forward):
UI builds multiple selections -> `fetchHistogramComposite` -> `/api/v1/series/histogram/composite` -> `DataStore.histogram_composite`.

## Glossary
- Selection: The filter payload (files/symbols/intervals/strategies/direction/date/contracts/margins/account equity) used across API calls.
- MTM: Mark-to-market daily series used for equity and daily return calculations.
- Per-file cache: Cached parquet artifacts for each file/override combination (`equity`, `daily_returns`, `daily_percent_returns`, intervals, spikes).
- Exposure view: Net position/margin shape selector (`portfolio_daily`, `portfolio_step`, `per_symbol`, `per_file`).
- Spike filter: Option to include or suppress spike overlays in some series paths.
- Walk-forward: UI mode that applies rolling in-sample/out-of-sample windows and can aggregate outputs across windows.
