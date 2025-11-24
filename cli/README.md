# CLI Skeleton

CLI will wrap the API so workflows can run headlessly (agents/humans). Implement commands matching the API surface:
- ingest (upload files)
- list-files/meta
- series (equity/drawdown/netpos/margin/histogram)
- metrics
- correlations
- cta
- optimizer (allocator/riskfolio) with job progress
- exports (CSV/Parquet)

Suggested approach: Typer or Click, using the OpenAPI client or simple HTTP calls. Read API base/token from env (`API_BASE_URL`, `API_TOKEN`).
