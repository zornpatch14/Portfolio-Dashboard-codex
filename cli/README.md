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

Suggested approach: Typer or Click, using the OpenAPI client or simple HTTP calls. Read API base/token from env (`API_BASE_URL`,
 `API_TOKEN`).

## Usage

The CLI reads configuration from the `API_BASE_URL` and optional `API_TOKEN` environment variables. You can override them per
 command with `--base-url` and `--token` flags.

```
python -m cli --help
python -m cli ingest data/*.xlsx
python -m cli list-files
python -m cli selection-meta
python -m cli series --kind equity --files f1,f2 --symbols ES,NQ
python -m cli metrics --files f1
python -m cli correlations --mode returns --files f1,f2
python -m cli cta --files f1,f2
python -m cli optimizer allocator --files f1 --param target=vol
python -m cli export --kind trades --format csv --files f1 --output trades.csv
```
