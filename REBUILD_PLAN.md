# Portfolio Dashboard Rebuild Plan

Authoritative plan for the multi-agent refactor from Dash to the API-first Next.js + FastAPI stack. This file is the single source of truth for contracts, scope, and milestones. Update it whenever interfaces change.

## Goals
- Faster, columnar compute for large trade sets (Polars + Arrow/Parquet).
- API-first: every UI action available via REST + CLI, with OpenAPI docs.
- Downsampled, cached series by default; raw available for exports/parity.
- Robust caching: per-file artifacts reused across selections; selection hash only recomputes cheap portfolio aggregation.
- Async for heavy work (optimizers, correlations) with SSE progress.
- Parity with current app outputs on a baseline selection matrix.
- CI-enforced quality (lint/type/tests/contracts/perf smoke).
- Legacy code note: keep existing Dash/`Portfolio_Dashboard7.py`/`src/` files as reference only. Do not delete or modify them unless explicitly needed for reference. All new implementation lives in `api/`, `web/`, `cli/`, `tests/`. Removal of legacy code happens later, after the new stack is validated.

## Stack (baseline)
- Front end: Next.js + React + TypeScript; ECharts; AG Grid; React Query; Tailwind.
- Backend: FastAPI + uvicorn; Polars + PyArrow + Parquet; Redis cache; Celery (or RQ) + Redis workers; SSE (WebSocket only if needed).
- Tooling: Poetry or uv; ruff + mypy + pytest; Docker; GitHub Actions; .env config; OpenAPI from FastAPI.

## Data model
- Per-file Parquet: parsed TradeStation Trades List + MTM.
  - Required columns (trades): File, Symbol, Interval, Strategy, entry_time, exit_time, direction, entry_type, net_profit, runup, drawdown_trade, gross_profit, commission, slippage, contracts, CumulativePL_raw, pct_profit_raw, notional_exposure.
  - MTM: File, mtm_date, mtm_net_profit.
  - Metadata index: file_id, filename, hash, symbol(s), interval(s), strategy(ies), date_min/max (exit_time), row counts, mtm presence.
- Selection tuple (hashed for combined caches):
  - files (list of file_ids), symbols, intervals, strategies, direction, start_date, end_date, contract_multipliers, margin_overrides, spike_flag, data_version.

## Caching rules
- Per-file caches (keyed by file_id [+ data_version]):
  - trades parquet, mtm parquet
  - daily returns, equity curve, percent-equity index, netpos series, margin usage series
- Combined caches (keyed by selection hash):
  - portfolio equity/percent-equity/drawdown/intraday-DD
  - per-file + portfolio metrics rows
  - netpos aggregation, margin aggregates, histogram bins
  - correlations, CTA tables
  - optimizer outputs (by selection + params)
- Downsampling: default LTTB/bucket to ~2k points; include raw_count and downsampled_count; allow downsample=false.

## API surface (v1)
- Upload: POST /api/v1/upload (multipart) → job id + file_ids; GET /api/v1/files; GET /api/v1/files/{id}.
- Selection helper: GET /api/v1/selection/meta (available symbols/intervals/strategies/date bounds for current files).
- Series:
  - GET /api/v1/series/equity
  - GET /api/v1/series/equity-percent
  - GET /api/v1/series/drawdown
  - GET /api/v1/series/intraday-dd
  - GET /api/v1/series/netpos
  - GET /api/v1/series/margin
  - GET /api/v1/series/histogram
  - All accept selection params + downsample flag; Arrow/Parquet for large payloads.
- Metrics: GET /api/v1/metrics (per-file + portfolio rows).
- Correlations: GET /api/v1/correlations (mode: drawdown_pct/returns/pl/slope).
- CTA: GET /api/v1/cta.
- Optimizers: POST /api/v1/optimizer/allocator; POST /api/v1/optimizer/riskfolio → job id; GET /api/v1/jobs/{id} for status/result; SSE stream at /api/v1/jobs/{id}/events.
- Exports: GET /api/v1/export/trades, /api/v1/export/metrics (CSV/Parquet).

## CLI/headless
- Python CLI package that wraps the API. Commands:
  - ingest (upload files)
  - list-files/meta
  - fetch series (equity/drawdown/netpos/margin/histogram)
  - metrics
  - correlations
  - cta
  - optimizer (allocator/riskfolio) with progress polling/stream
  - exports (CSV/Parquet)
- Uses same selection flags as API; reads API base URL + token from env.

## Testing and parity
- Unit: parsers, metrics, netpos, margin, CTA, correlations, selection filters, downsampler.
- Contract: OpenAPI schema snapshot; JSONSchema for major responses.
- Parity: run baseline selections (see tests/baseline/) through legacy outputs vs new API; compare hashes/metrics.
- E2E: Playwright for critical flows (upload → select → equity/drawdown/netpos/metrics/optimizer).
- Perf smoke: CI hits equity/drawdown/netpos endpoints on baseline data and enforces latency/size budgets.

## Performance targets (initial)
- P50 API latency on baseline selection with multiple files: < 300 ms for cached series; < 1.5 s for cold path per-file; optimizer async only.
- Payload default (downsampled): < 1 MB per series; raw available on request.
- Cache hit ratio goal: > 80% for repeated selections in session.

## Workstreams and owners
- A) Ingest layer: XLSX→Polars→Parquet, metadata index, file_id/hash.
- B) Core compute: per-file caches (daily returns/equity/percent/netpos/margin), portfolio aggregation, downsampling utils.
- C) API surface: FastAPI skeleton, Pydantic models, versioned routes, SSE jobs, exports.
- D) Front-end: Next.js skeleton, selection state, ECharts/AG Grid wiring to real endpoints.
- E) CLI/headless: Python CLI wrapping API; selection flags; streaming job progress.
- F) Parity/perf/E2E: baseline matrix, parity harness, perf smoke, Playwright.
- G) Integrator: rebases/merges, resolves overlaps, keeps plan and task board current, coordinates interface changes.

## Milestones
1) Skeletons: FastAPI with mock endpoints + OpenAPI; Next.js shell with stub data; CLI placeholder; Docker + CI scaffold.
2) Ingest + per-file caches: real XLSX→Parquet; metadata index; per-file daily/equity/percent/netpos/margin cached; basic series endpoints return real data; CLI can ingest and fetch equity.
3) Portfolio + metrics + downsampling: selection hashing; portfolio aggregation; metrics parity on baseline; downsample defaults; exports working.
4) Advanced: correlations, CTA, optimizers async with SSE; front end wired to live data; parity/perf passing; E2E green.
5) Hardening: observability (timings, cache hits/misses), config/envs, docs, polish CLI, stabilize contracts.

## Coordination rules
- Branches: feature/<area>. Small, scoped PRs.
- Update REBUILD_PLAN.md and TASK_BOARD.md with any interface or ownership change.
- CI must pass (lint/type/unit/contract/parity/perf smoke/E2E where applicable) before merge.
- Required PR check: `ci / build (pull_request)` must be green for auto-merge.
- Breaking contract changes require updating OpenAPI, schemas, and parity expectations.
- Integrator reviews cross-cutting changes and resolves conflicts.
