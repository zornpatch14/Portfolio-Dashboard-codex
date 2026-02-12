---
name: fullstack-wiring
description: Make end-to-end changes across FastAPI backend and Next.js frontend in this Portfolio Dashboard repo without missing layers. Use when adding/changing API-backed features that touch routes, schemas, data plumbing, client types, and UI charts/views.
---

# Fullstack Wiring Workflow

## When To Use
- Use for feature work that spans backend + frontend.
- Use when changing API contracts (query params, payloads, response fields).
- Use when wiring new analytics outputs into dashboard tabs/charts.

## When Not To Use
- Do not use for backend-only internal refactors with no API/UI impact.
- Do not use for frontend-only visual tweaks with unchanged API contracts.

## Steps
1. Find backend surface.
2. Find frontend/client surface.
3. Find state/store/data plumbing.
4. Find charts/views consuming the data.
5. Update shared types/contracts.
6. Add or adjust tests (if coverage exists).
7. Verify end-to-end.

## Step Details
1. Find backend surface:
```sh
rg -n "APIRouter|@router\\.|include_router|response_model" api/api/app/main.py api/api/app/routes
```
2. Find frontend/client surface:
```sh
rg -n "fetchSeries|fetchMetrics|fetchHistogram|fetchCta|submitRiskfolioJob|fetchJobStatus" web/lib/api.ts web/app/page.tsx
```
3. Find state/store/data plumbing:
```sh
rg -n "Selection|selection_hash|get_selection|DataStore|PerFileCache|PortfolioAggregator" api/api/app web/lib
```
4. Find charts/views:
```sh
rg -n "tabs|render[A-Za-z]+\\(|Chart|Grid|Heatmap" web/app/page.tsx web/components
```
5. Update contracts in both layers:
- Backend: `api/api/app/schemas.py`, `api/api/app/dependencies.py`, affected route/service files.
- Frontend: `web/lib/api.ts`, `web/lib/types/selection.ts`, `web/app/page.tsx` and affected components.
6. Add/adjust tests:
- Existing test targets: `tests/test_ingest.py`, `tests/test_compute.py`.
- Add focused tests only where current patterns already exist.
7. Verify:
```sh
python -m pytest tests/test_ingest.py tests/test_compute.py
cd web
npx tsc --noEmit
npm run build
```

## Do-Not-Forget Checklist (Repo-Specific)
- Query param parsing lives in `api/api/app/dependencies.py`; keep new params normalized and validated there.
- Selection hash excludes transient fields; confirm caching behavior when changing selection fields.
- `DATA_ROOT` and cache paths (`DATA_ROOT/.cache/per_file`) must stay coherent.
- `REDIS_URL` is optional; behavior must remain correct with in-memory fallback.
- `NEXT_PUBLIC_API_BASE` must be set for live frontend calls.
- Correlation/export endpoints currently return 501; avoid wiring UI paths as if implemented unless implementing backend too.
- If response field naming differs (`per_file` vs `perFile`), keep compatibility normalization in `web/lib/api.ts`.

## Minimal-Diff Reminder
- Change only required files.
- Avoid renames/moves unless requested.
- Keep existing request/response patterns and naming conventions.
