# API Agent Guide

## Scope
Use this file for backend-only work in `api/` or the backend half of fullstack changes. Repo-wide defaults still live at root `AGENTS.md`.

## Backend Entry Points
- App bootstrap: `api/api/app/main.py`
- Route surfaces: `api/api/app/routes/`
- Query parsing + selection hashing: `api/api/app/dependencies.py`
- Request/response models: `api/api/app/schemas.py`
- Core orchestration: `api/api/app/services/data_store.py`
- Compute/cache internals: `api/api/app/compute/`
- Optimizer service: `api/api/app/services/riskfolio_optimizer.py`
- Ingest path: `api/api/app/ingest.py`

## Run Backend
From repo root:
```sh
cd api
python -m pip install -r requirements.txt
python -m uvicorn api.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Verify Backend Changes
From repo root:
```sh
python -m pytest tests/test_ingest.py tests/test_compute.py
```

## Backend Wiring Checklist
1. Update route handler in `api/api/app/routes/` for the HTTP surface.
2. Keep selection/query parsing aligned in `api/api/app/dependencies.py`.
3. Update schemas in `api/api/app/schemas.py` when response/request shape changes.
4. Implement compute/data behavior in `api/api/app/services/data_store.py` and/or `api/api/app/compute/`.
5. If API contract changed, update frontend callers in the same rollout (`web/lib/api.ts`, `web/app/page.tsx`).

## Repo-Specific Invariants
- `selection_hash` intentionally excludes `account_equity`, `start_date`, and `end_date`; preserve this unless cache-key behavior is intentionally changing.
- New selection query params should be normalized/validated in `api/api/app/dependencies.py` before use in routes/services.
- `DataStore` defaults `DATA_ROOT` to `./data`; cache artifacts are under `DATA_ROOT/.cache`.
- Cache backend uses `REDIS_URL` when available, otherwise in-memory fallback must still work.
- Correlation/export endpoints are currently stubbed (`501`) unless you implement those backend paths.

## Edit Guardrails
- Prefer minimal diffs and existing patterns in routes/services/compute.
- Avoid adding dependencies unless explicitly requested.
- Do not rename/move backend files unless requested.
