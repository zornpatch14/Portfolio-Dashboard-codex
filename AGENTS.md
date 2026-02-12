# Portfolio Dashboard Agent Guide

## What This Repo Is
This repository hosts a futures portfolio analytics dashboard with a FastAPI backend and a Next.js frontend. Users upload TradeStation-style trade lists, the API ingests and caches normalized datasets, and the web app renders portfolio analytics (equity, drawdown, margin, histogram, CTA, metrics, optimizer views) from live API calls.

## Canonical Guidance
- Always-on repo guidance: `AGENTS.md` (this file)
- Navigation map and wiring paths: `REPO_MAP.md`
- Skill for fast file discovery: `.agents/skills/repo-orientation/SKILL.md`
- Skill for cross-layer feature wiring: `.agents/skills/fullstack-wiring/SKILL.md`

## Architecture At A Glance
- Frontend: Next.js app in `web/` (`web/app/page.tsx` is the primary UI surface).
- Backend: FastAPI app in `api/api/app/` (`api/api/app/main.py` bootstraps routers).
- Persistence: Parquet + metadata index under `DATA_ROOT` (`api/api/app/ingest.py`, `api/api/app/services/data_store.py`).
- Compute/caching: Per-file parquet cache + selection-level cache (Redis or in-memory fallback).

## Where Things Live
- `docker-compose.yml`: local multi-service orchestration (`redis`, `api`, `web`)
- `api/api/app/main.py`: API app startup, CORS, CSV mirror worker, router registration
- `api/api/app/routes/`: HTTP route surfaces (`upload`, `files`, `series`, `metrics`, `cta`, `optimizer`, `correlations`, `exports`)
- `api/api/app/services/data_store.py`: core ingest + query + analytics orchestration
- `api/api/app/compute/`: portfolio aggregation, exposure, downsampling, per-file cache logic
- `api/api/app/services/riskfolio_optimizer.py`: optimizer job execution and job-state API
- `web/app/page.tsx`: main dashboard tabs and query wiring
- `web/lib/api.ts`: frontend API client and response/request types
- `tests/`: backend ingest/compute tests and sample XLSX test data
- `storage/` and `data/`: runtime/generated data caches (ignored by git)

## Run Locally
Recommended (all services):
```sh
docker compose up --build
```

Manual backend (from repo root):
```sh
cd api
python -m pip install -r requirements.txt
python -m uvicorn api.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Manual frontend (new terminal, from repo root):
```sh
cd web
npm install
# PowerShell:
$env:NEXT_PUBLIC_API_BASE="http://localhost:8000"
npm run dev
```

## Test, Lint, Typecheck, Build
- Backend tests:
```sh
python -m pytest tests/test_ingest.py tests/test_compute.py
```
- Frontend typecheck:
```sh
cd web
npx tsc --noEmit
```
- Frontend production build:
```sh
cd web
npm run build
```
- Frontend lint:
```sh
cd web
npm run lint
```
Currently prompts for ESLint setup interactively.
- Backend lint/typecheck/build: `Unknown`
Checked: `api/pyproject.toml`, `api/requirements.txt`, repo root for `Makefile`, `pytest.ini`, `tox.ini`, `.github/`.
What would confirm: committed lint/typecheck config and/or scripts.

## Conventions And Guardrails For Edits
- Prefer minimal diffs; do not refactor unrelated code.
- Reuse existing helpers/patterns in `data_store.py`, `compute/`, and `web/lib/api.ts`.
- Do not add dependencies unless explicitly requested.
- Do not rename/move files unless requested.
- When changing an API contract (route params, schema fields, response keys), update downstream callers in the same rollout (routes, service layer, `web/lib/api.ts`, and UI usage).

## Definition Of Done
- Change is scoped to requested behavior only.
- Affected backend path is validated (endpoint call path and no obvious schema mismatch).
- Run relevant checks:
  - Backend change: `python -m pytest tests/test_ingest.py tests/test_compute.py` (if `pytest` is installed).
  - Frontend change: `npx tsc --noEmit` and `npm run build` in `web/`.
- If API contracts changed, verify frontend compile/build still passes.
- Update docs/skills when repo structure or wiring changed.

## Common Gotchas
- `NEXT_PUBLIC_API_BASE` is required for live frontend API calls (`web/lib/api.ts` hard-fails without it).
- `web/next.config.mjs` disables lint/type errors during `npm run build`; build success does not imply lint/type clean.
- `/api/v1/correlations` and `/api/v1/export/*` currently return `501 Not Implemented`.
- `DATA_ROOT` controls backend read/write location; if unset, backend defaults to `./data`.
- Series/equity computations expect MTM parquet for files; missing MTM can raise errors from cache compute paths.
