# Dev Quickstart and Guardrails

This repo is being refactored to an API-first stack with multiple agents working in parallel. Follow these steps to keep a consistent environment and CI signal.

## Prereqs
- Docker and docker-compose
- Python 3.11+
- Node 20+ (for the Next.js front end)

## Environment
1) Copy `.env.example` to `.env` and adjust paths if needed.
2) Baseline data: place representative Trades List/MTM files under `tests/data/` (see `tests/baseline/README.md`).

## Run the stack locally (scaffold)
```sh
docker-compose up --build
```
- Services: `api` (FastAPI skeleton), `web` (Next.js skeleton), `redis`, `worker` (Celery placeholder).
- Mounts: `./api`, `./web`. Parquet/storage under `STORAGE_DIR` (`/data` in containers by default).

## CI expectations
- Protected main; all merges go through branches + CI.
- CI should run (to be expanded by agents):
  - lint/type/tests
  - contract snapshot (OpenAPI)
  - parity (baseline selections vs goldens)
  - perf smoke (equity/drawdown/netpos endpoints)
  - Playwright E2E for key flows

## Baseline/parity
- Keep small, representative XLSX/MTM in `tests/data/`.
- Store goldens/hashes for a fixed selection matrix in `tests/baseline/`.
- Parity harness compares legacy outputs vs new API for those selections.

## Branching/merging
- Short-lived feature branches (`feature/<area>`). No direct-to-main.
- Auto-merge when CI green; Integrator handles rebases/conflicts.
- Update `TASK_BOARD.md` and `REBUILD_PLAN.md` when changing contracts/ownership.

## Branch protection (recommend)
- Protect `main`: require PR + CI green before merge; enable auto-merge on green.
- Give the Integrator (bot/human) permission to rebase/merge.
- Keep secrets out of the repo; use `.env` locally and CI secrets for real creds.

## Useful commands (once services are fleshed out)
- API dev server: `uvicorn api.app.main:app --reload`
- Front end: `npm install && npm run dev` in `web/`
- CLI (future): `python -m cli ...`

## Notes
- Current `api/` and `web/` are skeletons for agents to fill in. Do not assume production readiness.
- Avoid committing real credentials; use `.env` and secrets in CI.
