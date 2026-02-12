---
name: repo-orientation
description: Quickly locate the correct files and commands in this Portfolio Dashboard repository. Use when a request starts with codebase discovery, "where does this live?", bug triage, or selecting edit targets before making changes across API, frontend, compute, or data layers.
---

# Repo Orientation Workflow

## When To Use
- Use for first-pass discovery before editing.
- Use when mapping a feature/bug to concrete files.
- Use when you must confirm run/test/build commands for this repo.

## When Not To Use
- Do not use for isolated edits where file targets are already explicit.
- Do not use for non-code tasks (copywriting, translation, general chat).

## Steps
1. Confirm scope from the user request.
2. Snapshot relevant structure quickly:
```sh
rg --files -g "!web/node_modules/**" -g "!web/.next/**" -g "!**/__pycache__/**"
```
3. Locate backend surfaces:
```sh
rg -n "include_router|APIRouter|@router\\." api/api/app/main.py api/api/app/routes
```
4. Locate frontend surfaces:
```sh
rg -n "tabs|useQuery|fetchSeries|fetchMetrics|fetchHistogram|fetchCta|submitRiskfolioJob" web/app/page.tsx web/lib/api.ts
```
5. Locate data and compute plumbing:
```sh
rg -n "DATA_ROOT|API_CACHE_DIR|REDIS_URL|selection_hash|PerFileCache|DataStore|IngestService" api/api/app
```
6. Map requested behavior to the smallest set of files, then edit only those files.

## Verification Checklist
- Backend logic touched:
```sh
python -m pytest tests/test_ingest.py tests/test_compute.py
```
- Frontend logic/types touched:
```sh
cd web
npx tsc --noEmit
npm run build
```
- If route/schema changed, verify `web/lib/api.ts` and `web/app/page.tsx` remain aligned with backend request/response models.

## Minimal-Diff Reminder
- Preserve existing structure and naming.
- Reuse existing helpers before adding new modules.
- Avoid unrelated formatting/refactors.
