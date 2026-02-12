# Web Agent Guide

## Scope
Use this file for frontend-only work in `web/` or the frontend half of fullstack changes. Repo-wide defaults still live at root `AGENTS.md`.

## Frontend Entry Points
- Main UI and tab wiring: `web/app/page.tsx`
- API client + response normalization: `web/lib/api.ts`
- Selection model/types: `web/lib/types/selection.ts`
- Shared components/charts: `web/components/`
- Build/runtime config: `web/next.config.mjs`, `web/package.json`

## Run Frontend
From repo root:
```sh
cd web
npm install
# PowerShell:
$env:NEXT_PUBLIC_API_BASE="http://localhost:8000"
npm run dev
```

## Verify Frontend Changes
From `web/`:
```sh
npx tsc --noEmit
npm run build
```

Optional lint (currently interactive):
```sh
npm run lint
```

## Frontend Wiring Checklist
1. Update the tab/view surface in `web/app/page.tsx` or relevant `web/components/*`.
2. Keep API request/query construction aligned in `web/lib/api.ts`.
3. Keep response types and runtime normalization aligned (for example `per_file` to `perFile` handling).
4. If backend contract changes, update frontend types/calls in the same rollout.
5. Validate loading/error/no-data behavior for the touched view.

## Repo-Specific Invariants
- `NEXT_PUBLIC_API_BASE` is required; `web/lib/api.ts` throws if unset.
- Query param names in `selectionQuery` must match backend expectations in `api/api/app/dependencies.py` (`start_date`, `end_date`, `contract_multipliers`, `margin_overrides`, `spike_flag`, `account_equity`).
- Keep endpoint constants centralized in `web/lib/api.ts` unless there is a strong reason to split.
- `web/next.config.mjs` allows builds despite lint/type settings; rely on explicit `npx tsc --noEmit` for type safety.
- Correlation/export backend endpoints are currently not implemented; avoid assuming successful data responses unless backend is added.

## Edit Guardrails
- Prefer minimal diffs and existing patterns.
- Avoid adding dependencies unless explicitly requested.
- Do not rename/move frontend files unless requested.
