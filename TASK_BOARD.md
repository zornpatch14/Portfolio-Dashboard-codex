# Task Board

Lightweight tracker for active work. Keep this in sync with actual status. Roles can be reassigned as needed. The Integrator is responsible for rebases/merges and keeping this file current.

## Team roles
- Integrator (G): rebases/merges, resolves overlaps, approves contract changes, keeps REBUILD_PLAN.md and TASK_BOARD.md updated.
- A) Ingest: XLSX -> Polars -> Parquet; metadata index; file_id/hash.
- B) Core compute: per-file caches (daily/equity/percent/netpos/margin), portfolio aggregation, downsampling.
- C) API: FastAPI routes/models, SSE jobs, exports.
- D) Front-end: Next.js shell, selection state, ECharts/AG Grid wiring.
- E) CLI/headless: Python CLI wrapping API; selection flags; job streaming.
- F) Parity/Perf/E2E: baseline matrix, parity harness, perf smoke, Playwright.

## Milestones (summary)
1) Skeletons (API + FE + CLI + Docker/CI).
2) Ingest + per-file caches + real series endpoints.
3) Portfolio + metrics + downsampling + exports.
4) Correlations/CTA/optimizers async + wired FE + parity/perf passing.
5) Hardening (observability/config/docs/CLI polish).

## Status board
| Workstream | Owner           | Branch             | Status  | Notes |
|------------|-----------------|--------------------|---------|-------|
| Integrator | agent-integrator | feature/integrator | Pending | Rebase/merge gatekeeper; resolves conflicts; updates plan/board. |
| A) Ingest  | agent-ingest    | feature/ingest     | Pending | Build Parquet ingest + metadata index. |
| B) Compute | agent-compute   | feature/compute    | Pending | Per-file caches; portfolio aggregation; downsampling. |
| C) API     | agent-api       | feature/api        | Pending | FastAPI v1 routes; SSE jobs; exports; OpenAPI. |
| D) FE      | agent-fe        | feature/fe         | Pending | Next.js shell; selection state; wire sample endpoints. |
| E) CLI     | agent-cli       | feature/cli        | Pending | CLI commands for ingest/series/metrics/optimizer/exports. |
| F) Parity  | agent-parity    | feature/parity     | Pending | Baseline data; parity harness; perf smoke; Playwright. |

## Working agreements
- Branching: short-lived feature branches (`feature/<area>`). Trunk-based flow; main is protected.
- Auto-merge: merges happen only when CI is green and the Integrator (bot/human) rebases/fast-forwards cleanly. No manual interaction from the owner once pushed.
- Direct-to-main: not allowed; all changes go through branches + CI gate.
- Conflicts: Integrator resolves; if automatic resolution fails, branch is paused until fixed.
- Small PRs; scope per workstream.
- Update this board and REBUILD_PLAN.md when interfaces/ownership change.
- CI must pass before merge (lint/type/unit/contract/parity/perf smoke/E2E where applicable).
- Breaking contract changes require Integrator sign-off and updated OpenAPI/schemas/goldens.
