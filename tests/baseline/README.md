# Baseline Data and Goldens

Purpose: parity testing between the legacy Dash app and the new API.

## What to include
- Small, representative Trades List XLSX files + MTM sheets under `tests/data/`.
- A fixed selection matrix (e.g., combinations of files, symbols, intervals, strategies, directions, date windows, multipliers, margins, spike flag).
- Golden outputs (hashes/metrics/series summaries) from the legacy app for those selections stored here (`tests/baseline/`).

## How to use
- Parity harness should load `tests/data/`, run both legacy and new API for each selection, and compare:
  - equity/percent-equity hashes
  - metrics rows
  - netpos aggregates
  - correlations/CTA summaries (where applicable)
- Perf smoke should hit equity/drawdown/netpos on the baseline selections and enforce latency/payload budgets.

## Notes
- Keep datasets small to keep CI fast.
- Do not commit proprietary/large datasets; anonymize if needed.
