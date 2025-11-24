# Baseline Selections Stub

`selections.json` lists the fixed selections used for parity and perf tests. Fill in with the actual filenames you place under `tests/data/`.

Suggested fields per selection:
- `name`: identifier for the selection (used to name golden files).
- `files`: list of filenames (relative to `tests/data/`).
- `symbols`: list of symbols to include.
- `intervals`: list of intervals/timeframes.
- `strategies`: list of strategies (can be empty).
- `direction`: "All" | "Long" | "Short".
- `start`, `end`: ISO date strings or null for full range.
- `contracts`: object mapping filename -> multiplier (can be empty).
- `margins`: object mapping filename -> override (can be empty).
- `spike`: boolean for spike toggle.

Workflow:
1) Drop small representative XLSX trades/MTM files into `tests/data/`.
2) Update `selections.json` to reference those files.
3) Run a legacy-side script to produce goldens (hashes/metrics) per selection; store outputs in `tests/baseline/<name>.json`.
4) Parity harness loads `selections.json`, calls the new API, and compares to `<name>.json` goldens.
