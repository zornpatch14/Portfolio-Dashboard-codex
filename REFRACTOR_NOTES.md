# Portfolio Dashboard Refactor Notes

This document captures the current state of the `build_core_artifact` refactor so the work can continue seamlessly in a new session. It summarizes the user’s requirements, what has been completed, and the precise remaining steps (with the “no shortcuts, lean & robust” philosophy we’ve been following).

## User Instructions / Expectations

- The refactor must keep the app lean, fast, and robust—no shortcuts, hacks, or bandaids.
- Before writing code for any new step, produce a clear explanation/plan of what will be done.
- Maintain cache correctness and determinism; add diagnostics (like parity logs) when needed, but remove temporary tooling only after the new path is proven.
- Prefer vectorized/pandas or other efficient approaches over Python loops where practical.
- Keep validation in place (e.g., parity logging) until the entire refactor is complete.

## Completed Steps (1‑5)

1. **Documented the current artifact contract**  
   - Enumerated every key returned by `build_core_artifact` and its consumers (tabs, metrics, allocator, CTA, etc.).  
   - Captured baseline artifacts/hashes for representative selections.  
   - Added guards/assumptions so downstream code doesn’t mutate artifact outputs.

2. **Designed the layered cache architecture**  
   - Formalized per-file slice cache, derived per-file caches, portfolio aggregator, and validation hooks.  
   - Defined cache keys, invalidation rules, and data contracts for each layer.

3. **Created the integration/migration plan**  
   - Sequenced the work (slice → netpos → equity bundles → capital/returns → percent equity → portfolio view → orchestrator → consumer migration → validation).  
   - Established parity logging/testing strategy to ensure correctness at every stage.

4. **Introduced canonical per-file slice cache** (`FileSlice`, `get_file_slice`)  
   - Each slice includes filtered trades (windowed), full-history trades, MTM rows, label/metadata, contract multipliers, and margin overrides.  
   - All downstream work now reuses the slice cache, eliminating redundant store lookups.

5. **Replaced the net-position engine with a deterministic/vectorized version**  
   - `_build_netpos_series_for_slice` operates on the full trade history, handles zero contracts, and mirrors the legacy logic.  
   - `cached_netpos_per_file` uses the slice cache and selection window to produce the per-file series.  
   - Parity logging (`NETPOS_PARITY_CHECK` + `netpos_parity.log`) compares new vs. old outputs when enabled.

6. **Added the daily MTM + equity bundle cache** (`EquityBundle`, `get_equity_bundle`)  
   - `_hybrid_equity_from_trades_and_mtm` is now called once per file per selection; the cached bundle provides the cumulative equity curve and normalized daily P&L.  
   - `build_core_artifact` reads equity/daily from this helper instead of recomputing.

7. **Implemented the capital & returns cache** (`MarginUsage`, `get_margin_usage`)  
   - Encapsulates `_contract_time_usage`, margin resolution, capital series, and `_daily_returns_from_pnl_and_capital`.  
   - Consumers now pull capital and daily percent returns directly from this helper instead of via the artifact.  
   - `MarginUsage` also reports the resolved `margin_per_contract` for diagnostics.

*(Steps 6 & 7 correspond to roadmap bullets “Daily MTM + Equity Derivation Layer” and “Capital & Returns Layer.”)*

## Remaining Steps (8‑12)

8. **Deferred Percent-Equity Construction**  
   - Provide a helper (e.g., `get_percent_equity(sel_key, fname, demand_flag)`) that generates the additive percent-equity index only when a consumer explicitly needs it (percent tabs, net-contract charts).  
   - Memoize per file/selection (keyed by daily return hash) so percent curves aren’t rebuilt on every artifact change.  
   - Portfolio percent equity (next step) will reuse the same lazy infrastructure.

9. **Portfolio Aggregation Layer**  
   - Build `build_portfolio_view(sel_key)` that gathers per-file daily return/capital series from the caches above, runs `_compute_portfolio_from_maps` once, and offers lazy percent equity for the portfolio.  
   - Add sanity assertions (aligned indexes, non-negative capital) and expose light-weight helpers for tabs that need only parts of the portfolio data (e.g., allocator).  
   - This layer should also surface metadata (which files contributed, etc.) for future diagnostics.

10. **Recompose `build_core_artifact` as Orchestrator**  
    - Once all per-file/portfolio caches exist, refactor `build_core_artifact` to simply gather cached slices, bundles, and views.  
    - Keep the parity flag/logging available until new vs. old orchestrators match for all selections; then retire redundant logic.  
    - ✅ Status: The artifact contract has been slimmed to `{trades_by_file, equity_by_file, label/symbol/strategy/interval maps, files tuple, portfolio diagnostics}`. Percent-equity curves, daily returns/P&L, and capital must now be accessed via the cache helpers (`get_percent_equity`, `get_daily_returns_by_file`, `get_daily_pnl_by_file`, `get_margin_usage`, `build_portfolio_view`).

11. **Consumer Migration & Cleanup**  
    - Update tabs, metrics, allocator, correlation, CTA, etc., to call the new helpers directly where appropriate (e.g., metrics pulling equity bundles, allocator grabbing portfolio view).  
    - Remove obsolete memoized helpers (`cached_hybrid_equity`, etc.) once all consumers are on the new path.  
    - Delete temporary logging (like `NETPOS_PARITY_CHECK`) after parity is fully proven.

12. **Validation & Regression Testing**  
    - Re-run the baseline snapshot matrix (artifact hashes, Riskfolio weights, CTA outputs) comparing old vs. new orchestrators.  
    - Stress-test cache invalidation by mutating selections (files, symbols, intervals, dates, contracts, margin overrides).  
    - Instrument cache hit/miss counts and timings to prove the refactor delivers the expected performance gains.  
    - Only after this step should the parity tooling be removed/disabled.

## Ongoing Practices

- **Before coding each step:** explain the plan (inputs, outputs, cache keys, invariants, validation).  
- **No shortcuts:** favor durable solutions even if they require more upfront work.  
- **Testing:** after each major change, rerun representative selections (with `NETPOS_PARITY_CHECK` as needed) to ensure behavior remains identical.  
- **Encoding:** all files are now UTF‑8; keep it that way when editing.  
- **Parity logging:** leave `NETPOS_PARITY_CHECK` plumbing in place until Step 12 is complete.

With this document in the repo, a future chat can read `REFRACTOR_NOTES.md`, immediately see what has been done, what remains, and continue the refactor without rehashing earlier context.
