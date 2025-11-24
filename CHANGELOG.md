# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning (https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.9.1] - 2025-11-17

### Fixed
- Restored the Margin tab purchasing-power and drawdown charts to use the true cumulative portfolio P&L. The cached portfolio view now stores the combined equity curve, and `_purchasing_power_series` consumes that series instead of the margin-capital footprint, so the chart once again reflects `starting balance + portfolio P&L - initial margin used`.

## [1.9.0] - 2025-11-12

### Added
- Baseline + stress-test automation tools (`scripts/baseline_snapshot.py`, `scripts/cache_stress_test.py`) so cache regressions can be checked without going through the UI; the stress test also reports cache hit/miss timings for the hottest helpers.

### Changed
- Rebuilt `build_core_artifact` as a thin orchestrator over the new cache layers (file slices, equity bundles, margin usage, portfolio view). All consumers—tabs, allocator, CTA, Riskfolio—now pull their data directly from those helpers instead of artifact side channels.
- Finalised the vectorised `cached_netpos_per_file` pipeline and `MarginUsage` daily-return/capital layer; every tab that needs daily P&L/returns now reuses those cached results.
- Slimmed the artifact contract to just `trades_by_file`, `equity_by_file`, metadata maps, and portfolio diagnostics; percent-equity, daily-return/P&L, and capital series are now strictly helper-provided.

### Removed
- Retired the old parity tooling (`NETPOS_PARITY_CHECK`, legacy net-position builder, parity logger/output) now that the new deterministic net-pos engine has been validated.
- Dropped deprecated artifact fields (`daily_return_pct_by_file`, `daily_pnl_by_file`, `capital_by_file`, `pct_equity_*`, `portfolio_*`) and all related consumer code.

## [1.8.0] - 2025-11-06

### Added
- Riskfolio “Stage 2” contract conversion table that freezes the latest optimisation weights, displays max/avg weight gaps, and lets you rescale contracts instantly with a dedicated account-equity input.
- Shared “Apply Suggested Contracts” workflow so both the Riskfolio and Inverse Volatility tabs push their suggested counts straight into the Load Trade Lists contract multipliers.

### Changed
- Separated Riskfolio optimisation settings from contract sizing: the contracts table now reacts to live margin overrides and equity edits without rerunning the optimiser, and the old optimiser equity control was removed.

### Fixed
- Eliminated overlapping wildcard-callback errors by consolidating contract-apply logic into a single callback, keeping both IVP and Riskfolio buttons functional.

## [1.7.0] - 2025-11-05

### Added
- Volatility targeting controls (target %, lookback months, update cadence) on the dashboard sidebar so every tab can be rescaled to a volatility sleeve without leaving the main workflow.
- Server-side sleeve caches, percent-equity builders, and spike-aware helpers that translate daily MTM dollars into reusable percent-return curves for charts, tables, Riskfolio, and downloads.
- POINT_VALUE constants plus TradeStation ETL enhancements that derive notional exposure directly from entry price, contracts, and contract point size; bundled new MNQ trade lists, reference PDFs, and a profiling snapshot used while tuning the sleeve engine.

### Changed
- All equity, drawdown, allocation, exposure, and download callbacks now consume the cached sleeve data so dollar and percent analytics remain in sync with the selected volatility regime and spike toggle.
- Inverse Volatility tab callback registration now plugs into the shared selection key + cache plumbing, keeping its optimisations consistent with the rest of the app.

### Fixed
- Notional exposure is computed even when TradeStation leaves percent-profit fields blank, preventing downstream analytics from seeing zero notionals and improving spike scaling.

## [1.6.0] - 2025-11-04

### Added
- CTA Report tab presenting the Rule 4.35 capsule summary, five-year monthly rate-of-return grid, performance charts, and downloadable CSV for monthly returns.
- src/cta_report.py helper module that composes daily percent returns into NAV paths, additive vs compounded monthly tables, drawdown analytics, and rolling statistics ready for future reinvestment workflows.
- Portfolio helper _compute_cta_report so CTA analytics reuse the existing selection filters, contract multipliers, and spike overlays.

### Changed
- Equity, drawdown, and intraday charts now render both dollar and percent views with consistent margins and shared axes for easier comparison.
- CTA charts and tables align with the new helper outputs, including additive versus compounded metrics and clearer styling.
- Updated monthly grouping logic and DataFrame formatting calls to modern pandas APIs (e.g., req="ME", DataFrame.map).

### Fixed
- Percent equity curves keep flat periods horizontal (no interpolated slopes) and respect spike toggles in both currency and percent plots.
- CTA reporting guards against missing/duplicate month columns, NaN monthly bars, and deprecated pandas warnings.
## [1.5.0] - 2025-11-04

### Added
- Captured TradeStation `% Profit`, gross P&L, and implied notional per leg during ETL so trade-level analytics now include fee-adjusted percent returns.
- Derived daily percent return, daily notional exposure, and cumulative percent equity series from mark-to-market data; cached both strategy-level and portfolio-level streams for downstream consumers.
- Appended percent-based performance metrics to the summary table (avg/std daily return and a daily Sharpe), keeping all legacy dollar statistics intact.

### Changed
- Reworked the Equity, Portfolio Drawdown, and Intraday Drawdown tabs to render both dollar and percent charts simultaneously (stacked), with shared x-axis ranges and consistent margins so the plots align visually.
- Updated the intraday, drawdown, and equity chart builders to honour uniform left/right padding and to reuse the new percent equity index, ensuring spike overlays affect dollar and percent views identically.
- Retooled Riskfolio’s returns preparation to prioritise the new percent-return overrides while preserving dollar fallbacks for legacy workflows.
- Normalised purchasing power, margin, and net-position charts with fixed axes/margins and shared ranges for cleaner comparisons across the margin tab.

### Fixed
- Eliminated deprecated `fillna(method="ffill")` usage in percent pipelines to avoid forthcoming pandas errors.
- Ensured spike-adjusted percent curves remain stepwise during flat periods, preventing sloped interpolation between daily anchors.
- Locked equity/drawdown/intraday percent charts to the same timeline as their dollar counterparts so the two representations stay in sync even when spikes insert intra-day timestamps.

## [1.4.0] - 2025-11-XX

### Added
- Parsed the optional TradeStation “Daily” sheet alongside trades, storing per-file mark-to-market rows for daily MTM equity reconstruction.
- Introduced hybrid equity and daily P/L helpers that blend MTM deltas with trade-book anchors, including a settings toggle to overlay trade-level drawdown spikes on equity, drawdown, intraday, and purchasing-power visualisations.

### Changed
- Rebuilt daily equity series to include start-of-day anchors and MTM closes so intraday drawdown charts reflect day-over-day swings even without spikes.
- Updated purchasing power to recompute from the hybrid (or spiked) equity path while keeping Riskfolio / Inverse Vol optimisations tied to close-to-close returns.

### Fixed
- Ensured initial-margin plots render as proper step functions aligned with net contracts.
- Restored intraday drawdown bars for MTM-enabled files by supplying per-day open/close points and correctly dating spike overlays.

## [1.3.0] - 2025-10-31

### Added
- Split the Riskfolio tab into dedicated subtabs (Mean-Risk live, Risk Parity / Hierarchical placeholders) with contextual tooltips and documentation for every input.
- Exposed full mean-risk configuration controls: objective, return model, risk measure, risk-free rate, and utility risk aversion.
- Surfaced the selected optimisation configuration in the results summary so each run is self-documented.
- Introduced strategy-level weight caps alongside per-asset bounds and symbol caps, all wired into the constraint system.
- Added backend toolbox helpers and a formal config builder so future Riskfolio toolboxes can reuse the same plumbing.

### Changed
- Rebuilt the mean-risk pathway on Riskfolio’s native constraint framework, translating UI selections into `asset_classes`/`constraints` DataFrames and removing the prior manual inequality plumbing.
- Normalised constraint inputs into structured rule sets before optimisation and enriched the selection metadata with per-file symbol/strategy/interval maps.
- Parked the turnover control for future use and clarified its reserved status in the UI.
- Updated the strategy caps helper text to reflect real strategy names (e.g., `35X=0.30`).
- Enabled the covariance method dropdown only when a variance-family risk measure is selected.
- Tweaked the results layout and tabs to reflect the new Riskfolio structure and inputs.

### Fixed
- Resolved the “unsupported model 'classic'” error by ensuring the adapter translates UI selections into Riskfolio-compatible options.
- Guarded against pandas index truthiness errors in constraint building by coercing asset lists before iteration.

### Removed
- Dropped unused L1/L2 regularisation controls from the Mean-Risk card.
- Removed cardinality support (UI, callbacks, and config wiring) pending a future discrete optimisation rewrite.

## [1.2.2] - 2025-10-31

### Fixed
- Refresh the Inverse Volatility table's current contracts immediately when contract multipliers change or suggestions are applied, eliminating stale values unless the multiplier input is touched.


## [1.2.1] - 2025-10-30

### Added
- Introduced the Inverse Volatility tab to provide an alternate allocation view focused on inverse volatility weights.

### Changed
- Refined the Inverse Volatility tab layout with tighter margin, bounded weight gap percentages, updated multiplier styling, and smoother increment controls for the multiplier input.
- Reworked the Load Trade Lists tab cards so contract and margin controls are left-justified, spaced consistently, and the included files checklist breathes better.
- Updated the overall tab arrangement so the main tabs load in their revised order and layout.

### Fixed
- Ensured contract multipliers respect a value of zero and adjusted related toggle behavior to prevent accidental changes when clicking around the checkbox area.

## [1.1.0] - 2025-10-29

### Added
- Nothing yet.

### Changed
- <!-- UI request: flatten per-file contract controls into a single-row layout within the Load Trade Lists tab. -->
- Restructured the Load Trade Lists tab so per-file controls sit in single-row cards.
- <!-- UI request: reorganize allocator settings beside results with tighter spacing and a narrower controls pane. -->
- Split the Allocator tab into side-by-side settings and results cards with compact controls.
- <!-- UI request: ensure the dashboard opens with the Load Trade Lists tab selected -->
- Defaulted the main tab set to open on Load Trade Lists.

### Fixed
- Nothing yet.

### Removed
- Removed the intermediate Charts tab so individual chart views are first-level tabs.

## [1.1.1] - 2025-10-29

### Changed

- Inlined the Riskfolio tab layout within the main app layout so riskfolio-results and related components always exist, eliminating missing-id callback errors.
- Added a dedicated callback to toggle Riskfolio controls based on available trade data and updated long-callback running outputs to use allow_duplicate.

## [1.0.0] - 2025-10-29

### Added
- Initial stable release of the Portfolio Dashboard.
- Main script `Portfolio_Dashboard7.py`.
- Core `src` modules: `allocator.py`, `charts.py`, `constants.py`, `corr.py`, `equity.py`, `etl.py`, `helpers.py`, `margin.py`, `metrics.py`, `netpos.py`, `riskfolio_adapter.py`, `tabs.py`.

