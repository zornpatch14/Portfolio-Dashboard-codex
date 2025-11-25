'use client';

import { useMemo, useState } from 'react';
import { useQueries, useQuery } from '@tanstack/react-query';
import MetricsGrid from '../components/MetricsGrid';
import SeriesChart from '../components/SeriesChart';
import { SelectionList } from '../components/SelectionList';
import { HistogramChart } from '../components/HistogramChart';
import { SelectionControls } from '../components/SelectionControls';
import {
  fetchCorrelations,
  fetchCta,
  fetchHistogram,
  fetchMetrics,
  fetchOptimizer,
  fetchSeries,
  mockCorrelations,
  mockCta,
  mockHistogram,
  mockOptimizer,
  mockSeries,
  SeriesKind,
} from '../lib/api';
import { loadSampleSelections, Selection } from '../lib/selections';
import { CorrelationHeatmap } from '../components/CorrelationHeatmap';

const selections = loadSampleSelections();
const tabs = [
  { key: 'overview', label: 'Overview' },
  { key: 'correlations', label: 'Correlations' },
  { key: 'optimizer', label: 'Optimizer' },
  { key: 'cta', label: 'CTA & Reports' },
  { key: 'ingest', label: 'Uploads & Exports' },
] as const;

type TabKey = (typeof tabs)[number]['key'];

export default function HomePage() {
  const [activeSelection, setActiveSelection] = useState<Selection>(selections[0]);
  const [activeTab, setActiveTab] = useState<TabKey>('overview');
  const [corrMode, setCorrMode] = useState('drawdown_pct');
  const [accountEquity, setAccountEquity] = useState(50000);
  const [includeDownsample, setIncludeDownsample] = useState(true);
  const [exportFormat, setExportFormat] = useState<'csv' | 'parquet'>('parquet');
  const apiBase = process.env.NEXT_PUBLIC_API_BASE;

  const availableFiles = useMemo(
    () => Array.from(new Set(selections.flatMap((s) => s.files))).sort(),
    [],
  );

  const seriesKinds: SeriesKind[] = ['equity', 'equityPercent', 'drawdown', 'intradayDrawdown', 'netpos', 'margin'];

  const [
    equityQuery,
    equityPctQuery,
    drawdownQuery,
    intradayDdQuery,
    netposQuery,
    marginQuery,
    histogramQuery,
    metricsQuery,
  ] = useQueries({
    queries: [
      ...seriesKinds.map((kind) => ({
        queryKey: [kind, activeSelection.name, activeSelection.files.join(',')],
        queryFn: () => fetchSeries(activeSelection, kind),
        initialData: mockSeries(activeSelection, kind),
      })),
      {
        queryKey: ['histogram', activeSelection.name, activeSelection.files.join(',')],
        queryFn: () => fetchHistogram(activeSelection),
        initialData: mockHistogram(activeSelection),
      },
      {
        queryKey: ['metrics', activeSelection.name, activeSelection.files.join(',')],
        queryFn: () => fetchMetrics(activeSelection),
        initialData: [],
      },
    ],
  });

  const correlationQuery = useQuery({
    queryKey: ['correlations', activeSelection.name, activeSelection.files.join(','), corrMode],
    queryFn: () => fetchCorrelations(activeSelection, corrMode),
    initialData: mockCorrelations(activeSelection, corrMode),
  });

  const optimizerQuery = useQuery({
    queryKey: ['optimizer', activeSelection.name, activeSelection.files.join(',')],
    queryFn: () => fetchOptimizer(activeSelection),
    initialData: mockOptimizer(activeSelection),
  });

  const ctaQuery = useQuery({
    queryKey: ['cta', activeSelection.name, activeSelection.files.join(',')],
    queryFn: () => fetchCta(activeSelection),
    initialData: mockCta(activeSelection),
  });

  const metricsSummary = useMemo(() => {
    const rows = metricsQuery.data || [];
    if (!rows.length) return null;
    const totals = rows.reduce(
      (acc, row) => {
        acc.netProfit += row.netProfit;
        acc.drawdown += row.drawdown;
        acc.trades += row.trades;
        acc.winRate += row.winRate;
        return acc;
      },
      { netProfit: 0, drawdown: 0, trades: 0, winRate: 0 },
    );
    const avgWin = totals.winRate / rows.length;
    const avgTrades = Math.round(totals.trades / rows.length);
    return { netProfit: totals.netProfit, drawdown: totals.drawdown, avgWin, avgTrades };
  }, [metricsQuery.data]);

  const busy =
    equityQuery.isFetching ||
    drawdownQuery.isFetching ||
    metricsQuery.isFetching ||
    equityPctQuery.isFetching ||
    intradayDdQuery.isFetching ||
    netposQuery.isFetching ||
    marginQuery.isFetching ||
    histogramQuery.isFetching ||
    correlationQuery.isFetching ||
    optimizerQuery.isFetching ||
    ctaQuery.isFetching;

  const activeBadge = busy ? <div className="badge">Loading...</div> : <div className="badge">Live</div>;

  const renderOverview = () => (
    <div>
      <div className="flex gap-md" style={{ alignItems: 'center', justifyContent: 'space-between' }}>
        <h2 className="section-title" style={{ margin: 0 }}>
          Equity, drawdown, and metrics
        </h2>
        {activeBadge}
      </div>

      <div className="metric-cards" style={{ marginTop: 14 }}>
        <div className="metric-card">
          <span className="text-muted small">Net Profit (sum)</span>
          <strong>{metricsSummary ? metricsSummary.netProfit.toLocaleString() : "--"}</strong>
        </div>
        <div className="metric-card">
          <span className="text-muted small">Drawdown (sum)</span>
          <strong>{metricsSummary ? metricsSummary.drawdown.toLocaleString() : "--"}</strong>
        </div>
        <div className="metric-card">
          <span className="text-muted small">Avg Trades</span>
          <strong>{metricsSummary ? metricsSummary.avgTrades : "--"}</strong>
        </div>
        <div className="metric-card">
          <span className="text-muted small">Avg Win %</span>
          <strong>{metricsSummary ? `${metricsSummary.avgWin.toFixed(1)}%` : "--"}</strong>
        </div>
        <div className="metric-card">
          <span className="text-muted small">Files</span>
          <strong>{activeSelection.files.length}</strong>
        </div>
        <div className="metric-card">
          <span className="text-muted small">Account Equity</span>
          <strong>{`$${accountEquity.toLocaleString()}`}</strong>
        </div>
      </div>

      <div className="charts-grid" style={{ marginTop: 18 }}>
        <div className="card">
          <SeriesChart title="Equity Curve" series={equityQuery.data -- mockSeries(activeSelection, 'equity')} color="#4cc3ff" />
          <div className="text-muted small">Points: {equityQuery.data-.downsampledCount -- equityQuery.data-.points.length}</div>
        </div>
        <div className="card">
          <SeriesChart
            title="Percent Equity"
            series={equityPctQuery.data -- mockSeries(activeSelection, 'equityPercent')}
            color="#8fe3c7"
          />
          <div className="text-muted small">Points: {equityPctQuery.data-.downsampledCount -- equityPctQuery.data-.points.length}</div>
        </div>
        <div className="card">
          <SeriesChart
            title="Drawdown"
            series={drawdownQuery.data -- mockSeries(activeSelection, 'drawdown')}
            color="#ff8f6b"
          />
          <div className="text-muted small">Points: {drawdownQuery.data-.downsampledCount -- drawdownQuery.data-.points.length}</div>
        </div>
        <div className="card">
          <SeriesChart
            title="Intraday Drawdown"
            series={intradayDdQuery.data -- mockSeries(activeSelection, 'intradayDrawdown')}
            color="#f4c95d"
          />
          <div className="text-muted small">Points: {intradayDdQuery.data-.downsampledCount -- intradayDdQuery.data-.points.length}</div>
        </div>
        <div className="card">
          <SeriesChart title="Net Position" series={netposQuery.data -- mockSeries(activeSelection, 'netpos')} color="#9f8bff" />
          <div className="text-muted small">Points: {netposQuery.data-.downsampledCount -- netposQuery.data-.points.length}</div>
        </div>
        <div className="card">
          <SeriesChart title="Margin Usage" series={marginQuery.data -- mockSeries(activeSelection, 'margin')} color="#54ffd0" />
          <div className="text-muted small">Points: {marginQuery.data-.downsampledCount -- marginQuery.data-.points.length}</div>
        </div>
        <div className="card">
          <HistogramChart histogram={histogramQuery.data -- mockHistogram(activeSelection)} />
          <div className="text-muted small">Distribution: {histogramQuery.data-.buckets.length -- 0} buckets</div>
        </div>
      </div>

      <div style={{ marginTop: 18 }}>
        <h3 className="section-title">Per-file metrics</h3>
        {metricsQuery.data && metricsQuery.data.length - (
          <MetricsGrid rows={metricsQuery.data} />
        ) : (
          <div className="placeholder-text">No metrics returned yet.</div>
        )}
      </div>
    </div>
  );

  const renderCorrelations = () => (
    <div className="panel" style={{ marginTop: 8 }}>
      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="section-title" style={{ margin: 0 }}>Correlation Heatmap</h3>
        {activeBadge}
      </div>
      <p className="text-muted small" style={{ marginTop: 4 }}>
        Mirrors the Dash correlations tab (drawdown %, returns, P/L, slope). Use the mode and slope controls below to switch views.
      </p>
      <div className="grid-2" style={{ marginTop: 10 }}>
        <div>
          <label className="field-label" htmlFor="corr-mode">Correlation mode</label>
          <select
            id="corr-mode"
            className="input"
            value={corrMode}
            onChange={(event) => setCorrMode(event.target.value)}
          >
            <option value="drawdown_pct">Daily Drawdown % (recommended)</option>
            <option value="returns_z">Z-scored Daily Returns</option>
            <option value="pl">Daily $ P/L</option>
            <option value="slope">Rolling Slope</option>
          </select>
        </div>
        <div>
          <label className="field-label" htmlFor="corr-window">Slope window (if slope mode)</label>
          <input
            id="corr-window"
            className="input"
            type="number"
            min={5}
            step={1}
            value={corrMode === 'slope' - 20 : ''}
            placeholder="20"
            disabled={corrMode !== 'slope'}
          />
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <CorrelationHeatmap data={correlationQuery.data -- mockCorrelations(activeSelection, corrMode)} />
      </div>
      <div className="text-muted small" style={{ marginTop: 10 }}>
        {correlationQuery.data-.notes.map((note) => (
          <div key={note}>- {note}</div>
        ))}
      </div>
    </div>
  );

  const renderOptimizer = () => (
    <div className="panel" style={{ marginTop: 8 }}>
      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="section-title" style={{ margin: 0 }}>Riskfolio-style Optimizer</h3>
        {activeBadge}
      </div>
      <p className="text-muted small" style={{ marginTop: 4 }}>
        Carries over the legacy Riskfolio controls: objective, return model, risk measure, risk-free rate, utility lambda, bounds,
        and contract sizing. Backend wiring comes later; the UI matches all Dash controls.
      </p>

      <div className="grid-2" style={{ marginTop: 12 }}>
        <div className="card">
          <label className="field-label" htmlFor="objective">Objective</label>
          <select id="objective" className="input" defaultValue="sharpe">
            <option value="sharpe">Max Risk-Adjusted Return (Sharpe)</option>
            <option value="min_risk">Min Risk</option>
            <option value="max_return">Max Return</option>
            <option value="utility">Utility</option>
          </select>
          <label className="field-label" htmlFor="return-model" style={{ marginTop: 12 }}>Return Model</label>
          <select id="return-model" className="input" defaultValue="arithmetic">
            <option value="arithmetic">Arithmetic</option>
            <option value="approx_log">Approximate Log</option>
            <option value="log">Exact Log</option>
          </select>
          <label className="field-label" htmlFor="risk-measure" style={{ marginTop: 12 }}>Risk Measure</label>
          <select id="risk-measure" className="input" defaultValue="variance">
            <option value="variance">Variance</option>
            <option value="semi_variance">Semi-Variance</option>
            <option value="cvar">CVaR</option>
            <option value="cdar">CDaR</option>
            <option value="evar">EVaR</option>
          </select>
        </div>

        <div className="card">
          <label className="field-label" htmlFor="rf">Risk-Free Rate (annual %)</label>
          <input id="rf" className="input" type="number" step={0.05} defaultValue={0} />
          <label className="field-label" htmlFor="risk-aversion" style={{ marginTop: 12 }}>Risk Aversion (Utility)</label>
          <input id="risk-aversion" className="input" type="number" step={0.1} defaultValue={2} />
          <label className="field-label" htmlFor="alpha" style={{ marginTop: 12 }}>Confidence Level alpha</label>
          <input id="alpha" className="input" type="number" min={0} max={1} step={0.01} defaultValue={0.95} />
          <label className="field-label" htmlFor="bounds" style={{ marginTop: 12 }}>Weight Bounds (min / max)</label>
          <div className="flex gap-sm">
            <input className="input" type="number" defaultValue={0} />
            <input className="input" type="number" defaultValue={0.6} />
          </div>
        </div>
      </div>

      <div className="grid-2" style={{ marginTop: 16 }}>
        <div className="card">
          <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
            <strong>Allocation summary</strong>
            <button type="button" className="button" disabled>
              Run optimization
            </button>
          </div>
          <div className="grid-2" style={{ marginTop: 12 }}>
            <div className="metric-card">
              <span className="text-muted small">Expected Return</span>
              <strong>{optimizerQuery.data?.summary.expectedReturn}%</strong>
            </div>
            <div className="metric-card">
              <span className="text-muted small">Risk</span>
              <strong>{optimizerQuery.data?.summary.risk}%</strong>
            </div>
            <div className="metric-card">
              <span className="text-muted small">Sharpe</span>
              <strong>{optimizerQuery.data?.summary.sharpe}</strong>
            </div>
            <div className="metric-card">
              <span className="text-muted small">Max Drawdown</span>
              <strong>{optimizerQuery.data?.summary.maxDrawdown}%</strong>
            </div>
          </div>
          <div className="text-muted small" style={{ marginTop: 8 }}>
            Objective: {optimizerQuery.data?.summary.objective} | Capital: {`$${optimizerQuery.data?.summary.capital.toLocaleString()}`}
          </div>
        </div>

        <div className="card">
          <strong>Contract sizing (margin aware)</strong>
          <table className="compact-table">
            <thead>
              <tr>
                <th>Asset</th>
                <th>Contracts</th>
                <th>Notional</th>
                <th>Margin</th>
              </tr>
            </thead>
            <tbody>
              {optimizerQuery.data-.contracts.map((row) => (
                <tr key={row.asset}>
                  <td>{row.asset}</td>
                  <td>{row.contracts}</td>
                  <td>{`$${row.notional.toLocaleString()}`}</td>
                  <td>{`$${row.margin.toLocaleString()}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <strong>Weights</strong>
        <table className="compact-table">
          <thead>
            <tr>
              <th>Asset</th>
              <th>Weight</th>
              <th>Contracts</th>
              <th>Margin/Contract</th>
            </tr>
          </thead>
          <tbody>
            {optimizerQuery.data-.weights.map((row) => (
              <tr key={row.asset}>
                <td>{row.asset}</td>
                <td>{row.weight}</td>
                <td>{row.contracts}</td>
                <td>{`$${row.marginPerContract.toLocaleString()}`}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="text-muted small" style={{ marginTop: 8 }}>
          Includes the Dash inverse-vol fallback (set contracts to zero to exclude) and mirrors the allocation & contract tables.
        </div>
      </div>
    </div>
  );

  const renderCta = () => (
    <div className="panel" style={{ marginTop: 8 }}>
      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="section-title" style={{ margin: 0 }}>CTA-Style Report</h3>
        {activeBadge}
      </div>
      <p className="text-muted small" style={{ marginTop: 4 }}>
        Mirrors monthly/annual ROR tables, ROI/Annual ROR/ROA summary, time-in-market, longest flat, and download prompts from the
        Dash CTA tab.
      </p>

      <div className="metric-cards" style={{ marginTop: 12 }}>
        <div className="metric-card">
          <span className="text-muted small">ROI</span>
          <strong>{ctaQuery.data?.summary.roi}%</strong>
        </div>
        <div className="metric-card">
          <span className="text-muted small">Annual ROR</span>
          <strong>{ctaQuery.data?.summary.annualRor}%</strong>
        </div>
        <div className="metric-card">
          <span className="text-muted small">Time in Market</span>
          <strong>{ctaQuery.data?.summary.timeInMarket}%</strong>
        </div>
        <div className="metric-card">
          <span className="text-muted small">Longest Flat (days)</span>
          <strong>{ctaQuery.data?.summary.longestFlat}</strong>
        </div>
        <div className="metric-card">
          <span className="text-muted small">Max Run-up</span>
          <strong>{ctaQuery.data?.summary.maxRunup}%</strong>
        </div>
        <div className="metric-card">
          <span className="text-muted small">Max Drawdown</span>
          <strong>{ctaQuery.data?.summary.maxDrawdown}%</strong>
        </div>
      </div>

      <div className="grid-2" style={{ marginTop: 16 }}>
        <div className="card">
          <strong>Monthly Returns (additive vs compounded)</strong>
          <table className="compact-table">
            <thead>
              <tr>
                <th>Month</th>
                <th>Additive %</th>
                <th>Compounded %</th>
              </tr>
            </thead>
            <tbody>
              {ctaQuery.data-.monthly.map((row) => (
                <tr key={row.month}>
                  <td>{row.month}</td>
                  <td>{row.additive}%</td>
                  <td>{row.compounded}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card">
          <strong>Annual ROR</strong>
          <table className="compact-table">
            <thead>
              <tr>
                <th>Year</th>
                <th>Additive %</th>
                <th>Compounded %</th>
              </tr>
            </thead>
            <tbody>
              {ctaQuery.data-.annual.map((row) => (
                <tr key={row.year}>
                  <td>{row.year}</td>
                  <td>{row.additive}%</td>
                  <td>{row.compounded}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="text-muted small" style={{ marginTop: 8 }}>
            Additive vs compounded view matches the Dash helper text.
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <strong>Download actions</strong>
        <div className="flex gap-md" style={{ marginTop: 10, flexWrap: 'wrap' }}>
          <button className="button" type="button" disabled>
            Export monthly ROR (CSV)
          </button>
          <button className="button" type="button" disabled>
            Export annual ROR (CSV)
          </button>
          <button className="button" type="button" disabled>
            Export metrics (CSV/Parquet)
          </button>
        </div>
        <div className="text-muted small" style={{ marginTop: 8 }}>
          Hooks mirror Dash exports; backend wiring will route to /api/v1/export once implemented.
        </div>
      </div>
    </div>
  );

  const renderIngest = () => (
    <div className="panel" style={{ marginTop: 8 }}>
      <h3 className="section-title" style={{ margin: 0 }}>Load Trade Lists & Exports</h3>
      <p className="text-muted small" style={{ marginTop: 4 }}>
        Parity with the Dash "Load Trade Lists" and "Settings" tabs: upload area, included files grid, contract/margin overrides,
        account equity, spike toggle, downsample/export switches, and file metadata preview.
      </p>

      <div className="card">
        <div className="upload-area">Drag & drop or select .xlsx files</div>
        <div className="text-muted small" style={{ marginTop: 6 }}>Uploads mirror Dash (multiple files, persistent list).</div>
      </div>

      <div className="grid-2" style={{ marginTop: 14 }}>
        <div className="card">
          <strong>Included files</strong>
          <div className="chips" style={{ marginTop: 8 }}>
            {availableFiles.map((file) => {
              const active = activeSelection.files.includes(file);
              return (
                <button
                  key={file}
                  type="button"
                  className={`chip ${active ? 'chip-active' : ''}`}
                  onClick={() =>
                    setActiveSelection((prev) => ({
                      ...prev,
                      files: active
                        - prev.files.filter((f) => f !== file)
                        : [...prev.files, file],
                    }))
                  }
                >
                  {file}
                </button>
              );
            })}
          </div>
          <div className="text-muted small" style={{ marginTop: 8 }}>Matches Dash file checklist + contract inputs.</div>
        </div>

        <div className="card">
          <strong>Export options</strong>
          <div className="flex gap-md" style={{ marginTop: 10, alignItems: 'center' }}>
            <label className="field-label" htmlFor="downsample" style={{ margin: 0 }}>Downsample series</label>
            <input
              id="downsample"
              type="checkbox"
              checked={includeDownsample}
              onChange={(event) => setIncludeDownsample(event.target.checked)}
            />
            <label className="field-label" htmlFor="format" style={{ margin: 0 }}>Export format</label>
            <select id="format" className="input" value={exportFormat} onChange={(event) => setExportFormat(event.target.value as 'csv' | 'parquet')}>
              <option value="csv">CSV</option>
              <option value="parquet">Parquet</option>
            </select>
          </div>
          <div className="text-muted small" style={{ marginTop: 8 }}>Maps to /api/v1/export/trades and /api/v1/export/metrics.</div>
        </div>
      </div>

      <div className="grid-2" style={{ marginTop: 14 }}>
        <div className="card">
          <strong>Account equity</strong>
          <input
            className="input"
            type="number"
            value={accountEquity}
            onChange={(event) => setAccountEquity(Number(event.target.value))}
            style={{ marginTop: 8, maxWidth: 240 }}
          />
          <div className="text-muted small" style={{ marginTop: 8 }}>
            Mirrors Dash account equity input (used for purchasing power, allocator seed, and margin tab).
          </div>
        </div>
        <div className="card">
          <strong>Spike handling</strong>
          <div className="flex gap-md" style={{ alignItems: 'center', marginTop: 10 }}>
            <label className="field-label" htmlFor="spikes" style={{ margin: 0 }}>Include trade-level spikes</label>
            <input
              id="spikes"
              type="checkbox"
              checked={activeSelection.spike}
              onChange={(event) => setActiveSelection({ ...activeSelection, spike: event.target.checked })}
            />
          </div>
          <div className="text-muted small" style={{ marginTop: 8 }}>
            Mirrors the Dash spike toggle (overlays on equity/drawdown/margin charts).
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <strong>Metadata preview</strong>
        <table className="compact-table">
          <thead>
            <tr>
              <th>File</th>
              <th>Symbols</th>
              <th>Intervals</th>
              <th>Strategies</th>
              <th>Date range</th>
            </tr>
          </thead>
          <tbody>
            {availableFiles.map((file) => (
              <tr key={file}>
                <td>{file}</td>
                <td>Auto-parsed</td>
                <td>15 / 30 / 60</td>
                <td>Strategy A/B</td>
                <td>Rolling</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="text-muted small" style={{ marginTop: 8 }}>
          Acts as the Dash helper showing parsed symbol/interval/strategy/date metadata per file.
        </div>
      </div>
    </div>
  );

  const renderTab = () => {
    if (activeTab === 'overview') return renderOverview();
    if (activeTab === 'correlations') return renderCorrelations();
    if (activeTab === 'optimizer') return renderOptimizer();
    if (activeTab === 'cta') return renderCta();
    return renderIngest();
  };

  return (
    <div className="two-column">
      <div className="panel" style={{ position: 'sticky', top: 16 }}>
        <div className="flex gap-sm" style={{ alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div className="text-muted small">Portfolio Dashboard</div>
            <h1 style={{ margin: '6px 0 8px 0' }}>Selection driver</h1>
            <div className="text-muted small">
              {apiBase - (
                <span>
                  <span className="status-dot" /> API base configured: {apiBase}
                </span>
              ) : (
                <span>
                  <span className="status-dot" style={{ background: '#ffcb6b' }} /> Using mock data until NEXT_PUBLIC_API_BASE is set
                </span>
              )}
            </div>
          </div>
          <button
            type="button"
            className="button"
            onClick={() => {
              equityQuery.refetch();
              equityPctQuery.refetch();
              drawdownQuery.refetch();
              intradayDdQuery.refetch();
              netposQuery.refetch();
              marginQuery.refetch();
              histogramQuery.refetch();
              metricsQuery.refetch();
              correlationQuery.refetch();
              optimizerQuery.refetch();
              ctaQuery.refetch();
            }}
            disabled={busy}
          >
            {busy ? "Refreshing..." : "Refresh" }
          </button>
        </div>

        <div style={{ marginTop: 18 }}>
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="flex gap-md" style={{ alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <div className="text-muted small">Active selection</div>
                <strong>{activeSelection.name}</strong>
              </div>
              <div className="badge">{activeSelection.files.length} files</div>
            </div>
            <div className="text-muted small" style={{ marginTop: 8 }}>
              {activeSelection.files.join(', ')}
            </div>
          </div>
          <SelectionList selections={selections} active={activeSelection} onSelect={setActiveSelection} />
          <div style={{ marginTop: 12 }}>
            <SelectionControls
              selection={activeSelection}
              availableFiles={availableFiles}
              onChange={setActiveSelection}
            />
          </div>
        </div>
      </div>

      <div className="panel" style={{ overflow: 'hidden' }}>
        <div className="tabs">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              type="button"
              className={`tab ${activeTab === tab.key ? 'tab-active' : ''}`}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <div style={{ marginTop: 12 }}>{renderTab()}</div>
      </div>
    </div>
  );
}
