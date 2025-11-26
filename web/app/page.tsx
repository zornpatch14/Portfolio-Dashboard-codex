'use client';

import { useEffect, useMemo, useState } from 'react';
import { useQueries, useQuery } from '@tanstack/react-query';
import MetricsGrid from '../components/MetricsGrid';
import SeriesChart from '../components/SeriesChart';
import { HistogramChart } from '../components/HistogramChart';
import { SelectionControls } from '../components/SelectionControls';
import { CorrelationHeatmap } from '../components/CorrelationHeatmap';
import { EquityMultiChart } from '../components/EquityMultiChart';
import {
  fetchCorrelations,
  fetchCta,
  fetchHistogram,
  fetchMetrics,
  fetchOptimizer,
  fetchSeries,
  uploadFiles,
  getSelectionMeta,
  listFiles,
  SeriesKind,
} from '../lib/api';
import { loadSampleSelections, Selection } from '../lib/selections';

const selections = loadSampleSelections();
const SELECTION_STORAGE_KEY = 'portfolio-selection-state';
const UPLOAD_INPUT_ID = 'upload-input';
const tabs = [
  { key: 'load-trade-lists', label: 'Load Trade Lists' },
  { key: 'summary', label: 'Summary' },
  { key: 'equity-curves', label: 'Equity Curves' },
  { key: 'portfolio-drawdown', label: 'Portfolio Drawdown' },
  { key: 'margin', label: 'Margin' },
  { key: 'trade-pl-histogram', label: 'Trade P/L Histogram' },
  { key: 'correlations', label: 'Correlations' },
  { key: 'riskfolio', label: 'Riskfolio' },
  { key: 'cta-report', label: 'CTA Report' },
  { key: 'metrics', label: 'Metrics' },
  { key: 'inverse-volatility', label: 'Inverse Volatility' },
] as const;

type TabKey = (typeof tabs)[number]['key'];

export default function HomePage() {
  const [activeSelection, setActiveSelection] = useState<Selection>(selections[0]);
  const [activeTab, setActiveTab] = useState<TabKey>('load-trade-lists');
  const [corrMode, setCorrMode] = useState('drawdown_pct');
  const [accountEquity, setAccountEquity] = useState(50000);
  const [includeDownsample, setIncludeDownsample] = useState(true);
  const [exportFormat, setExportFormat] = useState<'csv' | 'parquet'>('parquet');
  const [plotEnabled, setPlotEnabled] = useState<Record<string, boolean>>({});
  const [plotDrawdownEnabled, setPlotDrawdownEnabled] = useState<Record<string, boolean>>({});
  const [plotMarginEnabled, setPlotMarginEnabled] = useState<Record<string, boolean>>({});
  const [plotHistogramEnabled, setPlotHistogramEnabled] = useState<Record<string, boolean>>({});
  const [riskfolioMode, setRiskfolioMode] = useState<'mean-risk' | 'risk-parity' | 'hierarchical'>('mean-risk');
  const [selectionMeta, setSelectionMeta] = useState<Awaited<ReturnType<typeof getSelectionMeta>> | null>(null);
  const [filesMeta, setFilesMeta] = useState<Awaited<ReturnType<typeof listFiles>>>([]);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const apiBase = process.env.NEXT_PUBLIC_API_BASE;
  const apiMissing = !apiBase;
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    try {
      const stored = typeof window !== 'undefined' ? localStorage.getItem(SELECTION_STORAGE_KEY) : null;
      if (stored) {
        const parsed = JSON.parse(stored) as { selection: Selection; includeDownsample?: boolean };
        if (parsed.selection) {
          setActiveSelection(parsed.selection);
        }
        if (typeof parsed.includeDownsample === 'boolean') {
          setIncludeDownsample(parsed.includeDownsample);
        }
      }
    } catch {
      // ignore corrupted storage
    }
  }, []);

  useEffect(() => {
    try {
      const payload = JSON.stringify({ selection: activeSelection, includeDownsample });
      if (typeof window !== 'undefined') {
        localStorage.setItem(SELECTION_STORAGE_KEY, payload);
      }
    } catch {
      // ignore write failures
    }
  }, [activeSelection, includeDownsample]);

  useEffect(() => {
    if (apiMissing) return;
    (async () => {
      try {
        const meta = await getSelectionMeta();
        setSelectionMeta(meta);
        setFilesMeta(meta.files);
        if (!activeSelection.fileIds || !activeSelection.fileIds.length) {
          const ids = meta.files.map((f) => f.file_id);
          const labels = Object.fromEntries(meta.files.map((f) => [f.file_id, f.filename]));
          setActiveSelection((prev) => ({ ...prev, fileIds: ids, fileLabels: labels, files: ids }));
        }
      } catch (error: any) {
        console.warn('Failed to load selection metadata', error);
        setErrorMessage(error?.message || 'Failed to load selection metadata');
      }
    })();
  }, [apiMissing]);

  const availableFiles = useMemo(
    () => (filesMeta.length ? filesMeta.map((f) => f.file_id).sort() : Array.from(new Set(selections.flatMap((s) => s.files))).sort()),
    [filesMeta],
  );

  const fileLabelMap = useMemo(
    () => Object.fromEntries(filesMeta.map((f) => [f.file_id, f.filename])),
    [filesMeta],
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
        queryKey: [kind, activeSelection.name, (activeSelection.fileIds || activeSelection.files).join(',')],
        queryFn: () => fetchSeries(activeSelection, kind, includeDownsample),
      })),
      {
        queryKey: ['histogram', activeSelection.name, (activeSelection.fileIds || activeSelection.files).join(',')],
        queryFn: () => fetchHistogram(activeSelection),
      },
      {
        queryKey: ['metrics', activeSelection.name, (activeSelection.fileIds || activeSelection.files).join(',')],
        queryFn: () => fetchMetrics(activeSelection),
      },
    ],
  });

  useEffect(() => {
    const points = equityQuery.data?.data ?? [];
    const timestamps = points.map((p) => p.timestamp).filter(Boolean) ?? [];
    if (!timestamps.length) return;
    const sorted = [...timestamps].sort();
    const rangeStart = sorted[0];
    const rangeEnd = sorted[sorted.length - 1];
    setActiveSelection((prev) => {
      const nextStart = prev.start ?? rangeStart;
      const nextEnd = prev.end ?? rangeEnd;
      if (nextStart === prev.start && nextEnd === prev.end) return prev;
      return { ...prev, start: nextStart, end: nextEnd };
    });
  }, [equityQuery.data, activeSelection.name]);

  const deriveFileMeta = useMemo(() => {
    return (fileId: string) => {
      const label = fileLabelMap[fileId] || fileId;
      const base = label.split('/').pop() || label;
      const parts = base.replace(/\.[^.]+$/, '').split('_');
      return {
        symbol: parts[1] || '',
        interval: parts[2] || '',
        strategy: parts.slice(3).join('_') || '',
      };
    };
  }, [fileLabelMap]);

  const equityLines = useMemo(() => {
    const portfolioPoints = (equityQuery.data?.data ?? []).map((p) => ({ timestamp: p.timestamp, value: p.value }));
    return { perFile: [], portfolio: portfolioPoints };
  }, [equityQuery.data]);

  const equityPercentLines = useMemo(() => {
    const portfolioPoints = (equityPctQuery.data?.data ?? []).map((p) => ({ timestamp: p.timestamp, value: p.value }));
    return { perFile: [], portfolio: portfolioPoints };
  }, [equityPctQuery.data]);

  useEffect(() => {
    const names = new Set<string>();
    equityLines.perFile.forEach((s) => names.add(s.name));
    equityPercentLines.perFile.forEach((s) => names.add(s.name));
    names.add('Portfolio');
    setPlotEnabled((prev) => {
      const next = { ...prev };
      names.forEach((name) => {
        if (next[name] === undefined) next[name] = true;
      });
      Object.keys(next).forEach((key) => {
        if (!names.has(key)) delete next[key];
      });
      return next;
    });
  }, [equityLines.perFile, equityPercentLines.perFile]);

  const drawdownLines = useMemo(() => {
    const portfolioPoints = (drawdownQuery.data?.data ?? []).map((p) => ({ timestamp: p.timestamp, value: p.value }));
    return { perFile: [], portfolio: portfolioPoints };
  }, [drawdownQuery.data]);

  const drawdownPercentLines = useMemo(() => {
    const equityBase = accountEquity || 1;
    return {
      perFile: drawdownLines.perFile.map((s) => ({
        name: s.name,
        points: s.points.map((p) => ({ ...p, value: (p.value / equityBase) * 100 })),
      })),
      portfolio: drawdownLines.portfolio.map((p) => ({ ...p, value: (p.value / equityBase) * 100 })),
    };
  }, [drawdownLines, accountEquity]);

  useEffect(() => {
    const names = new Set<string>();
    drawdownLines.perFile.forEach((s) => names.add(s.name));
    names.add('Portfolio');
    setPlotDrawdownEnabled((prev) => {
      const next = { ...prev };
      names.forEach((name) => {
        if (next[name] === undefined) next[name] = true;
      });
      Object.keys(next).forEach((key) => {
        if (!names.has(key)) delete next[key];
      });
      return next;
    });
  }, [drawdownLines.perFile]);

  const marginLines = useMemo(() => {
    const portfolioPoints = (marginQuery.data?.data ?? []).map((p) => ({ timestamp: p.timestamp, value: p.value }));
    return { perFile: [], portfolio: portfolioPoints };
  }, [marginQuery.data]);

  const netposLines = useMemo(() => {
    const portfolioPoints = (netposQuery.data?.data ?? []).map((p) => ({ timestamp: p.timestamp, value: p.value }));
    return { perFile: [], portfolio: portfolioPoints };
  }, [netposQuery.data]);

  const purchasingPowerLines = useMemo(() => {
    const timeline = Array.from(
      new Set([
        ...equityLines.portfolio.map((p) => p.timestamp),
        ...marginLines.portfolio.map((p) => p.timestamp),
      ]),
    ).sort();

    const perFile = equityLines.perFile.map((eq) => {
      const margin = marginLines.perFile.find((m) => m.name === eq.name);
      const points = timeline.map((ts) => {
        const equityVal = eq.points.find((p) => p.timestamp === ts)?.value ?? 0;
        const marginVal = margin?.points.find((p) => p.timestamp === ts)?.value ?? 0;
        const value = accountEquity + equityVal - marginVal;
        return { timestamp: ts, value };
      });
      return { name: eq.name, points };
    });

    const portfolioPoints = timeline.map((ts) => {
      const equityVal = equityLines.portfolio.find((p) => p.timestamp === ts)?.value ?? 0;
      const marginVal = marginLines.portfolio.find((p) => p.timestamp === ts)?.value ?? 0;
      return { timestamp: ts, value: accountEquity + equityVal - marginVal };
    });

    return { perFile, portfolio: portfolioPoints };
  }, [equityLines, marginLines, accountEquity]);

  const purchasingPowerDrawdownLines = useMemo(() => {
    const toDrawdown = (points: { timestamp: string; value: number }[]) => {
      let maxSoFar = Number.NEGATIVE_INFINITY;
      return points.map((p) => {
        maxSoFar = Math.max(maxSoFar, p.value);
        const dd = maxSoFar === 0 ? 0 : ((p.value - maxSoFar) / maxSoFar) * 100;
        return { ...p, value: dd };
      });
    };
    return {
      perFile: purchasingPowerLines.perFile.map((s) => ({ name: s.name, points: toDrawdown(s.points) })),
      portfolio: toDrawdown(purchasingPowerLines.portfolio),
    };
  }, [purchasingPowerLines]);

  useEffect(() => {
    const names = new Set<string>();
    purchasingPowerLines.perFile.forEach((s) => names.add(s.name));
    names.add('Portfolio');
    setPlotMarginEnabled((prev) => {
      const next = { ...prev };
      names.forEach((name) => {
        if (next[name] === undefined) next[name] = true;
      });
      Object.keys(next).forEach((key) => {
        if (!names.has(key)) delete next[key];
      });
      return next;
    });
  }, [purchasingPowerLines.perFile]);

  const histogramData = useMemo(() => {
    const portfolioBuckets = histogramQuery.data?.buckets ?? [];
    return portfolioBuckets.length ? [{ name: 'Portfolio', buckets: portfolioBuckets }] : [];
  }, [histogramQuery.data]);

  useEffect(() => {
    const names = new Set<string>();
    histogramData.forEach((h) => names.add(h.name));
    names.add('Portfolio');
    setPlotHistogramEnabled((prev) => {
      const next = { ...prev };
      names.forEach((name) => {
        if (next[name] === undefined) next[name] = true;
      });
      Object.keys(next).forEach((key) => {
        if (!names.has(key)) delete next[key];
      });
      return next;
    });
  }, [histogramData]);

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

  const renderSummary = () => (
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
          <strong>{metricsSummary ? `$${metricsSummary.netProfit.toLocaleString()}` : '--'}</strong>
        </div>
        <div className="metric-card">
          <span className="text-muted small">Drawdown (sum)</span>
          <strong>{metricsSummary ? `$${metricsSummary.drawdown.toLocaleString()}` : '--'}</strong>
        </div>
        <div className="metric-card">
          <span className="text-muted small">Avg Trades</span>
          <strong>{metricsSummary ? metricsSummary.avgTrades : '--'}</strong>
        </div>
        <div className="metric-card">
          <span className="text-muted small">Avg Win %</span>
          <strong>{metricsSummary ? `${metricsSummary.avgWin.toFixed(1)}%` : '--'}</strong>
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
          {equityQuery.data ? (
            <SeriesChart title="Equity Curve" series={equityQuery.data} color="#4cc3ff" />
          ) : (
            <div className="placeholder-text">No equity data.</div>
          )}
          <div className="text-muted small">
            Points: {equityQuery.data?.downsampled_count ?? equityQuery.data?.data.length ?? 0}
          </div>
        </div>
        <div className="card">
          {equityPctQuery.data ? (
            <SeriesChart title="Percent Equity" series={equityPctQuery.data} color="#8fe3c7" />
          ) : (
            <div className="placeholder-text">No percent equity data.</div>
          )}
          <div className="text-muted small">
            Points: {equityPctQuery.data?.downsampled_count ?? equityPctQuery.data?.data.length ?? 0}
          </div>
        </div>
        <div className="card">
          {drawdownQuery.data ? (
            <SeriesChart title="Drawdown" series={drawdownQuery.data} color="#ff8f6b" />
          ) : (
            <div className="placeholder-text">No drawdown data.</div>
          )}
          <div className="text-muted small">
            Points: {drawdownQuery.data?.downsampled_count ?? drawdownQuery.data?.data.length ?? 0}
          </div>
        </div>
        <div className="card">
          {intradayDdQuery.data ? (
            <SeriesChart title="Intraday Drawdown" series={intradayDdQuery.data} color="#f4c95d" />
          ) : (
            <div className="placeholder-text">No intraday drawdown data.</div>
          )}
          <div className="text-muted small">
            Points: {intradayDdQuery.data?.downsampled_count ?? intradayDdQuery.data?.data.length ?? 0}
          </div>
        </div>
        <div className="card">
          {netposQuery.data ? (
            <SeriesChart title="Net Position" series={netposQuery.data} color="#9f8bff" />
          ) : (
            <div className="placeholder-text">No net position data.</div>
          )}
          <div className="text-muted small">
            Points: {netposQuery.data?.downsampled_count ?? netposQuery.data?.data.length ?? 0}
          </div>
        </div>
        <div className="card">
          {marginQuery.data ? (
            <SeriesChart title="Margin Usage" series={marginQuery.data} color="#54ffd0" />
          ) : (
            <div className="placeholder-text">No margin data.</div>
          )}
          <div className="text-muted small">
            Points: {marginQuery.data?.downsampled_count ?? marginQuery.data?.data.length ?? 0}
          </div>
        </div>
        <div className="card">
          {histogramQuery.data ? (
            <HistogramChart histogram={histogramQuery.data} />
          ) : (
            <div className="placeholder-text">No histogram data.</div>
          )}
          <div className="text-muted small">
            Distribution: {histogramQuery.data?.buckets.length ?? 0} buckets
          </div>
        </div>
      </div>

      <div style={{ marginTop: 18 }}>
        <h3 className="section-title">Per-file metrics</h3>
        {metricsQuery.data && metricsQuery.data.length ? (
          <MetricsGrid rows={metricsQuery.data} />
        ) : (
          <div className="placeholder-text">No metrics returned yet.</div>
        )}
      </div>
    </div>
  );

  const renderEquityCurves = () => (
    <div className="panel" style={{ marginTop: 8 }}>
      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="section-title" style={{ margin: 0 }}>Equity Curves</h3>
        {activeBadge}
      </div>
      <div className="text-muted small" style={{ marginTop: 6 }}>
        Equity curves honor filters (symbols/intervals/strategies), contract multipliers, and date range. Setting contracts to zero or toggling off filters will exclude that file from the chart and portfolio line.
      </div>
      <div className="card" style={{ marginTop: 12 }}>
        <strong>Plot lines</strong>
        <div className="chips" style={{ marginTop: 10, flexWrap: 'wrap' }}>
          {[...equityLines.perFile.map((s) => s.name), 'Portfolio'].map((name) => {
            const active = plotEnabled[name] ?? true;
            return (
              <button
                key={name}
                type="button"
                className={`chip ${active ? 'chip-active' : ''}`}
                onClick={() => setPlotEnabled((prev) => ({ ...prev, [name]: !active }))}
              >
                {name}
              </button>
            );
          })}
        </div>
      </div>

      <div style={{ marginTop: 12, display: 'grid', gap: 12 }}>
        <EquityMultiChart
          title="Equity Curve ($)"
          series={[
            ...equityLines.perFile.filter((s) => plotEnabled[s.name] !== false),
            ...(plotEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: equityLines.portfolio }]),
          ]}
        />
        <EquityMultiChart
          title="Equity Curve (%)"
          series={[
            ...equityPercentLines.perFile.filter((s) => plotEnabled[s.name] !== false),
            ...(plotEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: equityPercentLines.portfolio }]),
          ]}
        />
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
            value={corrMode === 'slope' ? 20 : ''}
            placeholder="20"
            disabled={corrMode !== 'slope'}
          />
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <CorrelationHeatmap data={correlationQuery.data ?? mockCorrelations(activeSelection, corrMode)} />
      </div>
      <div className="text-muted small" style={{ marginTop: 10 }}>
        {correlationQuery.data?.notes.map((note) => (
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
        Upload and select files to enable optimization. Controls mirror the legacy Riskfolio tab (objective, return model, risk
        measure, covariance method, bounds, budget, caps, and utility inputs). Backend wiring comes later.
      </p>

      <div className="card" style={{ marginTop: 10 }}>
        <div className="tabs" style={{ borderBottom: 'none', gap: 6 }}>
          <button
            type="button"
            className={`tab ${riskfolioMode === 'mean-risk' ? 'tab-active' : ''}`}
            onClick={() => setRiskfolioMode('mean-risk')}
          >
            Mean-Risk
          </button>
          <button type="button" className="tab" style={{ opacity: 0.5 }} disabled>
            Risk Parity (coming soon)
          </button>
          <button type="button" className="tab" style={{ opacity: 0.5 }} disabled>
            Hierarchical (coming soon)
          </button>
        </div>
      </div>

      {riskfolioMode === 'mean-risk' ? (
        <>
          <div className="grid-2" style={{ marginTop: 12 }}>
            <div className="card">
              <label className="field-label" htmlFor="objective">Objective</label>
              <select id="objective" className="input" defaultValue="max_rar">
                <option value="max_return">Maximum Return</option>
                <option value="min_risk">Minimum Risk</option>
                <option value="max_rar">Maximum Risk-Adjusted Return Ratio</option>
                <option value="utility">Maximum Utility</option>
              </select>
              <label className="field-label" htmlFor="return-model" style={{ marginTop: 12 }}>Return Model</label>
              <select id="return-model" className="input" defaultValue="arithmetic">
                <option value="arithmetic">Mean Historical Return (Arithmetic)</option>
                <option value="approx_log">Approximate Log</option>
                <option value="log">Exact Log</option>
              </select>
              <label className="field-label" htmlFor="risk-measure" style={{ marginTop: 12 }}>Risk Measure</label>
              <select id="risk-measure" className="input" defaultValue="variance">
                <option value="variance">Variance</option>
                <option value="std">Standard Deviation</option>
                <option value="semi_variance">Semi-Variance</option>
                <option value="cvar">CVaR</option>
                <option value="cdar">CDaR</option>
                <option value="evar">EVaR</option>
              </select>
              <label className="field-label" htmlFor="cov-method" style={{ marginTop: 12 }}>Covariance Method</label>
              <select id="cov-method" className="input" defaultValue="sample">
                <option value="sample">Sample (Normal)</option>
                <option value="ewma">Exponentially Weighted (EWMA)</option>
                <option value="ledoit-wolf">Ledoit-Wolf</option>
              </select>
            </div>

            <div className="card">
              <label className="field-label" htmlFor="rf">Risk-Free Rate (annual %)</label>
              <input id="rf" className="input" type="number" step={0.05} defaultValue={0} />
              <label className="field-label" htmlFor="risk-aversion" style={{ marginTop: 12 }}>Risk Aversion (Utility only)</label>
              <input id="risk-aversion" className="input" type="number" step={0.1} defaultValue={2} />
              <label className="field-label" htmlFor="alpha" style={{ marginTop: 12 }}>Confidence Level alpha</label>
              <input id="alpha" className="input" type="number" min={0} max={1} step={0.01} defaultValue={0.95} />
              <label className="field-label" htmlFor="bounds" style={{ marginTop: 12 }}>Weight Bounds (min / max)</label>
              <div className="flex gap-sm">
                <input className="input" type="number" defaultValue={0} />
                <input className="input" type="number" defaultValue={0.6} />
              </div>
              <label className="field-label" htmlFor="budget" style={{ marginTop: 12 }}>Budget (sum of weights)</label>
              <input id="budget" className="input" type="number" step={0.1} defaultValue={1} />
              <label className="field-label" htmlFor="group-caps" style={{ marginTop: 12 }}>Group Caps (symbol=cap, comma separated)</label>
              <input id="group-caps" className="input" placeholder="e.g. MNQ,MES=0.5" />
              <label className="field-label" htmlFor="strategy-caps" style={{ marginTop: 12 }}>Strategy Caps (strategy=cap, comma separated)</label>
              <input id="strategy-caps" className="input" placeholder="e.g. trend=0.6, mean_reversion=0.4" />
              <label className="field-label" htmlFor="turnover" style={{ marginTop: 12 }}>Turnover Cap (reserved)</label>
              <input id="turnover" className="input" placeholder="Reserved" disabled />
            </div>
          </div>

          <div className="card" style={{ marginTop: 14 }}>
            <strong>Optimize</strong>
            <div className="flex gap-md" style={{ marginTop: 10, alignItems: 'center', flexWrap: 'wrap' }}>
              <button type="button" className="button" disabled>
                Optimize
              </button>
              <div className="text-muted small">Progress</div>
              <div style={{ flex: 1, minWidth: 180, background: '#1f2f4a', borderRadius: 6, height: 10, overflow: 'hidden' }}>
                <div style={{ width: '12%', background: '#4cc3ff', height: '100%' }} />
              </div>
            </div>
            <div className="text-muted small" style={{ marginTop: 8 }}>
              Riskfolio results will appear below (efficient frontier, risk contribution, backtested equity, asset correlation).
            </div>
          </div>

          <div className="grid-2" style={{ marginTop: 16 }}>
            <div className="card">
              <strong>Allocation summary</strong>
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
                  {optimizerQuery.data?.contracts.map((row) => (
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
                {optimizerQuery.data?.weights.map((row) => (
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

          <div className="card" style={{ marginTop: 14 }}>
            <strong>Suggested Contracts</strong>
            <div className="flex gap-md" style={{ alignItems: 'center', marginTop: 10, flexWrap: 'wrap' }}>
              <label className="field-label" htmlFor="riskfolio-equity" style={{ margin: 0 }}>Account Equity</label>
              <input id="riskfolio-equity" className="input" type="number" defaultValue={accountEquity} style={{ maxWidth: 200 }} />
              <button type="button" className="button" disabled>
                Apply Suggested Contracts
              </button>
            </div>
            <div className="text-muted small" style={{ marginTop: 10 }}>
              Columns: File, Symbol, Optimized Weight, Suggested Weight Gap, Current Weight Gap, Margin / Contract, Suggested Contracts,
              Suggested Margin, Current Contracts, Current Margin, Delta.
            </div>
            <div className="table-wrapper" style={{ marginTop: 10 }}>
              <table className="compact-table">
                <thead>
                  <tr>
                    <th>File</th>
                    <th>Symbol</th>
                    <th>Optimized Weight</th>
                    <th>Suggested Weight Gap</th>
                    <th>Current Weight Gap</th>
                    <th>Margin / Contract</th>
                    <th>Suggested Contracts</th>
                    <th>Suggested Margin</th>
                    <th>Current Contracts</th>
                    <th>Current Margin</th>
                    <th>Delta</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Example.xlsx</td>
                    <td>MNQ</td>
                    <td>0.24</td>
                    <td>+0.05</td>
                    <td>-0.02</td>
                    <td>$3,500</td>
                    <td>3</td>
                    <td>$10,500</td>
                    <td>2</td>
                    <td>$7,000</td>
                    <td>+1</td>
                  </tr>
                  <tr>
                    <td>Example2.xlsx</td>
                    <td>MES</td>
                    <td>0.18</td>
                    <td>+0.03</td>
                    <td>-0.01</td>
                    <td>$2,800</td>
                    <td>2</td>
                    <td>$5,600</td>
                    <td>1</td>
                    <td>$2,800</td>
                    <td>+1</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="grid-2" style={{ marginTop: 16 }}>
            <div className="card">
              <strong>Efficient Frontier</strong>
              <div className="placeholder-text">Frontier chart placeholder</div>
            </div>
            <div className="card">
              <strong>Risk Contribution (%)</strong>
              <div className="placeholder-text">Risk contribution bar chart placeholder</div>
            </div>
            <div className="card">
              <strong>Backtested Portfolio Equity</strong>
              <div className="placeholder-text">Backtested equity chart placeholder</div>
            </div>
            <div className="card">
              <strong>Asset Correlation</strong>
              <div className="placeholder-text">Asset correlation heatmap placeholder</div>
            </div>
          </div>
        </>
      ) : (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="text-muted small">This optimisation method is coming soon. Switch to Mean-Risk to configure controls.</div>
        </div>
      )}
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
              {ctaQuery.data?.monthly.map((row) => (
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
              {ctaQuery.data?.annual.map((row) => (
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

  const renderPortfolioDrawdown = () => (
    <div className="panel" style={{ marginTop: 8 }}>
      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="section-title" style={{ margin: 0 }}>Portfolio Drawdown</h3>
        {activeBadge}
      </div>
      <div className="card" style={{ marginTop: 12 }}>
        <strong>Plot lines</strong>
        <div className="chips" style={{ marginTop: 10, flexWrap: 'wrap' }}>
          {[...drawdownLines.perFile.map((s) => s.name), 'Portfolio'].map((name) => {
            const active = plotDrawdownEnabled[name] ?? true;
            return (
              <button
                key={name}
                type="button"
                className={`chip ${active ? 'chip-active' : ''}`}
                onClick={() => setPlotDrawdownEnabled((prev) => ({ ...prev, [name]: !active }))}
              >
                {name}
              </button>
            );
          })}
        </div>
      </div>

      <div style={{ marginTop: 12, display: 'grid', gap: 12 }}>
        <EquityMultiChart
          title="Portfolio Drawdown ($)"
          series={[
            ...drawdownLines.perFile.filter((s) => plotDrawdownEnabled[s.name] !== false),
            ...(plotDrawdownEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: drawdownLines.portfolio }]),
          ]}
        />
        <EquityMultiChart
          title="Portfolio Drawdown (%)"
          series={[
            ...drawdownPercentLines.perFile.filter((s) => plotDrawdownEnabled[s.name] !== false),
            ...(plotDrawdownEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: drawdownPercentLines.portfolio }]),
          ]}
        />
      </div>
    </div>
  );

  const renderIntradayDrawdown = () => (
    <div className="panel" style={{ marginTop: 8 }}>
      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="section-title" style={{ margin: 0 }}>Intraday Drawdown</h3>
        {activeBadge}
      </div>
      <div className="card" style={{ marginTop: 12 }}>
        {intradayDdQuery.data ? (
          <>
            <SeriesChart title="Intraday Drawdown" series={intradayDdQuery.data} color="#f4c95d" />
            <div className="text-muted small">
              Points: {intradayDdQuery.data?.downsampled_count ?? intradayDdQuery.data?.data.length ?? 0}
            </div>
          </>
        ) : (
          <div className="placeholder-text">No intraday drawdown data.</div>
        )}
      </div>
    </div>
  );

  const renderMargin = () => (
    <div className="panel" style={{ marginTop: 8 }}>
      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="section-title" style={{ margin: 0 }}>Margin</h3>
        {activeBadge}
      </div>
      <div className="text-muted small" style={{ marginTop: 6 }}>
        Margin views honor filters (symbols/intervals/strategies), contract multipliers, and date range. Setting contracts to zero or toggling off filters will exclude that file from the chart and portfolio line.
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <strong>Plot lines</strong>
        <div className="chips" style={{ marginTop: 10, flexWrap: 'wrap' }}>
          {[...purchasingPowerLines.perFile.map((s) => s.name), 'Portfolio'].map((name) => {
            const active = plotMarginEnabled[name] ?? true;
            return (
              <button
                key={name}
                type="button"
                className={`chip ${active ? 'chip-active' : ''}`}
                onClick={() => setPlotMarginEnabled((prev) => ({ ...prev, [name]: !active }))}
              >
                {name}
              </button>
            );
          })}
        </div>
      </div>

      <div style={{ marginTop: 12, display: 'grid', gap: 12 }}>
        <EquityMultiChart
          title="Purchasing Power ($)"
          description="Purchasing Power = Starting Balance + Portfolio P&L - Initial Margin Used"
          series={[
            ...purchasingPowerLines.perFile.filter((s) => plotMarginEnabled[s.name] !== false),
            ...(plotMarginEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: purchasingPowerLines.portfolio }]),
          ]}
        />
        <EquityMultiChart
          title="Purchasing Power Drawdown (%)"
          description="Purchasing Power Drawdown (percentage from peak)"
          series={[
            ...purchasingPowerDrawdownLines.perFile.filter((s) => plotMarginEnabled[s.name] !== false),
            ...(plotMarginEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: purchasingPowerDrawdownLines.portfolio }]),
          ]}
        />
        <EquityMultiChart
          title="Initial Margin Used ($)"
          description="Initial Margin Used = |net_sym| x IM(sym)"
          series={[
            ...marginLines.perFile.filter((s) => plotMarginEnabled[s.name] !== false),
            ...(plotMarginEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: marginLines.portfolio }]),
          ]}
        />
        <EquityMultiChart
          title="Net Contracts"
          description="Net contracts over time (per file and portfolio)"
          series={[
            ...netposLines.perFile.filter((s) => plotMarginEnabled[s.name] !== false),
            ...(plotMarginEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: netposLines.portfolio }]),
          ]}
        />
      </div>
    </div>
  );

  const renderHistogram = () => (
    <div className="panel" style={{ marginTop: 8 }}>
      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="section-title" style={{ margin: 0 }}>Trade P/L Histogram</h3>
        {activeBadge}
      </div>
      <div className="card" style={{ marginTop: 12 }}>
        <strong>Plot lines</strong>
        <div className="chips" style={{ marginTop: 10, flexWrap: 'wrap' }}>
          {[...histogramData.map((h) => h.name), 'Portfolio'].map((name) => {
            const active = plotHistogramEnabled[name] ?? true;
            return (
              <button
                key={name}
                type="button"
                className={`chip ${active ? 'chip-active' : ''}`}
                onClick={() => setPlotHistogramEnabled((prev) => ({ ...prev, [name]: !active }))}
              >
                {name}
              </button>
            );
          })}
        </div>
      </div>

      {(() => {
        const activeHists = histogramData.filter((h) => plotHistogramEnabled[h.name] !== false);
        const bucketOrder = activeHists[0]?.buckets.map((b) => b.bucket) ?? [];
        const bucketMap = new Map(bucketOrder.map((b) => [b, 0]));
        activeHists.forEach((hist) => {
          hist.buckets.forEach((bucket) => {
            bucketMap.set(bucket.bucket, (bucketMap.get(bucket.bucket) || 0) + bucket.count);
          });
        });
        const parsePct = (label: string) => {
          const cleaned = label.replace('%', '');
          const val = Number.parseFloat(cleaned);
          return Number.isFinite(val) ? val : 0;
        };
        const portfolioBuckets = bucketOrder.map((bucket) => ({
          bucket,
          count: bucketMap.get(bucket) || 0,
        }));
        const portfolioBucketsDollar = bucketOrder.map((bucket) => {
          const pct = parsePct(bucket);
          const dollars = (pct / 100) * accountEquity;
          const label = dollars >= 0 ? `$${dollars.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : `-$${Math.abs(dollars).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
          return { bucket: label, count: bucketMap.get(bucket) || 0 };
        });
        return (
          <div style={{ marginTop: 12, display: 'grid', gap: 12 }}>
            <div className="card">
              <strong>Portfolio Histogram ($)</strong>
              <div className="text-muted small" style={{ marginTop: 4 }}>
                Portfolio distribution; toggles exclude selected files from the sum.
              </div>
              <HistogramChart histogram={{ label: 'Portfolio Histogram ($)', buckets: portfolioBucketsDollar }} />
              <div className="text-muted small" style={{ marginTop: 8 }}>Buckets: {portfolioBucketsDollar.length}</div>
            </div>
            <div className="card">
              <strong>Portfolio Histogram (%)</strong>
              <div className="text-muted small" style={{ marginTop: 4 }}>
                Portfolio return distribution; toggles exclude selected files from the sum.
              </div>
              <HistogramChart histogram={{ label: 'Portfolio Histogram (%)', buckets: portfolioBuckets }} />
              <div className="text-muted small" style={{ marginTop: 8 }}>Buckets: {portfolioBuckets.length}</div>
            </div>
          </div>
        );
      })()}
    </div>
  );

  const renderMetrics = () => (
    <div className="panel" style={{ marginTop: 8 }}>
      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="section-title" style={{ margin: 0 }}>Metrics</h3>
        {activeBadge}
      </div>
      <div style={{ marginTop: 12 }}>
        {metricsQuery.data && metricsQuery.data.length ? (
          <MetricsGrid rows={metricsQuery.data} />
        ) : (
          <div className="placeholder-text">No metrics returned yet.</div>
        )}
      </div>
    </div>
  );

  const renderInverseVolatility = () => (
    <div className="panel" style={{ marginTop: 8 }}>
      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="section-title" style={{ margin: 0 }}>Inverse Volatility</h3>
        {activeBadge}
      </div>
      <div className="card" style={{ marginTop: 12 }}>
        <div className="text-muted small">Placeholder for inverse-volatility controls and tables from the legacy app.</div>
      </div>
    </div>
  );

  const renderAllocator = () => (
    <div className="panel" style={{ marginTop: 8 }}>
      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="section-title" style={{ margin: 0 }}>Allocator</h3>
        {activeBadge}
      </div>
      <div className="card" style={{ marginTop: 12 }}>
        <div className="text-muted small">Allocator UI placeholder to mirror the legacy tab (sizing, cash, contracts).</div>
      </div>
    </div>
  );

  const renderSettings = () => (
    <div className="panel" style={{ marginTop: 8 }}>
      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="section-title" style={{ margin: 0 }}>Settings</h3>
        {activeBadge}
      </div>
      <div className="card" style={{ marginTop: 12 }}>
        <div className="text-muted small">
          Selection filters live in the left rail (symbols, timeframes, strategies, direction, date range, spike toggle). This tab will
          mirror the legacy settings layout as needed.
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

      <div className="card" style={{ marginTop: 14 }}>
        <div className="upload-area">
          <label htmlFor={UPLOAD_INPUT_ID} style={{ cursor: 'pointer', display: 'block' }}>
            Drag & drop or select .xlsx files
          </label>
          <input
            id={UPLOAD_INPUT_ID}
            type="file"
            accept=".xlsx"
            multiple
            style={{ display: 'none' }}
            onChange={async (event) => {
              const files = event.target.files;
              if (!files || !files.length) return;
              const formData = new FormData();
              Array.from(files).forEach((file) => formData.append('files', file));
              try {
                setUploadStatus('Uploading...');
                const response = await uploadFiles(formData);
                setUploadStatus(`Upload job ${response.job_id} completed (${response.files.length} files ingested)`);
                const meta = await getSelectionMeta();
                setSelectionMeta(meta);
                setFilesMeta(meta.files);
                const ids = meta.files.map((f) => f.file_id);
                const labels = Object.fromEntries(meta.files.map((f) => [f.file_id, f.filename]));
                setActiveSelection((prev) => ({
                  ...prev,
                  fileIds: ids,
                  fileLabels: labels,
                  files: ids,
                }));
              } catch (error) {
                setUploadStatus('Upload failed');
                setErrorMessage(error instanceof Error ? error.message : 'Upload failed');
              } finally {
                if (event.target) {
                  event.target.value = '';
                }
              }
            }}
          />
        </div>
        <div className="text-muted small" style={{ marginTop: 6 }}>
          Uploads mirror Dash (multiple files, persistent list).
          {uploadStatus ? ` ${uploadStatus}` : ''}
        </div>
      </div>

      <div className="grid-2" style={{ marginTop: 14 }}>
        <div className="card">
          <strong>Included files</strong>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 120px 140px',
              gap: 10,
              alignItems: 'center',
              marginTop: 8,
            }}
          >
            <div className="text-muted small" />
            <button
              type="button"
              className="button"
              style={{ width: '100%' }}
              onClick={() =>
                setActiveSelection((prev) => ({
                  ...prev,
                  contracts: Object.fromEntries(availableFiles.map((file) => [file, 1])),
                }))
              }
            >
              Use default contracts (1)
            </button>
            <button type="button" className="button" style={{ width: '100%' }} disabled>
              Use default margin (placeholder)
            </button>
          </div>
          <div className="file-rows" style={{ marginTop: 10, display: 'grid', gap: 10 }}>
            {availableFiles.map((fileId) => {
              const label = fileLabelMap[fileId] || fileId;
              const active = activeSelection.files.includes(fileId);
              const contractValue = activeSelection.contracts[fileId] ?? 1;
              const marginValue = activeSelection.margins[fileId] ?? '';
              return (
                <div
                  key={fileId}
                  className="card"
                  style={{
                    padding: '10px 12px',
                    display: 'grid',
                    gridTemplateColumns: '1fr 120px 140px',
                    gap: 10,
                    alignItems: 'center',
                  }}
                >
                  <button
                    type="button"
                    className={`chip ${active ? 'chip-active' : ''}`}
                    onClick={() =>
                      setActiveSelection((prev) => ({
                        ...prev,
                        files: active ? prev.files.filter((f) => f !== fileId) : [...prev.files, fileId],
                      }))
                    }
                    style={{ justifyContent: 'flex-start' }}
                  >
                    {label}
                  </button>
                  <input
                    type="number"
                    min={0}
                    step={0.25}
                    className="input"
                    value={contractValue}
                    onChange={(event) => {
                      const next = Number(event.target.value);
                      setActiveSelection((prev) => ({
                        ...prev,
                        contracts: { ...prev.contracts, [fileId]: Number.isNaN(next) ? 0 : next },
                      }));
                    }}
                    placeholder="Contracts"
                  />
                  <input
                    type="number"
                    min={0}
                    step={100}
                    className="input"
                    value={marginValue}
                    onChange={(event) => {
                      const next = Number(event.target.value);
                      setActiveSelection((prev) => ({
                        ...prev,
                        margins: { ...prev.margins, [fileId]: Number.isNaN(next) ? 0 : next },
                      }));
                    }}
                    placeholder="Margin $/contract"
                  />
                </div>
              );
            })}
          </div>
          <div className="text-muted small" style={{ marginTop: 8 }}>Toggle files and edit contracts/margin per file.</div>
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
      </div>

      <div style={{ marginTop: 12 }}>
        <SelectionControls selection={activeSelection} availableFiles={availableFiles} onChange={setActiveSelection} />
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
            {filesMeta.map((file) => (
              <tr key={file.file_id}>
                <td>{file.filename}</td>
                <td>{file.symbols.join(', ')}</td>
                <td>{file.intervals.join(', ')}</td>
                <td>{file.strategies.join(', ')}</td>
                <td>
                  {file.date_min || file.date_max ? `${file.date_min || '—'} to ${file.date_max || '—'}` : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="text-muted small" style={{ marginTop: 8 }}>
          Acts as the Dash helper showing parsed symbol/interval/strategy/date metadata per file.
        </div>
      </div>

      <div className="card" style={{ marginTop: 18 }}>
        <strong>Export options</strong>
        <div className="flex gap-md" style={{ marginTop: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <label className="field-label" htmlFor="downsample" style={{ margin: 0 }}>Downsample series</label>
          <input
            id="downsample"
            type="checkbox"
            checked={includeDownsample}
            onChange={(event) => setIncludeDownsample(event.target.checked)}
          />
          <label className="field-label" htmlFor="format" style={{ margin: 0 }}>Export format</label>
          <select
            id="format"
            className="input"
            value={exportFormat}
            onChange={(event) => setExportFormat(event.target.value as 'csv' | 'parquet')}
            style={{ maxWidth: 200 }}
          >
            <option value="csv">CSV</option>
            <option value="parquet">Parquet</option>
          </select>
        </div>
        <div className="text-muted small" style={{ marginTop: 8 }}>Maps to /api/v1/export/trades and /api/v1/export/metrics.</div>
      </div>
    </div>
  );

  const renderTab = () => {
    if (activeTab === 'summary') return renderSummary();
    if (activeTab === 'equity-curves') return renderEquityCurves();
    if (activeTab === 'correlations') return renderCorrelations();
    if (activeTab === 'riskfolio') return renderOptimizer();
    if (activeTab === 'cta-report') return renderCta();
    if (activeTab === 'load-trade-lists') return renderIngest();
    if (activeTab === 'metrics') return renderMetrics();
    if (activeTab === 'portfolio-drawdown') return renderPortfolioDrawdown();
    if (activeTab === 'margin') return renderMargin();
    if (activeTab === 'trade-pl-histogram') return renderHistogram();
    if (activeTab === 'inverse-volatility') return renderInverseVolatility();
    return renderEquityCurves();
  };

  return (
    <div className="page">
      {errorMessage && (
        <div className="panel" style={{ borderColor: '#ff6b6b', background: 'rgba(255, 107, 107, 0.08)' }}>
          <div className="text-muted small">
            {errorMessage}
            <button
              type="button"
              className="button"
              style={{ marginLeft: 10 }}
              onClick={() => setErrorMessage(null)}
            >
              Dismiss
            </button>
          </div>
        </div>
      )}
      <div className="panel" style={{ marginBottom: 12 }}>
        <div className="flex gap-sm" style={{ alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h1 style={{ margin: '6px 0 8px 0' }}>Futures Portfolio Dashboard</h1>
            <div className="text-muted small">
              {apiBase ? (
                <span>
                  <span className="status-dot" /> API base configured: {apiBase}
                </span>
              ) : (
                <span>
                  <span className="status-dot" style={{ background: '#ffcb6b' }} /> API base not configured; set NEXT_PUBLIC_API_BASE to load live data
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
            {busy ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
      </div>

      {apiMissing && (
        <div className="panel" style={{ borderColor: '#ffcb6b', background: 'rgba(255, 203, 107, 0.06)' }}>
          <div className="text-muted small">
            Live uploads, series, metrics, and histogram calls are disabled until NEXT_PUBLIC_API_BASE is set.
          </div>
        </div>
      )}

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
