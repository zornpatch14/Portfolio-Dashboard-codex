'use client';

import { useMemo, useState } from 'react';
import { useQueries } from '@tanstack/react-query';
import MetricsGrid from '../components/MetricsGrid';
import SeriesChart from '../components/SeriesChart';
import { SelectionList } from '../components/SelectionList';
import { HistogramChart } from '../components/HistogramChart';
import { SelectionControls } from '../components/SelectionControls';
import {
  fetchHistogram,
  fetchMetrics,
  fetchSeries,
  mockHistogram,
  mockSeries,
  SeriesKind,
} from '../lib/api';
import { loadSampleSelections, Selection } from '../lib/selections';

const selections = loadSampleSelections();

export default function HomePage() {
  const [activeSelection, setActiveSelection] = useState<Selection>(selections[0]);
  const apiBase = process.env.NEXT_PUBLIC_API_BASE;

  const availableFiles = useMemo(
    () => Array.from(new Set(selections.flatMap((s) => s.files))).sort(),
    [],
  );

  const seriesKinds: SeriesKind[] = ['equity', 'equityPercent', 'drawdown', 'intradayDrawdown', 'netpos', 'margin'];

  const [equityQuery, equityPctQuery, drawdownQuery, intradayDdQuery, netposQuery, marginQuery, histogramQuery, metricsQuery] =
    useQueries({
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
    histogramQuery.isFetching;

  return (
    <div className="two-column">
      <div className="panel" style={{ position: 'sticky', top: 16 }}>
        <div className="flex gap-sm" style={{ alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div className="text-muted small">Portfolio Dashboard</div>
            <h1 style={{ margin: '6px 0 8px 0' }}>Selection driver</h1>
            <div className="text-muted small">
              {apiBase ? (
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
            }}
            disabled={busy}
          >
            {busy ? 'Refreshing…' : 'Refresh' }
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

      <div className="panel">
        <div className="flex gap-md" style={{ alignItems: 'center', justifyContent: 'space-between' }}>
          <h2 className="section-title" style={{ margin: 0 }}>
            Equity, drawdown, and metrics
          </h2>
          {busy ? <div className="badge">Loading…</div> : <div className="badge">Live</div>}
        </div>

        <div className="metric-cards" style={{ marginTop: 14 }}>
          <div className="metric-card">
            <span className="text-muted small">Net Profit (sum)</span>
            <strong>${metricsSummary ? metricsSummary.netProfit.toLocaleString() : '—'}</strong>
          </div>
          <div className="metric-card">
            <span className="text-muted small">Drawdown (sum)</span>
            <strong>${metricsSummary ? metricsSummary.drawdown.toLocaleString() : '—'}</strong>
          </div>
          <div className="metric-card">
            <span className="text-muted small">Avg Trades</span>
            <strong>{metricsSummary ? metricsSummary.avgTrades : '—'}</strong>
          </div>
          <div className="metric-card">
            <span className="text-muted small">Avg Win %</span>
            <strong>{metricsSummary ? `${metricsSummary.avgWin.toFixed(1)}%` : '—'}</strong>
          </div>
          <div className="metric-card">
            <span className="text-muted small">Files</span>
            <strong>{activeSelection.files.length}</strong>
          </div>
          <div className="metric-card">
            <span className="text-muted small">Direction</span>
            <strong>{activeSelection.direction}</strong>
          </div>
        </div>

        <div className="charts-grid" style={{ marginTop: 18 }}>
          <div className="card">
            <SeriesChart title="Equity Curve" series={equityQuery.data ?? mockSeries(activeSelection, 'equity')} color="#4cc3ff" />
            <div className="text-muted small">Points: {equityQuery.data?.downsampledCount ?? equityQuery.data?.points.length}</div>
          </div>
          <div className="card">
            <SeriesChart
              title="Percent Equity"
              series={equityPctQuery.data ?? mockSeries(activeSelection, 'equityPercent')}
              color="#8fe3c7"
            />
            <div className="text-muted small">Points: {equityPctQuery.data?.downsampledCount ?? equityPctQuery.data?.points.length}</div>
          </div>
          <div className="card">
            <SeriesChart
              title="Drawdown"
              series={drawdownQuery.data ?? mockSeries(activeSelection, 'drawdown')}
              color="#ff8f6b"
            />
            <div className="text-muted small">Points: {drawdownQuery.data?.downsampledCount ?? drawdownQuery.data?.points.length}</div>
          </div>
          <div className="card">
            <SeriesChart
              title="Intraday Drawdown"
              series={intradayDdQuery.data ?? mockSeries(activeSelection, 'intradayDrawdown')}
              color="#f4c95d"
            />
            <div className="text-muted small">Points: {intradayDdQuery.data?.downsampledCount ?? intradayDdQuery.data?.points.length}</div>
          </div>
          <div className="card">
            <SeriesChart title="Net Position" series={netposQuery.data ?? mockSeries(activeSelection, 'netpos')} color="#9f8bff" />
            <div className="text-muted small">Points: {netposQuery.data?.downsampledCount ?? netposQuery.data?.points.length}</div>
          </div>
          <div className="card">
            <SeriesChart title="Margin Usage" series={marginQuery.data ?? mockSeries(activeSelection, 'margin')} color="#54ffd0" />
            <div className="text-muted small">Points: {marginQuery.data?.downsampledCount ?? marginQuery.data?.points.length}</div>
          </div>
          <div className="card">
            <HistogramChart histogram={histogramQuery.data ?? mockHistogram(activeSelection)} />
            <div className="text-muted small">Distribution: {histogramQuery.data?.buckets.length ?? 0} buckets</div>
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
    </div>
  );
}
