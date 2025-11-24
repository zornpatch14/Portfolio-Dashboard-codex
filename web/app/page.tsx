'use client';

import { useMemo, useState } from 'react';
import { useQueries } from '@tanstack/react-query';
import MetricsGrid from '../components/MetricsGrid';
import SeriesChart from '../components/SeriesChart';
import { SelectionList } from '../components/SelectionList';
import { fetchMetrics, fetchSeries, mockSeries } from '../lib/api';
import { loadSampleSelections, Selection } from '../lib/selections';

const selections = loadSampleSelections();

export default function HomePage() {
  const [activeSelection, setActiveSelection] = useState<Selection>(selections[0]);
  const apiBase = process.env.NEXT_PUBLIC_API_BASE;

  const [equityQuery, drawdownQuery, metricsQuery] = useQueries({
    queries: [
      {
        queryKey: ['equity', activeSelection.name, activeSelection.files.join(',')],
        queryFn: () => fetchSeries(activeSelection, 'equity'),
        initialData: mockSeries(activeSelection, 'equity'),
      },
      {
        queryKey: ['drawdown', activeSelection.name, activeSelection.files.join(',')],
        queryFn: () => fetchSeries(activeSelection, 'drawdown'),
        initialData: mockSeries(activeSelection, 'drawdown'),
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

  const busy = equityQuery.isFetching || drawdownQuery.isFetching || metricsQuery.isFetching;

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
              drawdownQuery.refetch();
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
        </div>

        <div className="charts-grid" style={{ marginTop: 18 }}>
          <div className="card">
            <SeriesChart title="Equity Curve" series={equityQuery.data ?? mockSeries(activeSelection, 'equity')} color="#4cc3ff" />
            <div className="text-muted small">Points: {equityQuery.data?.downsampledCount ?? equityQuery.data?.points.length}</div>
          </div>
          <div className="card">
            <SeriesChart
              title="Drawdown"
              series={drawdownQuery.data ?? mockSeries(activeSelection, 'drawdown')}
              color="#ff8f6b"
            />
            <div className="text-muted small">Points: {drawdownQuery.data?.downsampledCount ?? drawdownQuery.data?.points.length}</div>
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
