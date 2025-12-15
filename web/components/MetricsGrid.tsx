'use client';

import { AgGridReact } from 'ag-grid-react';
import { useMemo, useCallback } from 'react';
import type { ColDef } from 'ag-grid-community';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-quartz.css';
import type { MetricsBlock } from '../lib/api';

type MetricFormat = 'currency' | 'percent' | 'number' | 'date';

type MetricColumn = {
  key: string;
  label: string;
  format?: MetricFormat;
  maxWidth?: number;
  minWidth?: number;
};

type GridRow = Record<string, number | string | null> & {
  key: string;
  label: string;
};

const METRIC_COLUMNS: MetricColumn[] = [
  { key: 'total_net_profit', label: 'Total Net Profit', format: 'currency' },
  { key: 'account_size_required', label: 'Account Size Required', format: 'currency' },
  { key: 'close_to_close_drawdown_value', label: 'Close-to-close DD', format: 'currency' },
  { key: 'intraday_peak_to_valley_drawdown_value', label: 'Intraday DD', format: 'currency' },
  { key: 'total_trades', label: 'Trades', format: 'number', maxWidth: 120 },
  { key: 'percent_profitable', label: 'Win %', format: 'percent', maxWidth: 110 },
  { key: 'profit_factor', label: 'Profit Factor', format: 'number' },
  { key: 'avg_trade_net_profit', label: 'Avg Trade', format: 'currency' },
  { key: 'avg_winning_trade', label: 'Avg Win', format: 'currency' },
  { key: 'avg_losing_trade', label: 'Avg Loss', format: 'currency' },
  { key: 'largest_winning_trade', label: 'Largest Win', format: 'currency' },
  { key: 'largest_losing_trade', label: 'Largest Loss', format: 'currency' },
  { key: 'max_consecutive_wins', label: 'Max Win Streak', format: 'number' },
  { key: 'max_consecutive_losses', label: 'Max Loss Streak', format: 'number' },
  { key: 'max_contracts_held', label: 'Max Contracts', format: 'number' },
  { key: 'total_contracts_held', label: 'Total Contracts', format: 'number' },
  { key: 'total_commission', label: 'Commission', format: 'currency' },
  { key: 'total_slippage', label: 'Slippage', format: 'currency' },
  { key: 'max_trade_drawdown', label: 'Max Trade DD', format: 'currency' },
  { key: 'close_to_close_drawdown_date', label: 'Close DD Date', format: 'date', minWidth: 150 },
  { key: 'intraday_peak_to_valley_drawdown_date', label: 'Intraday DD Date', format: 'date', minWidth: 150 },
];

const formatMetricValue = (value: unknown, format?: MetricFormat) => {
  if (value === null || value === undefined || value === '') return '--';
  if (format === 'date') {
    const raw = String(value);
    const date = new Date(raw);
    if (Number.isNaN(date.getTime())) return raw;
    return date.toLocaleDateString();
  }
  const num = Number(value);
  if (!Number.isFinite(num)) return '--';
  switch (format) {
    case 'currency':
      return `$${num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    case 'percent':
      return `${num.toFixed(1)}%`;
    case 'number':
    default:
      return num.toLocaleString();
  }
};

type Props = {
  blocks: MetricsBlock[];
};

export default function MetricsGrid({ blocks }: Props) {
  const rows = useMemo<GridRow[]>(
    () =>
      blocks.map((block) => {
        const row: GridRow = { key: block.key, label: block.label };
        METRIC_COLUMNS.forEach((col) => {
          row[col.key] = block.metrics[col.key] ?? null;
        });
        return row;
      }),
    [blocks],
  );

  const columnDefs = useMemo<ColDef<GridRow>[]>(
    () => [
      { headerName: 'Scope', field: 'label', flex: 1, minWidth: 200, pinned: 'left' },
      ...METRIC_COLUMNS.map((col) => ({
        headerName: col.label,
        field: col.key,
        minWidth: col.minWidth ?? 140,
        maxWidth: col.maxWidth,
        valueFormatter: (params: any) => formatMetricValue(params.value, col.format),
      })),
    ],
    [],
  );

  const defaultColDef = useMemo(
    () => ({ sortable: true, filter: true, resizable: true, suppressMovable: true }),
    [],
  );

  const getRowId = useCallback((params: { data: GridRow }) => params.data.key, []);

  return (
    <div className="table-wrapper ag-theme-quartz" style={{ width: '100%', height: 320 }}>
      <AgGridReact
        rowData={rows}
        columnDefs={columnDefs}
        defaultColDef={defaultColDef}
        animateRows
        suppressDragLeaveHidesColumns
        getRowId={getRowId}
      />
    </div>
  );
}
