'use client';

import { AgGridReact } from 'ag-grid-react';
import { useMemo } from 'react';
import type { ColDef } from 'ag-grid-community';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-quartz.css';
import { MetricsRow } from '../lib/api';

type Props = {
  rows: MetricsRow[];
};

export default function MetricsGrid({ rows }: Props) {
  const columnDefs = useMemo<ColDef<MetricsRow>[]>(
    () => [
      { headerName: 'Scope', field: 'scope', flex: 1, minWidth: 180 },
      { headerName: 'Net Profit', field: 'netProfit', valueFormatter: ({ value }: any) => `$${value.toLocaleString()}` },
      { headerName: 'Drawdown', field: 'drawdown', valueFormatter: ({ value }: any) => `$${value.toLocaleString()}` },
      { headerName: 'Trades', field: 'trades', maxWidth: 120 },
      { headerName: 'Win %', field: 'winRate', maxWidth: 110, valueFormatter: ({ value }: any) => `${value}%` },
      { headerName: 'Expectancy', field: 'expectancy', maxWidth: 140 },
      { headerName: 'Exposure', field: 'exposure', maxWidth: 130 },
    ],
    [],
  );

  const defaultColDef = useMemo(
    () => ({ sortable: true, filter: true, resizable: true, suppressMovable: true }),
    [],
  );

  return (
    <div className="table-wrapper ag-theme-quartz" style={{ width: '100%', height: 320 }}>
      <AgGridReact rowData={rows} columnDefs={columnDefs} defaultColDef={defaultColDef} animateRows suppressDragLeaveHidesColumns />
    </div>
  );
}
