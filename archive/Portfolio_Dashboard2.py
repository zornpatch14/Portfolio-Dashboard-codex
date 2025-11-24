#!/usr/bin/env python3
"""
Trade Analyzer (Dash)
=====================

A browser-based GUI for analyzing TradeStation "Trades List" Excel exports.
- Upload multiple .xlsx files (the TradeStation two-row-per-trade format).
- Toggle files on/off.
- Choose a date range to filter.
- See portfolio cumulative P/L and per-file equity curves.
- See summary metrics (total net profit, max drawdown) for the current view.

How to run:
  1) Install deps:
     pip install dash plotly pandas numpy openpyxl
  2) Run:
     python trade_analyzer_dash.py
  3) Open the browser at http://127.0.0.1:8050 (it should auto-open).

Packaging (optional):
  You can bundle this into an .exe with PyInstaller on your machine:
     pip install pyinstaller
     pyinstaller --onefile trade_analyzer_dash.py
  Then run the produced executable.

Notes:
  - All computation is done locally in the browser session; uploads stay in memory.
  - We parse the header dynamically and pair entry/exit to recover cumulative P/L.
"""

import base64
import io
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, dash_table
import plotly.graph_objects as go

# ---------------- Parsing & Metrics ----------------

def parse_trades_from_bytes(xlsx_bytes: bytes, file_display_name: str) -> pd.DataFrame:
    """Parse a TradeStation Trades List workbook (bytes) into tidy trades.

    Returns a DataFrame with columns:
      - EntryDateTime (datetime64[ns])
      - ExitDateTime (datetime64[ns] or NaT for open trades)
      - CumulativePL_raw (float)  # cumulative equity at exit
      - NetProfit_incl_costs (float)  # realized profit per trade
      - File (str)
    """
    # Read sheet as raw to find header row
    raw_df = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name='Trades List', header=None)
    hdr_idx = None
    for i, row in raw_df.iterrows():
        if row.isnull().all():
            continue
        row_str = row.astype(str)
        if row_str.str.contains('^#', na=False).any() and row_str.str.contains('Type', na=False).any():
            hdr_idx = i
            break
    if hdr_idx is None:
        hdr_idx = 2  # fallback commonly works

    df = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name='Trades List', header=hdr_idx)

    # Clean repeated headers and empties
    if 'Type' in df.columns:
        df = df[df['Type'] != 'Type']
    df = df.dropna(subset=['Type'])
    # Forward-fill trade number so entry+exit share the same ID
    if '#' in df.columns:
        df['TradeNo'] = pd.to_numeric(df['#'], errors='coerce').ffill().astype('Int64')
    else:
        # Fallback: use first column as ID
        first_col = df.columns[0]
        df['TradeNo'] = pd.to_numeric(df[first_col], errors='coerce').ffill().astype('Int64')

    # Times and cumulative equity
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')
    df['CumulativePL_raw'] = pd.to_numeric(df['Net Profit - Cum Net Profit'], errors='coerce')

    entry_types = {'Buy', 'Sell Short'}
    exit_types  = {'Sell', 'Buy to Cover'}

    trades = []
    last_exit_raw = 0.0
    open_trades: Dict[int, Dict] = {}

    for _, row in df.iterrows():
        tno = row['TradeNo']
        rtype = str(row['Type']).strip()
        raw_val = row['CumulativePL_raw']
        if pd.isna(raw_val):
            raw_val = last_exit_raw  # carry forward

        if rtype in entry_types:
            open_trades[int(tno)] = {
                'EntryDateTime': row['Date/Time'],
                'EntryRawPL': raw_val,
            }
        elif rtype in exit_types:
            base = open_trades.pop(int(tno), {'EntryDateTime': None, 'EntryRawPL': last_exit_raw})
            net = raw_val - last_exit_raw
            trades.append({
                'EntryDateTime': base['EntryDateTime'],
                'ExitDateTime': row['Date/Time'],
                'CumulativePL_raw': float(raw_val),
                'NetProfit_incl_costs': float(net),
                'File': file_display_name,
            })
            last_exit_raw = raw_val

    # Include open trades with zero P/L at last known cum value
    for _, base in open_trades.items():
        trades.append({
            'EntryDateTime': base['EntryDateTime'],
            'ExitDateTime': pd.NaT,
            'CumulativePL_raw': float(last_exit_raw),
            'NetProfit_incl_costs': 0.0,
            'File': file_display_name,
        })

    out = pd.DataFrame(trades)
    # Sort by exit time for consistency
    if not out.empty:
        out['ExitDateTime'] = pd.to_datetime(out['ExitDateTime'], errors='coerce')
        out = out.sort_values('ExitDateTime')
    return out

def max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    roll_max = series.cummax()
    dd = series - roll_max
    return float(dd.min())

def reconstruct_equity_df(equity_dict: dict, selected_files: List[str]) -> pd.DataFrame:
    """Rebuild a combined equity DataFrame from stored per-file series."""
    if not selected_files:
        return pd.DataFrame()
    frames = []
    for name in selected_files:
        series = equity_dict.get(name, [])
        if not series:
            continue
        s = pd.Series({pd.to_datetime(rec['Date']): rec['Value'] for rec in series}, name=name)
        s = s.sort_index()
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).sort_index()
    df = df.ffill()  # forward-fill between trade exits
    return df

def filter_by_dates(df: pd.DataFrame, start_iso: str, end_iso: str) -> pd.DataFrame:
    if df.empty or not start_iso or not end_iso:
        return df
    s, e = pd.to_datetime(start_iso), pd.to_datetime(end_iso)
    return df[(df.index >= s) & (df.index <= e)]

def metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute total net and max drawdown for each column, plus portfolio."""
    if df.empty:
        return pd.DataFrame(columns=['Series', 'Total Net', 'Max Drawdown'])
    rows = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            tn = 0.0; dd = 0.0
        else:
            tn = float(series.iloc[-1] - series.iloc[0])
            dd = max_drawdown(series)
        rows.append({'Series': col, 'Total Net': round(tn, 2), 'Max Drawdown': round(dd, 2)})
    # portfolio
    port = df.sum(axis=1)
    if not port.empty:
        tn = float(port.iloc[-1] - port.iloc[0])
        dd = max_drawdown(port)
        rows.append({'Series': 'Portfolio (Selected)', 'Total Net': round(tn, 2), 'Max Drawdown': round(dd, 2)})
    return pd.DataFrame(rows)

# ---------------- Dash App ----------------

from dash import callback  # separate decorator (Dash 2.0+)

app = Dash(__name__)
app.title = "Trade Analyzer"

app.layout = html.Div([
    html.H2("Trade Analyzer (Dash)"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select TradeStation Trades List .xlsx Files')
        ]),
        multiple=True,
        style={
            'width': '100%', 'height': '80px', 'lineHeight': '80px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin-bottom': '10px'
        },
    ),

    html.Div(id='file-summary', style={'margin': '6px 0', 'fontSize': '14px'}),

    html.Div([
        html.Div([
            html.H4("Select Files"),
            dcc.Checklist(id='file-checklist', options=[], value=[], inline=False,
                          inputStyle={'margin-right': '6px', 'margin-left': '0.5rem'}),
            html.Hr(),
            html.H4("Date Range"),
            dcc.DatePickerRange(
                id='date-range',
                display_format='YYYY-MM-DD',
                with_portal=True,
            ),
            html.Div(id='range-help', style={'fontSize': '12px', 'color': '#555', 'marginTop': '6px'}),
        ], style={'flex': '0 0 260px', 'padding': '10px', 'borderRight': '1px solid #ddd'}),

        html.Div([
            dcc.Tabs(id='tabs', value='tab-portfolio', children=[
                dcc.Tab(label='Portfolio Cumulative P/L', value='tab-portfolio'),
                dcc.Tab(label='Per-File Equity Curves', value='tab-files'),
                dcc.Tab(label='Summary Metrics', value='tab-metrics'),
            ]),
            html.Div(id='tab-content', style={'padding': '10px'}),
        ], style={'flex': '1 1 auto', 'padding': '10px'}),
    ], style={'display': 'flex', 'border': '1px solid #eee', 'borderRadius': '6px'}),

    # Hidden stores to keep parsed data client-side
    dcc.Store(id='store-trades'),     # list of records for each file (unused but handy for future)
    dcc.Store(id='store-equity'),     # dict: { filename: [{'Date': iso, 'Value': float}, ...], ... }
    dcc.Store(id='store-index-bounds')# dict: {'min': iso, 'max': iso}
])

@callback(
    Output('store-trades', 'data'),
    Output('store-equity', 'data'),
    Output('store-index-bounds', 'data'),
    Output('file-checklist', 'options'),
    Output('file-checklist', 'value'),
    Output('file-summary', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_upload(list_of_contents, list_of_names):
    if not list_of_contents:
        return None, None, None, [], [], "No files uploaded."

    all_trades = []
    equity_dict = {}
    min_ts, max_ts = None, None

    for content, name in zip(list_of_contents, list_of_names):
        try:
            header, b64data = content.split(',', 1)
            file_bytes = base64.b64decode(b64data)
            trades = parse_trades_from_bytes(file_bytes, name)
            if trades.empty:
                continue
            trades['File'] = name
            all_trades.append(trades)

            series = trades.dropna(subset=['ExitDateTime'])[['ExitDateTime', 'CumulativePL_raw']]
            series = series.drop_duplicates(subset='ExitDateTime', keep='last').set_index('ExitDateTime').sort_index()
            eq_records = [{'Date': ts.isoformat(), 'Value': float(val)} for ts, val in series['CumulativePL_raw'].items()]
            equity_dict[name] = eq_records

            if not series.empty:
                smin, smax = series.index.min(), series.index.max()
                min_ts = smin if (min_ts is None or smin < min_ts) else min_ts
                max_ts = smax if (max_ts is None or smax > max_ts) else max_ts
        except Exception as e:
            print(f"Error parsing {name}: {e}")

    if not all_trades:
        return None, None, None, [], [], "No valid trades found in uploaded files."

    tot_trades = sum(len(t) for t in all_trades)
    uniq_files = [t['File'].iloc[0] for t in all_trades]
    summary = f"Loaded {len(uniq_files)} file(s), {tot_trades} trade(s)."

    options = [{'label': f, 'value': f} for f in equity_dict.keys()]
    values = list(equity_dict.keys())

    bounds = None
    if min_ts is not None and max_ts is not None:
        bounds = {'min': min_ts.isoformat(), 'max': max_ts.isoformat()}

    trades_concat = pd.concat(all_trades, ignore_index=True)
    trades_json = trades_concat.to_dict(orient='records')

    return trades_json, equity_dict, bounds, options, values, summary

@callback(
    Output('date-range', 'min_date_allowed'),
    Output('date-range', 'max_date_allowed'),
    Output('date-range', 'start_date'),
    Output('date-range', 'end_date'),
    Output('range-help', 'children'),
    Input('store-index-bounds', 'data')
)
def set_date_bounds(bounds):
    if not bounds:
        return None, None, None, None, "Upload files to enable date filtering."
    min_iso, max_iso = bounds['min'], bounds['max']
    return min_iso, max_iso, min_iso, max_iso, f"Available: {min_iso[:10]} → {max_iso[:10]}"

def reconstruct_equity_df(equity_dict: dict, selected_files: List[str]) -> pd.DataFrame:
    if not selected_files:
        return pd.DataFrame()
    frames = []
    for name in selected_files:
        series = equity_dict.get(name, [])
        if not series:
            continue
        s = pd.Series({pd.to_datetime(rec['Date']): rec['Value'] for rec in series}, name=name)
        s = s.sort_index()
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).sort_index()
    df = df.ffill()
    return df

def filter_by_dates(df: pd.DataFrame, start_iso: str, end_iso: str) -> pd.DataFrame:
    if df.empty or not start_iso or not end_iso:
        return df
    s, e = pd.to_datetime(start_iso), pd.to_datetime(end_iso)
    return df[(df.index >= s) & (df.index <= e)]

def metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['Series', 'Total Net', 'Max Drawdown'])
    rows = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            tn = 0.0; dd = 0.0
        else:
            tn = float(series.iloc[-1] - series.iloc[0])
            dd = max_drawdown(series)
        rows.append({'Series': col, 'Total Net': round(tn, 2), 'Max Drawdown': round(dd, 2)})
    port = df.sum(axis=1)
    if not port.empty:
        tn = float(port.iloc[-1] - port.iloc[0])
        dd = max_drawdown(port)
        rows.append({'Series': 'Portfolio (Selected)', 'Total Net': round(tn, 2), 'Max Drawdown': round(dd, 2)})
    return pd.DataFrame(rows)

@callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('file-checklist', 'value'),
    Input('store-equity', 'data'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
)
def render_tab(tab_value, selected_files, equity_dict, start_date, end_date):
    if not equity_dict or not selected_files:
        return html.Div("Upload files and select at least one file to view charts.", style={'padding': '10px'})

    eq = reconstruct_equity_df(equity_dict, selected_files)
    eq = filter_by_dates(eq, start_date, end_date)

    if eq.empty:
        return html.Div("No data in the selected date range.", style={'padding': '10px'})

    if tab_value == 'tab-portfolio':
        portfolio = eq.sum(axis=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio.values, mode='lines', name='Portfolio'))
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=520,
                          title='Cumulative P/L (Selected Files)',
                          xaxis_title='Date', yaxis_title='Cumulative P/L')
        return dcc.Graph(figure=fig, config={'displaylogo': False})

    elif tab_value == 'tab-files':
        fig = go.Figure()
        for col in eq.columns:
            fig.add_trace(go.Scatter(x=eq.index, y=eq[col], mode='lines', name=col))
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=520,
                          title='Equity Curves by File (Selected)',
                          xaxis_title='Date', yaxis_title='Cumulative P/L')
        return dcc.Graph(figure=fig, config={'displaylogo': False})

    elif tab_value == 'tab-metrics':
        mt = metrics_table(eq)
        return dash_table.DataTable(
            data=mt.to_dict('records'),
            columns=[{'name': c, 'id': c} for c in mt.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'padding': '6px', 'fontSize': 14},
            style_header={'fontWeight': 'bold'},
        )

    return html.Div("Unknown tab.")

if __name__ == '__main__':
    try:
        import webbrowser
        webbrowser.open_new('http://127.0.0.1:8050/')
    except Exception:
        pass
    app.run(debug=True, port=8050)
