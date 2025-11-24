#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Trade Analysis Dashboard — Single File
================================================

CHANGELOG (2025-10-03)
----------------------
- Integrated "Trades List"-only metrics directly into this single file.
- Added filename parsing: symbol, timeframe (interval), strategy (regex + fallbacks).
- New Controls:
    • Symbols (multi-select), Timeframes (multi-select), Strategies (multi-select)
    • Direction radio (All / Long / Short)
- Charts unchanged in spirit but now reflect ALL filters (files + symbol + interval + strategy + direction + date).
- Metrics table:
    • Adds: Total Commission, Total Slippage, Account Size Required (abs max close-to-close DD),
            Intra-Day Peak→Valley Drawdown Value/Date (from Run-up/Drawdown columns),
            plus the existing P&L stats.
    • “Portfolio” row also includes: ROI, Annual ROR, ROA, Avg/Std Monthly Return (full months only),
      Trading Period, % Time in Market, Time in Market, Longest Flat Period, Max Equity Run-up and %.
- Exports:
    • Trades CSV now includes Symbol, Interval, Strategy, Direction, etc.
    • Metrics CSV includes new fields above.
- Notes:
    • Assumes two-row-per-trade format as exported by TradeStation (“Trades List” sheet).
    • No timezone handling, per user instruction.
"""

from __future__ import annotations

import base64, io, os, re, webbrowser, itertools
from datetime import datetime, date, timedelta
from typing import Any, Iterable, Tuple
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update, ctx
from flask_caching import Cache
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from src.constants import APP_TITLE, COLOR_PORTFOLIO, SHEET_HINTS, DEFAULT_INITIAL_CAPITAL, MARGIN_SPEC, MARGIN_BUFFER, FILENAME_RE
from src.etl import parse_tradestation_trades, parse_filename_meta, decode_upload, canonicalize_columns, load_sheet_guess, find_header_row
from src.equity import equity_from_trades_subset, combine_equity, max_drawdown_series, max_drawdown_value, intraday_drawdown_series
from src.metrics import max_consecutive_runs, compute_max_held, compute_closed_trade_dd, compute_intraday_p2v, monthly_stats_full_months, compute_metrics
from src.corr import _to_daily_equity, _drawdown_pct, _zscore, _rolling_slope, _compute_pl_dataframe_from_equity, build_correlation_heatmap
from src.netpos import _aggregate_netpos_per_symbol_from_series, _netpos_figure_from_series, _netpos_timeseries_from_store, build_net_contracts_figure
from src.helpers import _symbol_from_first_row, _within_dates, _display_label_from_df, _safe_unique_name, _files_key, _keyify_list, _keyify_contracts_map, normalize_date
from src.margin import _purchasing_power_series, _margin_series
from src.allocator import _portfolio_equity_from_weights, _peak_drawdown_value, _annualized_return, find_margin_aware_weights
from src.charts import build_equity_figure, build_drawdown_figure, build_intraday_dd_figure, build_pl_histogram_figure
from src.tabs import _single_graph_child, _two_graphs_child, pretty_card




# ------------------------------ Dash UI --------------------------------

app = Dash(__name__)
app.title = APP_TITLE


import src.netpos as _netpos_mod
_netpos_mod.app = app

import src.allocator as _alloc_mod
_alloc_mod.app = app



# Server-side stores & cache
if not hasattr(app.server, "trade_store"):
    app.server.trade_store = {}  # {filename: DataFrame}
cache = Cache(app.server, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300})



app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial", "padding": "18px", "background": "#f6f7fb"},
    children=[
        html.H2(APP_TITLE, style={"marginTop": 0}),

        # Upload
        pretty_card([
            html.H4("1) Load Trade Lists"),
            dcc.Upload(
                id="upload",
                children=html.Div(["Drag & drop or ", html.A("select .xlsx files")]),
                multiple=True,
                style={
                    "width": "100%", "height": "70px", "lineHeight": "70px",
                    "borderWidth": "2px", "borderStyle": "dashed",
                    "borderRadius": "8px", "textAlign": "center", "background": "#fafafa"
                },
            ),
            html.Div(id="file-list", style={"marginTop": "10px", "fontSize": "14px"}),
        ]),

        # Controls + Charts
        html.Div([
            pretty_card([
                html.H4("2) Controls"),
                html.Div([
                    html.Div("Included files:", style={"fontWeight": 600, "marginBottom": "6px"}),
                    html.Div(
                        id="file-includes-row",
                        style={"display": "grid", "gridTemplateColumns": "1fr auto", "gap": "8px", "alignItems": "start"},
                        children=[
                            dcc.Checklist(id="file-toggle", options=[], value=[], inline=False),
                            html.Div(
                                id="file-contracts",   # NEW: will be populated with one input per file
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fill, minmax(180px, 1fr))",
                                    "gap": "6px"
                                }
                            ),
                        ]
                    ),
                    html.Div("Tip: leave at 1 for native one-contract P/L; set 0 to exclude without unchecking.",
                            style={"fontSize": "12px", "color": "#555", "marginTop": "6px"}),
                ], style={"marginBottom": "10px"}),

                html.Div([
                    html.Div("Symbols:", style={"fontWeight": 600, "marginBottom": "6px"}),
                    dcc.Checklist(id="symbol-toggle", options=[], value=[], inline=True),
                ], style={"marginBottom": "10px"}),

                html.Div([
                    html.Div("Timeframes:", style={"fontWeight": 600, "marginBottom": "6px"}),
                    dcc.Checklist(id="interval-toggle", options=[], value=[], inline=True),
                ], style={"marginBottom": "10px"}),

                html.Div([
                    html.Div("Strategies:", style={"fontWeight": 600, "marginBottom": "6px"}),
                    dcc.Checklist(id="strategy-toggle", options=[], value=[], inline=False),
                ], style={"marginBottom": "10px"}),

                html.Div([
                    html.Div("Direction:", style={"fontWeight": 600, "marginBottom": "6px"}),
                    dcc.RadioItems(
                        id="direction-radio",
                        options=[
                            {"label": "All", "value": "All"},
                            {"label": "Long", "value": "Long"},
                            {"label": "Short", "value": "Short"},
                        ],
                        value="All",
                        inline=True
                    ),
                ], style={"marginBottom": "10px"}),

                html.Div([
                    html.Div("Date range:", style={"fontWeight": 600, "marginBottom": "6px"}),
                    dcc.DatePickerRange(
                        id="date-range", start_date=None, end_date=None,
                        display_format="YYYY-MM-DD", minimum_nights=0
                    ),

                html.Hr(),
                html.Div("Allocator settings:", style={"fontWeight": 600, "marginBottom": "6px"}),
                html.Div([
                    html.Div(["Account equity: ",
                            dcc.Input(id="alloc-equity", type="number", value=DEFAULT_INITIAL_CAPITAL, step=1000)]),
                    html.Div(["Max margin use (% of equity): ",
                            dcc.Input(id="alloc-margin-buffer", type="number", value=int(MARGIN_BUFFER*100), step=5)]),
                    html.Div(["Objective: ",
                            dcc.Dropdown(id="alloc-objective",
                                        options=[
                                            {"label": "Max Return / Drawdown", "value": "max_return_over_dd"},
                                            {"label": "Risk Parity (projected)", "value": "risk_parity"},
                                        ],
                                        value="max_return_over_dd", clearable=False, style={"width":"260px"})]),
                    html.Div(["Leverage cap (sum of weights): ",
                            dcc.Input(id="alloc-lev-cap", type="number", value=1.0, step=0.25)]),
                    html.Div(["Grid step (0.25=coarse, 0.1=finer): ",
                            dcc.Input(id="alloc-step", type="number", value=0.25, step=0.05)]),
                    html.Button("Run Allocator", id="btn-run-alloc", n_clicks=0, style={"marginTop":"6px"}),
                ], style={"display":"grid","gridTemplateColumns":"repeat(2, minmax(200px, 1fr))","gap":"8px"}),
 
                html.Div([
                    html.Div("Correlation view:", style={"fontWeight": 600, "marginBottom": "6px"}),
                    dcc.Dropdown(
                        id="corr-mode",
                        options=[
                            {"label": "Daily Drawdown % (recommended)", "value": "drawdown_pct"},
                            {"label": "Z-scored Daily Returns", "value": "returns_z"},
                            {"label": "Daily $ P/L", "value": "pl"},
                            {"label": "Rolling Slope", "value": "slope"},
                        ],
                        value="drawdown_pct",
                        clearable=False,
                        style={"maxWidth": "340px"}
                    ),
                    html.Div([
                        html.Span("Slope window: "),
                        dcc.Input(id="corr-slope-window", type="number", value=20, step=1, min=5, style={"width": "100px"})
                    ], style={"marginTop": "6px"})
                ], style={"marginBottom": "10px"}),



                ]),
            ]),

            pretty_card([
                html.H4("3) Charts"),
                dcc.Tabs(
                    id="chart-tabs",
                    value="equity",
                    children=[
                        dcc.Tab(label="Equity Curves", value="equity"),
                        dcc.Tab(label="Portfolio Drawdown", value="ddcurve"),
                        dcc.Tab(label="Intraday Drawdown", value="idd"),
                        dcc.Tab(label="Trade P/L Histogram", value="hist"),
                        dcc.Tab(label="Margin", value="margin"),
                        dcc.Tab(label="Correlations", value="corr"),
                    ],
                ),
                html.Div(id="chart-panel")  # NEW: we will inject one or multiple graphs into this
            ]),
        ], style={"display": "grid", "gridTemplateColumns": "0.9fr 2fr", "gap": "14px"}),


            pretty_card([
                html.H4("Allocator Result"),
                html.Div(id="alloc-summary", style={"marginBottom":"8px"}),
                dcc.Graph(id="alloc-portfolio-graph", style={"height":"360px"}),
                dash_table.DataTable(id="alloc-weights-table",
                                    style_header={"fontWeight":"bold"},
                                    style_cell={"padding":"6px", "whiteSpace":"normal"},
                                    page_size=20)
            ]),


        # Metrics
        html.Div([
            pretty_card([
                html.H4("Metrics"),
                dash_table.DataTable(
                    id="metrics-table",
                    style_header={"fontWeight": "bold"},
                    style_cell={"padding": "6px", "whiteSpace": "normal"},
                    page_size=20,
                    sort_action="native",
                    export_format="none"
                ),
                html.Div([
                    html.Button("Export filtered trades (CSV)", id="btn-export-trades"),
                    html.Button("Export metrics (CSV)", id="btn-export-metrics", style={"marginLeft": "10px"}),
                    dcc.Download(id="download-trades"),
                    dcc.Download(id="download-metrics"),
                ], style={"marginTop": "8px"})
            ])
        ], style={"marginTop": "14px"}),

        # Stores
        dcc.Store(id="store-trades"),
        dcc.Store(id="store-equity-minmax"),
        dcc.Store(id="store-contracts"),   # NEW: per-file contract multipliers {filename: float}
        dcc.Store(id="store-version", data=0),

    ]
)

# ------------------------------ Helpers --------------------------------


def make_selection_key(
    selected_files,
    symbols,
    intervals,
    strategies,
    direction,
    start_date,
    end_date,
    contracts_map,
    store_version,
):
    """
    Build a stable, hashable cache key for the current selection.
    - Lists are sorted & tuple-ized.
    - Dates normalized to 'YYYY-MM-DD' (or empty string).
    - Dicts (contracts_map) turned into a sorted tuple.
    - Direction coerced to a simple string.
    The return value is a tuple of labeled parts for readability in logs.
    """
    files_part      = _keyify_list([str(f) for f in (selected_files or [])])
    symbols_part    = _keyify_list(symbols or [])
    intervals_part  = _keyify_list(intervals or [])
    strategies_part = _keyify_list(strategies or [])

    start_part = normalize_date(start_date) or ""
    end_part   = normalize_date(end_date) or ""

    contracts_part = _keyify_contracts_map(contracts_map or {})

    return (
        ("v", int(store_version) if store_version is not None else 0),
        ("files", files_part),
        ("symbols", symbols_part),
        ("intervals", intervals_part),
        ("strategies", strategies_part),
        ("direction", str(direction or "All")),
        ("start", start_part),
        ("end", end_part),
        ("contracts", contracts_part),
    )

@cache.memoize()
def build_core_artifact(sel_key) -> dict:
    """
    Build and cache the 'core selection artifact' for the current selection.
    Relies on existing cached per-file functions, so repeated use is fast.

    Returns a dict with:
      - 'trades_by_file': {fname: DataFrame}
      - 'equity_by_file': {fname: Series}
      - 'label_map':      {fname: str}
      - 'files':          tuple[str, ...]
    """
    # sel_key is a tuple of labeled parts, e.g. (("v", 3), ("files", (...)), ...)
    parts = dict(sel_key)

    data_version = parts.get("v", 0)
    files = tuple(parts.get("files", ()))
    symbol_sel = tuple(parts.get("symbols", ()))
    interval_sel = tuple(parts.get("intervals", ()))
    strat_sel = tuple(parts.get("strategies", ()))
    direction = parts.get("direction", "All")
    start_date = parts.get("start", None)
    end_date = parts.get("end", None)

    if not start_date:
        start_date = None
    if not end_date:
        end_date = None


    # contracts: tuple[(file, mult)] -> dict
    contracts_kv = tuple(parts.get("contracts", ()))
    contracts_map = {fname: float(mult) for fname, mult in contracts_kv}

    trades_by_file: dict[str, pd.DataFrame] = {}
    equity_by_file: dict[str, pd.Series] = {}
    label_map: dict[str, str] = {}

    for fname in files:
        mult = float(contracts_map.get(fname, 1.0))

        # Use your existing cached per-file helpers
        tdf = cached_filtered_df(
            data_version, fname,
            symbol_sel, interval_sel, strat_sel,
            direction, start_date, end_date, mult
        )
        trades_by_file[fname] = tdf

        eq = cached_equity_series(
            data_version, fname,
            symbol_sel, interval_sel, strat_sel,
            direction, start_date, end_date, mult
        )
        equity_by_file[fname] = eq

        # Pretty label per file (falls back gracefully if df is empty)
        try:
            label_map[fname] = _display_label_from_df(tdf, fallback=fname)
        except TypeError:
            # If your _display_label_from_df has signature (df, fname) instead:
            label_map[fname] = _display_label_from_df(tdf, fname) if tdf is not None else str(fname)

    return {
        "trades_by_file": trades_by_file,
        "equity_by_file": equity_by_file,
        "label_map": label_map,
        "files": files,
    }


# ------------------------------ Net position caching & plotting ------------------------------



@cache.memoize()
def cached_netpos_per_file(data_version: int,
                           files_key: tuple,
                           symbol_sel: tuple, interval_sel: tuple, strat_sel: tuple,
                           direction: str,
                           start_date, end_date,
                           contracts_kv: tuple) -> dict[str, pd.Series]:
    """
    Compute per-file signed net contracts step series WITH:
      • correct carry-in at start_date (sum of deltas before the window)
      • explicit start/end anchors in the timeline
    Returns {filename -> pd.Series(index=datetime, values=signed net contracts)}.
    """
    # Rebuild contracts map from KV tuple
    contracts_map = {k: v for k, v in (contracts_kv or [])}

    # Helper: filter by symbol/interval/strategy/direction (but NOT the date yet)
    def base_filter_no_date(df: pd.DataFrame) -> pd.DataFrame:
        out = df
        if symbol_sel:   out = out[out["Symbol"].isin(symbol_sel)]
        if interval_sel: out = out[out["Interval"].isin(interval_sel)]
        if strat_sel:    out = out[out["Strategy"].isin(strat_sel)]
        if direction in ("Long", "Short"):
            out = out[out["direction"] == direction]
        return out

    # Resolve window bounds
    # We still need a global min/max to define anchors if the file has events.
    all_events = []
    for f in files_key:
        tdf = app.server.trade_store.get(f)
        if tdf is None or tdf.empty:
            continue
        df = base_filter_no_date(tdf)
        if df.empty:
            continue
        ent = pd.to_datetime(df["entry_time"], errors="coerce")
        exi = pd.to_datetime(df["exit_time"],  errors="coerce")
        all_events.append(ent.dropna())
        all_events.append(exi.dropna())
    if all_events:
        ts_min = pd.concat(all_events).min()
        ts_max = pd.concat(all_events).max()
    else:
        ts_min = ts_max = None

    # Normalize window inputs
    start_dt = pd.to_datetime(start_date) if start_date is not None else ts_min
    end_dt   = pd.to_datetime(end_date)   if end_date   is not None else ts_max
    if start_dt is None or pd.isna(start_dt):
        start_dt = ts_min
    if end_dt is None or pd.isna(end_dt):
        end_dt = ts_max
    if start_dt is None or end_dt is None:
        return {}

    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    out: dict[str, pd.Series] = {}

    for f in files_key:
        tdf = app.server.trade_store.get(f)
        if tdf is None or tdf.empty:
            continue
        df_all = base_filter_no_date(tdf)
        if df_all.empty:
            continue

        mult = float(contracts_map.get(f, 1.0))
        if mult == 0.0:
            continue

        # Build events for this file only (no date filter yet)
        ev_rows = []
        for _, r in df_all.iterrows():
            qty = float(r.get("contracts") or 0.0) * mult
            if qty == 0:
                continue
            ent, exi = r.get("entry_time"), r.get("exit_time")
            dirn = str(r.get("direction") or "")
            # ENTRY
            if pd.notna(ent):
                delta = qty if dirn == "Long" else (-qty if dirn == "Short" else 0.0)
                ev_rows.append({"ts": ent, "delta": delta, "kind_rank": 1})
            # EXIT
            if pd.notna(exi):
                delta = -qty if dirn == "Long" else (+qty if dirn == "Short" else 0.0)
                ev_rows.append({"ts": exi, "delta": delta, "kind_rank": 0})

        if not ev_rows:
            continue

        ev = pd.DataFrame(ev_rows)
        ev["ts"] = pd.to_datetime(ev["ts"], errors="coerce")
        ev = ev.dropna(subset=["ts"]).sort_values(["ts", "kind_rank"])

        # Carry-in = sum of all deltas strictly before start_dt
        carry0 = float(ev.loc[ev["ts"] < start_dt, "delta"].sum())

        # In-window deltas grouped at each timestamp
        inwin = ev.loc[(ev["ts"] >= start_dt) & (ev["ts"] <= end_dt)].copy()
        deltas_by_ts = inwin.groupby("ts")["delta"].sum().sort_index()

        # Build explicit timeline with anchors [start_dt] + event times + [end_dt]
        times = pd.Index([start_dt]).append(deltas_by_ts.index).append(pd.Index([end_dt])).unique().sort_values()

        # Accumulate along the timeline: start at carry0, then add all deltas on each event time
        acc = carry0
        vals = []
        last_t = None
        for t in times:
            if last_t is None:
                # first point -> just anchor with carry
                vals.append(acc)
            else:
                # add any deltas that happen exactly at this t
                acc += float(deltas_by_ts.get(t, 0.0))
                vals.append(acc)
            last_t = t

        s = pd.Series(vals, index=times)
        out[f] = s

    return out








# ------------------------------ Caching helpers ------------------------------



@cache.memoize()
def cached_filtered_df(data_version: int, fname: str,
                       symbol_sel: tuple, interval_sel: tuple, strat_sel: tuple,
                       direction: str, start_date, end_date, mult: float) -> pd.DataFrame:
    """
    Return a filtered (and scaled) view of a file's DataFrame.
    Uses a version key so uploads invalidate cache.
    """
    df = app.server.trade_store.get(fname)
    if df is None or df.empty:
        return pd.DataFrame()
    out = df
    if symbol_sel:
        out = out[out["Symbol"].isin(symbol_sel)]
    if interval_sel:
        out = out[out["Interval"].isin(interval_sel)]
    if strat_sel:
        out = out[out["Strategy"].isin(strat_sel)]
    if direction in ("Long", "Short"):
        out = out[out["direction"] == direction]
    out = _within_dates(out, start_date, end_date, col="exit_time")
    if mult and mult != 1.0 and not out.empty:
        for c in ["net_profit", "runup", "drawdown_trade", "commission", "slippage", "contracts", "CumulativePL_raw"]:
            if c in out.columns:
                out = out.copy()
                out.loc[:, c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0) * float(mult)
    return out

@cache.memoize()
def cached_equity_series(data_version: int, fname: str,
                         symbol_sel: tuple, interval_sel: tuple, strat_sel: tuple,
                         direction: str, start_date, end_date, mult: float) -> pd.Series:
    tdf = cached_filtered_df(data_version, fname, symbol_sel, interval_sel, strat_sel, direction, start_date, end_date, mult)
    return equity_from_trades_subset(tdf)



# ------------------------------ Callbacks ------------------------------

# -------------------- Per-file Contracts Inputs (UI) --------------------

from dash.dependencies import ALL

@app.callback(
    Output("file-contracts", "children"),
    Input("file-toggle", "options"),
    State("store-contracts", "data"),
    prevent_initial_call=False
)

def _render_contract_inputs(file_options, contracts_map):

    # Render one numeric input per file (defaults to 1.0)
    if not file_options:
        return []
    kids = []
    for opt in file_options:
        fname = opt["value"]
        kids.append(
            html.Div([
                html.Label(fname, style={"fontSize": "12px", "fontWeight": 600, "display": "block", "marginBottom": "2px"}),
                dcc.Input(
                    id={"type": "contracts-input", "index": fname},
                    type="number",
                    value=float((contracts_map or {}).get(fname, 1.0)),
                    min=0, step=1,
                    style={"width": "100%"}
                )

            ], style={"border": "1px solid #eee", "borderRadius": "8px", "padding": "6px", "background": "#fafafa"})
        )
    return kids

# -------------------- Per-file Contracts Map (Store) --------------------

@app.callback(
    Output("store-contracts", "data"),
    Input({"type": "contracts-input", "index": ALL}, "value"),
    State({"type": "contracts-input", "index": ALL}, "id"),
    prevent_initial_call=False
)
def _collect_contracts_map(values, ids):
    # Build { filename: multiplier } from the pattern IDs and current values
    if not ids:
        return {}
    out = {}
    for val, _id in zip(values, ids):
        fname = _id.get("index")
        try:
            mul = float(val) if val is not None else 1.0
        except Exception:
            mul = 1.0
        out[str(fname)] = mul
    return out



@app.callback(
    Output("store-trades", "data"),     # now a tiny token dict, not big JSON
    Output("file-list", "children"),
    Output("file-toggle", "options"),
    Output("file-toggle", "value"),
    Output("symbol-toggle", "options"),
    Output("symbol-toggle", "value"),
    Output("interval-toggle", "options"),
    Output("interval-toggle", "value"),
    Output("strategy-toggle", "options"),
    Output("strategy-toggle", "value"),
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
    Output("store-equity-minmax", "data"),
    Output("store-version", "data"),    # bump to invalidate caches
    Input("upload", "contents"),
    State("upload", "filename"),
    State("store-trades", "data"),      # previously stored tokens: {fname: True}
    State("file-toggle", "value"),
    State("symbol-toggle", "value"),
    State("interval-toggle", "value"),
    State("strategy-toggle", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    prevent_initial_call=True
)
def ingest_files(contents_list, names_list, prev_tokens, prev_file_sel, prev_sym_sel, prev_int_sel, prev_strat_sel, prev_start, prev_end):
    if not contents_list:
        raise PreventUpdate

    # Normalize to lists
    if isinstance(contents_list, str):
        contents_list = [contents_list]
        names_list = [names_list]

    # Start from previously stored tokens
    tokens: dict[str, bool] = dict(prev_tokens) if prev_tokens else {}
    existing_names = set(tokens.keys())
    newly_added: list[str] = []

    # Parse and stash DataFrames server-side; store tiny tokens in client store
    for contents, fname in zip(contents_list, names_list):
        raw_bytes, disp_name = decode_upload(contents, fname)
        disp_name = _safe_unique_name(disp_name, existing_names)
        existing_names.add(disp_name)
        tdf = parse_tradestation_trades(raw_bytes, disp_name)

        # Keep the real DF only on the server
        app.server.trade_store[disp_name] = tdf

        # Client only gets a tiny token
        tokens[disp_name] = True
        newly_added.append(disp_name)

    # File options and selected
    file_names = list(tokens.keys())
    file_options = [{"label": n, "value": n} for n in file_names]
    if prev_file_sel:
        file_selected = prev_file_sel + [n for n in newly_added if n not in prev_file_sel]
    else:
        file_selected = file_names

    # File list UI
    file_list_view = html.Ul([html.Li(n) for n in file_names])

    # Compute global min/max dates + all symbols/intervals/strategies from server DFs
    all_min, all_max = None, None
    all_symbols: set[str] = set()
    all_intervals: set[int] = set()
    all_strategies: set[str] = set()

    for fname in file_names:
        tdf = app.server.trade_store.get(fname)
        if tdf is None or tdf.empty:
            continue
        if "Symbol" in tdf:
            all_symbols.update(tdf["Symbol"].dropna().astype(str).unique().tolist())
        if "Interval" in tdf:
            all_intervals.update(pd.to_numeric(tdf["Interval"], errors="coerce").dropna().astype(int).unique().tolist())
        if "Strategy" in tdf:
            all_strategies.update(tdf["Strategy"].dropna().astype(str).unique().tolist())

        dt = pd.to_datetime(tdf.get("exit_time"), errors="coerce").dropna()
        if dt.empty:
            continue
        mn, mx = dt.min().date(), dt.max().date()
        all_min = mn if (all_min is None or mn < all_min) else all_min
        all_max = mx if (all_max is None or mx > all_max) else all_max

    sym_options = [{"label": s, "value": s} for s in sorted(all_symbols)]
    int_options = [{"label": str(i), "value": int(i)} for i in sorted(all_intervals)]
    strat_options = [{"label": s, "value": s} for s in sorted(all_strategies)]

    sym_selected = [v for v in (prev_sym_sel or []) if v in all_symbols] or [s for s in sorted(all_symbols)]
    int_selected = [v for v in (prev_int_sel or []) if v in all_intervals] or [i for i in sorted(all_intervals)]
    strat_selected = [v for v in (prev_strat_sel or []) if v in all_strategies] or [s for s in sorted(all_strategies)]

    start_date = prev_start if prev_start else (all_min if all_min else None)
    end_date   = prev_end   if prev_end   else (all_max if all_max else None)

    eq_minmax = {"min": str(all_min) if all_min else None, "max": str(all_max) if all_max else None}
    version_stamp = int(datetime.now().timestamp())

    return (
        tokens,                # tiny dict {filename: True}
        file_list_view,
        file_options,
        file_selected,
        sym_options, sym_selected,
        int_options, int_selected,
        strat_options, strat_selected,
        start_date, end_date,
        eq_minmax,
        version_stamp,
    )


@app.callback(
    Output("chart-panel", "children"),
    Input("store-trades", "data"),
    Input("file-toggle", "value"),
    Input("symbol-toggle", "value"),
    Input("interval-toggle", "value"),
    Input("strategy-toggle", "value"),
    Input("direction-radio", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("chart-tabs", "value"),
    Input("corr-mode", "value"),
    Input("corr-slope-window", "value"),
    Input("alloc-equity", "value"),        
    Input("store-contracts", "data"),
    Input("store-version", "data"),
)
def update_analysis(store_trades, selected_files, selected_symbols, selected_intervals,
                    selected_strategies, direction, start_date, end_date,
                    active_tab, corr_mode, corr_slope_window,
                    equity_val,                            
                    contracts_map, store_version):

    fig_empty = go.Figure().update_layout(margin=dict(l=10, r=10, t=30, b=10))
    if not store_trades:
        return [dcc.Graph(figure=fig_empty, style={"height": "520px"})]

    available_files = list(store_trades.keys())
    selected_files = [f for f in (selected_files or []) if f in available_files]
    if not selected_files:
        return [dcc.Graph(figure=fig_empty, style={"height": "520px"})]

    # ---- Build once from the selection artifact ----
    sel_key = make_selection_key(
        selected_files, selected_symbols, selected_intervals, selected_strategies,
        direction, start_date, end_date, contracts_map, store_version
    )
    art = build_core_artifact(sel_key)
    trades_by_file = {k: v for k, v in art["trades_by_file"].items() if v is not None and not v.empty}
    equity_by_file = {k: v for k, v in art["equity_by_file"].items() if v is not None and not v.empty}
    if not trades_by_file:
        return [dcc.Graph(figure=fig_empty, style={"height": "520px"})]


    # Display labels
    label_map = art["label_map"]


    # Combine equity
    eq_all = combine_equity(equity_by_file, list(equity_by_file.keys()))


    # ---- Tab switch ----
    if active_tab == "equity":
        return _single_graph_child(build_equity_figure(eq_all, label_map, COLOR_PORTFOLIO))

    if active_tab == "ddcurve":
        return _single_graph_child(build_drawdown_figure(eq_all.get("Portfolio") if eq_all is not None else None))

    if active_tab == "idd":
        return _single_graph_child(build_intraday_dd_figure(eq_all.get("Portfolio") if eq_all is not None else None))

    if active_tab == "hist":
        return _single_graph_child(build_pl_histogram_figure(trades_by_file))


    if active_tab == "corr":
        fig_corr = build_correlation_heatmap(
            equity_by_file,
            label_map=label_map,
            mode=corr_mode or "drawdown_pct",
            slope_window=int(corr_slope_window or 20),
            method="spearman"
        )
        return _single_graph_child(fig_corr)

    if active_tab == "margin":
        # Rebuild the selection keys locally (since we removed the earlier loop)
        symbol_key   = _keyify_list(selected_symbols)
        interval_key = _keyify_list(selected_intervals)
        strat_key    = _keyify_list(selected_strategies)
        # ---- cached net positions (per-file), same basis as allocator ----
        contracts_kv = _keyify_contracts_map(contracts_map)
        files_key = _files_key(selected_files)
        per_file_netpos = cached_netpos_per_file(
            store_version, files_key,
            symbol_key, interval_key, strat_key,
            direction, start_date, end_date,
            contracts_kv
        )

        # Per-symbol aggregation
        by_symbol_net = _aggregate_netpos_per_symbol_from_series(per_file_netpos)

        # Total Initial Margin (Σ |net_sym| × IM(sym))
        idx_union = None
        for s in by_symbol_net.values():
            idx_union = s.index if idx_union is None else idx_union.union(s.index)
        idx_union = (idx_union.sort_values() if idx_union is not None else pd.DatetimeIndex([]))
        total_init_margin = pd.Series(0.0, index=idx_union)
        for sym, s in by_symbol_net.items():
            spec = MARGIN_SPEC.get(sym.upper())
            if spec is None:
                continue
            init_margin = float(spec[0])
            total_init_margin = total_init_margin.add(
                s.reindex(idx_union).ffill().fillna(0.0).abs() * init_margin,
                fill_value=0.0
            )

        # Portfolio equity for purchasing power
        port_eq = eq_all.get("Portfolio") if (eq_all is not None and "Portfolio" in eq_all) else None
        starting_balance = float(equity_val or DEFAULT_INITIAL_CAPITAL)
        pp_series = _purchasing_power_series(port_eq, total_init_margin, starting_balance)

        # A) Purchasing Power
        fig_pp = go.Figure()
        if pp_series is not None and not pp_series.empty:
            fig_pp.add_trace(go.Scattergl(x=pp_series.index, y=pp_series, name="Purchasing Power", line=dict(width=3)))
        fig_pp.update_layout(
            title="Purchasing Power = Starting Balance + Portfolio P&L − Initial Margin Used",
            xaxis_title="Date/Time", yaxis_title="Purchasing Power ($)",
            hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10)
        )

        # B) Initial Margin Used
        fig_im = go.Figure()
        if total_init_margin is not None and not total_init_margin.empty:
            fig_im.add_trace(go.Scattergl(x=total_init_margin.index, y=total_init_margin,
                                          name="Initial Margin Used", line=dict(width=2)))
        fig_im.update_layout(
            title="Initial Margin Used (Σ symbols |net_sym| × IM(sym))",
            xaxis_title="Date/Time", yaxis_title="Initial Margin ($)",
            hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10)
        )

        # C) Net Contracts using precomputed series (no recompute)
        fig_netpos = _netpos_figure_from_series(per_file_netpos, by_symbol_net)

        return [
            dcc.Graph(figure=fig_pp, style={"height": "320px"}),
            dcc.Graph(figure=fig_im, style={"height": "260px"}),
            dcc.Graph(figure=fig_netpos, style={"height": "520px"}),
        ]

    return _single_graph_child(fig_empty)



@app.callback(
    Output("metrics-table", "data"),
    Output("metrics-table", "columns"),
    Input("store-trades", "data"),
    Input("file-toggle", "value"),
    Input("symbol-toggle", "value"),
    Input("interval-toggle", "value"),
    Input("strategy-toggle", "value"),
    Input("direction-radio", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("store-contracts", "data"),
    Input("store-version", "data"),  
    prevent_initial_call=True
)
def update_metrics_table(store_trades, selected_files, selected_symbols, selected_intervals,
                         selected_strategies, direction, start_date, end_date, contracts_map, store_version):

    if not store_trades or not selected_files:
        return [], []

    # Build once from the selection artifact
    sel_key = make_selection_key(
        selected_files, selected_symbols, selected_intervals, selected_strategies,
        direction, start_date, end_date, contracts_map, store_version
    )
    art = build_core_artifact(sel_key)

    rows = []
    frames_for_port = []

    for f in selected_files:
        tdf = art["trades_by_file"].get(f)
        if tdf is None or tdf.empty:
            continue
        frames_for_port.append(tdf)
        row = {
            "File": f,
            "Symbol": (tdf["Symbol"].iloc[0] if "Symbol" in tdf and not tdf.empty else ""),
            "Interval": (int(tdf["Interval"].iloc[0]) if "Interval" in tdf and not tdf.empty and not pd.isna(tdf["Interval"].iloc[0]) else None),
            "Strategy": (tdf["Strategy"].iloc[0] if "Strategy" in tdf and not tdf.empty else ""),
            "Direction": direction,
        }
        row.update(compute_metrics(tdf, DEFAULT_INITIAL_CAPITAL))
        rows.append(row)


    if frames_for_port:
        pf_trades = pd.concat(frames_for_port, ignore_index=True)
        pf_row = {"File": "Portfolio", "Symbol": "-", "Interval": None, "Strategy": "-", "Direction": direction}
        pf_row.update(compute_metrics(pf_trades, DEFAULT_INITIAL_CAPITAL))
        rows.append(pf_row)

    col_order = [
        "File", "Symbol", "Interval", "Strategy", "Direction",
        "Total Net Profit", "Gross Profit", "Gross Loss", "Profit Factor",
        "Total Number of Trades", "Percent Profitable", "Winning Trades", "Losing Trades",
        "Avg. Trade Net Profit", "Avg. Winning Trade", "Avg. Losing Trade", "Ratio Avg. Win:Avg. Loss",
        "Largest Winning Trade", "Largest Losing Trade",
        "Max. Consecutive Winning Trades", "Max. Consecutive Losing Trades",
        "Max. Shares/Contracts Held", "Total Shares/Contracts Held",
        "Account Size Required",
        "Total Commission", "Total Slippage",
        "Max. Drawdown (Trade Close to Trade Close) Value", "Max. Drawdown (Trade Close to Trade Close) Date",
        "Max. Drawdown (Intra-Day Peak to Valley) Value", "Max. Drawdown (Intra-Day Peak to Valley) Date",
        "Max. Trade Drawdown",
        "Return on Initial Capital", "Annual Rate of Return", "Return on Account",
        "Avg. Monthly Return (full months)", "Std. Deviation of Monthly Return (full months)",
        "Trading Period", "Percent of Time in the Market", "Time in the Market", "Longest Flat Period",
        "Max. Equity Run-up", "Max Equity Run-up as % of Initial Capital",
    ]

    rows_df = pd.DataFrame(rows)
    for c in col_order:
        if c not in rows_df.columns:
            rows_df[c] = np.nan
    rows_df = rows_df[col_order]

    def _guess_type(series: pd.Series) -> str:
        return "numeric" if pd.api.types.is_numeric_dtype(series) else "text"

    cols = [{"name": c, "id": c, "type": _guess_type(rows_df[c])} for c in rows_df.columns]
    return rows_df.to_dict("records"), cols


# ------------------------------- Exports -------------------------------

@app.callback(
    Output("alloc-summary", "children"),
    Output("alloc-portfolio-graph", "figure"),
    Output("alloc-weights-table", "data"),
    Output("alloc-weights-table", "columns"),
    Input("btn-run-alloc", "n_clicks"),
    State("store-trades", "data"),
    State("file-toggle", "value"),
    State("symbol-toggle", "value"),
    State("interval-toggle", "value"),
    State("strategy-toggle", "value"),
    State("direction-radio", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("alloc-equity", "value"),
    State("alloc-margin-buffer", "value"),
    State("alloc-objective", "value"),
    State("alloc-lev-cap", "value"),
    State("alloc-step", "value"),
    State("store-contracts", "data"),
    State("store-version", "data"),  
    prevent_initial_call=True
)
def run_allocator(n_clicks, store_trades, selected_files,
                  selected_symbols, selected_intervals, selected_strategies,
                  direction, start_date, end_date,
                  equity_val, margin_pct, objective, lev_cap, step,
                  contracts_map, store_version):

    if not n_clicks or not store_trades or not selected_files:
        return "Upload and select files, then run.", go.Figure(), [], []

    # Build once from the selection artifact
    sel_key = make_selection_key(
        selected_files, selected_symbols, selected_intervals, selected_strategies,
        direction, start_date, end_date, contracts_map, store_version
    )
    art = build_core_artifact(sel_key)

    equity_by_file = {k: v for k, v in art["equity_by_file"].items() if v is not None and not v.empty}
    frames_for_port = [v for v in art["trades_by_file"].values() if v is not None and not v.empty]

    symbol_key   = _keyify_list(selected_symbols)
    interval_key = _keyify_list(selected_intervals)
    strat_key    = _keyify_list(selected_strategies)



    # Use the SAME carry-in + anchors series as the Margin tab
    contracts_kv = _keyify_contracts_map(contracts_map)
    files_key    = _files_key(selected_files)
    netpos = cached_netpos_per_file(
        store_version, files_key,
        symbol_key, interval_key, strat_key,
        direction, start_date, end_date,
        contracts_kv
    )

    init_cap   = float(equity_val or DEFAULT_INITIAL_CAPITAL)
    margin_buf = max(0.0, min(1.0, (margin_pct or 60)/100.0))
    lev_cap    = float(lev_cap or 1.0)
    step       = float(step or 0.25)

    weights, diag = find_margin_aware_weights(
        equity_by_file=equity_by_file,
        per_file_netpos=netpos,
        store_trades=store_trades,
        initial_capital=init_cap,
        margin_spec=MARGIN_SPEC,
        margin_buffer=margin_buf,
        objective=objective,
        step=step,
        leverage_cap=lev_cap
    )

    if not weights:
        return "No feasible allocation under margin constraints. Try raising margin headroom, lowering leverage cap, or coarsening step.", go.Figure(), [], []

    port   = _portfolio_equity_from_weights(equity_by_file, weights)
    ann    = _annualized_return(port, init_cap)
    dd_mag = abs(_peak_drawdown_value(port))  # positive magnitude for display

    # Margin series at chosen weights (same netpos basis as above)
    idx_union = None
    for ser in netpos.values():
        idx_union = ser.index if idx_union is None else idx_union.union(ser.index)
    idx_union = idx_union.sort_values()
    total_margin = pd.Series(0.0, index=idx_union)
    for f, ser in netpos.items():
        df_trades = app.server.trade_store.get(f)
        sym_raw = _symbol_from_first_row(df_trades)
        sym = (sym_raw or "").upper()

        spec = MARGIN_SPEC.get(sym)
        if spec is None:
            continue
        init_margin = float(spec[0])
        total_margin = total_margin.add(
            ser.reindex(idx_union).ffill().fillna(0.0).abs() * init_margin * float(weights.get(f, 0.0)),
            fill_value=0.0
        )

    summary = (f"Best allocation ({diag.get('mode','grid')}): "
               f"Ann. Return ~ {ann*100:0.2f}% | Max DD {dd_mag:,.0f} | "
               f"Max Margin Used ${total_margin.max():,.0f} "
               f"({(total_margin.max()/init_cap*100):0.1f}% of equity, cap {margin_buf*100:0.0f}%)")

    fig = go.Figure()
    for f, s in equity_by_file.items():
        w = float(weights.get(f, 0.0))
        if w > 0 and s is not None and not s.empty:
            fig.add_trace(go.Scattergl(x=s.index, y=s * w, name=f"{f} × {w:0.2f}", line=dict(width=1)))
    fig.add_trace(go.Scattergl(x=port.index, y=port, name="Portfolio (alloc)", line=dict(width=3)))
    fig.update_layout(title="Allocated Portfolio Equity", xaxis_title="Date", yaxis_title="Cumulative P/L",
                      hovermode="x unified", margin=dict(l=10,r=10,t=40,b=10))

    rows = []
    for f, w in sorted(weights.items()):
        if w <= 0:
            continue
        df_trades = app.server.trade_store.get(f)
        sym_raw = _symbol_from_first_row(df_trades)
        rows.append({
            "File": f,
            "Symbol": (sym_raw or "").upper(),
            "Weight": float(w),
            "Note": "scale factor for P&L; convert to contracts via margin",
        })


    cols = [
        {"name": "File", "id": "File"},
        {"name": "Symbol", "id": "Symbol"},
        {"name": "Weight", "id": "Weight", "type": "numeric"},
        {"name": "Note", "id": "Note"},
    ]

    return summary, fig, rows, cols



def export_trades(n_clicks, store_trades, sel_files, sel_syms, sel_ints, sel_strats, direction,
                  start_date, end_date, contracts_map, store_version):
    if not n_clicks or not store_trades or not sel_files:
        return None

    # Build once from the selection artifact
    sel_key = make_selection_key(
        sel_files, sel_syms, sel_ints, sel_strats,
        direction, start_date, end_date, contracts_map, store_version
    )
    art = build_core_artifact(sel_key)

    frames = [tdf for f, tdf in art["trades_by_file"].items() if tdf is not None and not tdf.empty and f in (sel_files or [])]

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True).sort_values(["File", "exit_time"])
    export_cols = [
        "File", "Symbol", "Interval", "Strategy", "direction",
        "entry_time", "exit_time",
        "entry_type", "entry_price", "exit_price",
        "contracts", "net_profit", "runup", "drawdown_trade",
        "commission", "slippage", "CumulativePL_raw",
    ]
    export_cols = [c for c in export_cols if c in out.columns]
    out = out[export_cols]
    return dcc.send_data_frame(out.to_csv, "filtered_trades.csv", index=False)

@app.callback(
    Output("download-trades", "data"),
    Input("btn-export-trades", "n_clicks"),
    State("store-trades", "data"),
    State("file-toggle", "value"),
    State("symbol-toggle", "value"),
    State("interval-toggle", "value"),
    State("strategy-toggle", "value"),
    State("direction-radio", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("store-contracts", "data"),
    State("store-version", "data"),  
    prevent_initial_call=True
)
def _cb_export_trades(n_clicks, store_trades, sel_files, sel_syms, sel_ints, sel_strats, direction, start_date, end_date, contracts_map, store_version):
    return export_trades(n_clicks, store_trades, sel_files, sel_syms, sel_ints, sel_strats, direction, start_date, end_date, contracts_map, store_version)


@app.callback(
    Output("download-metrics", "data"),
    Input("btn-export-metrics", "n_clicks"),
    State("metrics-table", "data"),
    prevent_initial_call=True
)
def _cb_export_metrics(n_clicks, rows):
    if not n_clicks or not rows:
        return None
    df = pd.DataFrame(rows)
    return dcc.send_data_frame(df.to_csv, "metrics.csv", index=False)

# ----------------------------- Entrypoint ------------------------------
if __name__ == "__main__":
    webbrowser.open_new("http://127.0.0.1:8050/")
    app.run(debug=True, use_reloader=False, port=8050)
