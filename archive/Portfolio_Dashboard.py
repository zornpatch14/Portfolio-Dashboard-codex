#!/usr/bin/env python3
"""
Portfolio Trade Analysis Dashboard (Dash)
========================================

Features
--------
- Drag & drop multiple TradeStation "Trades List" .xlsx exports
- Robust parser for 2-row per trade format (header auto-detect)
- Per-trade net profit (incl. costs) and cumulative equity
- Include/exclude files with toggles
- Global date filter with DatePickerRange
- Charts (toggle which to show):
    * Equity curves (per-file + portfolio)
    * Portfolio max drawdown over time
    * Portfolio intraday drawdown (worst per day)
    * Histogram of trade P/L
- Metrics table (per file + portfolio)
- Export filtered trades & metrics as CSV

Packaging (optional)
--------------------
- Build a one-file executable (Windows):
    pip install pyinstaller
    pyinstaller --noconfirm --onefile --name TradeAnalysisDash portfolio_dash.py

Then run: TradeAnalysisDash.exe

Notes
-----
- No external services required. Runs on localhost in your default browser.
- If your "Trades List" sheet is named differently, the parser will try to
  find a sheet containing "Trades List". Otherwise, set SHEET_HINTS below.

"""

import base64
import io
import os
from datetime import datetime, date

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update, ctx
import plotly.graph_objects as go

# ----------------------------- Config ---------------------------------

SHEET_HINTS = ["Trades List", "Trades", "Sheet1"]  # parser will search these names
APP_TITLE   = "Portfolio Trade Analysis"
COLOR_PORTFOLIO = "black"  # portfolio equity color


# -------------------------- Parsing & ETL ------------------------------

def _find_header_row(raw_df: pd.DataFrame) -> int | None:
    """Find header row by locating a row containing both '#' and 'Type'."""
    for i in range(min(50, len(raw_df))):  # scan first 50 rows defensively
        row = raw_df.iloc[i].astype(str)
        if row.str.contains(r"^#").any() and row.str.contains("Type", case=False).any():
            return i
        # looser fallback: any '#' cell + a 'Type' cell
        if ("#" in row.values) and row.str.contains("Type", case=False).any():
            return i
    return None


def _load_sheet_guess(xls: pd.ExcelFile) -> str:
    """Pick a sheet name likely to be the 'Trades List'."""
    sheets = xls.sheet_names
    # look for any sheet that contains our hints
    for hint in SHEET_HINTS:
        for s in sheets:
            if hint.lower() in s.lower():
                return s
    # fallback to first sheet
    return sheets[0]


def parse_tradestation_trades(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse a TradeStation 'Trades List' Excel file into a tidy trade list.

    Returns DataFrame with columns:
        File, EntryDateTime, ExitDateTime, NetProfit_incl_costs, CumulativePL_raw
    """
    # Read without header to find header row dynamically
    bio = io.BytesIO(file_bytes)
    xls = pd.ExcelFile(bio, engine="openpyxl")
    sheet = _load_sheet_guess(xls)
    raw_df = pd.read_excel(xls, sheet_name=sheet, header=None)
    hdr_idx = _find_header_row(raw_df)
    if hdr_idx is None:
        # fallback: hope row 2 is a header, common in TS exports
        hdr_idx = 2

    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet, header=hdr_idx, engine="openpyxl")

    # Clean rows that are repeated headers, blanks, etc.
    if "Type" not in df.columns:
        # try to locate a 'Type' column by case-insensitive
        matches = [c for c in df.columns if isinstance(c, str) and c.lower() == "type"]
        if matches:
            df.rename(columns={matches[0]: "Type"}, inplace=True)

    df = df.dropna(subset=["Type"])
    df = df[df["Type"] != "Type"]

    # Standardize column aliases
    col_map = {}
    # '#'
    for c in df.columns:
        if isinstance(c, str) and c.strip().startswith("#"):
            col_map[c] = "#"
    # Date/Time
    if "Date/Time" not in df.columns:
        for c in df.columns:
            if isinstance(c, str) and "date" in c.lower():
                col_map[c] = "Date/Time"
                break
    # Cumulative Net
    cum_candidates = [
        "Net Profit - Cum Net Profit", "Cum Net Profit", "Cumulative Net",
        "NetProfitCum", "Cum P&L", "Cum Profit"
    ]
    cum_name = None
    for cand in cum_candidates:
        if cand in df.columns:
            cum_name = cand
            break
    if cum_name is None:
        # try fuzzy
        for c in df.columns:
            s = str(c).lower()
            if "cum" in s and ("net" in s or "profit" in s or "p&l" in s):
                cum_name = c
                break
    if cum_name is None:
        raise ValueError(f"Could not find cumulative P/L column in {filename}")

    if col_map:
        df = df.rename(columns=col_map)

    # Build trade numbers (forward-fill from '#')
    if "#" in df.columns:
        df["TradeNo"] = pd.to_numeric(df["#"], errors="coerce").ffill().astype("Int64")
    else:
        # If missing '#', create a progressive ID
        df["TradeNo"] = pd.Series(np.arange(1, len(df) + 1), dtype="Int64")

    # Parse date/time
    df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="coerce")

    df["CumulativePL_raw"] = pd.to_numeric(df[cum_name], errors="coerce")

    entry_types = {"Buy", "Sell Short"}
    exit_types  = {"Sell", "Buy to Cover"}

    trades = []
    last_exit_raw = 0.0
    open_trades = {}

    for _, row in df.iterrows():
        tno = row["TradeNo"]
        rtype = str(row["Type"]).strip()
        raw_val = row["CumulativePL_raw"]
        if pd.isna(raw_val):
            raw_val = last_exit_raw

        if rtype in entry_types:
            open_trades[tno] = {
                "EntryDateTime": row["Date/Time"],
                "EntryType": rtype,
                "EntryRawPL": raw_val,
            }
        elif rtype in exit_types:
            tr = open_trades.pop(tno, {
                "EntryDateTime": None,
                "EntryType": None,
                "EntryRawPL": last_exit_raw,
            })
            netp = float(raw_val) - float(last_exit_raw)
            tr.update({
                "ExitDateTime": row["Date/Time"],
                "CumulativePL_raw": float(raw_val),
                "NetProfit_incl_costs": float(netp),
            })
            last_exit_raw = float(raw_val)
            trades.append(tr)

    # Include open trades (0 profit at last known equity)
    for _, tr in open_trades.items():
        trades.append({
            "EntryDateTime": tr["EntryDateTime"],
            "ExitDateTime": pd.NaT,
            "CumulativePL_raw": float(last_exit_raw),
            "NetProfit_incl_costs": 0.0,
        })

    out = pd.DataFrame(trades)
    out.insert(0, "File", os.path.basename(filename))
    return out


def equity_from_trades(trades_df: pd.DataFrame) -> pd.Series:
    """Series of cumulative equity indexed by ExitDateTime (drop NaT)."""
    s = (
        trades_df.dropna(subset=["ExitDateTime"])
                 .drop_duplicates(subset=["ExitDateTime"], keep="last")
                 .set_index("ExitDateTime")["CumulativePL_raw"]
                 .sort_index()
    )
    return s


def combine_equity(equity_by_file: dict[str, pd.Series], files_selected: list[str]) -> pd.DataFrame:
    """Outer-join per-file equity curves; forward-fill; add 'Portfolio' sum."""
    if not files_selected:
        return pd.DataFrame()
    series_list = []
    for f in files_selected:
        if f in equity_by_file and not equity_by_file[f].empty:
            series_list.append(equity_by_file[f].rename(f))
    if not series_list:
        return pd.DataFrame()
    eq = pd.concat(series_list, axis=1).sort_index().ffill()
    eq["Portfolio"] = eq.sum(axis=1)
    return eq


def max_drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown time series: equity - running_max (<= 0)."""
    if equity.empty:
        return equity
    run_max = equity.cummax()
    return equity - run_max


def max_drawdown_value(equity: pd.Series) -> float:
    dd = max_drawdown_series(equity)
    return float(dd.min()) if not dd.empty else 0.0


def intraday_drawdown_series(equity: pd.Series) -> pd.Series:
    """
    For each day: worst (most negative) peak-to-trough decline within that day.
    Returns a daily Series indexed by date with negative values (drawdowns).
    """
    if equity.empty:
        return pd.Series(dtype=float)
    # Ensure datetime index without tz
    s = equity.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    out = {}
    for d, g in s.groupby(s.index.date):
        g = g.sort_index()
        run_max = g.cummax()
        dd = g - run_max
        out[d] = dd.min() if not dd.empty else 0.0
    return pd.Series(out, dtype=float)


def trade_metrics(trades: pd.DataFrame, equity: pd.Series | None) -> dict:
    """
    Compute summary metrics from trades (and equity for max DD).
    Returns a dict of metrics.
    """
    trades = trades.copy()
    # Filter to rows with actual exits when computing win/loss stats
    t = trades.dropna(subset=["ExitDateTime"])
    pnl = t["NetProfit_incl_costs"].astype(float)

    total = float(pnl.sum()) if not pnl.empty else 0.0
    n_trades = int(len(pnl))
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    win_rate = (wins / n_trades * 100.0) if n_trades else 0.0
    avg_trade = float(pnl.mean()) if n_trades else 0.0
    avg_win = float(pnl[pnl > 0].mean()) if wins else 0.0
    avg_loss = float(pnl[pnl < 0].mean()) if losses else 0.0
    gross_win = float(pnl[pnl > 0].sum()) if wins else 0.0
    gross_loss = abs(float(pnl[pnl < 0].sum())) if losses else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else np.inf if gross_win > 0 else 0.0
    expectancy = (win_rate/100.0)*avg_win - (1 - win_rate/100.0)*abs(avg_loss)
    median_trade = float(pnl.median()) if n_trades else 0.0
    max_dd = max_drawdown_value(equity) if equity is not None else 0.0

    return {
        "Net Profit": total,
        "# Trades": n_trades,
        "Win Rate %": win_rate,
        "Avg Trade": avg_trade,
        "Profit Factor": profit_factor,
        "Max Drawdown": max_dd,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss,
        "Expectancy": expectancy,
        "Median Trade": median_trade,
    }


# ------------------------------ Dash UI --------------------------------

app = Dash(__name__)
app.title = APP_TITLE

def pretty_card(children):
    return html.Div(
        children,
        style={
            "border": "1px solid #ddd",
            "borderRadius": "12px",
            "padding": "14px",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.06)",
            "background": "white",
        },
    )

# --- keep pretty_card as-is ---

app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial", "padding": "18px", "background": "#f6f7fb"},
    children=[
        html.H2(APP_TITLE, style={"marginTop": 0}),

        # Top row: Upload (full width)
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

        # Second row: sidebar controls (left) + charts (right)
        html.Div([
            pretty_card([
                html.H4("2) Controls"),
                html.Div([
                    html.Div("Included files:", style={"fontWeight": 600, "marginBottom": "6px"}),
                    dcc.Checklist(id="file-toggle", options=[], value=[], inline=False),
                ], style={"marginBottom": "12px"}),

                html.Div([
                    html.Div("Date range:", style={"fontWeight": 600, "marginBottom": "6px"}),
                    dcc.DatePickerRange(
                        id="date-range", start_date=None, end_date=None,
                        display_format="YYYY-MM-DD", minimum_nights=0
                    ),
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
                    ],
                ),
                dcc.Graph(id="chart-graph", style={"height": "520px"}),
            ]),
        ], style={"display": "grid", "gridTemplateColumns": "0.9fr 2fr", "gap": "14px"}),

        # Metrics (unchanged)
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

        # Hidden stores (unchanged)
        dcc.Store(id="store-trades"),
        dcc.Store(id="store-equity-minmax"),
    ]
)


# ------------------------------ Helpers --------------------------------

def _safe_unique_name(name: str, existing: set[str]) -> str:
    base = name
    n = 2
    while name in existing:
        name = f"{base} ({n})"
        n += 1
    return name


def _decode_upload(contents: str, filename: str) -> tuple[bytes, str]:
    """Decode a dcc.Upload content string to raw bytes and a sanitized display name."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    display = os.path.basename(filename) or "uploaded.xlsx"
    return decoded, display


def _within_dates(df: pd.DataFrame, start_d: date | None, end_d: date | None, col="ExitDateTime") -> pd.DataFrame:
    if start_d is None and end_d is None:
        return df
    out = df.copy()
    if start_d is not None:
        out = out[out[col] >= pd.Timestamp(start_d)]
    if end_d is not None:
        out = out[out[col] <= pd.Timestamp(end_d) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)]
    return out


# ------------------------------ Callbacks -------------------------------

from dash.exceptions import PreventUpdate

@app.callback(
    Output("store-trades", "data"),
    Output("file-list", "children"),
    Output("file-toggle", "options"),
    Output("file-toggle", "value"),
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
    Output("store-equity-minmax", "data"),
    Input("upload", "contents"),
    State("upload", "filename"),
    State("store-trades", "data"),      # <-- keep previously stored files
    State("file-toggle", "value"),      # <-- keep previous selections
    State("date-range", "start_date"),  # <-- preserve user-set dates
    State("date-range", "end_date"),
    prevent_initial_call=True
)
def ingest_files(contents_list, names_list, prev_store, prev_selected, prev_start, prev_end):
    if not contents_list:
        raise PreventUpdate

    # Normalize to lists (dcc.Upload may send a single string if one file)
    if isinstance(contents_list, str):
        contents_list = [contents_list]
        names_list = [names_list]

    # Start from whatever we already had
    parsed_by_file = dict(prev_store) if prev_store else {}

    # Track all names we already have to avoid collisions
    existing_names = set(parsed_by_file.keys())
    newly_added = []

    # Parse each newly uploaded file and add/merge into the store
    for contents, fname in zip(contents_list, names_list):
        raw_bytes, disp_name = _decode_upload(contents, fname)
        disp_name = _safe_unique_name(disp_name, existing_names)
        existing_names.add(disp_name)

        trades_df = parse_tradestation_trades(raw_bytes, disp_name)
        parsed_by_file[disp_name] = trades_df.to_dict(orient="records")
        newly_added.append(disp_name)

    # Build checklist options and selected values:
    options = [{"label": n, "value": n} for n in parsed_by_file.keys()]
    # Keep whatever the user had selected and also select the newly added files
    if prev_selected:
        selected = prev_selected + [n for n in newly_added if n not in prev_selected]
    else:
        selected = list(parsed_by_file.keys())  # first time: select all

    # File list UI
    file_list_view = html.Ul([html.Li(n) for n in parsed_by_file.keys()])

    # Compute global min/max across ALL stored files
    all_min, all_max = None, None
    for recs in parsed_by_file.values():
        if not recs:
            continue
        tdf = pd.DataFrame(recs)
        dt = pd.to_datetime(tdf["ExitDateTime"], errors="coerce").dropna()
        if dt.empty:
            continue
        mn, mx = dt.min().date(), dt.max().date()
        all_min = mn if (all_min is None or mn < all_min) else all_min
        all_max = mx if (all_max is None or mx > all_max) else all_max

    # Respect the user's current date selection if they already set one
    start_date = prev_start if prev_start else (all_min if all_min else None)
    end_date   = prev_end   if prev_end   else (all_max if all_max else None)

    return (
        parsed_by_file,
        file_list_view,
        options,
        selected,
        start_date,
        end_date,
        {"min": str(all_min) if all_min else None, "max": str(all_max) if all_max else None},
    )



from dash import no_update

@app.callback(
    Output("chart-graph", "figure"),
    Output("metrics-table", "data"),
    Output("metrics-table", "columns"),
    Input("store-trades", "data"),
    Input("file-toggle", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("chart-tabs", "value"),
)
def update_analysis(store_trades, selected_files, start_date, end_date, active_tab):
    fig_empty = go.Figure().update_layout(margin=dict(l=10, r=10, t=30, b=10))
    if not store_trades:
        return fig_empty, [], []

    # Keep only selections that actually exist
    available = list(store_trades.keys())
    selected_files = [f for f in (selected_files or []) if f in available]
    if not selected_files:
        return fig_empty, [], []

    # Rehydrate trades + equity by file
    trades_by_file, equity_by_file = {}, {}
    for fname in selected_files:
        recs = store_trades.get(fname, [])
        if not recs:
            continue
        tdf = pd.DataFrame(recs)
        tdf["ExitDateTime"] = pd.to_datetime(tdf["ExitDateTime"], errors="coerce")
        tdf["EntryDateTime"] = pd.to_datetime(tdf["EntryDateTime"], errors="coerce")

        # Date-filter trades for metrics/hist
        tdf_f = _within_dates(tdf, start_date, end_date, col="ExitDateTime")
        trades_by_file[fname] = tdf_f

        # Equity (filter on index)
        eq_series = equity_from_trades(tdf)
        if start_date or end_date:
            eq_series = eq_series.loc[
                (eq_series.index >= (pd.Timestamp(start_date) if start_date else eq_series.index.min())) &
                (eq_series.index <= (pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                                     if end_date else eq_series.index.max()))
            ]
        equity_by_file[fname] = eq_series

    # Combine equity → portfolio
    eq_all = combine_equity(equity_by_file, list(equity_by_file.keys()))

    # -------- Build figures for each tab --------
    # Equity Curves
    def build_equity_fig():
        if eq_all is None or eq_all.empty:
            return fig_empty
        f = go.Figure()
        for col in [c for c in eq_all.columns if c != "Portfolio"]:
            f.add_trace(go.Scatter(x=eq_all.index, y=eq_all[col], name=col, mode="lines", line=dict(width=1)))
        if "Portfolio" in eq_all:
            f.add_trace(go.Scatter(x=eq_all.index, y=eq_all["Portfolio"], name="Portfolio",
                                   mode="lines", line=dict(width=2.5, color=COLOR_PORTFOLIO)))
        f.update_layout(
            title="Equity Curves",
            xaxis_title="Date", yaxis_title="Cumulative P/L",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        return f

    # Portfolio Drawdown (equity - running max)
    def build_dd_fig():
        if eq_all is None or "Portfolio" not in eq_all or eq_all["Portfolio"].empty:
            return fig_empty
        dd = max_drawdown_series(eq_all["Portfolio"])
        f = go.Figure()
        f.add_trace(go.Scatter(x=dd.index, y=dd, name="Drawdown", mode="lines", line=dict(width=2)))
        f.update_layout(
            title="Portfolio Drawdown Over Time",
            xaxis_title="Date", yaxis_title="Drawdown",
            hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10)
        )
        return f

    # Intraday Drawdown (worst per day)
    def build_idd_fig():
        if eq_all is None or "Portfolio" not in eq_all or eq_all["Portfolio"].empty:
            return fig_empty
        idd = intraday_drawdown_series(eq_all["Portfolio"])
        f = go.Figure()
        f.add_trace(go.Bar(x=idd.index, y=idd))
        f.update_layout(
            title="Portfolio Intraday Drawdown (worst per day)",
            xaxis_title="Date", yaxis_title="Intraday Drawdown",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return f

    # Histogram of Trade P/L
    def build_hist_fig():
        all_pnl = []
        for tdf in trades_by_file.values():
            pnl = pd.to_numeric(tdf["NetProfit_incl_costs"], errors="coerce")
            pnl = pnl[~pnl.isna()]
            all_pnl.append(pnl)
        pnl_all = pd.concat(all_pnl) if all_pnl else pd.Series(dtype=float)
        f = go.Figure()
        if not pnl_all.empty:
            f.add_trace(go.Histogram(x=pnl_all, nbinsx=50, name="Trade P/L"))
        f.update_layout(
            title="Histogram of Trade P/L",
            xaxis_title="P/L per Trade", yaxis_title="Count",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return f

    # Pick figure by tab
    if active_tab == "equity":
        chart_fig = build_equity_fig()
    elif active_tab == "ddcurve":
        chart_fig = build_dd_fig()
    elif active_tab == "idd":
        chart_fig = build_idd_fig()
    elif active_tab == "hist":
        chart_fig = build_hist_fig()
    else:
        chart_fig = fig_empty

    # -------- Metrics table (unchanged logic) --------
    rows = []
    for f, tdf in trades_by_file.items():
        eq = equity_by_file.get(f, pd.Series(dtype=float))
        row = {"File": f}
        row.update(trade_metrics(tdf, eq))
        rows.append(row)

    if eq_all is not None and "Portfolio" in eq_all and not eq_all.empty:
        pf_trades = pd.concat([tdf.assign(File=f) for f, tdf in trades_by_file.items()], ignore_index=True)
        pf_trades = pf_trades.sort_values("ExitDateTime")
        pf_row = {"File": "Portfolio"}
        pf_row.update(trade_metrics(pf_trades, eq_all["Portfolio"]))
        rows.append(pf_row)

    col_order = ["File", "Net Profit", "# Trades", "Win Rate %", "Avg Trade",
                 "Profit Factor", "Max Drawdown", "Avg Win", "Avg Loss",
                 "Expectancy", "Median Trade"]
    cols = [{"name": c, "id": c, "type": "numeric" if c != "File" else "text"} for c in col_order]
    rows_df = pd.DataFrame(rows)[col_order] if rows else pd.DataFrame(columns=col_order)

    return chart_fig, rows_df.to_dict("records"), cols



@app.callback(
    Output("download-trades", "data"),
    Input("btn-export-trades", "n_clicks"),
    State("store-trades", "data"),
    State("file-toggle", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    prevent_initial_call=True
)
def export_trades(n_clicks, store_trades, selected_files, start_date, end_date):
    if not n_clicks or not store_trades or not selected_files:
        return None
    df_list = []
    for f in selected_files:
        recs = store_trades.get(f, [])
        if not recs:
            continue
        tdf = pd.DataFrame(recs)
        tdf["ExitDateTime"] = pd.to_datetime(tdf["ExitDateTime"], errors="coerce")
        tdf["EntryDateTime"] = pd.to_datetime(tdf["EntryDateTime"], errors="coerce")
        tdf = _within_dates(tdf, start_date, end_date, col="ExitDateTime").assign(File=f)
        df_list.append(tdf)

    if not df_list:
        return None
    out = pd.concat(df_list, ignore_index=True)
    out = out.sort_values(["File", "ExitDateTime"])
    return dcc.send_data_frame(out.to_csv, "filtered_trades.csv", index=False)


@app.callback(
    Output("download-metrics", "data"),
    Input("btn-export-metrics", "n_clicks"),
    State("metrics-table", "data"),
    prevent_initial_call=True
)
def export_metrics(n_clicks, rows):
    if not n_clicks or not rows:
        return None
    df = pd.DataFrame(rows)
    return dcc.send_data_frame(df.to_csv, "metrics.csv", index=False)


# ----------------------------- Entrypoint -------------------------------

#if __name__ == "__main__":
#    # Launch the Dash app
#    # Visit http://127.0.0.1:8050/ in your browser if it doesn't auto-open
#    app.run(debug=True)

import webbrowser

if __name__ == "__main__":
    webbrowser.open_new("http://127.0.0.1:8050/")
    app.run(debug=True, use_reloader=False, port=8050)

