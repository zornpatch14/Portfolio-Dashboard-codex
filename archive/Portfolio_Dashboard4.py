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

import base64
import io
import os
import re
from datetime import datetime, date, timedelta
from typing import Any, Iterable, Tuple
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update, ctx
import plotly.graph_objects as go
import plotly.express as px  
from dash.exceptions import PreventUpdate
import webbrowser
import itertools  



# ----------------------------- Config ---------------------------------

APP_TITLE = "Portfolio Trade Analysis"
COLOR_PORTFOLIO = "black"              # portfolio equity color
SHEET_HINTS = ["Trades List", "Trades", "Sheet1"]  # parser sheet guesses
DEFAULT_INITIAL_CAPITAL = 25_000.0

# ---- Margin & contract config (edit to your markets) ----
# Initial & maintenance margin per CONTRACT. Keep maintenance as a fraction of initial if you prefer.
MARGIN_SPEC = {
    # symbol_root: (initial_margin, maintenance_margin, big_point_value)
    "MNQ": (3250, 3250, 2.0),    # example values; replace with your broker's
    "MES": (2300.0, 2300.0, 5.0),
    "MYM": (1500.0, 1500.0, 0.5),
    "M2K":  (1000.0, 1000.0, 5.0),
    "CD":  (1100.0, 1100.0, 10.0),
    "JY":  (3100.0, 3100.0, 6.25),
    "NE1":  (1500.0, 1500.0, 10.0),
    "NG":  (3800.0, 3800.0, 10.0),
    # ...add more symbols here...
}

# Conservative headroom: cap total margin use to this fraction of account equity (e.g., 0.6 = 60%)
MARGIN_BUFFER = 1.00


# Filename parsing regex (from integration instructions)
FILENAME_RE = re.compile(
    r"(?i)tradeslist[_-](?P<symbol>[A-Za-z]+)[_-](?P<interval>\d+)[_-](?P<strategy>[^.]+)"
)

# -------------------------- Parsing & ETL ------------------------------

def _find_header_row(raw_df: pd.DataFrame) -> int | None:
    """Find header row by locating a row containing both '#', 'Type', and 'Date/Time'."""
    for i in range(min(50, len(raw_df))):
        row = raw_df.iloc[i].astype(str).str.strip()
        if ("#" in row.values) and row.str.contains("Type", case=False, na=False).any() \
           and row.str.contains("Date/Time", case=False, na=False).any():
            return i
    return None

def _load_sheet_guess(xls: pd.ExcelFile) -> str:
    """Pick the sheet likely to be the 'Trades List'."""
    for hint in SHEET_HINTS:
        for s in xls.sheet_names:
            if hint.lower() in s.lower():
                return s
    return xls.sheet_names[0]

def parse_filename_meta(filename: str) -> tuple[str, int | None, str]:
    """
    Extract (Symbol, Interval, Strategy) from filenames like:
      tradeslist_MNQ_15_3X.xlsx
    Fallbacks try underscore splitting.
    """
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    m = FILENAME_RE.search(base)
    if m:
        sym = m.group("symbol").upper()
        interval = int(m.group("interval"))
        strat = m.group("strategy")
        return sym, interval, strat
    # Fallback: split on underscores
    toks = name.split("_")
    if len(toks) >= 4 and toks[0].lower().startswith("tradeslist"):
        sym = toks[1].upper()
        try:
            interval = int(toks[2])
        except Exception:
            interval = None
        strat = "_".join(toks[3:])
        return sym, interval, strat
    # Last fallback
    return "UNKNOWN", None, name

def _decode_upload(contents: str, filename: str) -> tuple[bytes, str]:
    """Decode dcc.Upload content string -> raw bytes and display name."""
    _content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    display = os.path.basename(filename) or "uploaded.xlsx"
    return decoded, display

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names we care about (case-insensitive).
    Expected columns (or close variants):
      '#', 'Type', 'Date/Time', 'Signal', 'Price',
      'Shares/Ctrts - Profit/Loss',
      'Net Profit - Cum Net Profit',
      'Run-up/Drawdown', 'Comm.', 'Slippage'
    """
    col_map: dict[str, str] = {}

    def find_col(candidates: Iterable[str], required: bool = True) -> str | None:
        for c in df.columns:
            s = str(c).strip()
            low = s.lower()
            for cand in candidates:
                if low == cand.lower() or cand.lower() in low:
                    col_map[s] = candidates[0]  # map to primary name
                    return s
        if required:
            return None
        return None

    # map pound column
    pound = None
    for c in df.columns:
        s = str(c).strip()
        if s.startswith("#") or s == "#":
            pound = s
            col_map[s] = "#"
            break

    find_col(["Type"])
    find_col(["Date/Time", "date"])
    find_col(["Signal"], required=False)
    find_col(["Price"], required=False)
    find_col(["Shares/Ctrts - Profit/Loss", "Shares", "Ctrts", "Profit/Loss"], required=False)
    # cumulative P&L (many variants)
    find_col(["Net Profit - Cum Net Profit", "Cum Net Profit", "Cumulative Net", "Cum P&L", "Cum Profit"])
    find_col(["Run-up/Drawdown", "Run-up", "Drawdown"], required=False)
    find_col(["Comm.", "Commission"], required=False)
    find_col(["Slippage"], required=False)

    if col_map:
        df = df.rename(columns=col_map)
    return df

def parse_tradestation_trades(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse a TradeStation 'Trades List' Excel into one-row-per-trade with metadata.
    Returns DataFrame columns:
      File, Symbol, Interval, Strategy,
      entry_time, exit_time, entry_type, direction,
      net_profit, runup, drawdown_trade, contracts,
      commission, slippage,
      CumulativePL_raw (for convenience)
    Notes:
      - We pair rows by trade number (#): entry row followed by exit row.
      - Net profit computed robustly from cumulative net differences at exits.
      - Commission & Slippage assumed per-side; totals later multiply by 2.
    """
    bio = io.BytesIO(file_bytes)
    xls = pd.ExcelFile(bio, engine="openpyxl")
    sheet = _load_sheet_guess(xls)
    raw = pd.read_excel(xls, sheet_name=sheet, header=None, engine="openpyxl")

    hdr_idx = _find_header_row(raw)
    if hdr_idx is None:
        hdr_idx = 2  # common case

    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet, header=hdr_idx, engine="openpyxl")
    # Drop repeated header rows, blanks
    if "Type" not in df.columns:
        # try case-insensitive 'Type'
        matches = [c for c in df.columns if isinstance(c, str) and c.lower() == "type"]
        if matches:
            df.rename(columns={matches[0]: "Type"}, inplace=True)

    df = df.dropna(subset=["Type"])
    df = df[df["Type"] != "Type"]

    # Canonicalize column names
    df = _canonicalize_columns(df)

    # Trade number column -> forward fill
    if "#" in df.columns:
        df["TradeNo"] = pd.to_numeric(df["#"], errors="coerce").ffill().astype("Int64")
    else:
        df["TradeNo"] = pd.Series(np.arange(1, len(df) + 1), dtype="Int64")

    # Parse datetimes and numerics
    dtcol = "Date/Time" if "Date/Time" in df.columns else None
    if dtcol:
        df[dtcol] = pd.to_datetime(df[dtcol], errors="coerce")

    for numcol in ["Price",
                   "Shares/Ctrts - Profit/Loss",
                   "Net Profit - Cum Net Profit",
                   "Run-up/Drawdown",
                   "Comm.", "Slippage"]:
        if numcol in df.columns:
            df[numcol] = pd.to_numeric(df[numcol], errors="coerce")

    # Pair entry->exit rows by "TradeNo"
    # Track cumulative net to compute per-trade net P&L robustly
    entry_types = {"Buy", "Sell Short"}
    exit_types = {"Sell", "Buy to Cover"}

    sym, interval, strat = parse_filename_meta(filename)
    trades: list[dict[str, Any]] = []

    last_exit_cum = 0.0
    open_by_no: dict[int, dict[str, Any]] = {}

    for _, row in df.iterrows():
        tno = int(row["TradeNo"]) if not pd.isna(row["TradeNo"]) else None
        rtype = str(row["Type"]).strip()
        ts = row[dtcol] if dtcol else pd.NaT
        cum_val = float(row.get("Net Profit - Cum Net Profit", np.nan))
        if np.isnan(cum_val):
            cum_val = last_exit_cum

        if rtype in entry_types:
            open_by_no[tno] = {
                "entry_time": ts,
                "entry_type": rtype,
                "entry_price": float(row.get("Price", np.nan)) if "Price" in row else np.nan,
                "contracts": float(row.get("Shares/Ctrts - Profit/Loss", np.nan))
                             if "Shares/Ctrts - Profit/Loss" in row else np.nan,
                "runup": float(row.get("Run-up/Drawdown", 0.0)) if "Run-up/Drawdown" in row else 0.0,
            }
        elif rtype in exit_types:
            ent = open_by_no.pop(tno, {
                "entry_time": pd.NaT,
                "entry_type": None,
                "entry_price": np.nan,
                "contracts": np.nan,
                "runup": 0.0,
            })
            netp = float(cum_val) - float(last_exit_cum)
            last_exit_cum = float(cum_val)

            exit_price = float(row.get("Price", np.nan)) if "Price" in row else np.nan
            drawdown_trade = float(row.get("Run-up/Drawdown", 0.0)) if "Run-up/Drawdown" in row else 0.0
            comm = float(row.get("Comm.", 0.0)) if "Comm." in row else 0.0
            slip = float(row.get("Slippage", 0.0)) if "Slippage" in row else 0.0

            direction = "Long" if ent.get("entry_type") == "Buy" else ("Short" if ent.get("entry_type") == "Sell Short" else "Unknown")

            trades.append({
                "File": os.path.basename(filename),
                "Symbol": sym,
                "Interval": interval,
                "Strategy": strat,

                "entry_time": ent.get("entry_time"),
                "exit_time": ts,
                "entry_type": ent.get("entry_type"),
                "direction": direction,

                "entry_price": ent.get("entry_price"),
                "exit_price": exit_price,
                "contracts": ent.get("contracts"),

                "runup": float(ent.get("runup", 0.0)),
                "drawdown_trade": float(drawdown_trade),

                "commission": float(comm),
                "slippage": float(slip),

                "net_profit": float(netp),
                "CumulativePL_raw": float(last_exit_cum),
            })

    # Include any open trades as 0 P&L at last known equity (exit_time=NaT)
    for ent in open_by_no.values():
        trades.append({
            "File": os.path.basename(filename),
            "Symbol": sym,
            "Interval": interval,
            "Strategy": strat,

            "entry_time": ent.get("entry_time"),
            "exit_time": pd.NaT,
            "entry_type": ent.get("entry_type"),
            "direction": ("Long" if ent.get("entry_type") == "Buy"
                          else ("Short" if ent.get("entry_type") == "Sell Short" else "Unknown")),

            "entry_price": ent.get("entry_price"),
            "exit_price": np.nan,
            "contracts": ent.get("contracts"),
            "runup": float(ent.get("runup", 0.0)),
            "drawdown_trade": 0.0,
            "commission": 0.0,
            "slippage": 0.0,
            "net_profit": 0.0,
            "CumulativePL_raw": float(last_exit_cum),
        })

    out = pd.DataFrame(trades)

    # --- dtypes & ordering (do this once) ---
    if not out.empty:
        # datetimes
        if "exit_time" in out.columns:
            out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
        if "entry_time" in out.columns:
            out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")

        # numerics (single pass)
        num_cols = ["net_profit", "runup", "drawdown_trade",
                    "commission", "slippage", "contracts", "CumulativePL_raw"]
        for c in num_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        # categoricals (speeds up filters/groupbys)
        cat_cols = ["Symbol", "Strategy", "direction", "entry_type"]
        for c in cat_cols:
            if c in out.columns:
                out[c] = out[c].astype("category")

        # keep chronological order once
        if "exit_time" in out.columns:
            out.sort_values("exit_time", inplace=True)
            out.reset_index(drop=True, inplace=True)

    return out


# --- Equity helpers (filter-aware) ---

def equity_from_trades_subset(trades_df: pd.DataFrame) -> pd.Series:
    """
    Build cumulative equity from a *filtered subset* of trades using net_profit.
    Index = exit_time (NaT rows dropped). Starts at 0 and cumsum by exit order.
    """
    if trades_df is None or trades_df.empty:
        return pd.Series(dtype=float)
    sub = trades_df.dropna(subset=["exit_time"]).copy()
    sub = sub.sort_values("exit_time")
    eq = sub["net_profit"].cumsum()
    eq.index = sub["exit_time"]
    # Drop duplicate exit times keeping last (as TS can have same timestamp)
    eq = eq[~eq.index.duplicated(keep="last")]
    return eq

def combine_equity(equity_by_file: dict[str, pd.Series], files_selected: list[str]) -> pd.DataFrame:
    """Outer-join per-file equity curves; forward-fill; add 'Portfolio' sum."""
    if not files_selected:
        return pd.DataFrame()
    series_list = []
    for f in files_selected:
        s = equity_by_file.get(f)
        if s is not None and not s.empty:
            series_list.append(s.rename(f))
    if not series_list:
        return pd.DataFrame()
    eq = pd.concat(series_list, axis=1).sort_index().ffill()
    eq["Portfolio"] = eq.sum(axis=1)
    return eq

def max_drawdown_series(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    return equity - equity.cummax()

def max_drawdown_value(equity: pd.Series) -> float:
    dd = max_drawdown_series(equity)
    return float(dd.min()) if not dd.empty else 0.0

def intraday_drawdown_series(equity: pd.Series) -> pd.Series:
    """
    For each day: worst (most negative) peak-to-trough decline within that day.
    Returns a daily Series indexed by date with negative values (drawdowns).
    """
    if equity is None or equity.empty:
        return pd.Series(dtype=float)
    s = equity.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    out = {}
    for d, g in s.groupby(s.index.date):
        run_max = g.cummax()
        dd = g - run_max
        out[d] = dd.min() if not dd.empty else 0.0
    return pd.Series(out, dtype=float)

# === NEW: build per-file P/L series and a correlation heatmap ===
def _compute_pl_dataframe_from_equity(equity_by_file: dict, method: str = "diff"):
    """
    equity_by_file: dict[str, pd.Series] where each Series is an equity curve indexed by datetime.
    method: "diff" -> per-period P/L in dollars; "pct" -> per-period returns.
    """
    dfs = []
    for fname, eq in (equity_by_file or {}).items():
        if eq is None or len(eq) == 0:
            continue
        s = eq.sort_index()
        pl = s.diff().dropna() if method == "diff" else s.pct_change().dropna()
        pl.name = fname
        dfs.append(pl)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, axis=1).fillna(0.0)
    # Drop fully-constant columns (corr would be NaN)
    return df.loc[:, df.apply(lambda c: c.std(skipna=True) > 0)]

def build_correlation_heatmap(
    equity_by_file: dict,
    label_map: dict | None = None,
    mode: str = "drawdown_pct",       # 'drawdown_pct' | 'returns_z' | 'pl' | 'slope'
    slope_window: int = 20,
    method: str = "pearson"           # or 'spearman'
):
    """
    Build a correlation heatmap focusing on curve shape.
    - drawdown_pct: correlate daily drawdown % states (best for “are they in DD together?”)
    - returns_z:    correlate z-scored daily returns (scale-free co-movement)
    - pl:           correlate daily $ P/L (sum of per-exit P/L per day) — your old behavior
    - slope:        correlate rolling equity slope over 'slope_window' days
    """


    if not equity_by_file:
        return go.Figure()

    # 1) transform each series to the chosen representation
    frames = []
    for fname, eq in equity_by_file.items():
        if eq is None or eq.empty:
            continue
        eqd = _to_daily_equity(eq)
        if mode == "drawdown_pct":
            s = _drawdown_pct(eqd).dropna()
        elif mode == "returns_z":
            ret = eqd.pct_change().dropna()
            s = _zscore(ret)
        elif mode == "pl":
            # daily P/L ($): diff then sum by day
            s = eq.diff().resample("1D").sum().dropna()
        elif mode == "slope":
            slope = _rolling_slope(eqd, window=int(slope_window or 20)).dropna()
            # Optional: z-score slope so units are comparable
            s = _zscore(slope)
        else:
            s = _drawdown_pct(eqd).dropna()
        s.name = fname
        frames.append(s)

    if not frames:
        fig = go.Figure()
        fig.update_layout(
            title="No data for correlation",
            margin=dict(l=40, r=20, t=60, b=40)
        )
        return fig

    # 2) inner-join so we only compare overlapping days
    df = pd.concat(frames, axis=1, join="inner").dropna(how="any")
    if label_map:
        df = df.rename(columns=label_map)

    # Drop constant columns to avoid NaNs
    df = df.loc[:, df.apply(lambda c: c.std(ddof=0) > 0)]

    if df.shape[1] < 2:
        fig = go.Figure()
        fig.update_layout(
            title="Need at least two non-constant series for correlation",
            margin=dict(l=40, r=20, t=60, b=40)
        )
        return fig

    # 3) correlation
    corr = df.corr(method=method)

    # 4) plot
    fig = px.imshow(
        corr, aspect="equal", zmin=-1, zmax=1, color_continuous_scale="RdBu"
    )
    # readable text
    fig.update_traces(
        text=corr.round(2).astype(str).values,
        texttemplate="%{text}",
        hovertemplate="x:%{x}<br>y:%{y}<br>corr:%{z:.2f}<extra></extra>"
    )
    title_map = {
        "drawdown_pct": "Correlation of Daily Drawdown %",
        "returns_z": "Correlation of Z-scored Daily Returns",
        "pl": "Correlation of Daily $ P/L",
        "slope": f"Correlation of Rolling Equity Slope (window={slope_window})",
    }
    fig.update_layout(
        title=title_map.get(mode, "Correlation"),
        xaxis_title="File / Strategy",
        yaxis_title="File / Strategy",
        margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig


# === NEW: correlation transforms focused on curve shape ===
def _to_daily_equity(s: pd.Series) -> pd.Series:
    """Daily equity: last value per day, forward filled."""
    if s is None or s.empty:
        return pd.Series(dtype=float)
    return s.resample("1D").last().ffill()

def _drawdown_pct(s: pd.Series) -> pd.Series:
    """DD% = equity / rolling max - 1 (≤ 0)."""
    if s is None or s.empty:
        return pd.Series(dtype=float)
    rm = s.cummax()
    out = (s / rm) - 1.0
    return out

def _zscore(x: pd.Series) -> pd.Series:
    m = x.mean()
    sd = x.std(ddof=0)
    return (x - m) / sd if sd and np.isfinite(sd) and sd > 0 else x * 0.0

def _rolling_slope(y: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling OLS slope of equity vs. time index (0..window-1).
    Units are 'equity units per day'; comparable after per-series standardization.
    """
    if len(y) < window:
        return pd.Series(dtype=float)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(win: np.ndarray) -> float:
        yv = win.astype(float)
        y_mean = yv.mean()
        cov = ((x - x_mean) * (yv - y_mean)).sum()
        return cov / x_var if x_var != 0 else 0.0

    return y.rolling(window, min_periods=window).apply(_slope, raw=True)



# ------------------------------ Metrics --------------------------------

def max_consecutive_runs(pnl: Iterable[float]) -> tuple[int, int]:
    max_w = max_l = cur_w = cur_l = 0
    for x in pnl:
        if x > 0:
            cur_w += 1; cur_l = 0
        elif x < 0:
            cur_l += 1; cur_w = 0
        else:
            cur_w = 0; cur_l = 0
        max_w = max(max_w, cur_w); max_l = max(max_l, cur_l)
    return max_w, max_l

def compute_max_held(sub: pd.DataFrame) -> tuple[int, int]:
    """
    Sweep line over entry/exit to compute:
      - Max simultaneous contracts held
      - Total contracts held (sum abs(entry contracts))
    """
    events: list[tuple[pd.Timestamp, int, int]] = []
    for _, r in sub.iterrows():
        qty = int(r.get("contracts") or 0)
        if qty == 0 or pd.isna(r.get("entry_time")) or pd.isna(r.get("exit_time")):
            continue
        # Sort exits first when equal timestamps
        events.append((r["entry_time"], 1, qty))
        events.append((r["exit_time"], 0, -qty))
    if not events:
        return 0, int(sub.get("contracts", pd.Series(dtype=float)).fillna(0).abs().sum())

    events.sort(key=lambda x: (x[0], x[1]))
    cur = 0
    max_held = 0
    for _, _, delta in events:
        cur += delta
        max_held = max(max_held, abs(cur))
    total_held = int(sub.get("contracts", pd.Series(dtype=float)).fillna(0).abs().sum())
    return int(max_held), int(total_held)

def compute_closed_trade_dd(series: pd.Series) -> tuple[float, pd.Timestamp | None]:
    """
    From closed-trade equity series → (max DD value (positive magnitude), date at DD).
    We store positive magnitude for internal use; display as negative in table.
    """
    if series is None or series.empty:
        return 0.0, None
    eq = series.cumsum() if series.index.dtype != "datetime64[ns]" else series  # (safety)
    peak = 0.0
    max_dd = 0.0
    max_time = None
    for t, v in zip(series.index, series):
        cum = v
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
            max_time = t
    return float(max_dd), max_time

def compute_intraday_p2v(trades: pd.DataFrame) -> tuple[float, pd.Timestamp | None]:
    """
    Intra-trade (run-up/drawdown) peak→valley DD following integration spec.
    Returns positive magnitude, and a timestamp (exit_time) at occurrence.
    """
    if trades is None or trades.empty:
        return 0.0, None
    sub = trades.sort_values("exit_time").reset_index(drop=True)
    base = 0.0
    hwm = 0.0
    max_dd = 0.0
    max_dd_time = None
    for _, r in sub.iterrows():
        runup = float(r.get("runup") or 0.0)
        drawd = float(r.get("drawdown_trade") or 0.0)
        pnl = float(r.get("net_profit") or 0.0)
        peak_cand = base + max(0.0, runup, pnl, 0.0)
        hwm = max(hwm, base, peak_cand)
        valley_cand = base + min(0.0, drawd, pnl, 0.0)
        dd = hwm - valley_cand
        if dd > max_dd:
            max_dd = dd
            max_dd_time = r.get("exit_time")
        base += pnl
    return float(max_dd), max_dd_time

def _union_intervals_coverage(sub: pd.DataFrame) -> tuple[float, float, float]:
    """
    Union of [entry_time, exit_time] intervals:
      returns (coverage_seconds, period_seconds, longest_gap_seconds).
    """
    df = sub[["entry_time", "exit_time"]].dropna().sort_values("entry_time")
    if df.empty:
        return 0.0, 0.0, 0.0
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur_s = df.iloc[0, 0]; cur_e = df.iloc[0, 1]
    for s, e in df.iloc[1:].itertuples(index=False, name=None):
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    coverage = sum((e - s).total_seconds() for s, e in merged)
    period = (df["exit_time"].max() - df["entry_time"].min()).total_seconds()
    longest_gap = 0.0
    for (s1, e1), (s2, e2) in zip(merged, merged[1:]):
        gap = (s2 - e1).total_seconds()
        if gap > longest_gap:
            longest_gap = gap
    return float(coverage), float(period), float(longest_gap)

def _full_month_bounds(series_dt: pd.Series) -> tuple[pd.Period | None, pd.Period | None]:
    if series_dt.empty:
        return None, None
    first = series_dt.min()
    last = series_dt.max()
    first_full = (first + pd.offsets.MonthBegin(1)).normalize()
    last_full_start = (last.replace(day=1)) - pd.offsets.MonthBegin(1)
    return first_full.to_period("M"), last_full_start.to_period("M")

def monthly_stats_full_months(sub: pd.DataFrame) -> tuple[float, float, pd.Series]:
    """
    Exit-based monthly attribution; include only full calendar months.
    Returns (avg, std, monthly_series).
    """
    if sub is None or sub.empty:
        return 0.0, 0.0, pd.Series(dtype=float)
    s = sub.dropna(subset=["exit_time"]).copy()
    if s.empty:
        return 0.0, 0.0, pd.Series(dtype=float)
    monthly = s.groupby(pd.Grouper(key="exit_time", freq="M"))["net_profit"].sum()
    months = monthly.index.to_period("M")
    first_full, last_full = _full_month_bounds(s["exit_time"])
    if not first_full or not last_full or first_full > last_full:
        return 0.0, 0.0, pd.Series(dtype=float)
    mask = (months >= first_full) & (months <= last_full)
    monthly_full = monthly[mask]
    if monthly_full.empty:
        return 0.0, 0.0, pd.Series(dtype=float)
    return float(monthly_full.mean()), float(monthly_full.std(ddof=0)), monthly_full

def _fmt_td(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    days = td.days
    years = days // 365
    months = (days % 365) // 30
    rem_days = (days % 365) % 30
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    return f"{years} Years, {months} Months, {rem_days} days, {hours} hours, {minutes} Minutes"

def compute_metrics(sub: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    """
    Compute P&L + risk metrics for a filtered subset of trades.
    Includes:
      - Standard P&L stats
      - Commission/Slippage totals (double per trade)
      - Close→close DD (value & date) and Account Size Required (=abs DD)
      - Intra-day P2V DD (value & date) from run-up/drawdown
      - Max/Total contracts held
      - Portfolio-level stats (ROI, Annual ROR, ROA, monthly stats, time-in-market)
    """
    sub2 = sub.dropna(subset=["exit_time"]).copy()
    sub2 = sub2.sort_values("exit_time")
    pnl = sub2["net_profit"].astype(float)
    n = len(sub2)

    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    gp = float(pnl[pnl > 0].sum()) if wins else 0.0
    gl = float(pnl[pnl < 0].sum()) if losses else 0.0
    net = float(gp + gl)
    pf = (gp / abs(gl)) if gl < 0 else (np.inf if gp > 0 else 0.0)
    avg_trade = float(net / n) if n else 0.0
    avg_win = float(pnl[pnl > 0].mean()) if wins else 0.0
    avg_loss = float(pnl[pnl < 0].mean()) if losses else 0.0
    ratio_aw_al = (avg_win / abs(avg_loss)) if (losses and avg_loss) else np.nan
    lw = float(pnl.max()) if n else 0.0
    ll = float(pnl.min()) if n else 0.0
    max_w, max_l = max_consecutive_runs(pnl.tolist() if n else [])

    # Equity and DD (close→close)
    eq = pnl.cumsum()
    eq.index = sub2["exit_time"]
    close_dd_val_pos, close_dd_time = compute_closed_trade_dd(eq)

    # Intra-day (run-up/drawdown) P2V DD
    intra_dd_val_pos, intra_dd_time = compute_intraday_p2v(sub2)

    # Contracts held
    max_held, total_held = compute_max_held(sub2)

    # Time in market
    cover, period, longest_gap = _union_intervals_coverage(sub.dropna(subset=["entry_time", "exit_time"]))
    pct_time = (cover / period * 100.0) if period > 0 else 0.0

    # Costs
    total_comm = float(sub2.get("commission", pd.Series(dtype=float)).sum() * 2.0)
    total_slip = float(sub2.get("slippage", pd.Series(dtype=float)).sum() * 2.0)

    # Portfolio-level aggregates
    roi = (net / initial_capital * 100.0) if initial_capital > 0 else np.nan
    years = (period / (365.2425 * 24 * 3600)) if period > 0 else 0.0
    ann_ror = (((initial_capital + net) / initial_capital) ** (1 / years) - 1) * 100.0 if years > 0 else np.nan
    roa = (net / close_dd_val_pos) * 100.0 if close_dd_val_pos > 0 else np.nan
    avg_m, std_m, monthly_full = monthly_stats_full_months(sub2)

    return {
        # Core P&L
        "Total Net Profit": net,
        "Gross Profit": gp,
        "Gross Loss": gl,
        "Profit Factor": pf,
        "Total Number of Trades": int(n),
        "Percent Profitable": (wins / n * 100.0) if n else 0.0,
        "Winning Trades": wins,
        "Losing Trades": losses,
        "Avg. Trade Net Profit": avg_trade,
        "Avg. Winning Trade": avg_win,
        "Avg. Losing Trade": avg_loss,
        "Ratio Avg. Win:Avg. Loss": ratio_aw_al,
        "Largest Winning Trade": lw,
        "Largest Losing Trade": ll,
        "Max. Consecutive Winning Trades": int(max_w),
        "Max. Consecutive Losing Trades": int(max_l),

        # Contracts
        "Max. Shares/Contracts Held": int(max_held),
        "Total Shares/Contracts Held": int(total_held),

        # Risk & Costs
        "Account Size Required": float(close_dd_val_pos),
        "Total Commission": total_comm,
        "Total Slippage": total_slip,
        "Max. Drawdown (Trade Close to Trade Close) Value": -float(close_dd_val_pos),
        "Max. Drawdown (Trade Close to Trade Close) Date": close_dd_time,
        "Max. Trade Drawdown": float(sub2.get("drawdown_trade", pd.Series([0])).min()),
        "Max. Drawdown (Intra-Day Peak to Valley) Value": -float(intra_dd_val_pos),
        "Max. Drawdown (Intra-Day Peak to Valley) Date": intra_dd_time,

        # Portfolio-level (reported on all rows; interpret primarily for Portfolio)
        "Return on Initial Capital": roi,
        "Annual Rate of Return": ann_ror,
        "Return on Account": roa,
        "Avg. Monthly Return (full months)": avg_m,
        "Std. Deviation of Monthly Return (full months)": std_m,
        "Trading Period": _fmt_td(period),
        "Percent of Time in the Market": pct_time,
        "Time in the Market": _fmt_td(cover),
        "Longest Flat Period": _fmt_td(longest_gap),
        "Max. Equity Run-up": float((eq - eq.cummin()).max()) if not eq.empty else 0.0,
        "Max Equity Run-up as % of Initial Capital": (float((eq - eq.cummin()).max()) / initial_capital * 100.0) if initial_capital > 0 else np.nan,

        # For optional auditing
        "_Monthly Full Series": monthly_full,
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
    ]
)

# ------------------------------ Helpers --------------------------------


def _safe_unique_name(name: str, existing: set[str]) -> str:
    base = name
    n = 2
    out = name
    while out in existing:
        out = f"{base} ({n})"
        n += 1
    return out

def _within_dates(df: pd.DataFrame, start_d: date | None, end_d: date | None, col="exit_time") -> pd.DataFrame:
    if start_d is None and end_d is None:
        return df
    out = df.copy()
    if start_d is not None:
        out = out[out[col] >= pd.Timestamp(start_d)]
    if end_d is not None:
        out = out[out[col] <= pd.Timestamp(end_d) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)]
    return out

def _display_label_from_df(tdf: pd.DataFrame, fallback: str) -> str:
    """
    Build 'Strategy-SYMBOL-INTERVAL-min' from the first non-null row of tdf.
    Falls back gracefully if any field is missing.
    Ensures SYMBOL uppercased and INTERVAL is int-like.
    """
    if tdf is None or tdf.empty:
        return fallback
    strat = str(tdf["Strategy"].dropna().astype(str).iloc[0]) if "Strategy" in tdf and tdf["Strategy"].notna().any() else None
    sym   = str(tdf["Symbol"].dropna().astype(str).iloc[0])   if "Symbol"   in tdf and tdf["Symbol"].notna().any()   else None
    inter = tdf["Interval"].dropna().iloc[0] if "Interval" in tdf and tdf["Interval"].notna().any() else None
    try:
        inter = int(inter) if inter is not None else None
    except Exception:
        inter = None

    parts = []
    if strat: parts.append(strat)
    if sym:   parts.append(sym.upper())
    if inter is not None: parts.append(f"{inter}-min")

    label = "-".join(parts) if parts else fallback
    return label


def build_net_contracts_figure(store_trades: dict,
                               selected_files: list[str],
                               selected_symbols: list[str],
                               selected_intervals: list[int],
                               selected_strategies: list[str],
                               direction: str,
                               start_date,
                               end_date,
                               contracts_map: dict | None = None):

    """
    Net Contracts tab with globally aligned timestamps:
      • Per-file lines (thin)
      • Per-symbol lines (dashed, medium)
      • Portfolio line (thick black)
    Y = |net contracts|; hover shows direction and signed value.
    Sorting: by timestamp with EXITS (kind_rank=0) before ENTRIES (1) at identical times.
    """


    empty_fig = go.Figure().update_layout(margin=dict(l=10, r=10, t=30, b=10))
    if not store_trades or not selected_files:
        return empty_fig

    # ---------- Filter trades ----------
    def base_filter(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
        out["exit_time"]  = pd.to_datetime(out["exit_time"],  errors="coerce")
        if selected_symbols:
            out = out[out["Symbol"].isin(selected_symbols)]
        if selected_intervals:
            out = out[out["Interval"].isin(selected_intervals)]
        if selected_strategies:
            out = out[out["Strategy"].isin(selected_strategies)]
        if direction in ("Long", "Short"):
            out = out[out["direction"] == direction]
        out["contracts"] = pd.to_numeric(out["contracts"], errors="coerce").fillna(0.0).abs()
        return out

    trades_by_file = {}
    for f in selected_files:
        recs = store_trades.get(f, [])
        if not recs:
            continue
        tdf = base_filter(pd.DataFrame(recs))
        if not tdf.empty:
            tdf["File"] = f
            trades_by_file[f] = tdf

    if not trades_by_file:
        return empty_fig

    # Pretty labels for per-file lines
    per_file_label_map: dict[str, str] = {}
    used_labels: set[str] = set()
    for f, tdf in trades_by_file.items():
        base = _display_label_from_df(tdf, fallback=f)
        pretty = _safe_unique_name(base, used_labels)
        used_labels.add(pretty)
        per_file_label_map[f] = pretty



    # ---------- Build ENTRY/EXIT events with sign ----------
    ev_rows = []
    for f, df in trades_by_file.items():
        for _, r in df.iterrows():
            qty = float(r.get("contracts") or 0.0)
            # apply file-level multiplier (affects net position size)
            if contracts_map:
                qty *= float(contracts_map.get(f, 1.0))
            if qty == 0:
                continue
            sym = str(r.get("Symbol"))
            dirn = str(r.get("direction") or "")
            ent, exi = r.get("entry_time"), r.get("exit_time")

            # ENTRY
            if pd.notna(ent):
                delta = qty if dirn == "Long" else (-qty if dirn == "Short" else 0.0)
                ev_rows.append({"ts": ent, "delta": delta, "File": f, "Symbol": sym, "kind_rank": 1})
            # EXIT (reverse)
            if pd.notna(exi):
                delta = -qty if dirn == "Long" else (+qty if dirn == "Short" else 0.0)
                ev_rows.append({"ts": exi, "delta": delta, "File": f, "Symbol": sym, "kind_rank": 0})

    if not ev_rows:
        return empty_fig

    ev = pd.DataFrame(ev_rows).dropna(subset=["ts"])
    ev["ts"] = pd.to_datetime(ev["ts"], errors="coerce")
    ev = ev.dropna(subset=["ts"])
    ev = ev.sort_values(["ts", "kind_rank"])  # exits then entries at tie

    # ---------- Date window ----------
    ts_min, ts_max = ev["ts"].min(), ev["ts"].max()
    start_dt = pd.to_datetime(start_date) if start_date else ts_min
    end_dt   = pd.to_datetime(end_date)   if end_date   else ts_max
    if pd.isna(start_dt): start_dt = ts_min
    if pd.isna(end_dt):   end_dt   = ts_max
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    # Global timeline = union of all event times in-window + [start,end]  (strictly datetimes)
    inwin_ts = ev.loc[(ev["ts"] >= start_dt) & (ev["ts"] <= end_dt), "ts"]
    # Build a clean DatetimeIndex and sort
    times_all_idx = pd.DatetimeIndex(inwin_ts).append(pd.DatetimeIndex([start_dt, end_dt])).dropna().unique().sort_values()
    # Convert to plain list for plotting (but keep as timestamps only)
    times_all = list(times_all_idx.to_pydatetime())

    def make_series(group_ev: pd.DataFrame) -> pd.DataFrame:
        """
        Returns DataFrame with aligned timestamps:
          ts, y_signed, y_abs, dir
        Using global times_all so tooltips align across traces.
        """
        # Work on a local copy of the global times to avoid accidental mutation
        times = times_all

        if group_ev.empty:
            ser = pd.DataFrame({"ts": times, "y_signed": 0.0})
        else:
            g = group_ev.copy()
            g["ts"] = pd.to_datetime(g["ts"], errors="coerce")
            g = g.dropna(subset=["ts"])

            # Carry-in from events before start
            carry0 = float(g.loc[g["ts"] < start_dt, "delta"].sum())

            # deltas at exact timestamps (in-window), as a dict keyed by Timestamp
            deltas = g.loc[(g["ts"] >= start_dt) & (g["ts"] <= end_dt)].groupby("ts")["delta"].sum()

            # Accumulate along the global timeline
            acc = carry0
            pts = [{"ts": times[0], "y_signed": acc}]  # times[0] is start_dt
            for t in times[1:]:
                acc += float(deltas.get(pd.Timestamp(t), 0.0))
                pts.append({"ts": t, "y_signed": acc})

            ser = pd.DataFrame(pts).sort_values("ts")

        ser["y_abs"] = ser["y_signed"].abs()
        ser["dir"] = np.where(ser["y_signed"] > 0, "long",
                              np.where(ser["y_signed"] < 0, "short", "flat"))
        return ser

    # Per-file series
    per_file_series = {}
    for f in trades_by_file.keys():
        g = ev[ev["File"] == f][["ts", "delta", "kind_rank"]]
        per_file_series[f] = make_series(g)

    # Per-symbol series (sum across files)
    per_symbol_series = {}
    for sym, g in ev.groupby("Symbol", dropna=True):
        per_symbol_series[str(sym)] = make_series(g[["ts", "delta", "kind_rank"]])

    # Portfolio series (SUM over symbols of ABS(net)), i.e., no cross-symbol cancellation
    # Assumption: make_series() for each symbol used the same global 'times_all'
    # Build a portfolio DataFrame with the shared timeline
    # Portfolio series (Σ symbols |net|), with proper timestamp alignment
    if per_symbol_series:
        # Use a DatetimeIndex for alignment
        portfolio_index = pd.DatetimeIndex(times_all)
        port_df = pd.DataFrame(index=portfolio_index)
        port_df["y_abs"] = 0.0

        for sym, ser in per_symbol_series.items():
            aligned = ser.set_index("ts")["y_signed"].reindex(port_df.index).ffill().fillna(0.0)
            port_df["y_abs"] = port_df["y_abs"] + aligned.abs()

        portfolio_series = port_df.reset_index().rename(columns={"index": "ts"})
        portfolio_series["y_signed"] = portfolio_series["y_abs"]
        portfolio_series["dir"] = "n/a"
    else:
        portfolio_series = pd.DataFrame({"ts": list(times_all), "y_abs": 0.0, "y_signed": 0.0, "dir": "flat"})



    # ---------- Plot ----------
    fig = go.Figure()

    def add_curve(name, ser, width=2, dash=None, color=None):
        if ser is None or ser.empty:
            return
        customdata = np.column_stack([ser["dir"].astype(str), ser["y_signed"].astype(float)])
        fig.add_trace(go.Scatter(
            x=ser["ts"], y=ser["y_abs"], mode="lines", name=name,
            line=dict(width=width, dash=dash, color=color),
            line_shape="hv",
            customdata=customdata,
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>%{fullData.name}"
                          "<br>Net: %{y:.0f} (%{customdata[0]})"
                          "<br>Signed: %{customdata[1]:.0f}"
                          "<extra></extra>"
        ))

    # per-file lines (thin)
    for f, ser in sorted(per_file_series.items()):
        disp = per_file_label_map.get(f, f)
        add_curve(disp, ser, width=1)


    # per-symbol lines (dashed, medium)
    for sym, ser in sorted(per_symbol_series.items()):
        add_curve(f"Symbol: {sym}", ser, width=2, dash="dash")

    # portfolio line (thick black)
    add_curve("Portfolio (Net)", portfolio_series, width=3, color="black")

    fig.add_hline(y=0, line_width=1, line_dash="dot", opacity=0.6)

    fig.update_layout(
        title="Net Contracts — Per Symbol: |long−short|, Portfolio: Σ symbols |net| (no cross-symbol cancel)",
        xaxis_title="Date/Time",
        yaxis_title="|Net Contracts|",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(dtick=1)
    )
    return fig

def _netpos_timeseries_from_store(store_trades: dict,
                                  selected_files: list[str],
                                  selected_symbols: list[str],
                                  selected_intervals: list[int],
                                  selected_strategies: list[str],
                                  direction: str,
                                  start_date,
                                  end_date,
                                  contracts_map: dict | None = None) -> dict[str, pd.Series]:

    """
    Returns {file -> signed net contracts time-series} aligned on a shared DatetimeIndex.
    Mirrors the event construction used by build_net_contracts_figure, but returns data only.
    """

    if not store_trades or not selected_files:
        return {}

    # Filter per-file, similar to build_net_contracts_figure
    def base_filter(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
        out["exit_time"]  = pd.to_datetime(out["exit_time"],  errors="coerce")
        if selected_symbols:
            out = out[out["Symbol"].isin(selected_symbols)]
        if selected_intervals:
            out = out[out["Interval"].isin(selected_intervals)]
        if selected_strategies:
            out = out[out["Strategy"].isin(selected_strategies)]
        if direction in ("Long", "Short"):
            out = out[out["direction"] == direction]
        out["contracts"] = pd.to_numeric(out["contracts"], errors="coerce").fillna(0.0)
        return out

    trades_by_file = {}
    for f in selected_files:
        recs = store_trades.get(f, [])
        if not recs:
            continue
        tdf = base_filter(pd.DataFrame(recs))
        if not tdf.empty:
            tdf["File"] = f
            trades_by_file[f] = tdf

    if not trades_by_file:
        return {}

    # Build events
    ev_rows = []
    for f, df in trades_by_file.items():
        for _, r in df.iterrows():
            qty = float(r.get("contracts") or 0.0)
            if contracts_map:
                qty *= float(contracts_map.get(f, 1.0))
            if qty == 0:
                continue
            ent, exi = r.get("entry_time"), r.get("exit_time")
            dirn = str(r.get("direction") or "")
            sym  = str(r.get("Symbol") or "")
            # ENTRY
            if pd.notna(ent):
                delta = qty if dirn == "Long" else (-qty if dirn == "Short" else 0.0)
                ev_rows.append({"ts": ent, "delta": delta, "File": f, "Symbol": sym, "kind_rank": 1})
            # EXIT
            if pd.notna(exi):
                delta = -qty if dirn == "Long" else (+qty if dirn == "Short" else 0.0)
                ev_rows.append({"ts": exi, "delta": delta, "File": f, "Symbol": sym, "kind_rank": 0})

    ev = pd.DataFrame(ev_rows)
    if ev.empty:
        return {}

    ev["ts"] = pd.to_datetime(ev["ts"], errors="coerce")
    ev = ev.dropna(subset=["ts"]).sort_values(["ts", "kind_rank"])

    # Date window
    ts_min, ts_max = ev["ts"].min(), ev["ts"].max()
    start_dt = pd.to_datetime(start_date) if start_date else ts_min
    end_dt   = pd.to_datetime(end_date)   if end_date   else ts_max
    if pd.isna(start_dt): start_dt = ts_min
    if pd.isna(end_dt):   end_dt   = ts_max
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    # Global timeline
    idx = (pd.DatetimeIndex(ev.loc[(ev["ts"] >= start_dt) & (ev["ts"] <= end_dt), "ts"])
           .append(pd.DatetimeIndex([start_dt, end_dt])).dropna().unique().sort_values())

    # Build per-file cumulative signed net contracts
    out = {}
    for f, g in ev.groupby("File"):
        g = g[(g["ts"] >= start_dt) & (g["ts"] <= end_dt)]
        g = g.groupby("ts")["delta"].sum().reindex(idx, fill_value=0.0)
        out[f] = g.cumsum()
    return out


def _symbol_from_first_row(store_trades: dict, fname: str) -> str:
    """Pull the Symbol for a file from the first row (used to map to MARGIN_SPEC)."""
    recs = store_trades.get(fname, [])
    if not recs:
        return ""
    sym = pd.DataFrame(recs).get("Symbol")
    return str(sym.dropna().astype(str).iloc[0]) if sym is not None and sym.notna().any() else ""

# --- NEW: Net position aggregation per symbol (sum signed across files, then take abs later) ---
def _aggregate_netpos_per_symbol(per_file_netpos: dict[str, pd.Series],
                                 store_trades: dict) -> dict[str, pd.Series]:
    """
    Convert {file -> signed net contracts series} to {SYMBOL -> signed net contracts series},
    where each symbol series sums signed nets across all files of that symbol on a shared index.
    """

    if not per_file_netpos:
        return {}

    # Union index across all files
    idx = None
    for s in per_file_netpos.values():
        idx = s.index if idx is None else idx.union(s.index)
    idx = idx.sort_values()

    by_sym: dict[str, pd.Series] = {}
    for fname, ser in per_file_netpos.items():
        sym = _symbol_from_first_row(store_trades, fname).upper()
        if not sym:
            continue
        aligned = ser.reindex(idx).ffill().fillna(0.0)
        by_sym[sym] = (by_sym.get(sym, pd.Series(0.0, index=idx))).add(aligned, fill_value=0.0)
    return by_sym


# --- NEW: Purchasing Power series ---
def _purchasing_power_series(portfolio_equity: pd.Series,
                             init_margin_total: pd.Series,
                             starting_balance: float) -> pd.Series:
    """
    purchasing_power(t) = starting_balance + portfolio_equity(t) - init_margin_total(t)
    Aligns indices with forward-fill to keep curves continuous.
    """
    # If both are missing/empty, nothing to plot
    if (portfolio_equity is None or portfolio_equity.empty) and \
       (init_margin_total is None or init_margin_total.empty):
        return pd.Series(dtype=float)

    # Build the union time index
    idx = None
    if portfolio_equity is not None and not portfolio_equity.empty:
        idx = portfolio_equity.index
    if init_margin_total is not None and not init_margin_total.empty:
        idx = init_margin_total.index if idx is None else idx.union(init_margin_total.index)

    idx = pd.DatetimeIndex(idx).sort_values()

    # Use explicit emptiness checks instead of boolean `or` on Series
    eq_src = portfolio_equity if (portfolio_equity is not None and not portfolio_equity.empty) \
                               else pd.Series(0.0, index=idx)
    mr_src = init_margin_total if (init_margin_total is not None and not init_margin_total.empty) \
                                else pd.Series(0.0, index=idx)

    eq = eq_src.reindex(idx).ffill().fillna(0.0)
    mr = mr_src.reindex(idx).ffill().fillna(0.0)

    # Return as a Series aligned on idx
    return pd.Series(starting_balance, index=idx) + eq - mr




def _margin_series(per_file_netpos: dict[str, pd.Series],
                   store_trades: dict,
                   margin_spec: dict[str, tuple[float, float, float]]) -> pd.Series:
    """
    Convert per-file signed net contracts into a total **initial margin requirement** series.
    Sum across files using each file's Symbol → margin_spec.
    """

    if not per_file_netpos:
        return pd.Series(dtype=float)

    # Union index
    all_idx = None
    for s in per_file_netpos.values():
        all_idx = s.index if all_idx is None else all_idx.union(s.index)
    all_idx = all_idx.sort_values()

    total = pd.Series(0.0, index=all_idx)
    for fname, ser in per_file_netpos.items():
        sym = _symbol_from_first_row(store_trades, fname)
        spec = margin_spec.get(sym.upper())
        if spec is None:
            # unknown symbol → assume zero to avoid false positives; better: raise or warn
            continue
        init_margin = float(spec[0])
        ser_re = ser.reindex(all_idx).ffill().fillna(0.0).abs()  # contracts count
        total = total.add(ser_re * init_margin, fill_value=0.0)
    return total

def _portfolio_equity_from_weights(equity_by_file: dict[str, pd.Series],
                                   weights: dict[str, float]) -> pd.Series:
    """Weighted sum of per-file equity curves (curves must share the same index union; we ffill gaps)."""

    if not equity_by_file or not weights:
        return pd.Series(dtype=float)

    # Union index
    idx = None
    for s in equity_by_file.values():
        idx = s.index if idx is None else idx.union(s.index)
    idx = idx.sort_values()

    port = pd.Series(0.0, index=idx)
    for f, w in weights.items():
        s = equity_by_file.get(f)
        if s is None or s.empty or w == 0:
            continue
        port = port.add(s.reindex(idx).ffill().fillna(0.0) * float(w), fill_value=0.0)
    return port


def _peak_drawdown_value(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    run_max = series.cummax()
    dd = series - run_max
    return float(dd.min())


def _annualized_return(series: pd.Series, initial_capital: float) -> float:
    """Very simple annualization from first->last timestamp."""
    if series is None or series.empty or initial_capital <= 0:
        return float("nan")
    start = series.index.min()
    end   = series.index.max()
    if pd.isna(start) or pd.isna(end) or end <= start:
        return float("nan")
    years = (end - start).total_seconds() / (365.2425 * 24 * 3600)
    final_equity = initial_capital + float(series.iloc[-1])
    return (final_equity / initial_capital) ** (1/years) - 1.0


def find_margin_aware_weights(equity_by_file: dict[str, pd.Series],
                              per_file_netpos: dict[str, pd.Series],
                              store_trades: dict,
                              initial_capital: float,
                              margin_spec: dict[str, tuple[float, float, float]],
                              margin_buffer: float = 0.60,
                              objective: str = "max_return_over_dd",
                              step: float = 0.25,
                              leverage_cap: float = 1.0) -> tuple[dict[str, float], dict]:
    """
    Brute-force grid search over non-negative weights (0..leverage_cap) in increments of `step`,
    with sum(weights) <= leverage_cap. Enforces margin constraint:
        max(total_margin_series) <= margin_buffer * initial_capital

    Objective options:
      - "max_return_over_dd": maximize (AnnRet / abs(DD))
      - "risk_parity": set weights ∝ 1/vol (then project to constraints)

    Returns (best_weights, diagnostics)
    """


    files = list(equity_by_file.keys())
    if not files:
        return {}, {"reason": "no files"}

    # Risk-parity shortcut
    if objective == "risk_parity":
        vols = []
        for f in files:
            s = equity_by_file[f].diff().dropna()
            vols.append(s.std() if not s.empty else np.nan)
        inv = np.array([0 if (np.isnan(v) or v == 0) else 1.0/v for v in vols], dtype=float)
        if inv.sum() == 0:
            base_w = np.ones(len(files)) / len(files)
        else:
            base_w = inv / inv.sum()
        # Project to leverage cap
        w = base_w * leverage_cap
        weights = {f: float(wi) for f, wi in zip(files, w)}
        # Check margin; if fail, shrink uniformly until pass or zero
        shrink = 1.0
        while shrink > 0:
            test_w = {f: wi * shrink for f, wi in weights.items()}
            port = _portfolio_equity_from_weights(equity_by_file, test_w)
            # Margin check
            # scale netpos by weights, sum margins
            idx_union = None
            for ser in per_file_netpos.values():
                idx_union = ser.index if idx_union is None else idx_union.union(ser.index)
            idx_union = idx_union.sort_values()
            total_margin = pd.Series(0.0, index=idx_union)
            for f, ser in per_file_netpos.items():
                sym = _symbol_from_first_row(store_trades, f).upper()
                spec = margin_spec.get(sym)
                if spec is None: 
                    continue
                init_margin = float(spec[0])
                total_margin = total_margin.add(ser.reindex(idx_union).ffill().fillna(0.0).abs() * init_margin * test_w.get(f, 0.0),
                                                fill_value=0.0)
            if total_margin.max() <= margin_buffer * initial_capital:
                # compute score
                ann = _annualized_return(port, initial_capital)
                dd  = abs(_peak_drawdown_value(port))
                score = (ann / dd) if (dd > 0 and not np.isnan(ann)) else -np.inf
                diag = {"ann_return": ann, "dd": -dd, "score": score, "mode": "risk_parity_projected"}
                return test_w, diag
            shrink *= 0.9
        return {}, {"reason": "cannot satisfy margin with risk_parity base"}

    # Grid search (coarse but robust). For 10 strategies, try a larger step (0.5 or 0.25).
    grid_vals = np.arange(0.0, leverage_cap + 1e-9, step)
    best_score = -np.inf
    best = None
    best_diag = {}

    for combo in itertools.product(grid_vals, repeat=len(files)):
        if sum(combo) > leverage_cap + 1e-9:
            continue
        weights = {f: float(w) for f, w in zip(files, combo)}
        if all(w == 0.0 for w in weights.values()):
            continue

        # Portfolio equity
        port = _portfolio_equity_from_weights(equity_by_file, weights)

        # Margin constraint
        idx_union = None
        for ser in per_file_netpos.values():
            idx_union = ser.index if idx_union is None else idx_union.union(ser.index)
        idx_union = idx_union.sort_values()
        total_margin = pd.Series(0.0, index=idx_union)
        for f, ser in per_file_netpos.items():
            sym = _symbol_from_first_row(store_trades, f).upper()
            spec = margin_spec.get(sym)
            if spec is None:
                continue
            init_margin = float(spec[0])
            total_margin = total_margin.add(ser.reindex(idx_union).ffill().fillna(0.0).abs() * init_margin * weights.get(f, 0.0),
                                            fill_value=0.0)
        if total_margin.max() > margin_buffer * initial_capital:
            continue  # violates margin headroom

        # Objective
        ann = _annualized_return(port, initial_capital)
        dd  = abs(_peak_drawdown_value(port))
        if objective == "max_return_over_dd":
            score = (ann / dd) if (dd > 0 and not np.isnan(ann)) else -np.inf
        else:
            score = -dd

        if np.isfinite(score) and score > best_score:
            best_score = score
            best = weights
            best_diag = {"ann_return": ann, "dd": -dd, "score": score, "mode": "grid"}
    return (best or {}), best_diag


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
    Output("store-trades", "data"),
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
    Input("upload", "contents"),
    State("upload", "filename"),
    State("store-trades", "data"),      # keep previously stored files
    State("file-toggle", "value"),      # keep previous selection
    State("symbol-toggle", "value"),
    State("interval-toggle", "value"),
    State("strategy-toggle", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    prevent_initial_call=True
)
def ingest_files(contents_list, names_list, prev_store, prev_file_sel, prev_sym_sel, prev_int_sel, prev_strat_sel, prev_start, prev_end):
    if not contents_list:
        raise PreventUpdate

    # Normalize to lists
    if isinstance(contents_list, str):
        contents_list = [contents_list]
        names_list = [names_list]

    # Start from previously stored files
    parsed_by_file: dict[str, list[dict[str, Any]]] = dict(prev_store) if prev_store else {}
    existing_names = set(parsed_by_file.keys())
    newly_added: list[str] = []

    # Parse and merge
    for contents, fname in zip(contents_list, names_list):
        raw_bytes, disp_name = _decode_upload(contents, fname)
        disp_name = _safe_unique_name(disp_name, existing_names)
        existing_names.add(disp_name)
        tdf = parse_tradestation_trades(raw_bytes, disp_name)
        parsed_by_file[disp_name] = tdf.to_dict(orient="records")
        newly_added.append(disp_name)

    # Build file options and selected
    file_options = [{"label": n, "value": n} for n in parsed_by_file.keys()]
    if prev_file_sel:
        file_selected = prev_file_sel + [n for n in newly_added if n not in prev_file_sel]
    else:
        file_selected = list(parsed_by_file.keys())

    # File list UI
    file_list_view = html.Ul([html.Li(n) for n in parsed_by_file.keys()])

    # Compute global min/max dates across ALL stored files
    all_min, all_max = None, None
    all_symbols: set[str] = set()
    all_intervals: set[int] = set()
    all_strategies: set[str] = set()

    for recs in parsed_by_file.values():
        if not recs:
            continue
        tdf = pd.DataFrame(recs)
        # Collect meta
        if "Symbol" in tdf:
            all_symbols.update(tdf["Symbol"].dropna().astype(str).unique().tolist())
        if "Interval" in tdf:
            all_intervals.update(tdf["Interval"].dropna().astype(int).unique().tolist())
        if "Strategy" in tdf:
            all_strategies.update(tdf["Strategy"].dropna().astype(str).unique().tolist())
        # Dates
        dt = pd.to_datetime(tdf.get("exit_time", pd.Series(dtype="datetime64[ns]")), errors="coerce").dropna()
        if dt.empty:
            continue
        mn, mx = dt.min().date(), dt.max().date()
        all_min = mn if (all_min is None or mn < all_min) else all_min
        all_max = mx if (all_max is None or mx > all_max) else all_max

    # Build symbol/timeframe/strategy options
    sym_options = [{"label": s, "value": s} for s in sorted(all_symbols)]
    int_options = [{"label": str(i), "value": int(i)} for i in sorted(all_intervals)]
    strat_options = [{"label": s, "value": s} for s in sorted(all_strategies)]

    # Preserve previous selections where possible; otherwise select all
    sym_selected = [v for v in (prev_sym_sel or []) if v in all_symbols] or [s for s in sorted(all_symbols)]
    int_selected = [v for v in (prev_int_sel or []) if v in all_intervals] or [i for i in sorted(all_intervals)]
    strat_selected = [v for v in (prev_strat_sel or []) if v in all_strategies] or [s for s in sorted(all_strategies)]

    # Respect current date selection if set; else use global min/max
    start_date = prev_start if prev_start else (all_min if all_min else None)
    end_date = prev_end if prev_end else (all_max if all_max else None)

    return (
        parsed_by_file,
        file_list_view,
        file_options,
        file_selected,
        sym_options, sym_selected,
        int_options, int_selected,
        strat_options, strat_selected,
        start_date, end_date,
        {"min": str(all_min) if all_min else None, "max": str(all_max) if all_max else None},
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
    Input("store-contracts", "data"),
)


def update_analysis(store_trades, selected_files, selected_symbols, selected_intervals,
                    selected_strategies, direction, start_date, end_date,
                    active_tab, corr_mode, corr_slope_window,
                    contracts_map):


    fig_empty = go.Figure().update_layout(margin=dict(l=10, r=10, t=30, b=10))
    if not store_trades:
        return [dcc.Graph(figure=fig_empty, style={"height": "520px"})]

    # Keep only selections that exist
    available_files = list(store_trades.keys())
    selected_files = [f for f in (selected_files or []) if f in available_files]
    if not selected_files:
        return [dcc.Graph(figure=fig_empty, style={"height": "520px"})]

    # Rehydrate + filter per file
    trades_by_file: dict[str, pd.DataFrame] = {}
    equity_by_file: dict[str, pd.Series] = {}




    def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Symbol / Interval / Strategy filters (if any selected)
        if selected_symbols:
            out = out[out["Symbol"].isin(selected_symbols)]
        if selected_intervals:
            out = out[out["Interval"].isin(selected_intervals)]
        if selected_strategies:
            out = out[out["Strategy"].isin(selected_strategies)]
        # Direction
        if direction in ("Long", "Short"):
            out = out[out["direction"] == direction]
        # Date range (by exit_time)
        out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
        out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
        out = _within_dates(out, start_date, end_date, col="exit_time")
        return out

    for fname in selected_files:
        recs = store_trades.get(fname, [])
        if not recs:
            continue
        tdf = pd.DataFrame(recs)
        tdf_f = _apply_filters(tdf)

        # === SCALE by contracts multiplier (per-file) ===
        mul = float((contracts_map or {}).get(fname, 1.0))
        # If multiplier is 0 → exclude this file without unchecking
        if mul == 0.0:
            continue

        if mul != 1.0 and not tdf_f.empty:

            # Scale fields that represent $ or qty so all downstream calc/plots reflect N contracts
            for col in ["net_profit", "runup", "drawdown_trade", "commission", "slippage", "contracts", "CumulativePL_raw"]:
                if col in tdf_f.columns:
                    tdf_f[col] = pd.to_numeric(tdf_f[col], errors="coerce").fillna(0.0) * mul

        trades_by_file[fname] = tdf_f
        equity_by_file[fname] = equity_from_trades_subset(tdf_f)


    # Build a mapping: filename -> pretty display label
    label_map: dict[str, str] = {}
    used: set[str] = set()
    for fname, tdf in trades_by_file.items():
        base = _display_label_from_df(tdf, fallback=fname)
        # ensure uniqueness if multiple files resolve to same label
        pretty = _safe_unique_name(base, used)
        used.add(pretty)
        label_map[fname] = pretty

    # Combine equity → portfolio
    eq_all = combine_equity(equity_by_file, list(equity_by_file.keys()))

    # -------- Figures --------
    def build_equity_fig():
        if eq_all is None or eq_all.empty:
            return fig_empty
        f = go.Figure()
        for col in [c for c in eq_all.columns if c != "Portfolio"]:
            disp = label_map.get(col, col)  # <-- use display label
            f.add_trace(go.Scatter(x=eq_all.index, y=eq_all[col], name=disp, mode="lines", line=dict(width=1)))
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

    def build_hist_fig():
        all_pnl = []
        for tdf in trades_by_file.values():
            if tdf is None or tdf.empty:
                continue
            pnl = pd.to_numeric(tdf["net_profit"], errors="coerce")
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

    def _single_graph_child(fig, height=520):
        return [dcc.Graph(figure=fig, style={"height": f"{height}px"})]


    children = []
    if active_tab == "equity":
        children = _single_graph_child(build_equity_fig())

    elif active_tab == "ddcurve":
        children = _single_graph_child(build_dd_fig())

    elif active_tab == "idd":
        children = _single_graph_child(build_idd_fig())

    elif active_tab == "hist":
        children = _single_graph_child(build_hist_fig())

    elif active_tab == "corr":
        fig_corr = build_correlation_heatmap(
            equity_by_file,
            label_map=label_map,
            mode=corr_mode or "drawdown_pct",
            slope_window=int(corr_slope_window or 20),
            method="spearman"
        )
        children = _single_graph_child(fig_corr)

    elif active_tab == "margin":
        # --- Build inputs for all three charts ---

        # 1) Per-file signed net positions (already respects filters)
        per_file_netpos = _netpos_timeseries_from_store(
            store_trades, selected_files,
            selected_symbols, selected_intervals, selected_strategies,
            direction, start_date, end_date,
            contracts_map=contracts_map
        )

        # 2) Aggregate to per-symbol signed nets (sum across strategies/files for same symbol)
        by_symbol_net = _aggregate_netpos_per_symbol(per_file_netpos, store_trades)

        # 3) Total initial margin series:
        #    For each SYMBOL, margin used = |net_sym(t)| * initial_margin(symbol),
        #    then sum across symbols.
    
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
            total_init_margin = total_init_margin.add(s.reindex(idx_union).ffill().fillna(0.0).abs() * init_margin,
                                                      fill_value=0.0)

        # 4) Portfolio cumulative P&L for the filtered set (already computed as eq_all["Portfolio"])
        port_eq = eq_all.get("Portfolio") if (eq_all is not None and "Portfolio" in eq_all) else None

        # 5) Purchasing Power series (uses DEFAULT_INITIAL_CAPITAL per your instruction)
        pp_series = _purchasing_power_series(port_eq, total_init_margin, DEFAULT_INITIAL_CAPITAL)

        # --- Build the three figures ---

        # A) Purchasing Power (line)
        fig_pp = go.Figure()
        if pp_series is not None and not pp_series.empty:
            fig_pp.add_trace(go.Scatter(x=pp_series.index, y=pp_series, mode="lines", name="Purchasing Power", line=dict(width=3)))
        fig_pp.update_layout(
            title="Purchasing Power = Starting Balance + Portfolio P&L − Initial Margin Used",
            xaxis_title="Date/Time", yaxis_title="Purchasing Power ($)",
            hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10)
        )

        # B) Initial Margin Used (line)
        fig_im = go.Figure()
        if total_init_margin is not None and not total_init_margin.empty:
            fig_im.add_trace(go.Scatter(x=total_init_margin.index, y=total_init_margin, mode="lines", name="Initial Margin Used", line=dict(width=2)))
        fig_im.update_layout(
            title="Initial Margin Used (Σ symbols |net_sym| × IM(sym))",
            xaxis_title="Date/Time", yaxis_title="Initial Margin ($)",
            hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10)
        )

        # C) Existing Net Contracts chart (unchanged)
        fig_netpos = build_net_contracts_figure(
            store_trades=store_trades,
            selected_files=selected_files,
            selected_symbols=selected_symbols,
            selected_intervals=selected_intervals,
            selected_strategies=selected_strategies,
            direction=direction,
            start_date=start_date,
            end_date=end_date,
            contracts_map=contracts_map,
        )

        # Stack them vertically (top→bottom)
        children = [
            dcc.Graph(figure=fig_pp, style={"height": "320px"}),
            dcc.Graph(figure=fig_im, style={"height": "260px"}),
            dcc.Graph(figure=fig_netpos, style={"height": "520px"}),
        ]

    else:
        # default
        children = _single_graph_child(fig_empty)



    return children



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
    prevent_initial_call=True
)
def update_metrics_table(store_trades, selected_files, selected_symbols, selected_intervals,
                         selected_strategies, direction, start_date, end_date, contracts_map):

    if not store_trades or not selected_files:
        return [], []

    def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if selected_symbols:
            out = out[out["Symbol"].isin(selected_symbols)]
        if selected_intervals:
            out = out[out["Interval"].isin(selected_intervals)]
        if selected_strategies:
            out = out[out["Strategy"].isin(selected_strategies)]
        if direction in ("Long", "Short"):
            out = out[out["direction"] == direction]
        out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
        out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
        return _within_dates(out, start_date, end_date, col="exit_time")

    rows = []
    frames_for_port = []

    for f in selected_files:
        recs = store_trades.get(f, [])
        if not recs:
            continue
        tdf = pd.DataFrame(recs)
        tdf = _apply_filters(tdf)

        mul = float((contracts_map or {}).get(f, 1.0))
        if mul == 0.0:
            continue
        if mul != 1.0 and not tdf.empty:
            for col in ["net_profit", "runup", "drawdown_trade", "commission", "slippage", "contracts", "CumulativePL_raw"]:
                if col in tdf.columns:
                    tdf[col] = pd.to_numeric(tdf[col], errors="coerce").fillna(0.0) * mul

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
    prevent_initial_call=True
)
def run_allocator(n_clicks, store_trades, selected_files,
                  selected_symbols, selected_intervals, selected_strategies,
                  direction, start_date, end_date,
                  equity_val, margin_pct, objective, lev_cap, step,
                  contracts_map):


    if not n_clicks or not store_trades or not selected_files:
        return "Upload and select files, then run.", go.Figure(), [], []

    # Rebuild per-file equity curves using current filters (reuse your helpers)
    # NOTE: this mirrors pieces from update_analysis
    trades_by_file = {}
    equity_by_file = {}

    def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if selected_symbols:
            out = out[out["Symbol"].isin(selected_symbols)]
        if selected_intervals:
            out = out[out["Interval"].isin(selected_intervals)]
        if selected_strategies:
            out = out[out["Strategy"].isin(selected_strategies)]
        if direction in ("Long", "Short"):
            out = out[out["direction"] == direction]
        out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
        out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
        return _within_dates(out, start_date, end_date, col="exit_time")

    for fname in selected_files:
        recs = store_trades.get(fname, [])
        if not recs:
            continue
        tdf = _apply_filters(pd.DataFrame(recs))

        mul = float((contracts_map or {}).get(fname, 1.0))
        if mul == 0.0:
            continue
        if mul != 1.0 and not tdf.empty:

            for col in ["net_profit", "runup", "drawdown_trade", "commission", "slippage", "contracts", "CumulativePL_raw"]:
                if col in tdf.columns:
                    tdf[col] = pd.to_numeric(tdf[col], errors="coerce").fillna(0.0) * mul

        trades_by_file[fname] = tdf
        equity_by_file[fname] = equity_from_trades_subset(tdf)

    # Net position series
    netpos = _netpos_timeseries_from_store(store_trades, selected_files,
                                        selected_symbols, selected_intervals, selected_strategies,
                                        direction, start_date, end_date,
                                        contracts_map=contracts_map)


    init_cap = float(equity_val or DEFAULT_INITIAL_CAPITAL)
    margin_buf = max(0.0, min(1.0, (margin_pct or 60)/100.0))
    lev_cap = float(lev_cap or 1.0)
    step = float(step or 0.25)

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

    # Build combined equity
    port = _portfolio_equity_from_weights(equity_by_file, weights)
    ann = _annualized_return(port, init_cap)
    dd  = _peak_drawdown_value(port)

    # Build margin series for the chosen weights
    # (scale each file's netpos by its weight, then multiply by per-symbol initial margin)
    idx_union = None
    for ser in netpos.values():
        idx_union = ser.index if idx_union is None else idx_union.union(ser.index)
    idx_union = idx_union.sort_values()
    total_margin = pd.Series(0.0, index=idx_union)
    for f, ser in netpos.items():
        sym = _symbol_from_first_row(store_trades, f).upper()
        spec = MARGIN_SPEC.get(sym)
        if spec is None:
            continue
        init_margin = float(spec[0])
        total_margin = total_margin.add(ser.reindex(idx_union).ffill().fillna(0.0).abs() * init_margin * float(weights.get(f, 0.0)),
                                        fill_value=0.0)

    # Summary
    summary = (f"Best allocation ({diag.get('mode','grid')}): "
               f"Ann. Return ~ {ann*100:0.2f}% | Max DD {dd:,.0f} | "
               f"Max Margin Used ${total_margin.max():,.0f} "
               f"({(total_margin.max()/init_cap*100):0.1f}% of equity, cap {margin_buf*100:0.0f}%)")

    # Plot equity
    fig = go.Figure()
    for f, s in equity_by_file.items():
        w = float(weights.get(f, 0.0))
        if w > 0 and s is not None and not s.empty:
            fig.add_trace(go.Scatter(x=s.index, y=s * w, name=f"{f} × {w:0.2f}", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=port.index, y=port, name="Portfolio (alloc)", line=dict(width=3)))
    fig.update_layout(title="Allocated Portfolio Equity", xaxis_title="Date", yaxis_title="Cumulative P/L",
                      hovermode="x unified", margin=dict(l=10,r=10,t=40,b=10))

    # Weights table
    rows = [{"File": f,
             "Symbol": _symbol_from_first_row(store_trades, f).upper(),
             "Weight": float(w),
             "Note": "scale factor for P&L; convert to contracts via margin"}
            for f, w in sorted(weights.items()) if w > 0]
    cols = [{"name": c, "id": c} for c in ["File","Symbol","Weight","Note"]]

    return summary, fig, rows, cols

def export_trades(n_clicks, store_trades, sel_files, sel_syms, sel_ints, sel_strats, direction, start_date, end_date, contracts_map):
    if not n_clicks or not store_trades or not sel_files:
        return None

    frames = []
    for f in sel_files:
        recs = store_trades.get(f, [])
        if not recs:
            continue
        tdf = pd.DataFrame(recs)
        # Filters
        if sel_syms:
            tdf = tdf[tdf["Symbol"].isin(sel_syms)]
        if sel_ints:
            tdf = tdf[tdf["Interval"].isin(sel_ints)]
        if sel_strats:
            tdf = tdf[tdf["Strategy"].isin(sel_strats)]
        if direction in ("Long", "Short"):
            tdf = tdf[tdf["direction"] == direction]
        tdf["exit_time"] = pd.to_datetime(tdf["exit_time"], errors="coerce")
        tdf["entry_time"] = pd.to_datetime(tdf["entry_time"], errors="coerce")
        tdf = _within_dates(tdf, start_date, end_date, col="exit_time")

        # === SCALE by contracts multiplier (per-file) so exports match on-screen results ===
        mul = float((contracts_map or {}).get(f, 1.0))
        # If multiplier is 0 → exclude this file from the CSV
        if mul == 0.0:
            continue
        if mul != 1.0 and not tdf.empty:

            for col in ["net_profit", "runup", "drawdown_trade", "commission", "slippage", "contracts", "CumulativePL_raw"]:
                if col in tdf.columns:
                    tdf[col] = pd.to_numeric(tdf[col], errors="coerce").fillna(0.0) * mul

        frames.append(tdf)


    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["File", "exit_time"])
    # Consistent column order for export
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
    prevent_initial_call=True
)
def _cb_export_trades(n_clicks, store_trades, sel_files, sel_syms, sel_ints, sel_strats, direction, start_date, end_date, contracts_map):
    return export_trades(n_clicks, store_trades, sel_files, sel_syms, sel_ints, sel_strats, direction, start_date, end_date, contracts_map)

# NEW: metrics export is now a proper Dash callback
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
