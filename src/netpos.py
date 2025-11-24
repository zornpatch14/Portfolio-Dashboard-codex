# src/netpos.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Any, Dict
import plotly.graph_objects as go

from src.helpers import (
    _symbol_from_first_row,
    _within_dates,
    _display_label_from_df,
    _safe_unique_name,
)

# Module-level hook set from main:
#   import src.netpos as _netpos_mod
#   _netpos_mod.app = app
app: Optional[Any] = None




# If any pasted code uses these, uncomment them:
# from plotly.subplots import make_subplots
# import plotly.express as px
# import math
# from datetime import datetime, timedelta
# from src.constants import APP_TITLE  # or any constants you actually use

# ----- PASTE THESE FUNCTIONS FROM Portfolio_Dashboard6.py (unchanged) -----
# (Search in your main file for each def name and cut+paste the whole function
#  body including docstrings/comments. Do NOT tweak logic.)
#
# Keep the INNER 'base_filter(...)' closures exactly as-is.
#
# 1) def _aggregate_netpos_per_symbol_from_series(...):
#    - Aggregates per-symbol net contracts series and computes the
#      portfolio-wide value (sum of per-symbol |net|).
# 2) def _netpos_figure_from_series(...):
#    - Builds the Plotly figure for the Net Contracts tab from series dict.
# 3) def _netpos_timeseries_from_store(...):   (paste if you have it)
#    - Builds per-symbol net contract time series from your cached store.
# 4) def build_net_contracts_figure(...):
#    - Top-level builder your Dash callback calls for the Net Contracts tab.
#
# PASTE BELOW THIS LINE
# --------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# 1) Aggregate file-level nets into per-SYMBOL nets
# -----------------------------------------------------------------------------
def _aggregate_netpos_per_symbol_from_series(per_file_netpos: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """
    Convert {file -> signed net contracts Series} into {SYMBOL -> signed net Series}
    by summing across files that share the same symbol. Index is the union of times.
    """
    if not per_file_netpos:
        return {}

    # Union index
    idx = None
    for s in per_file_netpos.values():
        idx = s.index if idx is None else idx.union(s.index)
    idx = idx.sort_values()

    by_sym: dict[str, pd.Series] = {}
    for fname, ser in per_file_netpos.items():
        # Fetch the trades DF from the app store, then extract the symbol (PURE helper)
        df_trades = app.server.trade_store.get(fname) if (app is not None and hasattr(app, "server")) else None
        sym_raw = _symbol_from_first_row(df_trades)
        sym = (sym_raw or "").upper()
        if not sym:
            continue

        aligned = ser.reindex(idx).ffill().fillna(0.0)
        by_sym[sym] = (by_sym.get(sym, pd.Series(0.0, index=idx))).add(aligned, fill_value=0.0)

    return by_sym


# -----------------------------------------------------------------------------
# 2) Build figure directly from precomputed series (optional path)
# -----------------------------------------------------------------------------
def _netpos_figure_from_series(
    per_file_series: dict[str, pd.Series],
    per_symbol_series: dict[str, pd.Series]
) -> go.Figure:
    """
    Build the Net Contracts figure directly from precomputed series (no recompute).
    Expects Series values (signed net contracts).
    """
    fig = go.Figure()

    # Per-file thin lines
    for f, s in sorted(per_file_series.items()):
        if s is None or s.empty:
            continue
        fig.add_trace(go.Scattergl(
            x=s.index, y=s.abs(), mode="lines", name=f, line=dict(width=1), line_shape="hv",
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>%{fullData.name}<br>|Net|: %{y:.0f}<extra></extra>"
        ))

    # Per-symbol dashed
    for sym, s in sorted(per_symbol_series.items()):
        if s is None or s.empty:
            continue
        fig.add_trace(go.Scattergl(
            x=s.index, y=s.abs(), mode="lines", name=f"Symbol: {sym}",
            line=dict(width=2, dash="dash"), line_shape="hv",
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>%{fullData.name}<br>|Net|: %{y:.0f}<extra></extra>"
        ))

    # Portfolio = sum across symbols of |net|
    if per_symbol_series:
        idx = None
        for s in per_symbol_series.values():
            idx = s.index if idx is None else idx.union(s.index)
        idx = idx.sort_values()
        port = pd.Series(0.0, index=idx)
        for s in per_symbol_series.values():
            port = port.add(s.reindex(idx).ffill().fillna(0.0).abs(), fill_value=0.0)
        fig.add_trace(go.Scattergl(
            x=port.index, y=port, mode="lines", name="Portfolio (Net)",
            line=dict(width=3, color="black"), line_shape="hv",
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>%{fullData.name}<br>|Net|: %{y:.0f}<extra></extra>"
        ))

    fig.add_hline(y=0, line_width=1, line_dash="dot", opacity=0.6)
    fig.update_layout(
        title=dict(text="Net Contracts - Per Symbol |long-short|, Portfolio: sum |net| (no cross-symbol cancel)", y=0.97),
        xaxis_title="Date/Time",
        yaxis_title="|Net Contracts|",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=70, b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            borderwidth=0,
        ),
        yaxis=dict(dtick=1),
    )
    return fig


# -----------------------------------------------------------------------------
# 3) Build time series (file -> signed net contracts) from the app store
# -----------------------------------------------------------------------------
def _netpos_timeseries_from_store(
    store_tokens: dict,
    selected_files: list[str],
    selected_symbols: list[str],
    selected_intervals: list[int],
    selected_strategies: list[str],
    direction: str,
    start_date,
    end_date,
    contracts_map: dict | None = None
) -> dict[str, pd.Series]:
    """
    Returns {file -> signed net contracts time-series} aligned on a shared DatetimeIndex.
    Reads DataFrames from app.server.trade_store. Keeps your nested base_filter.
    """
    if not store_tokens or not selected_files:
        return {}

    def base_filter(df: pd.DataFrame) -> pd.DataFrame:
        out = df
        if selected_symbols:
            out = out[out["Symbol"].isin(selected_symbols)]
        if selected_intervals:
            out = out[out["Interval"].isin(selected_intervals)]
        if selected_strategies:
            out = out[out["Strategy"].isin(selected_strategies)]
        if direction in ("Long", "Short"):
            out = out[out["direction"] == direction]
        return _within_dates(out, start_date, end_date, col="exit_time")

    # Build entry/exit delta events
    ev_rows = []
    for f in selected_files:
        tdf = app.server.trade_store.get(f) if (app is not None and hasattr(app, "server")) else None
        if tdf is None or tdf.empty:
            continue
        df = base_filter(tdf)
        for _, r in df.iterrows():
            qty = float(r.get("contracts") or 0.0)
            if contracts_map:
                qty *= float(contracts_map.get(f, 1.0))
            if qty == 0:
                continue
            ent, exi = r.get("entry_time"), r.get("exit_time")
            dirn = str(r.get("direction") or "")
            sym = str(r.get("Symbol") or "")
            if pd.notna(ent):
                delta = qty if dirn == "Long" else (-qty if dirn == "Short" else 0.0)
                ev_rows.append({"ts": ent, "delta": delta, "File": f, "Symbol": sym, "kind_rank": 1})
            if pd.notna(exi):
                delta = -qty if dirn == "Long" else (+qty if dirn == "Short" else 0.0)
                ev_rows.append({"ts": exi, "delta": delta, "File": f, "Symbol": sym, "kind_rank": 0})

    ev = pd.DataFrame(ev_rows)
    if ev.empty:
        return {}

    ev["ts"] = pd.to_datetime(ev["ts"], errors="coerce")
    ev = ev.dropna(subset=["ts"]).sort_values(["ts", "kind_rank"])

    ts_min, ts_max = ev["ts"].min(), ev["ts"].max()
    start_dt = pd.to_datetime(start_date) if start_date else ts_min
    end_dt = pd.to_datetime(end_date) if end_date else ts_max
    if pd.isna(start_dt): start_dt = ts_min
    if pd.isna(end_dt):   end_dt = ts_max
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    idx = (
        pd.DatetimeIndex(ev.loc[(ev["ts"] >= start_dt) & (ev["ts"] <= end_dt), "ts"])
        .append(pd.DatetimeIndex([start_dt, end_dt])).dropna().unique().sort_values()
    )

    out: Dict[str, pd.Series] = {}
    for f, g in ev.groupby("File"):
        g = g[(g["ts"] >= start_dt) & (g["ts"] <= end_dt)]
        g = g.groupby("ts")["delta"].sum().reindex(idx, fill_value=0.0)
        out[f] = g.cumsum()
    return out


# -----------------------------------------------------------------------------
# 4) Full figure builder (reads from app store; keeps nested base_filter)
# -----------------------------------------------------------------------------
def build_net_contracts_figure(
    store_trades: dict,
    selected_files: list[str],
    selected_symbols: list[str],
    selected_intervals: list[int],
    selected_strategies: list[str],
    direction: str,
    start_date,
    end_date,
    contracts_map: dict | None = None
) -> go.Figure:

    empty_fig = go.Figure().update_layout(margin=dict(l=10, r=10, t=30, b=10))
    if not store_trades or not selected_files:
        return empty_fig

    def base_filter(df: pd.DataFrame) -> pd.DataFrame:
        out = df
        if selected_symbols:
            out = out[out["Symbol"].isin(selected_symbols)]
        if selected_intervals:
            out = out[out["Interval"].isin(selected_intervals)]
        if selected_strategies:
            out = out[out["Strategy"].isin(selected_strategies)]
        if direction in ("Long", "Short"):
            out = out[out["direction"] == direction]
        out = out.copy()
        out["contracts"] = pd.to_numeric(out["contracts"], errors="coerce").fillna(0.0).abs()
        return _within_dates(out, start_date, end_date, col="exit_time")

    # Gather filtered trades per selected file
    trades_by_file: Dict[str, pd.DataFrame] = {}
    for f in selected_files:
        tdf = app.server.trade_store.get(f) if (app is not None and hasattr(app, "server")) else None
        if tdf is None or tdf.empty:
            continue
        tdf2 = base_filter(tdf)
        if not tdf2.empty:
            tdf2["File"] = f
            trades_by_file[f] = tdf2

    if not trades_by_file:
        return empty_fig

    # Build display labels per file
    per_file_label_map: dict[str, str] = {}
    used_labels: set[str] = set()
    for f, tdf in trades_by_file.items():
        base = _display_label_from_df(tdf, fallback=f)
        pretty = _safe_unique_name(base, used_labels)
        used_labels.add(pretty)
        per_file_label_map[f] = pretty

    # Build entry/exit delta events
    ev_rows = []
    for f, df in trades_by_file.items():
        for _, r in df.iterrows():
            qty = float(r.get("contracts") or 0.0)
            if contracts_map:
                qty *= float(contracts_map.get(f, 1.0))
            if qty == 0:
                continue
            sym = str(r.get("Symbol"))
            dirn = str(r.get("direction") or "")
            ent, exi = r.get("entry_time"), r.get("exit_time")

            if pd.notna(ent):
                delta = qty if dirn == "Long" else (-qty if dirn == "Short" else 0.0)
                ev_rows.append({"ts": ent, "delta": delta, "File": f, "Symbol": sym, "kind_rank": 1})
            if pd.notna(exi):
                delta = -qty if dirn == "Long" else (+qty if dirn == "Short" else 0.0)
                ev_rows.append({"ts": exi, "delta": delta, "File": f, "Symbol": sym, "kind_rank": 0})

    if not ev_rows:
        return empty_fig

    ev = pd.DataFrame(ev_rows).dropna(subset=["ts"])
    ev["ts"] = pd.to_datetime(ev["ts"], errors="coerce")
    ev = ev.dropna(subset=["ts"]).sort_values(["ts", "kind_rank"])

    ts_min, ts_max = ev["ts"].min(), ev["ts"].max()
    start_dt = pd.to_datetime(start_date) if start_date else ts_min
    end_dt = pd.to_datetime(end_date) if end_date else ts_max
    if pd.isna(start_dt): start_dt = ts_min
    if pd.isna(end_dt):   end_dt = ts_max
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    inwin_ts = ev.loc[(ev["ts"] >= start_dt) & (ev["ts"] <= end_dt), "ts"]
    times_all_idx = (
        pd.DatetimeIndex(inwin_ts).append(pd.DatetimeIndex([start_dt, end_dt]))
        .dropna().unique().sort_values()
    )
    times_all = list(times_all_idx.to_pydatetime())

    def make_series(group_ev: pd.DataFrame) -> pd.DataFrame:
        times = times_all
        if group_ev.empty:
            ser = pd.DataFrame({"ts": times, "y_signed": 0.0})
        else:
            g = group_ev.copy()
            g["ts"] = pd.to_datetime(g["ts"], errors="coerce")
            g = g.dropna(subset=["ts"])
            carry0 = float(g.loc[g["ts"] < start_dt, "delta"].sum())
            deltas = g.loc[(g["ts"] >= start_dt) & (g["ts"] <= end_dt)].groupby("ts")["delta"].sum()
            acc = carry0
            pts = [{"ts": times[0], "y_signed": acc}]
            for t in times[1:]:
                acc += float(deltas.get(pd.Timestamp(t), 0.0))
                pts.append({"ts": t, "y_signed": acc})
            ser = pd.DataFrame(pts).sort_values("ts")
        ser["y_abs"] = ser["y_signed"].abs()
        ser["dir"] = np.where(
            ser["y_signed"] > 0, "long",
            np.where(ser["y_signed"] < 0, "short", "flat")
        )
        return ser

    per_file_series = {f: make_series(ev[ev["File"] == f][["ts", "delta", "kind_rank"]]) for f in trades_by_file.keys()}

    per_symbol_series = {}
    for sym, g in ev.groupby("Symbol", dropna=True):
        per_symbol_series[str(sym)] = make_series(g[["ts", "delta", "kind_rank"]])

    # Portfolio = sum across symbols of |net|
    if per_symbol_series:
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

    fig = go.Figure()

    def add_curve(name, ser, width=2, dash=None, color=None):
        if ser is None or ser.empty:
            return
        customdata = np.column_stack([ser["dir"].astype(str), ser["y_signed"].astype(float)])
        fig.add_trace(go.Scattergl(
            x=ser["ts"], y=ser["y_abs"], mode="lines", name=name,
            line=dict(width=width, dash=dash, color=color),
            line_shape="hv",
            customdata=customdata,
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>%{fullData.name}"
                          "<br>Net: %{y:.0f} (%{customdata[0]})"
                          "<br>Signed: %{customdata[1]:.0f}"
                          "<extra></extra>"
        ))

    # Per-file thin lines with pretty labels
    for f, ser in sorted(per_file_series.items()):
        disp = per_file_label_map.get(f, f)
        add_curve(disp, ser, width=1)

    # Per-symbol dashed
    for sym, ser in sorted(per_symbol_series.items()):
        add_curve(f"Symbol: {sym}", ser, width=2, dash="dash")

    # Portfolio bold
    add_curve("Portfolio (Net)", portfolio_series, width=3, color="black")

    fig.add_hline(y=0, line_width=1, line_dash="dot", opacity=0.6)
    fig.update_layout(
        title=dict(text="Net Contracts - Per Symbol |long-short|, Portfolio: sum |net| (no cross-symbol cancel)", y=0.97),
        xaxis_title="Date/Time",
        yaxis_title="|Net Contracts|",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=70, b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            borderwidth=0,
        ),
        yaxis=dict(dtick=1),
    )
    return fig