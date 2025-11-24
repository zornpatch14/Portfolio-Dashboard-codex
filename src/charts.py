# src/charts.py

# src/charts.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional
import plotly.graph_objects as go

from src.equity import (
    max_drawdown_series,
    intraday_drawdown_series,
)


# PURPOSE
#   Pure, reusable Plotly figure builders for the app. These functions only
#   take data as inputs and return `go.Figure` objects. No Dash callbacks,
#   no global state access—so they’re easy to test and reuse.
#
# WHAT LIVES HERE (current)
#   - build_equity_figure(eq_all, label_map, color_portfolio) -> go.Figure
#       One line per file + bold Portfolio line.
#   - build_drawdown_figure(portfolio_series) -> go.Figure
#       Portfolio drawdown over time (equity - rolling peak).
#   - build_intraday_dd_figure(portfolio_series) -> go.Figure
#       Worst intraday drawdown per day (bar chart).
#   - build_pl_histogram_figure(trades_by_file, nbins=50) -> go.Figure
#       Histogram of trade P/L across all files.
#
# INPUTS / OUTPUTS
#   Inputs are pandas Series/DataFrames and small dicts (e.g., label_map).
#   Output is always a `plotly.graph_objects.Figure`.
#
# DEPENDENCIES
#   - plotly.graph_objects as go
#   - src.equity: max_drawdown_series, intraday_drawdown_series
#   (Uncomment optional imports below only if a new chart needs them.)
#   # from plotly.subplots import make_subplots
#   # import plotly.express as px
#
# HOW THE MAIN FILE USES THIS
#   In Portfolio_Dashboard6.py:
#       from src.charts import (
#           build_equity_figure, build_drawdown_figure,
#           build_intraday_dd_figure, build_pl_histogram_figure,
#       )
#   Then inside `update_analysis(...)`, the tab handlers call these builders,
#   e.g.:
#       _single_graph_child(build_equity_figure(eq_all, label_map, COLOR_PORTFOLIO))
#
# ADDING NEW CHARTS
#   - Create a new `build_*_figure(...)` function here.
#   - Accept all needed inputs as parameters (do not read globals).
#   - Return a `go.Figure`.
#   - Import and call it from Portfolio_Dashboard6.py (no callbacks here).
#
# DO NOT
#   - Register Dash callbacks in this file.
#   - Read or mutate global app/server state.
#   - Depend on names closed over in the main file—pass everything in.




# 1) Equity curves -------------------------------------------------------
def build_equity_figure(
    eq_all: pd.DataFrame,
    label_map: Dict[str, str],
    color_portfolio: str,
    *,
    value_is_percent: bool = False,
) -> go.Figure:
    """
    Draws one line per file + bold Portfolio line.
    Expects eq_all with columns for each file and 'Portfolio'.
    """
    f = go.Figure()
    if eq_all is None or eq_all.empty:
        f.update_layout(margin=dict(l=80, r=20, t=30, b=10))
        return f

    scale = 100.0 if value_is_percent else 1.0
    yaxis_title = "Cumulative Return (%)" if value_is_percent else "Cumulative P/L"
    hover_fmt = "%{y:.2f}%" if value_is_percent else "%{y:.2f}"

    for col in [c for c in eq_all.columns if c != "Portfolio"]:
        disp = label_map.get(col, col)
        series = eq_all[col] * scale
        f.add_trace(go.Scattergl(
            x=eq_all.index,
            y=series,
            name=disp,
            line=dict(width=1),
            hovertemplate=hover_fmt + "<extra></extra>",
        ))
    if "Portfolio" in eq_all:
        series = eq_all["Portfolio"] * scale
        f.add_trace(go.Scattergl(
            x=eq_all.index,
            y=series,
            name="Portfolio",
            line=dict(width=2.5, color=color_portfolio),
            hovertemplate=hover_fmt + "<extra></extra>",
        ))
    f.update_layout(
        title=dict(text="Equity Curves", y=0.97),
        xaxis_title="Date", yaxis_title=yaxis_title,
        hovermode="x unified",
        margin=dict(l=80, r=20, t=60, b=70),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            borderwidth=0,
        ),
    )
    return f

# 2) Portfolio drawdown curve -------------------------------------------
def build_drawdown_figure(
    portfolio_series: Optional[pd.Series],
    *,
    value_is_percent: bool = False,
) -> go.Figure:
    """
    Drawdown(t) = equity(t) - rolling_peak(t). Returns a line chart.
    """
    f = go.Figure()
    if portfolio_series is None or portfolio_series.empty:
        f.update_layout(margin=dict(l=80, r=20, t=30, b=10))
        return f
    if value_is_percent:
        equity_index = (1.0 + portfolio_series).astype(float)
        equity_index = equity_index.sort_index()
        dd = (equity_index / equity_index.cummax()) - 1.0
        dd_plot = dd * 100.0
        yaxis_title = "Drawdown (%)"
        hover_fmt = "%{y:.2f}%"
    else:
        dd = max_drawdown_series(portfolio_series)
        dd_plot = dd
        yaxis_title = "Drawdown"
        hover_fmt = "%{y:.2f}"
    f.add_trace(go.Scattergl(
        x=dd.index,
        y=dd_plot,
        name="Drawdown",
        line=dict(width=2),
        hovertemplate=hover_fmt + "<extra></extra>",
    ))
    f.update_layout(
        title="Portfolio Drawdown Over Time",
        xaxis_title="Date", yaxis_title=yaxis_title,
        hovermode="x unified", margin=dict(l=80, r=20, t=40, b=10)
    )
    return f

# 3) Intraday drawdown (worst per day) ----------------------------------
def build_intraday_dd_figure(
    portfolio_series: Optional[pd.Series],
    *,
    value_is_percent: bool = False,
) -> go.Figure:
    """
    Plots worst intraday drawdown per day for the portfolio series as a bar chart.
    """
    f = go.Figure()
    if portfolio_series is None or portfolio_series.empty:
        f.update_layout(margin=dict(l=80, r=20, t=30, b=10))
        return f
    series = portfolio_series.sort_index()
    if value_is_percent:
        s = series.copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        out = {}
        for d, g in s.groupby(s.index.normalize()):
            if g.empty:
                out[d] = 0.0
                continue
            run_max = g.cummax()
            dd_pct = (g / run_max) - 1.0
            out[d] = dd_pct.min()
        idd = pd.Series(out, dtype=float) * 100.0
        hover_fmt = "%{y:.2f}%"
        yaxis_title = "Intraday Drawdown (%)"
    else:
        idd = intraday_drawdown_series(series)
        hover_fmt = "%{y:.2f}"
        yaxis_title = "Intraday Drawdown"
    f.add_trace(go.Bar(x=idd.index, y=idd, name="Intraday DD", hovertemplate=hover_fmt + "<extra></extra>"))
    f.update_layout(
        title="Portfolio Intraday Drawdown (worst per day)",
        xaxis_title="Date", yaxis_title=yaxis_title,
        margin=dict(l=80, r=20, t=40, b=10)
    )
    return f

# 4) Histogram of trade P/L ---------------------------------------------
def build_pl_histogram_figure(
    trades_by_file: Dict[str, pd.DataFrame],
    nbins: int = 50,
) -> go.Figure:
    """
    Concats trade P/L across files and draws a histogram.
    """
    f = go.Figure()
    if not trades_by_file:
        f.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        return f
    all_pnl = []
    for tdf in trades_by_file.values():
        pnl = pd.to_numeric(tdf.get("net_profit"), errors="coerce")
        pnl = pnl[~pnl.isna()]
        all_pnl.append(pnl)
    pnl_all = pd.concat(all_pnl) if all_pnl else pd.Series(dtype=float)
    if not pnl_all.empty:
        f.add_trace(go.Histogram(x=pnl_all, nbinsx=nbins, name="Trade P/L"))
    f.update_layout(
        title="Histogram of Trade P/L",
        xaxis_title="P/L per Trade", yaxis_title="Count",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return f


