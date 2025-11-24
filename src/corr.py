# src/corr.py

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px  

# ----- PASTE THESE FUNCTIONS FROM Portfolio_Dashboard6.py (unchanged) -----
# (Search in your main file for each def name and cut+paste the whole function
#  body including docstrings/comments. Do NOT tweak logic.)

# 1) def _to_daily_equity(...):
#    - Resamples equity to daily (or your chosen granularity) for correlation
# 2) def _drawdown_pct(...):
#    - Helper that turns equity series into a drawdown % series
# 3) def _zscore(...):
#    - Helper for z-normalization of a series (if used in correlation heatmap)
# 4) def _rolling_slope(...):
#    - Helper to compute rolling regression slope (if you support “slope windows”)
# 5) def _compute_pl_dataframe_from_equity(...):
#    - Produces a aligned DataFrame of per-series P/L changes
# 6) def build_correlation_heatmap(...):
#    - Main function your Dash callback calls to render the heatmap figure

# PASTE BELOW THIS LINE
# --------------------------------------------------------------------------


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







