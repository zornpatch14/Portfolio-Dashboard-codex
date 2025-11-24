# src/equity.py

from __future__ import annotations
import math
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

# ----- PASTE THESE FUNCTIONS FROM Portfolio_Dashboard6.py (unchanged) -----
# (Search in your main file for each def name and cut+paste the whole function
#  body including docstrings/comments. Do NOT tweak logic.)

# 1) def equity_from_trades_subset(...):
#    - Builds an equity time series from a filtered trades DataFrame
# 2) def combine_equity(...):
#    - Aligns multiple equity series and sums them
# 3) def max_drawdown_series(...):
#    - Computes running peak and drawdown %/value series
# 4) def max_drawdown_value(...):
#    - Returns max DD metrics (value/%, start/end dates)
# 5) def intraday_drawdown_series(...):
#    - (If present) constructs intra-day peak→valley drawdown series

# PASTE BELOW THIS LINE
# --------------------------------------------------------------------------

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


