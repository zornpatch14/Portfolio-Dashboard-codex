# src/margin.py

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Your existing margin spec lives here:
from src.constants import MARGIN_SPEC


from src.helpers import _symbol_from_first_row



# ----- PASTE THESE FUNCTIONS FROM Portfolio_Dashboard6.py (unchanged) -----
# (Search in your main file for each def name and cut+paste the whole function
#  body including docstrings/comments. Do NOT tweak logic.)
#
# 1) def _purchasing_power_series(portfolio_equity: pd.Series,
#                                 init_margin_total: pd.Series,
#                                 starting_balance: float) -> pd.Series
#    - Your current “PP = starting_balance + equity - margin” series.
#
# 2) def _margin_series(per_file_netpos: dict[str, pd.Series],
#                       store_trades: dict,
#                       margin_spec: dict[str, tuple[float, float, float]]) -> pd.Series
#    - Your current “Initial Margin Used” series built from |net contracts| × IM per symbol.
#
# PASTE BELOW THIS LINE
# --------------------------------------------------------------------------


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
        df_trades = store_trades.get(fname)
        sym = _symbol_from_first_row(df_trades) if df_trades is not None else "UNKNOWN"
        spec = margin_spec.get(sym.upper())
        if spec is None:
            # unknown symbol → assume zero to avoid false positives; better: raise or warn
            continue
        init_margin = float(spec[0])
        ser_re = ser.reindex(all_idx).ffill().fillna(0.0).abs()  # contracts count
        total = total.add(ser_re * init_margin, fill_value=0.0)
    return total



