# src/helpers.py

from __future__ import annotations
import pandas as pd
from datetime import date, datetime
from typing import Optional, Any, Dict, List, Tuple, Set, Iterable, Mapping

# NOTE:
# - All helpers here are PURE (no reads from app.server or other globals).
# - Callers must pass any needed DataFrame explicitly.



# ----- PASTE THESE FROM Portfolio_Dashboard6.py (verbatim) -----
# 1) def _symbol_from_first_row(df, *args, **kwargs) -> str | None:
# 2) def _within_dates(df, start_dt=None, end_dt=None):   # or your exact signature
# 3) def _display_label_from_df(df) -> str:               # if you use this
# 4) def _safe_unique_name(label: str, existing: set[str]) -> str
#
# PASTE BELOW THIS LINE
# ----------------------------------------------------------------



# --------------------------------------------------------------------------
# 1) Symbol extraction from a trades DataFrame (PURE)
# --------------------------------------------------------------------------
def _symbol_from_first_row(df: Optional[pd.DataFrame]) -> str | None:
    """
    Return a best-guess symbol from the first non-empty row of a trades DataFrame.
    Prefers an explicit 'Symbol' column; falls back to light heuristics if needed.
    PURE helper: does not touch app/server/global state.
    """
    if df is None or getattr(df, "empty", True):
        return None

    # Preferred: explicit Symbol column
    if "Symbol" in df.columns and df["Symbol"].notna().any():
        s = str(df["Symbol"].dropna().astype(str).iloc[0]).strip()
        return s if s else None

    # Fallbacks if Symbol is absent: infer from other columns you often have
    for col in ("File", "Strategy"):
        if col in df.columns and df[col].notna().any():
            val = str(df[col].dropna().astype(str).iloc[0]).strip()
            if val:
                # e.g., "MNQ_15m" -> "MNQ"
                return val.split("_")[0].upper()

    return None


# --------------------------------------------------------------------------
# 2) Date filtering (inclusive) on a given datetime-like column
# --------------------------------------------------------------------------
def _within_dates(
    df: pd.DataFrame,
    start_d: date | None,
    end_d: date | None,
    col: str = "exit_time",
) -> pd.DataFrame:
    """
    Return df filtered to [start_d, end_d] inclusive on column `col`.
    If both dates are None, returns df unchanged.
    """
    if start_d is None and end_d is None:
        return df

    out = df.copy()
    # Convert to pandas Timestamps for robust comparison
    if start_d is not None:
        out = out[out[col] >= pd.Timestamp(start_d)]
    if end_d is not None:
        # inclusive end-of-day: <= end_d 23:59:59.999999
        out = out[out[col] <= pd.Timestamp(end_d) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)]
    return out


# --------------------------------------------------------------------------
# 3) Friendly display label from a trades DataFrame
# --------------------------------------------------------------------------
def _display_label_from_df(tdf: pd.DataFrame, fallback: str) -> str:
    """
    Build 'Strategy-SYMBOL-INTERVAL-min' from the first non-null row of tdf.
    Falls back gracefully if any field is missing.
    Ensures SYMBOL uppercased and INTERVAL is int-like.
    """
    if tdf is None or tdf.empty:
        return fallback

    strat = (
        str(tdf["Strategy"].dropna().astype(str).iloc[0])
        if "Strategy" in tdf and tdf["Strategy"].notna().any()
        else None
    )
    sym = (
        str(tdf["Symbol"].dropna().astype(str).iloc[0])
        if "Symbol" in tdf and tdf["Symbol"].notna().any()
        else None
    )
    inter = (
        tdf["Interval"].dropna().iloc[0]
        if "Interval" in tdf and tdf["Interval"].notna().any()
        else None
    )
    try:
        inter = int(inter) if inter is not None else None
    except Exception:
        inter = None

    parts: list[str] = []
    if strat:
        parts.append(strat)
    if sym:
        parts.append(sym.upper())
    if inter is not None:
        parts.append(f"{inter}-min")

    label = "-".join(parts) if parts else fallback
    return label


# --------------------------------------------------------------------------
# 4) Make a name unique within a set
# --------------------------------------------------------------------------
def _safe_unique_name(name: str, existing: Set[str]) -> str:
    """
    If `name` already exists, append ' (n)' with the smallest n >= 2 to make it unique.
    """
    base = name
    n = 2
    out = name
    while out in existing:
        out = f"{base} ({n})"
        n += 1
    return out

# --------------------------------------------------------------------------
# 5) Selection key helpers for caching (PURE)
# --------------------------------------------------------------------------
def _files_key(files: list[str]) -> tuple:
    """Stable key for caching based on selected files. Returns a sorted tuple."""
    return tuple(sorted(files or []))


def _keyify_list(x):
    """Convert a list (or None) into a sorted tuple for use as a cache key."""
    if not x:
        return ()
    return tuple(sorted(x))


def _keyify_contracts_map(d):
    """
    Convert {file: multiplier} into a stable, sorted tuple of (str(file), float(mult)).
    Useful for cache keys where dict order is otherwise unstable.
    """
    if not d:
        return ()
    return tuple(sorted((str(k), float(v)) for k, v in d.items()))

def normalize_date(value):
    """
    Normalize any dash date value (string/date/None) to 'YYYY-MM-DD' or None.
    Safe to use as part of cache keys.
    """
    if value in (None, "", "null"):
        return None
    try:
        # supports pd.Timestamp, datetime.date, string
        dt = pd.to_datetime(value)
        # strip timezone and time portion; we only cache by date
        return dt.date().isoformat()
    except Exception:
        return None


def kv_tuple(d):
    """
    Convenience alias: stable, sorted tuple of (key, value) pairs for dicts.
    Uses the existing _keyify_contracts_map for {file: multiplier}.
    """
    if not d:
        return ()
    # if it's the contracts map specifically, keep its strict typing
    return _keyify_contracts_map(d)


def _currency_to_float(value):
    """
    Convert common currency-formatted strings (including $ and parentheses) to float.
    Returns NaN when conversion fails.
    """
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return float("nan")
    s = (
        s.replace("$", "")
         .replace(",", "")
         .replace("%", "")
         .replace("\xa0", "")
         .strip()
    )
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except Exception:
        return float("nan")
