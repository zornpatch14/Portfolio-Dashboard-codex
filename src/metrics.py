# src/metrics.py

from __future__ import annotations
import math
from typing import Dict, Tuple, Optional, Iterable, Any
import numpy as np
import pandas as pd
from datetime import timedelta

# ----- PASTE THESE FUNCTIONS FROM Portfolio_Dashboard6.py (unchanged) -----
# (Cut+paste the entire function bodies. Do NOT tweak logic.)

# 1) def max_consecutive_runs(...):
#    - Longest winning/losing streaks
# 2) def compute_max_held(...):
#    - Max concurrently-held contracts (per symbol or portfolio, depending on your impl)
# 3) def compute_closed_trade_dd(...):
#    - Close-to-close drawdown calc from closed trades
# 4) def compute_intraday_p2v(...):
#    - Peak→Valley drawdown calc using run-up/drawdown columns or equity
# 5) def _union_intervals_coverage(...):
#    - (Helper) coverage of trading intervals/time in market
# 6) def _full_month_bounds(...):
#    - (Helper) month-boundaries for “full months only” stats
# 7) def monthly_stats_full_months(...):
#    - Monthly stats (avg, std) using full months
# 8) def _fmt_td(...):
#    - (Helper) pretty format for timedeltas (if used)
# 9) def compute_metrics(...):
#    - Master metrics aggregator used by your metrics table

# PASTE BELOW THIS LINE
# --------------------------------------------------------------------------

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
    if series is None or series.empty:
        return 0.0, None
    eq = series  # assume equity series passed in (your call site already does this)
    peak = -np.inf
    max_dd = 0.0
    max_time = None
    for t, v in zip(eq.index, eq.values):
        if v > peak:
            peak = v
        dd = peak - v
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
    monthly = s.groupby(pd.Grouper(key="exit_time", freq="ME"))["net_profit"].sum()
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

def compute_metrics(sub: pd.DataFrame, initial_capital: float, daily_returns_pct: Optional[pd.Series] = None) -> dict[str, Any]:
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

    avg_daily_pct = np.nan
    std_daily_pct = np.nan
    sharpe_daily = np.nan
    if daily_returns_pct is not None:
        daily_vals = pd.to_numeric(daily_returns_pct, errors="coerce").dropna()
        if not daily_vals.empty:
            avg_daily_pct = float(daily_vals.mean() * 100.0)
            std_val = daily_vals.std(ddof=0)
            std_daily_pct = float(std_val * 100.0)
            if std_val > 0:
                sharpe_daily = float((daily_vals.mean() / std_val) * math.sqrt(252))

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

        # Daily percent performance
        "Avg. Daily Return (%)": avg_daily_pct,
        "Std. Dev. Daily Return (%)": std_daily_pct,
        "Daily Sharpe (252d)": sharpe_daily,

        # For optional auditing
        "_Monthly Full Series": monthly_full,
    }

