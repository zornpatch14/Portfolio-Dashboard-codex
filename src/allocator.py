# src/allocator.py

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import itertools  # used by find_margin_aware_weights

# If any of your pasted funcs need constants/helpers, import them here:
from src.constants import MARGIN_SPEC  # if referenced inside your code
from src.helpers import _symbol_from_first_row  # used inside find_margin_aware_weights



app: Optional[Any] = None





# ----- PASTE THESE FUNCTIONS FROM Portfolio_Dashboard6.py (unchanged) -----
# (Search in your main file for each def name and cut+paste the whole function
#  body including docstrings/comments. Do NOT tweak logic.)
#
# 1) def _portfolio_equity_from_weights(equity_by_file: dict[str, pd.Series],
#                                       weights: dict[str, float]) -> pd.Series
#    - Weighted sum of per-file equity curves.
#
# 2) def _peak_drawdown_value(series: pd.Series) -> float
#    - Returns the most negative drawdown value from an equity-like series.
#
# 3) def _annualized_return(series: pd.Series, initial_capital: float) -> float
#    - Annualizes return between first and last timestamps.
#
# 4) def find_margin_aware_weights(equity_by_file: dict[str, pd.Series],
#                                  per_file_netpos: dict[str, pd.Series],
#                                  store_trades: dict,
#                                  initial_capital: float,
#                                  margin_spec: dict[str, tuple[float, float, float]],
#                                  margin_buffer: float = 0.60,
#                                  objective: str = "max_return_over_dd",
#                                  step: float = 0.25,
#                                  leverage_cap: float = 1.0) -> tuple[dict[str, float], dict]
#    - Your grid/risk-parity search that enforces a margin headroom constraint.
#
# PASTE BELOW THIS LINE
# --------------------------------------------------------------------------


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
                              per_file_margin: dict[str, float] | None = None,
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

    # Interpret leverage_cap safely; we will branch on (lev_cap_int > 0)
    try:
        lev_cap_float = float(leverage_cap) if leverage_cap is not None else 0.0
    except Exception:
        lev_cap_float = 0.0
    lev_cap_int = int(lev_cap_float) if lev_cap_float > 0 else 0

    # Risk-parity (two modes):
    #   A) lev_cap_int > 0  -> keep your existing "project to cap" behavior
    #   B) lev_cap_int == 0 -> NEW: margin-only scaling (no "max contracts" cap)
    if objective == "risk_parity":
        # Build inverse-vol proportions
        vols = []
        for f in files:
            s = equity_by_file[f].diff().dropna()
            vols.append(s.std() if not s.empty else np.nan)
        inv = np.array([0 if (np.isnan(v) or v == 0) else 1.0 / v for v in vols], dtype=float)
        base_w = (inv / inv.sum()) if inv.sum() > 0 else (np.ones(len(files)) / len(files))

        # Helper: compute peak initial margin for an integer weight vector
        def _peak_init_margin_for(w_int_vec: np.ndarray) -> float:
            idx_union = None
            for ser in per_file_netpos.values():
                idx_union = ser.index if idx_union is None else idx_union.union(ser.index)
            if idx_union is None:
                return 0.0
            idx_union = idx_union.sort_values()
            total_margin = pd.Series(0.0, index=idx_union)
            for f, wi in zip(files, w_int_vec):
                if wi <= 0:
                    continue
                ser = per_file_netpos.get(f)
                if ser is None:
                    continue
                df_trades = store_trades.get(f) if store_trades is not None else None
                sym_raw = _symbol_from_first_row(df_trades)
                sym = (sym_raw or "").upper()
                # Per-file override, fallback to symbol spec
                im_override = None
                if per_file_margin:
                    try:
                        im_override = float(per_file_margin.get(f))
                    except Exception:
                        im_override = None
                if im_override is not None and im_override > 0:
                    init_margin = float(im_override)
                else:
                    spec = margin_spec.get(sym)
                    if spec is None:
                        continue
                    init_margin = float(spec[0])
                total_margin = total_margin.add(
                    ser.reindex(idx_union).ffill().fillna(0.0).abs() * init_margin * int(wi),
                    fill_value=0.0
                )
            return float(total_margin.max())

        max_allowed_margin = margin_buffer * initial_capital

        # ===== PATH A: project to a positive max-contracts cap (your existing behavior) =====
        if lev_cap_int > 0:
            # Project to leverage cap as whole contracts (floor), then trim if margin fails
            w = base_w * lev_cap_int
            w_int = np.floor(w).astype(int)

            # Safety: if sum exceeds cap (rare after floor), trim largest until it fits
            while w_int.sum() > lev_cap_int and w_int.sum() > 0:
                i = int(np.argmax(w_int))
                w_int[i] -= 1

            # If margin fails, decrement one contract at a time (largest-first) until it passes
            safe_guard = 100000
            while _peak_init_margin_for(w_int) > max_allowed_margin and w_int.sum() > 0 and safe_guard > 0:
                safe_guard -= 1
                i = int(np.argmax(w_int))
                if w_int[i] > 0:
                    w_int[i] -= 1
                else:
                    pos = np.where(w_int > 0)[0]
                    if pos.size == 0:
                        break
                    w_int[pos[0]] -= 1

            weights = {f: int(wi) for f, wi in zip(files, w_int)}
            if sum(weights.values()) == 0:
                return {}, {"reason": "risk_parity ended at zero due to margin"}

            port = _portfolio_equity_from_weights(equity_by_file, weights)
            ann = _annualized_return(port, initial_capital)
            dd  = abs(_peak_drawdown_value(port))
            score = (ann / dd) if (dd > 0 and not np.isnan(ann)) else -np.inf
            diag = {"ann_return": ann, "dd": -dd, "score": score, "mode": "risk_parity_projected"}
            return weights, diag

        # ===== PATH B: NEW margin-only scaling (no "max contracts" cap) =====
        # Scale the parity vector by a scalar s, floor to integers, and find the largest s
        # such that peak initial margin <= max_allowed_margin.
        def _w_int_from_scale(s: float) -> np.ndarray:
            if s <= 0:
                return np.zeros(len(files), dtype=int)
            w_float = base_w * s
            return np.floor(w_float).astype(int)  # NO Hamilton, by request

        # Exponential bracket for s
        s_lo, s_hi = 0.0, 1.0
        for _ in range(60):  # cap iterations defensively
            w_hi = _w_int_from_scale(s_hi)
            if w_hi.sum() == 0:
                s_hi *= 2.0
                continue
            peak_m = _peak_init_margin_for(w_hi)
            if peak_m <= max_allowed_margin:
                s_lo = s_hi
                s_hi *= 2.0
            else:
                break

        # Binary search for the largest feasible s
        for _ in range(60):
            s_mid = 0.5 * (s_lo + s_hi)
            w_mid = _w_int_from_scale(s_mid)
            if w_mid.sum() == 0:
                s_lo = s_mid
                continue
            peak_m = _peak_init_margin_for(w_mid)
            if peak_m <= max_allowed_margin:
                s_lo = s_mid
            else:
                s_hi = s_mid

        w_best = _w_int_from_scale(s_lo)

        # If zero (e.g., very tight margin), try to allocate 1 contract to the cheapest unit margin at peak
        if w_best.sum() == 0:
            # Estimate a "unit margin" per file at its peak-time participation
            idx_union = None
            for ser in per_file_netpos.values():
                idx_union = ser.index if idx_union is None else idx_union.union(ser.index)
            t_peak = None
            if idx_union is not None:
                idx_union = idx_union.sort_values()
                # Build 1-contract margin per file and take sum to identify an indicative peak time
                total_m = pd.Series(0.0, index=idx_union)
                for f in files:
                    ser = per_file_netpos.get(f)
                    if ser is None:
                        continue
                    df_trades = store_trades.get(f) if store_trades is not None else None
                    sym_raw = _symbol_from_first_row(df_trades)
                    sym = (sym_raw or "").upper()
                    spec = margin_spec.get(sym)
                    if spec is None:
                        continue
                    init_margin = float(spec[0])
                    total_m = total_m.add(ser.reindex(idx_union).ffill().fillna(0.0).abs() * init_margin,
                                          fill_value=0.0)
                if not total_m.empty:
                    t_peak = total_m.idxmax()
            # Pick the minimal unit-margin file at t_peak, if any
            best_idx, best_um = -1, np.inf
            for i, f in enumerate(files):
                ser = per_file_netpos.get(f)
                if ser is None or t_peak is None:
                    continue
                df_trades = store_trades.get(f) if store_trades is not None else None
                sym_raw = _symbol_from_first_row(df_trades)
                sym = (sym_raw or "").upper()
                spec = margin_spec.get(sym)
                if spec is None:
                    continue
                init_margin = float(spec[0])
                val = float(abs(ser.reindex([t_peak]).ffill().fillna(0.0).iloc[0])) if t_peak in ser.index else 0.0
                um = init_margin * val if val > 0 else np.inf
                if um < best_um:
                    best_um = um
                    best_idx = i
            if best_idx >= 0 and np.isfinite(best_um) and best_um <= max_allowed_margin:
                w_best[best_idx] = 1

        weights = {f: int(wi) for f, wi in zip(files, w_best)}
        if sum(weights.values()) == 0:
            return {}, {"reason": "risk_parity margin-only: zero feasible contracts"}

        port = _portfolio_equity_from_weights(equity_by_file, weights)
        ann = _annualized_return(port, initial_capital)
        dd  = abs(_peak_drawdown_value(port))
        score = (ann / dd) if (dd > 0 and np.isfinite(ann)) else -np.inf
        diag = {"ann_return": ann, "dd": -dd, "score": score, "mode": "risk_parity_margin_only"}
        return weights, diag

    # -------------------- Grid search objective (unchanged) --------------------
    # Use integer cap for grid search
    # -------------------- FAST GREEDY / BEAM SEARCH (replaces grid) --------------------
    # Goal: maximize Ann% / MaxDD$ subject to peak margin <= margin_buffer * initial_capital
    # Strategy: add 1 contract at a time where it improves the score and fits margin.
    # Much faster than full grid; usually near-optimal (often identical).

    # Pre-align equity series and unit-margin series on a common index for fast adds
    # Build index union
    idx_union = None
    for s in equity_by_file.values():
        idx_union = s.index if idx_union is None else idx_union.union(s.index)
    for s in per_file_netpos.values():
        idx_union = s.index if idx_union is None else idx_union.union(s.index)
    if idx_union is None or len(idx_union) == 0:
        return {}, {"reason": "no data timeline"}
    idx_union = idx_union.sort_values()

    # Aligned per-file equity increments (one-contract equity curves)
    eq_aligned: dict[str, pd.Series] = {}
    for f, s in equity_by_file.items():
        eq_aligned[f] = s.reindex(idx_union).ffill().fillna(0.0)

    # Aligned per-file unit initial-margin series: |netpos| * IM(symbol)
    um_aligned: dict[str, pd.Series] = {}
    for f, ser in per_file_netpos.items():
        df_trades = store_trades.get(f) if store_trades is not None else None
        sym_raw = _symbol_from_first_row(df_trades)
        sym = (sym_raw or "").upper()
        if per_file_margin and f in per_file_margin and per_file_margin[f] and per_file_margin[f] > 0:
            init_margin = float(per_file_margin[f])
        else:
            spec = margin_spec.get(sym)
            if spec is None:
                continue  # <- skip instead of treating as zero margin
            init_margin = float(spec[0])
        um_aligned[f] = ser.reindex(idx_union).ffill().fillna(0.0).abs() * init_margin

    # Only consider files that have a defined margin series
    alloc_files = [f for f in files if f in um_aligned]
    if not alloc_files:
        return {}, {"reason": "no files with known margin spec"}


    max_allowed_margin = margin_buffer * initial_capital

    # Portfolio state
    weights = {f: 0 for f in alloc_files}
    port_series = pd.Series(0.0, index=idx_union)
    margin_series = pd.Series(0.0, index=idx_union)

    # Helper: compute score quickly from current portfolio series
    def _score_from_series(port: pd.Series) -> tuple[float, float, float, float]:
        """
        Returns:
        ann_pct     : CAGR (unitless fraction, e.g., 0.20)
        dd_usd      : Max drawdown (dollars, positive magnitude)
        ratio       : (Avg Annual $ non-comp) / dd_usd   <-- primary objective
        annual_usd  : Avg Annual $ (non-comp)            <-- secondary objective
        """
        ann_pct = _annualized_return(port, initial_capital)
        dd_usd  = abs(_peak_drawdown_value(port))

        if port is None or port.empty:
            annual_usd = float("nan")
        else:
            start_dt = port.index.min()
            end_dt   = port.index.max()
            years = (end_dt - start_dt).total_seconds() / (365.2425 * 24 * 3600) if (end_dt > start_dt) else None
            total_return = float(port.iloc[-1])
            annual_usd = (total_return / years) if (years and years > 0) else float("nan")

        ratio = (annual_usd / dd_usd) if (dd_usd > 0 and np.isfinite(annual_usd)) else -np.inf
        return ann_pct, dd_usd, ratio, annual_usd




    # Initial score
    best_ann, best_dd, best_ratio, best_annual_usd = _score_from_series(port_series)


    # Greedy loop parameters
    MAX_STEPS = int(1e6)  # practically unbounded; loop will naturally stop when no improvement
    IMPROVEMENT_TOL = 1e-12  # accept only strictly better scores

    # Optional: beam width (set to 1 for pure greedy; >1 to track multiple partial solutions)
    BEAM_WIDTH = 2
    # We’ll keep a small list of states: (weights_dict, port_series, margin_series, ann, dd, score)
    beam = [(weights.copy(), port_series.copy(), margin_series.copy(),
            best_ann, best_dd, best_ratio, best_annual_usd)]




    def _try_add_one(state, f):
        # state: (weights, port_series, margin_series, ann_pct, dd_usd, ratio, annual_usd)
        w, port, marg, ann_prev, dd_prev, ratio_prev, annual_usd_prev = state

        # Enforce total-contracts cap if set (keep this if you still want the cap)
        if lev_cap_int > 0 and (sum(w.values()) + 1) > lev_cap_int:
            return None

        # Feasibility: adding 1 contract of f
        new_marg = marg.add(um_aligned[f], fill_value=0.0)
        if new_marg.max() > max_allowed_margin:
            return None

        new_port = port.add(eq_aligned.get(f, pd.Series(0.0, index=idx_union)), fill_value=0.0)
        ann2, dd2, ratio2, annual_usd2 = _score_from_series(new_port)

        new_w = w.copy()
        new_w[f] = new_w.get(f, 0) + 1
        return (new_w, new_port, new_marg, ann2, dd2, ratio2, annual_usd2)



    for _step in range(MAX_STEPS):
        # Expand each beam state by trying +1 on each file; keep best BEAM_WIDTH next states
        candidates = []
        for state in beam:
            for f in alloc_files:
                cand = _try_add_one(state, f)
                if cand is not None:
                    candidates.append(cand)
        if not candidates:
            break  # no feasible adds from any beam state

        # candidates is a list of tuples matching the beam state shape
        # tuple indices: 0=w, 1=port, 2=marg, 3=ann, 4=dd, 5=ratio, 6=annual_usd
        candidates.sort(
            key=lambda s: (
                -(s[5]),                                         # higher ratio first
                -(s[6] if np.isfinite(s[6]) else -1e300),        # higher annual_usd second
                float(s[2].max()),                               # lower peak margin third (optional)
                -sum(s[0].values()),                             # more contracts last (optional)
            )
        )



        next_beam = []
        seen = set()
        for cand in candidates:
            key = tuple(sorted(cand[0].items()))
            if key in seen:
                continue
            seen.add(key)
            next_beam.append(cand)
            if len(next_beam) >= BEAM_WIDTH:
                break

        best_prev_ratio     = beam[0][5]
        best_prev_annual_usd = beam[0][6]
        best_next           = next_beam[0]
        best_next_ratio     = best_next[5]
        best_next_annual_usd = best_next[6]

        # Accept if ratio strictly improves, OR if ratio ties (within tol) and annual_usd improves
        if (best_next_ratio > best_prev_ratio + IMPROVEMENT_TOL) or (
            abs(best_next_ratio - best_prev_ratio) <= IMPROVEMENT_TOL and
            (np.isfinite(best_next_annual_usd) and np.isfinite(best_prev_annual_usd) and best_next_annual_usd > best_prev_annual_usd)
        ):
            beam = next_beam
        else:
            break



    # Best final state
    best_state = beam[0]
    best_w, best_port, best_marg, best_ann, best_dd, best_ratio, best_annual_usd = best_state

    if all(v == 0 for v in best_w.values()):
        return {}, {"reason": "no feasible nonzero allocation found by greedy"}

    diag = {
        "ann_return": best_ann,
        "dd": -best_dd,
        "score": float(best_ratio),                 # ratio = annual_usd / dd_usd
        "annual_usd": float(best_annual_usd) if np.isfinite(best_annual_usd) else None,
        "mode": "greedy_max_return_over_dd",
        "peak_margin_used": float(best_marg.max()),
        "contracts_total": int(sum(best_w.values())),
    }
    return best_w, diag
