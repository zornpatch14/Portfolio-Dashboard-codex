from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd

from src.constants import DEFAULT_INITIAL_CAPITAL
from src.allocator import (
    _annualized_return,
    _portfolio_equity_from_weights,
    _peak_drawdown_value,
)
from src.equity import max_drawdown_series


# ---------------------------------------------------------------------------
# Module configuration & helpers
# ---------------------------------------------------------------------------

_ARTIFACT_FN: Optional[Callable[[Any], Dict[str, Any]]] = None
_DAILY_RETURNS_FN: Optional[Callable[[Any], Dict[str, pd.Series]]] = None
_DAILY_PNL_FN: Optional[Callable[[Any], Dict[str, pd.Series]]] = None
_INITIAL_CAPITAL: float = float(DEFAULT_INITIAL_CAPITAL)
_LAST_CONFIG: Dict[str, Any] = {}


class RiskfolioAdapterError(RuntimeError):
    """Raised when a recoverable Riskfolio-related error occurs."""


def configure(
    *,
    artifact_fn: Callable[[Any], Dict[str, Any]],
    initial_capital: Optional[float] = None,
    daily_returns_fn: Callable[[Any], Dict[str, pd.Series]],
    daily_pnl_fn: Callable[[Any], Dict[str, pd.Series]],
) -> None:
    """Inject dependencies supplied by the Dash layer once during app start."""

    global _ARTIFACT_FN, _INITIAL_CAPITAL, _DAILY_RETURNS_FN

    if not callable(artifact_fn):
        raise ValueError("artifact_fn must be callable")

    _ARTIFACT_FN = artifact_fn
    if initial_capital is not None and math.isfinite(initial_capital):
        _INITIAL_CAPITAL = float(initial_capital)
    if not callable(daily_returns_fn):
        raise ValueError("daily_returns_fn must be callable")
    _DAILY_RETURNS_FN = daily_returns_fn
    if not callable(daily_pnl_fn):
        raise ValueError("daily_pnl_fn must be callable")
    _DAILY_PNL_FN = daily_pnl_fn


def _require_artifact_fn() -> Callable[[Any], Dict[str, Any]]:
    if _ARTIFACT_FN is None:
        raise RuntimeError("riskfolio_adapter.configure() must be called before use")
    return _ARTIFACT_FN


def _get_artifact(sel_key) -> Dict[str, Any]:
    builder = _require_artifact_fn()
    artifact = builder(sel_key)
    if not isinstance(artifact, dict):
        raise RuntimeError("artifact_fn must return a dict-like object")
    return artifact


def _series_to_daily_pnl(eq: pd.Series) -> pd.Series:
    """Convert an equity curve (cumulative P/L) to daily P/L dollars."""

    if eq is None or eq.empty:
        return pd.Series(dtype=float)

    s = eq.sort_index()
    s.index = pd.to_datetime(s.index)
    delta = s.diff()
    daily = delta.resample("1D").sum().dropna()
    return daily


def _build_returns_matrix(
    equity_by_file: Dict[str, pd.Series],
    daily_override: Optional[Dict[str, pd.Series]] = None,
    pct_override: Optional[Dict[str, pd.Series]] = None,
    *,
    scale_to_pct: bool,
    initial_capital: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Prepare an aligned daily returns matrix from equity curves."""

    rets: list[pd.Series] = []
    symbols: Dict[str, str] = {}

    daily_override = daily_override or {}
    pct_override = pct_override or {}

    for fname, series in equity_by_file.items():
        values: Optional[pd.Series] = None

        if scale_to_pct:
            override_pct = pct_override.get(fname)
            if isinstance(override_pct, pd.Series) and not override_pct.empty:
                values = override_pct.sort_index()
            else:
                override = daily_override.get(fname)
                if isinstance(override, pd.Series) and not override.empty:
                    daily_pnl = override.sort_index()
                else:
                    daily_pnl = _series_to_daily_pnl(series)
                if daily_pnl.empty:
                    continue
                if initial_capital:
                    values = daily_pnl / initial_capital
                else:
                    values = daily_pnl.copy()
        else:
            override = daily_override.get(fname)
            if isinstance(override, pd.Series) and not override.empty:
                daily_pnl = override.sort_index()
            else:
                daily_pnl = _series_to_daily_pnl(series)
            if daily_pnl.empty:
                continue
            values = daily_pnl

        values.name = fname
        rets.append(values)

    if not rets:
        return pd.DataFrame(), {"files": [], "symbols": symbols}

    # Inner join across all assets so optimization shares a common timeline.
    R = pd.concat(rets, axis=1, join="inner").dropna(how="any")
    # Drop constant columns (no variance -> unusable for optimization).
    R = R.loc[:, R.std(ddof=0).replace(0.0, np.nan).notna()]

    # Build metadata (symbol inference best-effort from column name pattern "SYMBOL_*").
    for column in R.columns:
        sym = column.split("_")[0]
        symbols[column] = sym.upper()

    meta = {
        "files": list(R.columns),
        "symbols": symbols,
        "start": R.index.min(),
        "end": R.index.max(),
        "n_assets": R.shape[1],
    }

    return R, meta


def prepare_returns(sel_key, options: Dict[str, Any]):
    """Build the daily returns matrix and accompanying metadata."""

    artifact = _get_artifact(sel_key)

    equity_by_file = {
        k: v for k, v in (artifact.get("equity_by_file") or {}).items() if v is not None and not v.empty
    }
    if not equity_by_file:
        return pd.DataFrame(), {
            "status": "error",
            "error": "no_equity",
            "message": "No equity series available for the current selection.",
        }

    scale_to_pct = bool(options.get("scale_to_pct", False))
    initial_cap = float(options.get("initial_capital", _INITIAL_CAPITAL) or 0.0)

    daily_override = _DAILY_PNL_FN(sel_key) or {}
    pct_override = _DAILY_RETURNS_FN(sel_key) or {}

    returns, meta = _build_returns_matrix(
        equity_by_file,
        daily_override=daily_override,
        pct_override=pct_override,
        scale_to_pct=scale_to_pct,
        initial_capital=initial_cap,
    )

    if returns.empty:
        return returns, {
            "status": "error",
            "error": "empty_returns",
            "message": "Unable to derive daily returns for the selected files.",
        }

    label_map = artifact.get("label_map", {})
    symbol_map_art = artifact.get("symbol_map", {}) or {}
    strategy_map_art = artifact.get("strategy_map", {}) or {}
    interval_map_art = artifact.get("interval_map", {}) or {}

    def _subset(mapping: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in mapping.items() if k in returns.columns}

    symbol_subset = _subset(symbol_map_art)
    strategy_subset = _subset(strategy_map_art)
    interval_subset = _subset(interval_map_art)

    if symbol_subset:
        meta["symbols"] = symbol_subset

    meta.update(
        {
            "status": "ok",
            "label_map": label_map,
            "equity_by_file": equity_by_file,
            "trades_by_file": artifact.get("trades_by_file", {}),
            "symbol_map": symbol_subset,
            "strategy_map": strategy_subset,
            "interval_map": interval_subset,
        }
    )

    return returns, meta


# ---------------------------------------------------------------------------
# Riskfolio orchestration
# ---------------------------------------------------------------------------

_RM_ALIASES = {
    "variance": "MV",
    "mean_variance": "MV",
    "semi_variance": "MSV",
    "semivariance": "MSV",
    "mad": "MAD",
    "cvar": "CVaR",
    "expected_shortfall": "CVaR",
    "cdar": "CDaR",
    "evar": "EVaR",
}

_COV_ALIASES = {
    "sample": "hist",
    "historical": "hist",
    "ewma": "ewma",
    "ledoit": "ledoit",
    "ledoit-wolf": "ledoit",
}


def _select_risk_measure(config: Dict[str, Any]) -> Tuple[str, float]:
    rm_raw = str(config.get("risk_measure", config.get("model", "variance"))).lower()
    rm_code = _RM_ALIASES.get(rm_raw, "MV")
    alpha = float(config.get("alpha", 0.95))
    alpha = min(max(alpha, 0.5), 0.999)
    return rm_code, alpha


def _select_covariance(config: Dict[str, Any]) -> str:
    cov_raw = str(config.get("covariance", "sample")).lower()
    return _COV_ALIASES.get(cov_raw, "hist")


def _normalize_objective(value: Any) -> str:
    """Return a Riskfolio objective keyword from various user-facing labels."""
    if value is None:
        return "Sharpe"
    text = str(value).strip()
    if not text:
        return "Sharpe"
    key = text.replace("-", "").replace("_", "").lower()
    mapping = {
        "sharpe": "Sharpe",
        "maxratio": "Sharpe",
        "maxriskadjustedreturnratio": "Sharpe",
        "minrisk": "MinRisk",
        "minimumrisk": "MinRisk",
        "maxret": "MaxRet",
        "maximumreturn": "MaxRet",
        "utility": "Utility",
    }
    return mapping.get(key, text)


def _portfolio_accepts_kw_alpha(port_cls) -> bool:
    """Return True if Portfolio.optimization accepts 'a' (alpha) keyword."""
    try:
        sig = inspect.signature(port_cls.optimization)
        return "a" in sig.parameters
    except (ValueError, TypeError, AttributeError):
        return True


def _apply_bounds_constraints(port, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> None:
    """Translate per-asset bounds into inequality constraints for Riskfolio >=7."""
    n = lower_bounds.size
    lb_vec = np.asarray(lower_bounds, dtype=float).reshape(n, 1)
    ub_vec = np.asarray(upper_bounds, dtype=float).reshape(n, 1)

    # Stack +/- identity to enforce lb ≤ w ≤ ub.
    a_mat = np.vstack((np.eye(n), -np.eye(n)))
    b_vec = np.vstack((ub_vec, -lb_vec))

    port.ainequality = a_mat
    port.binequality = b_vec

    # Configure high-level long/short flags so other constraints stay consistent.
    allow_short = bool((lb_vec < 0).any())
    port.sht = allow_short
    if allow_short:
        max_short = float(np.abs(lb_vec).max())
        port.uppersht = max_short
        port.budgetsht = max(float(getattr(port, "budgetsht", 0.2)), float(np.abs(lb_vec[lb_vec < 0]).sum()))
    else:
        port.uppersht = 0.0
        port.budgetsht = 0.0
    port.upperlng = float(ub_vec.max())


def _vector_from_mapping(mapping: Dict[str, float], columns: Iterable[str]) -> Optional[np.ndarray]:
    if not mapping:
        return None
    vec = []
    for col in columns:
        vec.append(float(mapping.get(col, 0.0)))
    return np.asarray(vec, dtype=float)


def _build_assets_constraints_tables(rules: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Translate normalized constraint rules into Riskfolio asset_classes and constraints tables.
    """
    assets: List[str] = [str(a) for a in (rules.get("assets") or [])]
    asset_bounds: Dict[str, Dict[str, Optional[float]]] = rules.get("asset_bounds") or {}
    symbol_map: Dict[str, Optional[str]] = rules.get("symbol_map") or {}
    strategy_map: Dict[str, Optional[str]] = rules.get("strategy_map") or {}
    symbol_caps: Dict[str, Dict[str, Optional[float]]] = rules.get("symbol_caps") or {}
    strategy_caps: Dict[str, Dict[str, Optional[float]]] = rules.get("strategy_caps") or {}

    data: Dict[str, List[Optional[str]]] = {"Assets": assets}
    if symbol_map:
        data["Symbol"] = [symbol_map.get(a) for a in assets]
    if strategy_map:
        data["Strategy"] = [strategy_map.get(a) for a in assets]

    asset_classes_df = pd.DataFrame(data)

    rows: List[Dict[str, Any]] = []

    def _add_row(row: Dict[str, Any]) -> None:
        rows.append(
            {
                "Disabled": False,
                "Type": row.get("Type", ""),
                "Set": row.get("Set", ""),
                "Position": row.get("Position", ""),
                "Sign": row.get("Sign", ""),
                "Weight": row.get("Weight"),
                "Type Relative": "",
                "Relative Set": "",
                "Relative": "",
                "Factor": "",
            }
        )

    for asset in assets:
        bounds = asset_bounds.get(asset) or {}
        min_bound = bounds.get("min")
        max_bound = bounds.get("max")
        if min_bound is not None and np.isfinite(min_bound):
            _add_row({"Type": "Assets", "Position": asset, "Sign": ">=", "Weight": float(min_bound)})
        if max_bound is not None and np.isfinite(max_bound):
            _add_row({"Type": "Assets", "Position": asset, "Sign": "<=", "Weight": float(max_bound)})

    for sym, info in symbol_caps.items():
        max_cap = info.get("max")
        min_cap = info.get("min")
        position = str(sym)
        if min_cap is not None and np.isfinite(min_cap):
            _add_row({"Type": "Classes", "Set": "Symbol", "Position": position, "Sign": ">=", "Weight": float(min_cap)})
        if max_cap is not None and np.isfinite(max_cap):
            _add_row({"Type": "Classes", "Set": "Symbol", "Position": position, "Sign": "<=", "Weight": float(max_cap)})

    for strat, info in strategy_caps.items():
        max_cap = info.get("max")
        min_cap = info.get("min")
        position = str(strat)
        if min_cap is not None and np.isfinite(min_cap):
            _add_row({"Type": "Classes", "Set": "Strategy", "Position": position, "Sign": ">=", "Weight": float(min_cap)})
        if max_cap is not None and np.isfinite(max_cap):
            _add_row({"Type": "Classes", "Set": "Strategy", "Position": position, "Sign": "<=", "Weight": float(max_cap)})

    constraints_df = pd.DataFrame(rows, columns=[
        "Disabled",
        "Type",
        "Set",
        "Position",
        "Sign",
        "Weight",
        "Type Relative",
        "Relative Set",
        "Relative",
        "Factor",
    ]) if rows else pd.DataFrame(columns=[
        "Disabled",
        "Type",
        "Set",
        "Position",
        "Sign",
        "Weight",
        "Type Relative",
        "Relative Set",
        "Relative",
        "Factor",
    ])

    return asset_classes_df, constraints_df


def _series_from_weights(weights_obj: Any, columns: Iterable[str]) -> pd.Series:
    """Coerce Riskfolio weights output into a 1D Series aligned to columns."""
    cols = list(columns)
    if isinstance(weights_obj, pd.DataFrame):
        if weights_obj.shape[1] == 1:
            data = weights_obj.iloc[:, 0].reindex(cols).fillna(0.0)
        else:
            # Prefer column named 'weights' when available.
            if "weights" in weights_obj.columns:
                data = weights_obj["weights"].reindex(cols).fillna(0.0)
            else:
                data = weights_obj.iloc[:, 0].reindex(cols).fillna(0.0)
        return pd.Series(data.values, index=cols, dtype=float)
    if isinstance(weights_obj, pd.Series):
        return weights_obj.reindex(cols).fillna(0.0).astype(float)
    arr = np.asarray(weights_obj).reshape(-1)
    return pd.Series(arr[: len(cols)], index=cols, dtype=float)


def _risk_contribution_from_portfolio(port, weights: pd.Series, rm_code: str, alpha: float, returns: pd.DataFrame) -> pd.Series:
    """Compute per-asset risk contribution using Riskfolio's helper."""
    try:
        from riskfolio.src import RiskFunctions as rk  # type: ignore
    except Exception:
        return pd.Series(dtype=float)

    if weights is None or weights.empty:
        return pd.Series(dtype=float)

    w = pd.DataFrame(weights.values.reshape(-1, 1), index=weights.index)
    try:
        rc = rk.Risk_Contribution(
            w=w,
            returns=returns,
            cov=getattr(port, "cov", None),
            rm=rm_code,
            rf=float(getattr(port, "rf", 0.0) if hasattr(port, "rf") else 0.0),
            alpha=float(alpha),
            a_sim=float(getattr(port, "a_sim", 100)),
            beta=getattr(port, "beta", None),
            b_sim=getattr(port, "b_sim", None),
            kappa=float(getattr(port, "_kappa", 0.3)),
            kappa_g=getattr(port, "_kappa_g", None),
            solver="CLARABEL",
        )
        rc_flat = np.array(rc).reshape(-1)
        return pd.Series(rc_flat, index=weights.index)
    except Exception:
        return pd.Series(dtype=float)


def run_optimization(R: pd.DataFrame, config: Dict[str, Any], prev_weights: Optional[Dict[str, float]] = None):
    """Execute the chosen Riskfolio optimization workflow."""

    global _LAST_CONFIG

    if R is None or R.empty:
        return {"status": "error", "error": "empty_returns", "message": "Returns matrix is empty."}

    try:
        import riskfolio as rp
    except Exception as exc:  # pragma: no cover - dependency may be optional in CI
        return {
            "status": "error",
            "error": "import_error",
            "message": f"Riskfolio-Lib is not available: {exc}",
        }

    supports_kw_alpha = _portfolio_accepts_kw_alpha(rp.Portfolio)

    portfolio_kwargs = config.get("portfolio_kwargs") or {}
    model = str(config.get("model", "mean_variance")).lower()
    rm_code, alpha = _select_risk_measure(config)
    cov_method = _select_covariance(config)
    rf_rate = float(portfolio_kwargs.get("risk_free", config.get("risk_free", 0.0)))
    budget = float(config.get("budget", 1.0))
    periods_per_year = float(config.get("periods_per_year", 252))
    kelly_mode = portfolio_kwargs.get("kelly", config.get("kelly"))
    hist_flag = portfolio_kwargs.get("hist", config.get("hist", True))
    objective_value = portfolio_kwargs.get("objective", config.get("objective", "Sharpe"))
    objective_value = _normalize_objective(objective_value)
    risk_aversion = portfolio_kwargs.get("risk_aversion", config.get("risk_aversion"))
    portfolio_model = portfolio_kwargs.get("model", "Classic")
    if hist_flag is not None:
        hist_flag = bool(hist_flag)
    constraint_rules = config.get("constraint_rules") or {}

    columns = list(R.columns)
    prev_vec = _vector_from_mapping(prev_weights or {}, columns)

    asset_classes_df, constraints_df = _build_assets_constraints_tables(constraint_rules)
    constraints_tuple: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)
    if not asset_classes_df.empty and not constraints_df.empty:
        try:
            A_mat, B_vec = rp.ConstraintsFunctions.assets_constraints(constraints_df, asset_classes_df)
            constraints_tuple = (np.asarray(A_mat, dtype=float), np.asarray(B_vec, dtype=float))
        except Exception:
            constraints_tuple = (None, None)

    frontier: Optional[pd.DataFrame] = None
    weights_series: Optional[pd.Series] = None
    risk_contrib: Optional[pd.Series] = None
    ex_ante: Dict[str, Any] = {}

    try:
        if model in {"mean-variance", "mad", "cvar", "cdar", "evar", "mean_variance"}:
            port = rp.Portfolio(returns=R)
            port.assets_stats(method_mu="hist", method_cov=cov_method)
            port.budget = budget
            if prev_vec is not None:
                port.w_prev = prev_vec

            opt_kwargs = {
                "model": portfolio_model,
                "rm": rm_code,
                "obj": objective_value,
                "rf": rf_rate,
                "kelly": kelly_mode,
                "hist": hist_flag,
            }
            if kelly_mode is None:
                opt_kwargs.pop("kelly")
            if hist_flag is None:
                opt_kwargs.pop("hist")

            if supports_kw_alpha:
                opt_kwargs["a"] = alpha
            else:
                port.alpha = float(alpha)

            A_mat, B_vec = constraints_tuple
            if A_mat is not None and B_vec is not None:
                port.ainequality = A_mat
                port.binequality = B_vec

            if objective_value == "Utility" and risk_aversion is not None:
                opt_kwargs["l"] = float(risk_aversion)

            weights = port.optimization(**opt_kwargs)
            weights_series = _series_from_weights(weights, columns)

            risk_contrib = _risk_contribution_from_portfolio(port, weights_series, rm_code, alpha, R)

            # Efficient frontier only for MV-like objectives.
            if rm_code in {"MV", "CVaR", "CDaR"}:
                try:
                    frontier = port.efficient_frontier(rm=rm_code, points=int(config.get("frontier_points", 25)))
                except Exception:
                    frontier = None

        elif model in {"risk parity", "risk_parity", "erc"}:
            port = rp.Portfolio(returns=R)
            port.assets_stats(method_mu="hist", method_cov=cov_method)
            port.budget = budget
            rp_kwargs = {
                "model": "Classic",
                "rm": rm_code,
                "obj": "RiskParity",
                "rf": rf_rate,
            }

            if supports_kw_alpha:
                rp_kwargs["a"] = alpha
            else:
                port.alpha = float(alpha)

            A_mat, B_vec = constraints_tuple
            if A_mat is not None and B_vec is not None:
                port.ainequality = A_mat
                port.binequality = B_vec

            weights = port.optimization(**rp_kwargs)
            weights_series = _series_from_weights(weights, columns)
            risk_contrib = _risk_contribution_from_portfolio(port, weights_series, rm_code, alpha, R)

        elif model in {"hrp", "herc"}:
            hc = rp.HCPortfolio(returns=R)
            hc.covariance = cov_method
            codependence = str(config.get("codependence", "pearson"))
            linkage = str(config.get("linkage", "ward"))
            model_name = "HRP" if model == "hrp" else "HERC"
            weights = hc.optimization(
                model=model_name,
                codependence=codependence,
                covariance=cov_method,
                linkage=linkage,
                rf=rf_rate,
            )
            weights_series = _series_from_weights(weights, columns)

            temp_port = rp.Portfolio(returns=R)
            temp_port.assets_stats(method_mu="hist", method_cov=cov_method)
            risk_contrib = _risk_contribution_from_portfolio(temp_port, weights_series, rm_code, alpha, R)

        else:
            return {
                "status": "error",
                "error": "unknown_model",
                "message": f"Unsupported model '{model}'.",
            }

    except Exception as exc:
        return {
            "status": "error",
            "error": "optimization_failed",
            "message": str(exc),
        }

    if weights_series is None or weights_series.empty:
        return {
            "status": "error",
            "error": "no_solution",
            "message": "Optimization did not return any weights.",
        }

    weights_series = weights_series.astype(float)
    total_weight = float(weights_series.sum())
    if not math.isfinite(total_weight) or abs(total_weight) < 1e-12:
        total_weight = 1.0
    weights_series = weights_series / total_weight

    # Compute ex-ante statistics using the sample mean / covariance.
    port_returns = R.dot(weights_series)
    mu_port = port_returns.mean() * periods_per_year
    sigma_port = port_returns.std(ddof=0) * math.sqrt(periods_per_year)
    sharpe = mu_port / sigma_port if sigma_port > 0 else np.nan
    if rm_code in {"CVaR", "CDaR"}:
        tail = port_returns.sort_values()
        cutoff = int(max(1, math.floor((1 - alpha) * len(tail))))
        cvar_port = -tail.head(cutoff).mean() * periods_per_year if cutoff > 0 else np.nan
    else:
        cvar_port = np.nan

    ex_ante = {
        "expected_return": float(mu_port),
        "expected_volatility": float(sigma_port),
        "sharpe": float(sharpe) if math.isfinite(sharpe) else np.nan,
        "cvar": float(cvar_port) if math.isfinite(cvar_port) else np.nan,
    }

    _LAST_CONFIG = config.copy()

    result = {
        "status": "ok",
        "weights": weights_series,
        "risk_contribution": risk_contrib if risk_contrib is not None else pd.Series(dtype=float),
        "ex_ante": ex_ante,
        "frontier": frontier,
        "portfolio_returns": port_returns,
        "config_used": config,
    }

    return result


# ---------------------------------------------------------------------------
# Backtests & diagnostics
# ---------------------------------------------------------------------------


def _weights_from_input(weights: Any) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
    if isinstance(weights, dict):
        if all(isinstance(v, (int, float)) for v in weights.values()):
            return {k: float(v) for k, v in weights.items()}, None
        if "weights" in weights:
            inner = weights.get("weights") or {}
            cfg = weights.get("config") or weights.get("config_used")
            return {k: float(v) for k, v in inner.items()}, cfg
    raise ValueError("weights must be a mapping of asset -> weight")


def compute_ex_post_metrics(weights, sel_key):
    """Evaluate the optimized portfolio on historical equity curves."""

    weight_map, cfg = _weights_from_input(weights)
    artifact = _get_artifact(sel_key)
    equity_by_file = artifact.get("equity_by_file", {}) or {}

    if not equity_by_file:
        return {
            "status": "error",
            "error": "no_equity",
            "message": "No equity data available for ex-post evaluation.",
        }

    portfolio_equity = _portfolio_equity_from_weights(equity_by_file, weight_map)
    if portfolio_equity.empty:
        return {
            "status": "error",
            "error": "empty_equity",
            "message": "Weighted equity curve is empty.",
        }

    ann_return = _annualized_return(portfolio_equity, _INITIAL_CAPITAL)
    max_dd_value = _peak_drawdown_value(portfolio_equity)

    daily_pl = portfolio_equity.diff().dropna()
    daily_ret = daily_pl / _INITIAL_CAPITAL if _INITIAL_CAPITAL else daily_pl
    mean_ret = daily_ret.mean()
    std_ret = daily_ret.std(ddof=0)
    sharpe = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 0 else np.nan

    alpha = float((cfg or _LAST_CONFIG or {}).get("alpha", 0.95))
    alpha = min(max(alpha, 0.5), 0.999)
    if not daily_ret.empty:
        cutoff = int(max(1, math.floor((1 - alpha) * len(daily_ret))))
        cvar = -daily_ret.sort_values().head(cutoff).mean() if cutoff > 0 else np.nan
    else:
        cvar = np.nan

    metrics = {
        "status": "ok",
        "annual_return": float(ann_return) if math.isfinite(ann_return) else np.nan,
        "max_drawdown": float(max_dd_value),
        "annual_sharpe": float(sharpe) if math.isfinite(sharpe) else np.nan,
        "cvar": float(cvar) if math.isfinite(cvar) else np.nan,
    }

    return {
        "metrics": metrics,
        "equity": portfolio_equity,
        "daily_returns": daily_ret,
    }


# ---------------------------------------------------------------------------
# Convenience dataclasses (optional for Dash serialization helpers)
# ---------------------------------------------------------------------------


@dataclass
class RiskfolioProgress:
    step: int
    total: int
    label: str


__all__ = [
    "configure",
    "prepare_returns",
    "run_optimization",
    "compute_ex_post_metrics",
    "RiskfolioAdapterError",
    "RiskfolioProgress",
]
