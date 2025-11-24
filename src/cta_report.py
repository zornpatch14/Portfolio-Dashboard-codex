from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class CTAMonthlyTable:
    """Container for monthly/annual performance tables."""
    monthly_additive: pd.Series
    monthly_compounded: pd.Series
    matrix: pd.DataFrame            # Display table (years x months + YTD/Annual)
    monthly_long: pd.DataFrame      # Long-form table for charts
    annual_additive: pd.Series
    annual_compounded: pd.Series
    ytd_additive: float
    ytd_compounded: float
    last_year: Optional[int]


@dataclass
class CTASummary:
    """Key capsule statistics required by Rule 4.35."""
    start_date: Optional[pd.Timestamp]
    end_date: Optional[pd.Timestamp]
    largest_monthly_loss: Optional[Dict[str, Any]]
    best_month: Optional[Dict[str, Any]]
    worst_peak_to_valley_additive: Optional[Dict[str, Any]]
    worst_peak_to_valley_compounded: Optional[Dict[str, Any]]
    annual_ror_additive: pd.Series
    annual_ror_compounded: pd.Series
    ytd_ror_additive: float
    ytd_ror_compounded: float
    supplemental_metrics: Dict[str, Any]


@dataclass
class CTAReportResult:
    """Full CTA report payload used by the UI."""
    daily_returns: pd.Series
    nav_compounded: pd.Series
    nav_additive: pd.Series
    monthly: CTAMonthlyTable
    summary: CTASummary
    rolling_series: Dict[str, pd.Series]


def _normalise_daily_returns(daily_returns: pd.Series) -> pd.Series:
    if daily_returns is None:
        return pd.Series(dtype=float)
    series = pd.to_numeric(daily_returns, errors="coerce")
    index = pd.to_datetime(series.index).tz_localize(None)
    series.index = index
    series = series.sort_index()
    return series[~series.index.duplicated(keep="last")].fillna(0.0)


def _nav_from_daily(daily: pd.Series, initial: float = 100.0) -> pd.Series:
    if daily.empty:
        return pd.Series(dtype=float)
    nav = (1.0 + daily).cumprod() * float(initial)
    nav.name = "nav_compounded"
    return nav


def _nav_additive_from_daily(daily: pd.Series, initial: float = 100.0) -> pd.Series:
    if daily.empty:
        return pd.Series(dtype=float)
    # Non-reinvested NAV: start * (1 + cumulative simple return)
    nav = float(initial) * (1.0 + daily.cumsum())
    nav.name = "nav_additive"
    return nav


def _monthly_returns(daily: pd.Series) -> tuple[pd.Series, pd.Series]:
    if daily.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    groups = daily.groupby(pd.Grouper(freq="ME"))
    additive = groups.sum()
    compounded = groups.apply(lambda s: (1.0 + s).prod() - 1.0)
    additive.name = "monthly_additive"
    compounded.name = "monthly_compounded"
    return additive, compounded


def _monthly_long_frame(monthly_add: pd.Series,
                        monthly_comp: pd.Series) -> pd.DataFrame:
    index = monthly_add.index.union(monthly_comp.index)
    if len(index) == 0:
        return pd.DataFrame(columns=["month_end", "return_additive",
                                     "return_compounded", "year", "month",
                                     "month_name"])
    df = pd.DataFrame(index=index)
    df["return_additive"] = monthly_add.reindex(index)
    df["return_compounded"] = monthly_comp.reindex(index)
    df = df.reset_index()
    first_col = df.columns[0]
    if first_col != "month_end":
        df = df.rename(columns={first_col: "month_end"})
    df["month_end"] = pd.to_datetime(df["month_end"]).dt.normalize()
    df["year"] = df["month_end"].dt.year
    df["month"] = df["month_end"].dt.month
    df["month_name"] = df["month"].apply(lambda m: MONTH_NAMES[m - 1])
    return df


def _monthly_matrix(monthly_add: pd.Series,
                    monthly_comp: pd.Series) -> CTAMonthlyTable:
    if monthly_add.empty and monthly_comp.empty:
        empty_matrix = pd.DataFrame(columns=MONTH_NAMES + ["YTD", "Annual"])
        return CTAMonthlyTable(
            monthly_additive=pd.Series(dtype=float),
            monthly_compounded=pd.Series(dtype=float),
            matrix=empty_matrix,
            monthly_long=pd.DataFrame(columns=["month_end", "return_additive",
                                               "return_compounded", "year", "month",
                                               "month_name"]),
            annual_additive=pd.Series(dtype=float),
            annual_compounded=pd.Series(dtype=float),
            ytd_additive=float("nan"),
            ytd_compounded=float("nan"),
            last_year=None,
        )

    df = monthly_add.to_frame("value").copy()
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    pivot = df.pivot(index="Year", columns="Month", values="value")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = MONTH_NAMES
    pivot = pivot.sort_index()

    annual_additive = monthly_add.groupby(monthly_add.index.year).sum()
    annual_compounded = monthly_comp.groupby(monthly_comp.index.year).apply(
        lambda s: (1.0 + s).prod() - 1.0
    )
    ytd = pivot.sum(axis=1, skipna=True)
    pivot["YTD"] = ytd
    pivot["Annual"] = annual_additive.reindex(pivot.index)

    last_year = int(pivot.index.max()) if len(pivot.index) else None
    ytd_additive = float(ytd.loc[last_year]) if last_year in ytd else float("nan")
    if last_year is not None and last_year in annual_compounded.index:
        months_current_year = monthly_comp.loc[monthly_comp.index.year == last_year]
        ytd_compounded = (1.0 + months_current_year).prod() - 1.0
    else:
        ytd_compounded = float("nan")

    monthly_long = _monthly_long_frame(monthly_add, monthly_comp)

    return CTAMonthlyTable(
        monthly_additive=monthly_add,
        monthly_compounded=monthly_comp,
        matrix=pivot,
        monthly_long=monthly_long,
        annual_additive=annual_additive,
        annual_compounded=annual_compounded,
        ytd_additive=ytd_additive,
        ytd_compounded=float(ytd_compounded) if not math.isnan(ytd_compounded) else float("nan"),
        last_year=last_year,
    )


def _largest_monthly_drawdown(monthly_add: pd.Series) -> Optional[Dict[str, Any]]:
    if monthly_add.empty:
        return None
    idx = monthly_add.idxmin()
    value = float(monthly_add.loc[idx])
    return {
        "value": value,
        "month": idx.strftime("%Y-%m"),
    }


def _best_month(monthly_add: pd.Series) -> Optional[Dict[str, Any]]:
    if monthly_add.empty:
        return None
    idx = monthly_add.idxmax()
    value = float(monthly_add.loc[idx])
    return {
        "value": value,
        "month": idx.strftime("%Y-%m"),
    }


def _peak_to_valley_from_series(series: pd.Series) -> Optional[Dict[str, Any]]:
    if series.empty:
        return None
    running_max = series.cummax()
    drawdown = series - running_max
    min_drawdown = drawdown.min()
    if min_drawdown >= 0:
        return {
            "value": 0.0,
            "peak": None,
            "trough": None,
            "duration_months": 0,
        }
    trough = drawdown.idxmin()
    peak_value = running_max.loc[trough]
    peak_candidates = running_max.loc[:trough]
    peak_candidates = peak_candidates[peak_candidates == peak_value]
    peak = peak_candidates.index[-1]
    duration = (trough.year - peak.year) * 12 + (trough.month - peak.month)
    return {
        "value": float(min_drawdown),
        "peak": peak.strftime("%Y-%m"),
        "trough": trough.strftime("%Y-%m"),
        "peak_timestamp": peak,
        "trough_timestamp": trough,
        "duration_months": int(duration),
        "peak_level": float(series.loc[peak]),
        "trough_level": float(series.loc[trough]),
    }


def _peak_to_valley_additive(monthly_add: pd.Series) -> Optional[Dict[str, Any]]:
    if monthly_add.empty:
        return None
    cumulative = monthly_add.cumsum()
    return _peak_to_valley_from_series(cumulative)


def _peak_to_valley_compounded(monthly_comp: pd.Series) -> Optional[Dict[str, Any]]:
    if monthly_comp.empty:
        return None
    nav = (1.0 + monthly_comp).cumprod()
    drawdown_info = _peak_to_valley_from_series(nav)
    if drawdown_info:
        peak_level = drawdown_info.get("peak_level")
        trough_level = drawdown_info.get("trough_level")
        if peak_level and trough_level:
            drawdown_info["value"] = float((trough_level / peak_level) - 1.0)
    return drawdown_info


def _supplemental_metrics(daily: pd.Series,
                          monthly_add: pd.Series,
                          monthly_comp: pd.Series) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if not daily.empty:
        mean = float(daily.mean())
        std = float(daily.std(ddof=0))
        downside = float(daily[daily < 0].std(ddof=0))
        annual_return = mean * 252.0
        annual_vol = std * math.sqrt(252.0) if std > 0 else float("nan")
        sharpe = (annual_return / annual_vol) if annual_vol > 0 else float("nan")
        sortino = (annual_return / (downside * math.sqrt(252.0))) if downside > 0 else float("nan")
        var_95 = float(np.quantile(daily, 0.05))
        cvar_95 = float(daily[daily <= var_95].mean()) if not np.isnan(var_95) else float("nan")
        metrics.update({
            "annualised_return_daily": annual_return,
            "annualised_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "var_95": var_95,
            "cvar_95": cvar_95,
        })
    if not monthly_add.empty:
        total_months = len(monthly_add)
        positive_months = int((monthly_add > 0).sum())
        metrics.update({
            "total_months": total_months,
            "positive_months": positive_months,
            "percent_positive_months": (positive_months / total_months) if total_months else float("nan"),
            "average_monthly_return_additive": float(monthly_add.mean()),
        })
    if not monthly_comp.empty:
        metrics["average_monthly_return_compounded"] = float(monthly_comp.mean())
    return metrics


def _rolling_series(monthly_add: pd.Series,
                    monthly_comp: pd.Series,
                    window: int = 12) -> Dict[str, pd.Series]:
    result: Dict[str, pd.Series] = {}
    if not monthly_add.empty:
        rolling_add = monthly_add.rolling(window=window).sum()
        result["rolling_sum_additive"] = rolling_add
    if not monthly_comp.empty:
        rolling_comp = monthly_comp.rolling(window=window).apply(
            lambda s: (1.0 + s).prod() - 1.0, raw=False
        )
        result["rolling_compounded"] = rolling_comp
    return result


def build_cta_report(daily_returns: pd.Series,
                     *,
                     nav_start: float = 100.0) -> CTAReportResult:
    daily = _normalise_daily_returns(daily_returns)
    nav_comp = _nav_from_daily(daily, initial=nav_start)
    nav_add = _nav_additive_from_daily(daily, initial=nav_start)
    monthly_add, monthly_comp = _monthly_returns(daily)
    monthly_table = _monthly_matrix(monthly_add, monthly_comp)

    largest_monthly_loss = _largest_monthly_drawdown(monthly_add)
    best_month = _best_month(monthly_add)
    worst_pv_add = _peak_to_valley_additive(monthly_add)
    worst_pv_comp = _peak_to_valley_compounded(monthly_comp)

    supplemental = _supplemental_metrics(daily, monthly_add, monthly_comp)
    rolling = _rolling_series(monthly_add, monthly_comp)

    summary = CTASummary(
        start_date=daily.index.min() if not daily.empty else None,
        end_date=daily.index.max() if not daily.empty else None,
        largest_monthly_loss=largest_monthly_loss,
        best_month=best_month,
        worst_peak_to_valley_additive=worst_pv_add,
        worst_peak_to_valley_compounded=worst_pv_comp,
        annual_ror_additive=monthly_table.annual_additive,
        annual_ror_compounded=monthly_table.annual_compounded,
        ytd_ror_additive=monthly_table.ytd_additive,
        ytd_ror_compounded=monthly_table.ytd_compounded,
        supplemental_metrics=supplemental,
    )

    return CTAReportResult(
        daily_returns=daily,
        nav_compounded=nav_comp,
        nav_additive=nav_add,
        monthly=monthly_table,
        summary=summary,
        rolling_series=rolling,
    )

