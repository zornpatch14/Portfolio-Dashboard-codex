#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Trade Analysis Dashboard — Single File
================================================

CHANGELOG (2025-10-03)
----------------------
- Integrated "Trades List"-only metrics directly into this single file.
- Added filename parsing: symbol, timeframe (interval), strategy (regex + fallbacks).
- New Controls:
    • Symbols (multi-select), Timeframes (multi-select), Strategies (multi-select)
    • Direction radio (All / Long / Short)
- Charts unchanged in spirit but now reflect ALL filters (files + symbol + interval + strategy + direction + date).
- Metrics table:
    • Adds: Total Commission, Total Slippage, Account Size Required (abs max close-to-close DD),
            Intra-Day Peak?Valley Drawdown Value/Date (from Run-up/Drawdown columns),
            plus the existing P&L stats.
    • “Portfolio” row also includes: ROI, Annual ROR, ROA, Avg/Std Monthly Return (full months only),
      Trading Period, % Time in Market, Time in Market, Longest Flat Period, Max Equity Run-up and %.
- Exports:
    • Trades CSV now includes Symbol, Interval, Strategy, Direction, etc.
    • Metrics CSV includes new fields above.
- Notes:
    • Assumes two-row-per-trade format as exported by TradeStation (“Trades List” sheet).
    • No timezone handling, per user instruction.
"""

from __future__ import annotations

import base64, io, os, re, webbrowser, itertools, math, hashlib
import logging
from functools import lru_cache
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from threading import Lock
from typing import Any, Iterable, Tuple, Dict, Optional, Callable, List
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update, ctx
from flask_caching import Cache
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

_manager_ctor: Optional[Tuple[str, Any]] = None
_manager_kwarg_name: Optional[str] = None

try:
    import diskcache  # type: ignore

    try:
        from dash.background_callback import DiskcacheManager  # type: ignore[attr-defined]
    except ImportError:
        from dash.long_callback import DiskcacheLongCallbackManager  # type: ignore[import]

        _manager_ctor = ("long_callback_manager", DiskcacheLongCallbackManager)
        _manager_kwarg_name = "long_callback_manager"
    else:
        _manager_ctor = ("background_callback_manager", DiskcacheManager)
        _manager_kwarg_name = "background_callback_manager"
except Exception:  # pragma: no cover - optional dependency
    diskcache = None
    _manager_ctor = None
    _manager_kwarg_name = None

from src.constants import APP_TITLE, COLOR_PORTFOLIO, SHEET_HINTS, DEFAULT_INITIAL_CAPITAL, MARGIN_SPEC, MARGIN_BUFFER, FILENAME_RE
from src.etl import parse_tradestation_trades, parse_filename_meta, decode_upload, canonicalize_columns, load_sheet_guess, find_header_row
from src.equity import equity_from_trades_subset, combine_equity, max_drawdown_series, max_drawdown_value, intraday_drawdown_series
from src.metrics import max_consecutive_runs, compute_max_held, compute_closed_trade_dd, compute_intraday_p2v, monthly_stats_full_months, compute_metrics
from src.corr import _to_daily_equity, _drawdown_pct, _zscore, _rolling_slope, _compute_pl_dataframe_from_equity, build_correlation_heatmap
from src.netpos import _aggregate_netpos_per_symbol_from_series, _netpos_figure_from_series, _netpos_timeseries_from_store, build_net_contracts_figure
from src.helpers import _symbol_from_first_row, _within_dates, _display_label_from_df, _safe_unique_name, _files_key, _keyify_list, _keyify_contracts_map, normalize_date
from src.margin import _purchasing_power_series, _margin_series
from src.allocator import _portfolio_equity_from_weights, _peak_drawdown_value, _annualized_return, find_margin_aware_weights
from src.charts import build_equity_figure, build_drawdown_figure, build_intraday_dd_figure, build_pl_histogram_figure
from src import cta_report
from src.tabs import _single_graph_child, _two_graphs_child, pretty_card
import src.riskfolio_adapter as riskfolio_adapter
import src.inverse_vol as inverse_vol_tab


_PERCENT_EQUITY_CACHE: dict[tuple[Any, ...], tuple[str, PercentEquityCurve]] = {}
_PERCENT_EQUITY_CACHE_LOCK = Lock()
_PORTFOLIO_VIEW_CACHE: dict[tuple[Any, ...], tuple[str, PortfolioView]] = {}
_PORTFOLIO_VIEW_CACHE_LOCK = Lock()

# MTM rows represent end-of-day settlements. Use offsets to align the synthetic
# open/close timestamps with that settlement window (5 pm prior day -> 4:15 pm).
MTM_DAY_START_OFFSET = pd.Timedelta(hours=-7)    # 5:00pm previous calendar day
MTM_DAY_END_OFFSET = pd.Timedelta(hours=16, minutes=15)  # 4:15pm settlement
MTM_SESSION_DURATION = MTM_DAY_END_OFFSET - MTM_DAY_START_OFFSET
MTM_SESSION_HALF = MTM_SESSION_DURATION / 2


def _mtm_day_open(day: Any) -> pd.Timestamp:
    """Return the timestamp representing the MTM session open for a normalized day."""
    return pd.Timestamp(day).normalize() + MTM_DAY_START_OFFSET


def _mtm_day_close(day: Any) -> pd.Timestamp:
    """Return the timestamp representing the MTM session close for a normalized day."""
    return pd.Timestamp(day).normalize() + MTM_DAY_END_OFFSET


def _mtm_bucketize_index(index_like: Iterable) -> pd.DatetimeIndex:
    """
    Map timestamps to their MTM settlement day labels by removing the start offset
    before normalizing. Ensures open/close anchors fall into the same bucket.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index_like))
    return (idx - MTM_DAY_START_OFFSET).normalize()


@dataclass(slots=True)
class FileSlice:
    fname: str
    trades: pd.DataFrame
    trades_full: pd.DataFrame
    mtm: pd.DataFrame
    label: str
    symbol: Optional[str]
    strategy: Optional[str]
    interval: Optional[str]
    contracts_multiplier: float
    margin_override: Optional[float]


@dataclass(slots=True)
class EquityBundle:
    equity: pd.Series
    daily_pnl: pd.Series


@dataclass(slots=True)
class MarginUsage:
    capital_series: pd.Series
    daily_return_pct: pd.Series
    avg_contracts_abs: pd.Series
    avg_contracts_signed: pd.Series
    active_seconds: pd.Series
    duty_cycle: pd.Series
    margin_per_contract: float


@dataclass(slots=True)
class PercentEquityCurve:
    index_series: pd.Series
    cumulative_return_pct: pd.Series
    daily_hash: str


@dataclass(slots=True)
class PortfolioView:
    daily_return_pct: pd.Series
    pct_equity_index: pd.Series
    pct_equity: pd.Series
    total_nav: pd.Series
    portfolio_equity: pd.Series
    capital_by_file: dict[str, pd.Series]
    daily_returns_by_file: dict[str, pd.Series]
    files_included: tuple[str, ...]
    missing_daily: tuple[str, ...]
    missing_capital: tuple[str, ...]
    returns_digest: str

_riskfolio_cache = None
_riskfolio_manager = None

if diskcache and _manager_ctor:
    _riskfolio_cache = diskcache.Cache("./.dash-cache")
    _manager_kwarg_name, _manager_cls = _manager_ctor
    _riskfolio_manager = _manager_cls(_riskfolio_cache)  # type: ignore[misc]
else:  # pragma: no cover - fallback when diskcache unavailable
    _manager_kwarg_name = None

_app_kwargs: Dict[str, Any] = {}
if _riskfolio_manager and _manager_kwarg_name:
    _app_kwargs[_manager_kwarg_name] = _riskfolio_manager


# ------------------------------ Dash UI --------------------------------

app = Dash(__name__, suppress_callback_exceptions=True, **_app_kwargs)
app.config.setdefault("plotlyLoadTimeout", 120000)
app.title = APP_TITLE

_HAS_LONG_CALLBACK = bool(_riskfolio_manager and _manager_kwarg_name == "long_callback_manager")
_HAS_BACKGROUND_CALLBACK = bool(_riskfolio_manager and _manager_kwarg_name == "background_callback_manager")


RISKFOLIO_TOOLBOXES: Dict[str, Dict[str, Any]] = {
    "mean_risk": {
        "label": "Mean-Risk Optimization",
        "model": "Classic",
        "objectives": {
            "max_return": {
                "label": "Maximum Return",
                "obj": "MaxRet",
                "requires": [],
            },
            "min_risk": {
                "label": "Minimum Risk",
                "obj": "MinRisk",
                "requires": [],
            },
            "max_ratio": {
                "label": "Maximum Risk-Adjusted Return Ratio",
                "obj": "Sharpe",
                "requires": ["risk_free_rate"],
            },
            "max_utility": {
                "label": "Maximum Utility",
                "obj": "Utility",
                "requires": ["risk_free_rate", "risk_aversion"],
            },
        },
        "return_models": {
            "arithmetic": {
                "label": "Arithmetic Mean Return",
                "kelly": None,
            },
            "approx_log": {
                "label": "Approximate Log Return",
                "kelly": "approx",
            },
            "exact_log": {
                "label": "Exact Log Return",
                "kelly": "exact",
            },
        },
        "risk_measures": {
            "MV": "Standard Deviation",
            "KT": "Square Root Kurtosis",
            "MAD": "Mean Absolute Deviation",
            "GMD": "Gini Mean Difference",
            "CVRG": "Conditional Value at Risk Range",
            "TGRG": "Tail Gini Range",
            "EVRG": "Entropic Value at Risk Range",
            "RVRG": "Relativistic Value at Risk Range",
            "RG": "Range",
            "MSV": "Semi Standard Deviation",
            "SKT": "Square Root Semi Kurtosis",
            "FLPM": "First Lower Partial Moment",
            "SLPM": "Second Lower Partial Moment",
            "CVaR": "Conditional Value at Risk",
            "TG": "Tail Gini",
            "EVaR": "Entropic Value at Risk",
            "RLVaR": "Relativistic Value at Risk",
            "WR": "Worst Realization (Minimax)",
            "ADD": "Average Drawdown (Uncompounded)",
            "UCI": "Ulcer Index (Uncompounded)",
            "CDaR": "Conditional Drawdown at Risk (Uncompounded)",
            "EDaR": "Entropic Drawdown at Risk (Uncompounded)",
            "RLDaR": "Relativistic Drawdown at Risk (Uncompounded)",
            "MDD": "Maximum Drawdown (Uncompounded)",
        },
    },
}


def get_riskfolio_toolbox(toolbox_id: str) -> Dict[str, Any]:
    """Return the configuration dictionary for a given Riskfolio toolbox."""
    return RISKFOLIO_TOOLBOXES.get(toolbox_id, {})


def list_riskfolio_toolboxes() -> Dict[str, Dict[str, Any]]:
    """Return the mapping of all configured Riskfolio toolboxes."""
    return RISKFOLIO_TOOLBOXES


def build_toolbox_config(toolbox_id: str, selections: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize UI selections into Riskfolio optimization parameters.

    Currently supports the mean-risk toolbox and can be extended for others.
    """
    toolbox = get_riskfolio_toolbox(toolbox_id)
    if not toolbox:
        raise ValueError(f"Unknown Riskfolio toolbox '{toolbox_id}'.")

    if toolbox_id == "mean_risk":
        return _build_mean_risk_config(toolbox, selections)

    raise ValueError(f"No config builder registered for toolbox '{toolbox_id}'.")


def _build_mean_risk_config(toolbox: Dict[str, Any], selections: Dict[str, Any]) -> Dict[str, Any]:
    """Translate mean-risk selections into arguments for rp.Portfolio.optimization."""
    objective_key = selections.get("objective")
    return_key = selections.get("return_model")
    risk_key = selections.get("risk_measure")

    objectives = toolbox.get("objectives", {})
    return_models = toolbox.get("return_models", {})
    risk_measures = toolbox.get("risk_measures", {})

    if objective_key not in objectives:
        raise ValueError(f"Unsupported mean-risk objective '{objective_key}'.")
    if return_key not in return_models:
        raise ValueError(f"Unsupported mean-risk return model '{return_key}'.")
    if risk_key not in risk_measures:
        raise ValueError(f"Unsupported mean-risk risk measure '{risk_key}'.")

    objective_cfg = objectives[objective_key]
    return_cfg = return_models[return_key]

    config: Dict[str, Any] = {
        "model": toolbox.get("model", "Classic"),
        "objective": objective_cfg["obj"],
        "rm": risk_key,
        "kelly": return_cfg.get("kelly"),
    }

    rf = selections.get("risk_free_rate")
    lam = selections.get("risk_aversion")

    requirements = set(objective_cfg.get("requires", []))
    if "risk_free_rate" in requirements:
        if rf is None:
            raise ValueError("Risk-free rate is required for the selected objective.")
        config["risk_free"] = float(rf)
    elif rf is not None:
        config["risk_free"] = float(rf)

    if "risk_aversion" in requirements:
        if lam is None:
            raise ValueError("Risk aversion coefficient is required for the selected objective.")
        config["risk_aversion"] = float(lam)
    elif lam is not None:
        config["risk_aversion"] = float(lam)

    hist_flag = selections.get("hist", True)
    if hist_flag is not None:
        config["hist"] = bool(hist_flag)

    return config


_MEAN_RISK_TOOLBOX = get_riskfolio_toolbox("mean_risk")
_MEAN_RISK_OBJECTIVE_OPTIONS = [
    {"label": data["label"], "value": key} for key, data in (_MEAN_RISK_TOOLBOX.get("objectives") or {}).items()
]
_MEAN_RISK_RETURN_OPTIONS = [
    {"label": data["label"], "value": key} for key, data in (_MEAN_RISK_TOOLBOX.get("return_models") or {}).items()
]
_MEAN_RISK_DEFAULT_OBJECTIVE = (
    _MEAN_RISK_OBJECTIVE_OPTIONS[0]["value"] if _MEAN_RISK_OBJECTIVE_OPTIONS else "max_ratio"
)
_MEAN_RISK_DEFAULT_RETURN = (
    _MEAN_RISK_RETURN_OPTIONS[0]["value"] if _MEAN_RISK_RETURN_OPTIONS else "arithmetic"
)

def _legacy_risk_measure_code(value: Optional[str]) -> str:
    """
    Map existing Riskfolio risk measure dropdown values to mean-risk codes.
    Defaults to 'MV' when no mapping is available.
    """
    mapping = {
        "variance": "MV",
        "semi_variance": "MSV",
        "cvar": "CVaR",
        "cdar": "CDaR",
        "evar": "EVaR",
    }
    if value is None:
        return "MV"
    return mapping.get(str(value), "MV")


import src.netpos as _netpos_mod
_netpos_mod.app = app

import src.allocator as _alloc_mod
_alloc_mod.app = app



# Server-side stores & cache
if not hasattr(app.server, "trade_store"):
    app.server.trade_store = {}  # {filename: DataFrame}
if not hasattr(app.server, "mtm_store"):
    app.server.mtm_store = {}    # {filename: Daily MTM DataFrame}
cache = Cache(app.server, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300})



app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial", "padding": "18px", "background": "#f6f7fb"},
    children=[
        html.H2(APP_TITLE, style={"marginTop": 0}),

        pretty_card([
            dcc.Tabs(
                id="main-tabs",
                value="load",
                children=[
                    dcc.Tab(
                        label="Load Trade Lists",
                        value="load",
                        children=html.Div(
                            [
                                html.H4("1) Load Trade Lists"),
                                dcc.Upload(
                                    id="upload",
                                    children=html.Div(["Drag & drop or ", html.A("select .xlsx files")]),
                                    multiple=True,
                                    style={
                                        "width": "100%",
                                        "height": "70px",
                                        "lineHeight": "70px",
                                        "borderWidth": "2px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "8px",
                                        "textAlign": "center",
                                        "background": "#fafafa",
                                    },
                                ),
                                html.Div(id="file-list", style={"marginTop": "10px", "fontSize": "14px"}),
                                html.Div(
                                    [
                                        html.Div("Included files:", style={"fontWeight": 600, "marginBottom": "6px"}),
                                        html.Div(
                                            id="file-includes-row",
                                            style={
                                                "display": "grid",
                                                "gridTemplateColumns": "max-content 1fr",
                                                "gap": "8px",
                                                "alignItems": "start",
                                            },
                                            children=[
                                                dcc.Checklist(
                                                    id="file-toggle",
                                                    options=[],
                                                    value=[],
                                                    inline=False,
                                                    style={"display": "flex", "flexDirection": "column", "gap": "22px", "alignItems": "flex-start"},
                                                    inputStyle={"marginRight": "6px"},
                                                    labelStyle={"display": "inline-flex", "alignItems": "center", "cursor": "pointer", "width": "auto"},
                                                ),
                                                html.Div(
                                                    id="file-contracts",
                                                    style={
                                                        "display": "grid",
                                                        "gridTemplateColumns": "1fr",
                                                        "gap": "6px",
                                                    },
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            "Tip: leave at 1 for native one-contract P/L; set 0 to exclude without unchecking.",
                                            style={"fontSize": "12px", "color": "#555", "marginTop": "6px"},
                                        ),
                                    ],
                                    style={"marginTop": "18px"},
                                ),
                            ],
                            style={"padding": "12px 0"},
                        ),
                    ),
                    dcc.Tab(
                        label="Settings",
                        value="controls",
                        children=html.Div(
                            [
                                html.H4("Settings"),
                                html.Div(
                                    [
                                        html.Div("Symbols:", style={"fontWeight": 600, "marginBottom": "6px"}),
                                        dcc.Checklist(id="symbol-toggle", options=[], value=[], inline=True),
                                    ],
                                    style={"marginBottom": "10px"},
                                ),
                                html.Div(
                                    [
                                        html.Div("Timeframes:", style={"fontWeight": 600, "marginBottom": "6px"}),
                                        dcc.Checklist(id="interval-toggle", options=[], value=[], inline=True),
                                    ],
                                    style={"marginBottom": "10px"},
                                ),
                                html.Div(
                                    [
                                        html.Div("Strategies:", style={"fontWeight": 600, "marginBottom": "6px"}),
                                        dcc.Checklist(
                                            id="strategy-toggle",
                                            options=[],
                                            value=[],
                                            inline=False,
                                            style={"display": "flex", "flexDirection": "column", "gap": "6px", "alignItems": "flex-start"},
                                            inputStyle={"marginRight": "6px"},
                                            labelStyle={"display": "inline-flex", "alignItems": "center", "cursor": "pointer", "width": "auto"},
                                        ),
                                    ],
                                    style={"marginBottom": "10px"},
                                ),
                                  html.Div(
                                      [
                                          html.Div("Direction:", style={"fontWeight": 600, "marginBottom": "6px"}),
                                          dcc.RadioItems(
                                              id="direction-radio",
                                              options=[
                                                  {"label": "All", "value": "All"},
                                                  {"label": "Long", "value": "Long"},
                                                  {"label": "Short", "value": "Short"},
                                              ],
                                              value="All",
                                              inline=True,
                                          ),
                                      ],
                                      style={"marginBottom": "10px"},
                                  ),
                                html.Div(
                                    [
                                        html.Div("Date range:", style={"fontWeight": 600, "marginBottom": "6px"}),
                                        dcc.DatePickerRange(
                                            id="date-range",
                                            start_date=None,
                                            end_date=None,
                                            display_format="YYYY-MM-DD",
                                            minimum_nights=0,
                                        ),
                                    ],
                                    style={"marginBottom": "10px"},
                                ),
                                html.Div(
                                    [
                                        html.Div("Account equity:", style={"fontWeight": 600, "marginBottom": "6px"}),
                                        dcc.Input(
                                            id="alloc-equity",
                                            type="text",
                                            value=f"{DEFAULT_INITIAL_CAPITAL:g}",
                                            style={"width": "140px"},
                                        ),
                                    ],
                                    style={"marginBottom": "14px"},
                                ),
                                html.Div(
                                    [
                                        html.Div("Intraday spikes:", style={"fontWeight": 600, "marginBottom": "6px"}),
                                        dcc.Checklist(
                                            id="spike-toggle",
                                            options=[{"label": "Include trade-level drawdown spikes", "value": "include"}],
                                            value=[],
                                            inline=False,
                                            style={"display": "flex", "flexDirection": "column", "gap": "4px", "alignItems": "flex-start"},
                                            inputStyle={"marginRight": "6px"},
                                            labelStyle={"display": "inline-flex", "alignItems": "center", "cursor": "pointer"},
                                        ),
                                        html.Small(
                                            "Toggle on to overlay worst-case intraday drawdown spikes on equity, purchasing power, and drawdown charts.",
                                            style={"fontSize": "11px", "color": "#666"},
                                        ),
                                    ],
                                    style={"marginBottom": "14px"},
                                ),
                            ],
                            style={"padding": "12px 0"},
                        ),
                    ),
                    dcc.Tab(
                        label="Equity Curves",
                        value="equity",
                        children=html.Div(
                            [html.Div(id="equity_tab_content")],
                            style={"padding": "12px 0"},
                        ),
                    ),
                    dcc.Tab(
                        label="Portfolio Drawdown",
                        value="ddcurve",
                        children=html.Div(
                            [html.Div(id="drawdown_tab_content")],
                            style={"padding": "12px 0"},
                        ),
                    ),
                    dcc.Tab(
                        label="Intraday Drawdown",
                        value="idd",
                        children=html.Div(
                            [html.Div(id="intraday_tab_content")],
                            style={"padding": "12px 0"},
                        ),
                    ),
                    dcc.Tab(
                        label="Trade P/L Histogram",
                        value="hist",
                        children=html.Div(
                            [html.Div(id="hist_tab_content")],
                            style={"padding": "12px 0"},
                        ),
                    ),
                    dcc.Tab(
                        label="Margin",
                        value="margin",
                        children=html.Div(
                            [html.Div(id="margin_tab_content")],
                            style={"padding": "12px 0"},
                        ),
                    ),
                    dcc.Tab(
                        label="Correlations",
                        value="corr",
                        children=html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    "Correlation view:",
                                                    style={"fontWeight": 600, "marginBottom": "6px"},
                                                ),
                                                dcc.Dropdown(
                                                    id="corr-mode",
                                                    options=[
                                                        {"label": "Daily Drawdown % (recommended)", "value": "drawdown_pct"},
                                                        {"label": "Z-scored Daily Returns", "value": "returns_z"},
                                                        {"label": "Daily $ P/L", "value": "pl"},
                                                        {"label": "Rolling Slope", "value": "slope"},
                                                    ],
                                                    value="drawdown_pct",
                                                    clearable=False,
                                                    style={"maxWidth": "340px"},
                                                ),
                                                html.Div(
                                                    [
                                                        html.Span("Slope window: "),
                                                        dcc.Input(
                                                            id="corr-slope-window",
                                                            type="number",
                                                            value=20,
                                                            step=1,
                                                            min=5,
                                                            style={"width": "100px"},
                                                        ),
                                                    ],
                                                    style={"marginTop": "6px"},
                                                ),
                                            ],
                                            style={"marginBottom": "14px"},
                                        ),
                                        html.Div(id="corr_tab_content"),
                                    ]
                                ),
                            ],
                            style={"padding": "12px 0"},
                        ),
                    ),
                    dcc.Tab(
                        label="Riskfolio",
                        value="riskfolio",
                        children=html.Div(
                            [
                                html.Div(
                                    [
                                        pretty_card([
                                            html.H4("Riskfolio Controls"),
                                            dcc.Tabs(
                                                id="riskfolio-subtab",
                                                value="mean_risk",
                                                style={"marginTop": "6px"},
                                                children=[
                                                    dcc.Tab(
                                                        label="Mean-Risk",
                                                        value="mean_risk",
                                                        children=html.Div(
                                                            [
                                                                html.Div(
                                                                    "Upload and select files to enable optimization.",
                                                                    id="riskfolio-controls-hint",
                                                                    style={"fontSize": "12px", "color": "#555", "marginBottom": "8px"},
                                                                ),
                                                                html.Div([
                                                html.Div([
                                                    html.Label("Objective"),
                                                    html.Small(
                                                        "Objective controls what the optimiser is trying to maximise or minimise. 'Max Risk-Adjusted Return' (Sharpe) maximises (return - risk-free) per unit of risk. 'Min Risk' produces the lowest risk portfolio regardless of return. 'Max Return' chases raw expected return with no risk penalty. 'Utility' maximises expected return minus lambda * risk, where lambda is the risk-aversion input below. Pick the objective that matches your decision-making style.",
                                                        style={"fontSize": "11px", "color": "#666"},
                                                    ),
                                                    dcc.Dropdown(
                                                        id="riskfolio-mean-risk-objective",
                                                        options=_MEAN_RISK_OBJECTIVE_OPTIONS,
                                                        value=_MEAN_RISK_DEFAULT_OBJECTIVE,
                                                        clearable=False,
                                                        disabled=True,
                                                    ),
                                                ], style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
                                                html.Div([
                                                    html.Label("Return Model"),
                                                    html.Small(
                                                        "Return Model defines how expected returns are estimated. Arithmetic uses the straight historical average of daily P/L. Approximate Log applies a Taylor approximation to log returns, which is less sensitive to big jumps. Exact Log uses the full log compounding formula, appropriate when losses and gains are large and you want true compounded growth.",
                                                        style={"fontSize": "11px", "color": "#666"},
                                                    ),
                                                    dcc.Dropdown(
                                                        id="riskfolio-mean-risk-return",
                                                        options=_MEAN_RISK_RETURN_OPTIONS,
                                                        value=_MEAN_RISK_DEFAULT_RETURN,
                                                        clearable=False,
                                                        disabled=True,
                                                    ),
                                                ], style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
                                                html.Div([
                                                    html.Label("Risk Measure"),
                                                    html.Small(
                                                        "Risk Measure determines how volatility or downside risk is quantified. Variance captures overall dispersion; Semi-Variance only penalises downside moves; CVaR measures the expected loss beyond the chosen tail percentile; CDaR is similar but applied to drawdowns; EVaR (Entropic VaR) uses an exponential risk metric that emphasises the worst tails. Pick a measure that reflects the type of risk that matters most to you.",
                                                        style={"fontSize": "11px", "color": "#666"},
                                                    ),
                                                    dcc.Dropdown(
                                                        id="riskfolio-risk-measure",
                                                        options=[
                                                            {"label": "Variance", "value": "variance"},
                                                            {"label": "Semi-Variance", "value": "semi_variance"},
                                                            {"label": "CVaR", "value": "cvar"},
                                                            {"label": "CDaR", "value": "cdar"},
                                                            {"label": "EVaR", "value": "evar"},
                                                        ],
                                                        value="variance",
                                                        clearable=False,
                                                        disabled=True,
                                                    ),
                                                ], style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
                                                html.Div([
                                                    html.Label("Risk-Free Rate (annual %)"),
                                                    html.Small(
                                                        "Risk-free rate is an annual percentage representing the benchmark yield (e.g. Treasury). Typical values range from 0% to ~5%. The optimiser converts this to a daily decimal internally so it lines up with the daily return data.",
                                                        style={"fontSize": "11px", "color": "#666"},
                                                    ),
                                                    dcc.Input(
                                                        id="riskfolio-mean-risk-rf",
                                                        type="number",
                                                        value=0.0,
                                                        step=0.05,
                                                        min=0.0,
                                                        style={"width": "100%"},
                                                        disabled=True,
                                                    ),
                                                ], style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
                                                html.Div([
                                                    html.Label("Risk Aversion (Utility only)"),
                                                    html.Small(
                                                        "Risk aversion lambda is only used when the Utility objective is selected. It controls how much risk penalises the objective. Values between 1 and 5 are common: 1 keeps the focus on return, higher numbers prioritise risk reduction. Leave at the default if unsure.",
                                                        style={"fontSize": "11px", "color": "#666"},
                                                    ),
                                                    dcc.Input(
                                                        id="riskfolio-mean-risk-riskav",
                                                        type="number",
                                                        value=2.0,
                                                        step=0.1,
                                                        style={"width": "100%"},
                                                        disabled=True,
                                                    ),
                                                ], style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
                                                html.Div([
                                                    html.Label("Confidence Level alpha"),
                                                    html.Small(
                                                        "Confidence level alpha applies when using tail-based risk measures (CVaR, CDaR, EVaR). It represents the probability threshold for the tail (e.g. alpha = 0.95 means the worst 5% of outcomes). Higher alpha looks deeper into the tail; typical values are 0.90-0.99. For variance-style risk alpha is ignored.",
                                                        style={"fontSize": "11px", "color": "#666"},
                                                    ),
                                                    dcc.Input(
                                                        id="riskfolio-alpha",
                                                        type="number",
                                                        min=0.5,
                                                        max=0.999,
                                                        step=0.01,
                                                        value=0.95,
                                                        style={"width": "100%"},
                                                        disabled=True,
                                                    ),
                                                ], style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
                                                html.Div([
                                                    html.Label("Covariance Method"),
                                                    html.Small(
                                                        "Covariance captures how asset returns move together and drives diversification. 'Sample/Historical' uses the plain covariance of all observations—simple but sensitive to noise when you have few trades. 'EWMA' (Exponentially Weighted Moving Average) applies a decay so recent observations count more. 'Ledoit-Wolf Shrinkage' blends the sample matrix with a structured target to improve stability when sample size is small or correlations are unstable.",
                                                        style={"fontSize": "11px", "color": "#666"},
                                                    ),
                                                    dcc.Dropdown(
                                                        id="riskfolio-covariance",
                                                        options=[
                                                            {"label": "Sample", "value": "sample"},
                                                            {"label": "Exponentially-Weighted (EWMA)", "value": "ewma"},
                                                            {"label": "Ledoit-Wolf Shrinkage", "value": "ledoit"},
                                                        ],
                                                        value="ledoit",
                                                        clearable=False,
                                                        disabled=True,
                                                    ),
                                                ], style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
                                                html.Div([
                                                    html.Label("Weight Bounds"),
                                                    html.Small(
                                                        "Bounds limit the minimum/maximum allocation per asset. A min of 0.0 means long-only; a max below 1.0 forces diversification (e.g. 0.30 caps any single asset at 30%).",
                                                        style={"fontSize": "11px", "color": "#666"},
                                                    ),
                                                    html.Div([
                                                        dcc.Input(
                                                            id="riskfolio-bound-min",
                                                            type="number",
                                                            value=0.0,
                                                            step=0.05,
                                                            style={"width": "48%"},
                                                            disabled=True,
                                                        ),
                                                        dcc.Input(
                                                            id="riskfolio-bound-max",
                                                            type="number",
                                                            value=1.0,
                                                            step=0.05,
                                                            style={"width": "48%"},
                                                            disabled=True,
                                                        ),
                                                    ], style={"display": "flex", "justifyContent": "space-between", "gap": "4%"}),
                                                ], style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
                                                html.Div([
                                                    html.Label("Budget"),
                                                    html.Small(
                                                        "Budget is the sum of all portfolio weights. 1.0 means fully invested, >1.0 allows leverage, and <1.0 leaves cash on the sidelines.",
                                                        style={"fontSize": "11px", "color": "#666"},
                                                    ),
                                                    dcc.Input(
                                                        id="riskfolio-budget",
                                                        type="number",
                                                        value=1.0,
                                                        step=0.05,
                                                        style={"width": "100%"},
                                                        disabled=True,
                                                    ),
                                                ], style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
                                                html.Div([
                                                    html.Label("Group Caps (symbol=cap, comma separated)"),
                                                    html.Small(
                                                        "Provide upper limits for specific symbols or groups. Format: ES=0.40, NQ=0.30 (numbers are fractions of total budget). Leave empty if you don't need per-group caps.",
                                                        style={"fontSize": "11px", "color": "#666"},
                                                    ),
                                                    dcc.Input(
                                                        id="riskfolio-group-caps",
                                                        type="text",
                                                        placeholder="e.g. ES=0.4, NQ=0.3",
                                                        style={"width": "100%"},
                                                        disabled=True,
                                                    ),
                                                ], style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
                                                html.Div([
                                                    html.Label("Strategy Caps (strategy=cap, comma separated)"),
                                                    html.Small(
                                                        "Limit total allocation per strategy across all symbols/timeframes. Example: 35X=0.30, 5X=0.20.",
                                                        style={"fontSize": "11px", "color": "#666"},
                                                    ),
                                                    dcc.Input(
                                                        id="riskfolio-strategy-caps",
                                                        type="text",
                                                        placeholder="e.g. Momentum=0.6",
                                                        style={"width": "100%"},
                                                        disabled=True,
                                                    ),
                                                ], style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
                                                html.Div([
                                                    html.Label("Turnover Cap (reserved)"),
                                                    html.Small(
                                                        "Turnover constraints will return in a future update. Input is disabled and values are ignored for now.",
                                                        style={"fontSize": "11px", "color": "#666"},
                                                    ),
                                                    dcc.Input(
                                                        id="riskfolio-turnover",
                                                        type="number",
                                                        min=0.0,
                                                        step=0.05,
                                                        placeholder="Reserved for future use",
                                                        style={"width": "100%"},
                                                        disabled=True,
                                                    ),
                                                ], style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
                                                html.Div([
                                                    html.Button(
                                                        "Optimize",
                                                        id="riskfolio-btn-optimize",
                                                        n_clicks=0,
                                                        style={"width": "100%", "marginTop": "8px"},
                                                        disabled=True,
                                                    ),
                                                ], style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
                                                                ], style={"display": "flex", "flexDirection": "column", "gap": "12px"}),
                                                            ],
                                                            style={"display": "flex", "flexDirection": "column", "gap": "12px"},
                                                        ),
                                                    ),
                                                    dcc.Tab(
                                                        label="Risk Parity (coming soon)",
                                                        value="risk_parity",
                                                        disabled=True,
                                                        children=html.Div(
                                                            "Risk Parity optimisation will be available soon.",
                                                            style={"fontSize": "12px", "color": "#666", "padding": "10px"},
                                                        ),
                                                    ),
                                                    dcc.Tab(
                                                        label="Hierarchical (coming soon)",
                                                        value="hrp",
                                                        disabled=True,
                                                        children=html.Div(
                                                            "Hierarchical optimisation will be available soon.",
                                                            style={"fontSize": "12px", "color": "#666", "padding": "10px"},
                                                        ),
                                                    ),
                                                ],
                                            ),
                                        ]),
                                        pretty_card([
                                            html.H4("Optimization Results"),
                                            html.Div(
                                                [
                                                    html.Div("Progress", style={"fontWeight": 600, "marginBottom": "6px"}),
                                                    html.Div(
                                                        id="riskfolio-progress-label",
                                                        style={"fontSize": "12px", "color": "#555", "marginBottom": "6px"},
                                                    ),
                                                    html.Div(
                                                        html.Div(
                                                            id="riskfolio-progress-bar",
                                                            style={
                                                                "width": "0%",
                                                                "height": "100%",
                                                                "background": "#4c8bf5",
                                                                "transition": "width 0.3s",
                                                            },
                                                        ),
                                                        style={
                                                            "width": "100%",
                                                            "height": "8px",
                                                            "background": "#e5e7eb",
                                                            "borderRadius": "4px",
                                                        },
                                                    ),
                                                ],
                                                style={"marginBottom": "14px"},
                                            ),
                                            dcc.Loading(
                                                type="default",
                                                children=html.Div(
                                                    id="riskfolio-results",
                                                    children=html.Div(
                                                        "Run an optimization to populate results.",
                                                        style={"color": "#666", "fontSize": "14px"},
                                                    ),
                                                ),
                                            ),
                                            html.Div(
                                                [
                                                    html.H5("Riskfolio Suggested Contracts", style={"margin": "16px 0 6px"}),
                                                    html.Div(
                                                        [
                                                            html.Label("Account Equity", style={"fontWeight": 500}),
                                                            dcc.Input(
                                                                id="riskfolio-contract-equity",
                                                                type="number",
                                                                value=DEFAULT_INITIAL_CAPITAL,
                                                                min=0,
                                                                step=1000,
                                                                style={"width": "160px"},
                                                            ),
                                                            html.Button(
                                                                "Apply Suggested Contracts",
                                                                id="riskfolio-contracts-apply",
                                                                n_clicks=0,
                                                                style={"marginTop": "6px", "width": "100%"},
                                                            ),
                                                        ],
                                                        style={"display": "flex", "flexDirection": "column", "gap": "4px", "maxWidth": "200px"},
                                                    ),
                                                    html.Div(
                                                        id="riskfolio-contracts-note",
                                                        style={"fontSize": "12px", "color": "#555", "marginBottom": "6px"},
                                                    ),
                                                    dash_table.DataTable(
                                                        id="riskfolio-contracts-table",
                                                        data=[],
                                                        columns=[
                                                            {"name": "File", "id": "file"},
                                                            {"name": "Symbol", "id": "symbol"},
                                                            {"name": "Optimized Weight", "id": "weight"},
                                                            {"name": "Suggested Weight Gap", "id": "suggested_gap"},
                                                            {"name": "Current Weight Gap", "id": "current_gap"},
                                                            {"name": "Margin / Contract", "id": "margin_per_contract"},
                                                            {"name": "Suggested Contracts", "id": "suggested"},
                                                            {"name": "Suggested Margin", "id": "suggested_margin"},
                                                            {"name": "Current Contracts", "id": "current"},
                                                            {"name": "Current Margin", "id": "current_margin"},
                                                            {"name": "Delta", "id": "delta"},
                                                        ],
                                                        style_header={"fontWeight": "bold"},
                                                        style_cell={
                                                            "padding": "6px",
                                                            "fontSize": "12px",
                                                            "textAlign": "right",
                                                            "whiteSpace": "normal",
                                                        },
                                                        style_cell_conditional=[
                                                            {"if": {"column_id": "file"}, "textAlign": "left"},
                                                            {"if": {"column_id": "symbol"}, "textAlign": "left"},
                                                        ],
                                                        style_table={"overflowX": "auto"},
                                                    ),
                                            ],
                                            style={"display": "flex", "flexDirection": "column", "gap": "6px"},
                                        ),
                                    ]),
                                    ],
                                    style={"display": "grid", "gridTemplateColumns": "minmax(280px, 0.85fr) 2fr", "gap": "14px"},
                                ),
                            ],
                            style={"padding": "12px 0"},
                        ),
                    ),
                dcc.Tab(
                    label="Inverse Volatility",
                        value="inverse_vol",
                        children=inverse_vol_tab.layout(),
                    ),
                    dcc.Tab(
                        label="Allocator",
                        value="allocator",
                        children=html.Div(
                            [
                                html.Div(
                                    [
                                        pretty_card([
                                            html.Div("Allocator settings:", style={"fontWeight": 600, "marginBottom": "8px"}),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Span("Max margin use (% of equity):", style={"fontWeight": 500}),
                                                            dcc.Input(
                                                                id="alloc-margin-buffer",
                                                                type="number",
                                                                value=int(MARGIN_BUFFER * 100),
                                                                step=5,
                                                                style={"width": "110px"},
                                                            ),
                                                        ],
                                                        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Span("Objective:", style={"fontWeight": 500}),
                                                            dcc.Dropdown(
                                                                id="alloc-objective",
                                                                options=[
                                                                    {"label": "Max Return / Drawdown", "value": "max_return_over_dd"},
                                                                    {"label": "Risk Parity (projected)", "value": "risk_parity"},
                                                                ],
                                                                value="max_return_over_dd",
                                                                clearable=False,
                                                                style={"width": "220px", "fontSize": "14px"},
                                                            ),
                                                        ],
                                                        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Span("Maximum number of contracts:", style={"fontWeight": 500}),
                                                            dcc.Input(
                                                                id="alloc-lev-cap",
                                                                type="number",
                                                                value=1,
                                                                min=0,
                                                                step=1,
                                                                style={"width": "110px"},
                                                            ),
                                                        ],
                                                        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                                                    ),
                                                    html.Button(
                                                        "Run Allocator",
                                                        id="btn-run-alloc",
                                                        n_clicks=0,
                                                        style={"padding": "4px 12px", "height": "32px", "width": "140px", "fontSize": "13px"},
                                                    ),
                                                ],
                                                style={"display": "flex", "flexDirection": "column", "gap": "12px"},
                                            ),
                                        ]),
                                        pretty_card([
                                            html.Div(id="alloc-summary", style={"marginBottom": "8px"}),
                                            dcc.Graph(id="alloc-portfolio-graph", style={"height": "360px"}),
                                            dash_table.DataTable(
                                                id="alloc-weights-table",
                                                style_header={"fontWeight": "bold"},
                                                style_cell={"padding": "6px", "whiteSpace": "normal"},
                                                page_size=20,
                                            ),
                                        ]),
                                    ],
                                    style={"display": "grid", "gridTemplateColumns": "minmax(220px, 0.6fr) 2.4fr", "gap": "16px", "alignItems": "start"},
                                ),
                            ],
                            style={"padding": "12px 0"},
                        ),
                    ),
                dcc.Tab(
                    label="Metrics",
                    value="metrics",
                    children=html.Div(
                        [
                            html.H4("Metrics"),
                            dash_table.DataTable(
                                id="metrics-table",
                                style_header={"fontWeight": "bold"},
                                style_cell={"padding": "6px", "whiteSpace": "normal"},
                                page_size=20,
                                sort_action="native",
                                export_format="none",
                            ),
                            html.Div(
                                [
                                    html.Button("Export filtered trades (CSV)", id="btn-export-trades"),
                                    html.Button(
                                        "Export metrics (CSV)",
                                        id="btn-export-metrics",
                                        style={"marginLeft": "10px"},
                                    ),
                                    dcc.Download(id="download-trades"),
                                    dcc.Download(id="download-metrics"),
                                ],
                                style={"marginTop": "8px"},
                            ),
                        ],
                        style={"padding": "12px 0"},
                    ),
                ),
                dcc.Tab(
                    label="CTA Report",
                    value="cta-report",
                    children=html.Div(
                        [
                            html.Div(id="cta-summary-container"),
                            html.Div(id="cta-monthly-table-container", style={"marginTop": "16px"}),
                            html.Div(id="cta-charts-container", style={"marginTop": "16px"}),
                            html.Div(id="cta-disclosures-container", style={"marginTop": "24px"}),
                        ],
                        style={"padding": "12px 0"},
                    ),
                ),
            ],
        ),
        ]),

        # Stores
        dcc.Store(id="store-trades"),
        dcc.Store(id="store-contracts"),   # per-file contract multipliers {filename: float}
        dcc.Store(id="store-margins"),     # per-file margin $/contract overrides {filename: float}
        dcc.Store(id="store-version", data=0),
        dcc.Store(id="riskfolio-progress"),
        dcc.Store(id="riskfolio-last-weights"),
        dcc.Store(id="riskfolio-contracts-store"),
        dcc.Store(id="ivp-store-base"),
        dcc.Store(id="ivp-store-suggested"),

    ]
)

riskfolio_validation_stub = html.Div(
    [],
    style={"display": "none"},
)

app.validation_layout = html.Div([app.layout, riskfolio_validation_stub])

# ------------------------------ Helpers --------------------------------


def make_selection_key(
    selected_files,
    symbols,
    intervals,
    strategies,
    direction,
    start_date,
    end_date,
    contracts_map,
    margins_map,
    store_version,
):
    """
    Build a stable, hashable cache key for the current selection.
    - Lists are sorted & tuple-ized.
    - Dates normalized to 'YYYY-MM-DD' (or empty string).
    - Dicts (contracts_map) turned into a sorted tuple.
    - Direction coerced to a simple string.
    The return value is a tuple of labeled parts for readability in logs.
    """
    files_part      = _keyify_list([str(f) for f in (selected_files or [])])
    symbols_part    = _keyify_list(symbols or [])
    intervals_part  = _keyify_list(intervals or [])
    strategies_part = _keyify_list(strategies or [])

    start_part = normalize_date(start_date) or ""
    end_part   = normalize_date(end_date) or ""

    contracts_part = _keyify_contracts_map(contracts_map or {})
    margins_part   = _keyify_contracts_map(margins_map or {})
    return (
        ("v", int(store_version) if store_version is not None else 0),
        ("files", files_part),
        ("symbols", symbols_part),
        ("intervals", intervals_part),
        ("strategies", strategies_part),
        ("direction", str(direction or "All")),
        ("start", start_part),
        ("end", end_part),
        ("contracts", contracts_part),
        ("margins",   margins_part),
    )


def _coerce_float(value, default) -> float:
    """Convert arbitrary user input to a float, falling back to default on failure."""
    try:
        if value is None:
            return float(default)
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return float(default)
        text = text.replace(",", "")
        return float(text)
    except Exception:
        return float(default)


def _spike_toggle_enabled(value) -> bool:
    """Interpret checklist toggle payload -> bool."""
    if isinstance(value, (list, tuple, set)):
        return "include" in value
    if isinstance(value, str):
        return value.lower() in {"include", "on", "true", "yes"}
    if isinstance(value, bool):
        return value
    return False


def _resolve_margin_per_contract(
    fname: str,
    trades_df: Optional[pd.DataFrame],
    overrides: dict[str, float],
) -> float:
    override_val = overrides.get(fname) if overrides else None
    if override_val is not None:
        try:
            override_float = float(override_val)
            if override_float > 0:
                return override_float
        except Exception:
            pass

    sym = _symbol_from_first_row(trades_df)
    if not sym and isinstance(fname, str):
        sym_guess = fname.split("_")[0].strip()
        sym = sym_guess or None
    spec = MARGIN_SPEC.get((sym or "").upper())
    if spec is None:
        return 0.0
    try:
        return float(spec[0])
    except Exception:
        return 0.0


def _contract_time_usage(
    netpos_series: Optional[pd.Series],
    day_index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return avg abs contracts, active seconds, duty cycle, and signed avg contracts per day."""
    if day_index is None or len(day_index) == 0:
        empty_index = pd.DatetimeIndex([]) if day_index is None else pd.DatetimeIndex(day_index)
        empty = pd.Series(dtype=float, index=empty_index)
        return empty, empty.copy(), empty.copy(), empty.copy()

    normalized_days = pd.to_datetime(day_index).tz_localize(None).normalize()
    unique_days = pd.Index(sorted(set(normalized_days)))
    target_index = pd.DatetimeIndex(unique_days)
    day_set = set(unique_days)
    totals_abs: dict[pd.Timestamp, float] = {}
    totals_signed: dict[pd.Timestamp, float] = {}
    active_seconds: dict[pd.Timestamp, float] = {}

    if netpos_series is not None and not netpos_series.empty:
        s = pd.Series(
            pd.to_numeric(netpos_series, errors="coerce"),
            index=pd.to_datetime(netpos_series.index).tz_localize(None),
        ).sort_index()
        idx = s.index.to_list()
        vals = s.to_numpy(dtype=float)
        if len(idx) >= 2:
            for i in range(len(idx) - 1):
                start = idx[i]
                end = idx[i + 1]
                signed_value = vals[i]
                abs_value = abs(signed_value)
                if not np.isfinite(abs_value) or abs_value <= 0 or start >= end:
                    continue
                seg_start = start
                while seg_start < end:
                    day = pd.Timestamp(seg_start).normalize()
                    day_end = day + pd.Timedelta(days=1)
                    segment_end = day_end if day_end < end else end
                    dt = (segment_end - seg_start).total_seconds()
                    if dt <= 0:
                        break
                    if day in day_set:
                        totals_abs[day] = totals_abs.get(day, 0.0) + abs_value * dt
                        totals_signed[day] = totals_signed.get(day, 0.0) + signed_value * dt
                        active_seconds[day] = active_seconds.get(day, 0.0) + dt
                    seg_start = segment_end

    abs_values = []
    signed_values = []
    active_values = []
    duty_cycle_values = []
    seconds_in_day = 24 * 60 * 60
    for day in target_index:
        active = active_seconds.get(day, 0.0)
        total_abs = totals_abs.get(day, 0.0)
        total_signed = totals_signed.get(day, 0.0)
        abs_values.append(total_abs / active if active > 0 else 0.0)
        signed_values.append(total_signed / active if active > 0 else 0.0)
        active_values.append(active)
        duty_cycle = (active / seconds_in_day) if active > 0 else 0.0
        duty_cycle_values.append(max(0.0, min(1.0, duty_cycle)))

    avg_abs_series = pd.Series(abs_values, index=target_index)
    avg_signed_series = pd.Series(signed_values, index=target_index)
    active_seconds_series = pd.Series(active_values, index=target_index)
    duty_cycle_series = pd.Series(duty_cycle_values, index=target_index)
    return avg_abs_series, active_seconds_series, duty_cycle_series, avg_signed_series


def _active_time_avg_contracts(
    netpos_series: Optional[pd.Series],
    day_index: pd.DatetimeIndex,
) -> pd.Series:
    return _contract_time_usage(netpos_series, day_index)[0]


def _daily_returns_from_pnl_and_capital(
    daily_pnl: pd.Series,
    capital_series: Optional[pd.Series],
) -> pd.Series:
    if daily_pnl is None or daily_pnl.empty:
        return pd.Series(dtype=float)
    pnl = pd.Series(
        pd.to_numeric(daily_pnl, errors="coerce").fillna(0.0),
        index=pd.to_datetime(daily_pnl.index).tz_localize(None),
    )
    pnl.index = pnl.index.normalize()
    if capital_series is None or capital_series.empty:
        capital = pd.Series(0.0, index=pnl.index)
    else:
        capital = capital_series.reindex(pnl.index).fillna(0.0)
    capital_values = capital.to_numpy(dtype=float)
    pnl_values = pnl.to_numpy(dtype=float)
    returns = np.divide(
        pnl_values,
        capital_values,
        out=np.zeros_like(pnl_values),
        where=capital_values != 0.0,
    )
    return pd.Series(returns, index=pnl.index, name="daily_return_pct").fillna(0.0)


def _compute_portfolio_from_maps(
    returns_dict: dict[str, pd.Series],
    capital_dict: dict[str, pd.Series],
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if not returns_dict:
        empty = pd.Series(dtype=float)
        return empty, empty, empty, empty

    returns_df = pd.DataFrame(returns_dict).sort_index().fillna(0.0)
    capital_df = pd.DataFrame(capital_dict) if capital_dict else pd.DataFrame()
    if capital_df.empty:
        capital_df = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)
    else:
        capital_df = capital_df.reindex(returns_df.index).ffill().fillna(0.0)
        missing_cols = set(returns_df.columns) - set(capital_df.columns)
        for col in missing_cols:
            capital_df[col] = 0.0
        capital_df = capital_df[returns_df.columns]

    returns_np = returns_df.to_numpy(dtype=float)
    capital_np = capital_df.to_numpy(dtype=float)
    if returns_np.size == 0 or capital_np.size == 0:
        empty = pd.Series(dtype=float, index=returns_df.index)
        return empty, empty, empty, empty

    total_capital = capital_np.sum(axis=1, keepdims=True)
    weights = np.divide(
        capital_np,
        total_capital,
        out=np.zeros_like(capital_np),
        where=total_capital > 0.0,
    )
    portfolio_returns_np = np.einsum("ij,ij->i", returns_np, weights)
    portfolio_daily_return_pct = pd.Series(
        portfolio_returns_np,
        index=returns_df.index,
        name="portfolio_daily_return_pct",
    ).fillna(0.0)

    portfolio_total_nav = pd.Series(
        total_capital.flatten(),
        index=returns_df.index,
        name="portfolio_total_nav",
    ).ffill().fillna(0.0)

    if portfolio_daily_return_pct.empty:
        portfolio_pct_equity_index = pd.Series(dtype=float)
        portfolio_pct_equity = pd.Series(dtype=float)
    else:
        portfolio_pct_equity_index = _percent_equity_series_from_daily(portfolio_daily_return_pct)
        portfolio_pct_equity = portfolio_pct_equity_index - 1.0
        portfolio_pct_equity.name = "portfolio_cumulative_return_pct"

    return (
        portfolio_daily_return_pct,
        portfolio_pct_equity_index,
        portfolio_pct_equity,
        portfolio_total_nav,
    )


def _compute_portfolio_equity_series(sel_key, included_files: tuple[str, ...]) -> pd.Series:
    """Combine per-file equity curves into a normalized portfolio series."""
    if not included_files:
        return pd.Series(dtype=float)

    equity_by_file: dict[str, pd.Series] = {}
    for fname in included_files:
        bundle = get_equity_bundle(sel_key, fname)
        series = bundle.equity if bundle and isinstance(bundle.equity, pd.Series) else None
        if series is None or series.empty:
            continue
        equity_by_file[str(fname)] = series

    if not equity_by_file:
        return pd.Series(dtype=float)

    combined = combine_equity(equity_by_file, list(included_files))
    if combined is None or combined.empty:
        return pd.Series(dtype=float)

    portfolio_series = combined.get("Portfolio")
    if portfolio_series is None or portfolio_series.empty:
        return pd.Series(dtype=float)

    series = portfolio_series.copy()
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index)).tz_localize(None)
    return series.sort_index()


inverse_vol_tab.register_callbacks(
    app,
    make_selection_key=make_selection_key,
    coerce_float=_coerce_float,
    default_initial_capital=DEFAULT_INITIAL_CAPITAL,
)


def _coerce_contract_multiplier(value: Any) -> float:
    try:
        mult = float(value)
    except Exception:
        return 1.0
    return mult if math.isfinite(mult) else 1.0


def _coerce_margin_override(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        margin_val = float(value)
    except Exception:
        return None
    if not math.isfinite(margin_val) or margin_val <= 0:
        return None
    return margin_val


def _first_nonempty_str(df: Optional[pd.DataFrame], column: str) -> Optional[str]:
    if df is None or df.empty or column not in df.columns:
        return None
    series = df[column].dropna()
    if series.empty:
        return None
    value = str(series.iloc[0]).strip()
    return value or None


def _derive_symbol_from_df(df: Optional[pd.DataFrame], fname: Any) -> Optional[str]:
    sym_val = _symbol_from_first_row(df)
    if not sym_val and isinstance(fname, str):
        sym_guess = fname.split("_")[0].strip()
        sym_val = sym_guess or None
    if not sym_val:
        return None
    return str(sym_val).upper()


def _derive_label_from_df(df: Optional[pd.DataFrame], fname: Any) -> str:
    label = str(fname)
    if df is None:
        return label
    try:
        label = _display_label_from_df(df, fallback=str(fname))
    except TypeError:
        # Fallback for legacy helper signatures
        label = _display_label_from_df(df, fname) if df is not None else str(fname)
    return label or str(fname)


@cache.memoize()
def get_file_slice(sel_key, fname) -> FileSlice:
    """Return the canonical per-file slice for this selection."""
    parts = dict(sel_key)
    data_version = parts.get("v", 0)
    symbol_sel = tuple(parts.get("symbols", ()))
    interval_sel = tuple(parts.get("intervals", ()))
    strat_sel = tuple(parts.get("strategies", ()))
    direction = parts.get("direction", "All")
    start_date = parts.get("start") or None
    end_date = parts.get("end") or None
    contracts_kv = tuple(parts.get("contracts", ()))
    margins_kv = tuple(parts.get("margins", ()))

    contracts_map: dict[str, Any] = {str(k): v for k, v in contracts_kv}
    margins_map: dict[str, Any] = {str(k): v for k, v in margins_kv}

    multiplier = _coerce_contract_multiplier(contracts_map.get(fname, 1.0))
    margin_override = _coerce_margin_override(margins_map.get(fname))

    trades_full_df = cached_filtered_df(
        data_version,
        fname,
        symbol_sel,
        interval_sel,
        strat_sel,
        direction,
        None,
        None,
        multiplier,
    )
    trades_df = trades_full_df if (start_date is None and end_date is None) else cached_filtered_df(
        data_version,
        fname,
        symbol_sel,
        interval_sel,
        strat_sel,
        direction,
        start_date,
        end_date,
        multiplier,
    )
    mtm_df = cached_mtm_df(
        data_version,
        fname,
        symbol_sel,
        interval_sel,
        strat_sel,
        direction,
        start_date,
        end_date,
        multiplier,
    )

    label = _derive_label_from_df(trades_df, fname)
    symbol_val = _derive_symbol_from_df(trades_df, fname)
    strategy_val = _first_nonempty_str(trades_df, "Strategy")
    interval_val = _first_nonempty_str(trades_df, "Interval")
    interval_str = str(interval_val).strip() if interval_val is not None else None

    return FileSlice(
        fname=str(fname),
        trades=trades_df,
        trades_full=trades_full_df,
        mtm=mtm_df,
        label=label,
        symbol=symbol_val,
        strategy=strategy_val,
        interval=interval_str,
        contracts_multiplier=multiplier,
        margin_override=margin_override,
    )


def _compute_selection_window(sel_key, *, fallback_min=None, fallback_max=None) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Normalize the start/end dates for a selection, falling back to provided bounds."""
    parts = dict(sel_key)
    start_val = parts.get("start") or None
    end_val = parts.get("end") or None
    start_dt = pd.to_datetime(start_val) if start_val else None
    end_dt = pd.to_datetime(end_val) if end_val else None
    if start_dt is None or pd.isna(start_dt):
        start_dt = fallback_min
    if end_dt is None or pd.isna(end_dt):
        end_dt = fallback_max
    if start_dt is None or end_dt is None:
        return None, None
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt
    return pd.to_datetime(start_dt), pd.to_datetime(end_dt)


@cache.memoize()
def get_equity_bundle(sel_key, fname) -> EquityBundle:
    """Return cached cumulative equity and daily P&L series for a file."""
    file_slice = get_file_slice(sel_key, fname)
    eq_raw, daily_raw = _hybrid_equity_from_trades_and_mtm(file_slice.trades, file_slice.mtm)

    eq_series = eq_raw.copy() if isinstance(eq_raw, pd.Series) else pd.Series(dtype=float)
    if not eq_series.empty:
        eq_index = pd.DatetimeIndex(pd.to_datetime(eq_series.index)).tz_localize(None)
        eq_series.index = eq_index
        eq_series = eq_series.sort_index()

    daily_series = daily_raw.copy() if isinstance(daily_raw, pd.Series) else pd.Series(dtype=float)
    if not daily_series.empty:
        daily_series = pd.Series(pd.to_numeric(daily_series, errors="coerce").fillna(0.0))
        daily_index = pd.DatetimeIndex(pd.to_datetime(daily_series.index)).tz_localize(None).normalize()
        daily_series.index = daily_index
        daily_series = daily_series.sort_index()
    else:
        daily_series = pd.Series(dtype=float)
    daily_series.name = "daily_pnl"

    return EquityBundle(equity=eq_series, daily_pnl=daily_series)


@cache.memoize()
def get_margin_usage(sel_key, fname) -> MarginUsage:
    """Return cached capital/return stats for a file."""
    file_slice = get_file_slice(sel_key, fname)
    bundle = get_equity_bundle(sel_key, fname)

    parts = dict(sel_key)
    data_version = parts.get("v", 0)
    files_key = tuple(parts.get("files", ()))
    symbol_key = tuple(parts.get("symbols", ()))
    interval_key = tuple(parts.get("intervals", ()))
    strat_key = tuple(parts.get("strategies", ()))
    direction = parts.get("direction", "All")
    start_date = parts.get("start", None)
    end_date = parts.get("end", None)
    contracts_kv = tuple(parts.get("contracts", ()))

    netpos_map = cached_netpos_per_file(
        data_version,
        files_key,
        symbol_key,
        interval_key,
        strat_key,
        direction,
        start_date,
        end_date,
        contracts_kv,
    )
    netpos_series = netpos_map.get(fname)

    daily_index = bundle.daily_pnl.index
    avg_abs, active_seconds, duty_cycle, avg_signed = _contract_time_usage(netpos_series, daily_index)
    avg_abs = avg_abs.reindex(daily_index).fillna(0.0)
    avg_signed = avg_signed.reindex(daily_index).fillna(0.0)
    active_seconds = active_seconds.reindex(daily_index).fillna(0.0)
    duty_cycle = duty_cycle.reindex(daily_index).fillna(0.0)

    if file_slice.margin_override is not None:
        margin_per_contract = float(file_slice.margin_override)
    else:
        margins_kv = tuple(parts.get("margins", ()))
        margin_overrides = {str(k): float(v) for k, v in (margins_kv or [])}
        margin_per_contract = _resolve_margin_per_contract(fname, file_slice.trades, margin_overrides)

    capital_series = (avg_abs * float(margin_per_contract)).rename("margin_capital")
    daily_returns = _daily_returns_from_pnl_and_capital(bundle.daily_pnl, capital_series)

    return MarginUsage(
        capital_series=capital_series,
        daily_return_pct=daily_returns,
        avg_contracts_abs=avg_abs,
        avg_contracts_signed=avg_signed,
        active_seconds=active_seconds,
        duty_cycle=duty_cycle,
        margin_per_contract=float(margin_per_contract),
    )


def _normalize_daily_returns_series(daily_returns: Optional[pd.Series]) -> pd.Series:
    """Canonicalize daily percent series for hashing/alignment."""
    if daily_returns is None:
        return pd.Series(dtype=float)
    series = pd.Series(pd.to_numeric(daily_returns, errors="coerce"))
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index)).tz_localize(None)
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")].fillna(0.0)
    return series


def _daily_returns_digest(daily_returns: Optional[pd.Series]) -> str:
    """
    Deterministic digest for the per-file daily percent series.
    Ensures we rebuild percent equity only when the underlying data changes.
    """
    normalized = _normalize_daily_returns_series(daily_returns)
    if normalized.empty:
        return "empty"
    hashed = pd.util.hash_pandas_object(normalized, index=True).values
    return hashlib.sha1(hashed.tobytes()).hexdigest()


def _normalize_capital_series(capital_series: Optional[pd.Series]) -> pd.Series:
    """Canonicalize capital series for alignment/sanity checks."""
    if capital_series is None:
        return pd.Series(dtype=float)
    series = pd.Series(pd.to_numeric(capital_series, errors="coerce"))
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index)).tz_localize(None)
    return series.sort_index().fillna(0.0)


def _portfolio_returns_digest(daily_map: dict[str, pd.Series]) -> str:
    """Stable digest representing the set of per-file daily series."""
    if not daily_map:
        return "empty"
    parts: list[str] = []
    for fname in sorted(daily_map):
        parts.append(f"{fname}:{_daily_returns_digest(daily_map[fname])}")
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def get_percent_equity(sel_key, fname, demand_flag: str = "core") -> PercentEquityCurve:
    """
    Lazily build the additive percent-equity index for a file selection.
    demand_flag is a human-readable token for diagnostics (not part of the cache key).
    """
    usage = get_margin_usage(sel_key, fname)
    daily_returns = usage.daily_return_pct if usage else pd.Series(dtype=float)
    digest = _daily_returns_digest(daily_returns)
    cache_key = ("file", fname, sel_key)

    with _PERCENT_EQUITY_CACHE_LOCK:
        cached = _PERCENT_EQUITY_CACHE.get(cache_key)
        if cached and cached[0] == digest:
            return cached[1]

    if daily_returns is None or daily_returns.empty:
        empty_curve = PercentEquityCurve(
            index_series=pd.Series(dtype=float),
            cumulative_return_pct=pd.Series(dtype=float),
            daily_hash=digest,
        )
        with _PERCENT_EQUITY_CACHE_LOCK:
            _PERCENT_EQUITY_CACHE[cache_key] = (digest, empty_curve)
        return empty_curve

    aligned = _normalize_daily_returns_series(daily_returns)

    eq_index_series = _percent_equity_series_from_daily(aligned)
    pct_curve = (eq_index_series - 1.0).rename("cumulative_return_pct")

    curve = PercentEquityCurve(
        index_series=eq_index_series,
        cumulative_return_pct=pct_curve,
        daily_hash=digest,
    )
    with _PERCENT_EQUITY_CACHE_LOCK:
        _PERCENT_EQUITY_CACHE[cache_key] = (digest, curve)
    return curve


def _collect_per_file_series(
    sel_key,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """
    Gather sanitized per-file daily return and capital series for the selection.
    Returns:
        (daily_by_file, capital_by_file, included_files, missing_daily, missing_capital)
    """
    parts = dict(sel_key)
    files = tuple(parts.get("files", ()))
    if not files:
        empty_tuple: tuple[str, ...] = ()
        return {}, {}, empty_tuple, empty_tuple, empty_tuple

    daily_by_file: dict[str, pd.Series] = {}
    capital_by_file: dict[str, pd.Series] = {}
    included: list[str] = []
    missing_daily: list[str] = []
    missing_capital: list[str] = []

    for fname in files:
        fname_str = str(fname)
        usage = get_margin_usage(sel_key, fname_str)
        if usage is None:
            missing_daily.append(fname_str)
            missing_capital.append(fname_str)
            continue

        daily_series = _normalize_daily_returns_series(usage.daily_return_pct)
        if daily_series.empty:
            missing_daily.append(fname_str)
            continue

        capital_series_raw = usage.capital_series
        capital_series = _normalize_capital_series(capital_series_raw)
        capital_missing = capital_series.empty
        capital_series = capital_series.reindex(daily_series.index).ffill().fillna(0.0)
        if (capital_series < -1e-6).any():
            raise ValueError(f"Negative capital detected for file '{fname_str}'")

        if capital_missing:
            missing_capital.append(fname_str)

        included.append(fname_str)
        daily_by_file[fname_str] = daily_series
        capital_by_file[fname_str] = capital_series

    return (
        daily_by_file,
        capital_by_file,
        tuple(included),
        tuple(missing_daily),
        tuple(missing_capital),
    )


def get_daily_returns_by_file(sel_key) -> dict[str, pd.Series]:
    """
    Return the normalized per-file daily percent series for this selection.
    Consumers should migrate to this helper instead of reading the artifact map.
    """
    daily_by_file, _, _, _, _ = _collect_per_file_series(sel_key)
    return daily_by_file


def get_daily_pnl_by_file(sel_key) -> dict[str, pd.Series]:
    """
    Return normalized per-file daily P&L series for this selection.
    Derived from the cached EquityBundle.
    """
    parts = dict(sel_key)
    files = tuple(parts.get("files", ()))
    out: dict[str, pd.Series] = {}
    for fname in files:
        bundle = get_equity_bundle(sel_key, fname)
        daily_series = bundle.daily_pnl
        if daily_series is None or daily_series.empty:
            continue
        series = pd.Series(pd.to_numeric(daily_series, errors="coerce"))
        series.index = pd.DatetimeIndex(pd.to_datetime(series.index)).tz_localize(None)
        series = series.sort_index().fillna(0.0)
        out[fname] = series
    return out


def _empty_portfolio_view() -> PortfolioView:
    empty_series = pd.Series(dtype=float)
    return PortfolioView(
        daily_return_pct=empty_series,
        pct_equity_index=empty_series,
        pct_equity=empty_series,
        total_nav=empty_series,
        portfolio_equity=empty_series,
        capital_by_file={},
        daily_returns_by_file={},
        files_included=(),
        missing_daily=(),
        missing_capital=(),
        returns_digest="empty",
    )


def build_portfolio_view(sel_key) -> PortfolioView:
    """
    Aggregate per-file daily return/capital series into a cached portfolio view.
    Provides portfolio daily returns, percent equity, NAV, and diagnostic metadata.
    """
    daily_by_file, capital_by_file, included, missing_daily, missing_capital = _collect_per_file_series(sel_key)
    digest = _portfolio_returns_digest(daily_by_file)
    cache_key = ("portfolio", sel_key)

    with _PORTFOLIO_VIEW_CACHE_LOCK:
        cached = _PORTFOLIO_VIEW_CACHE.get(cache_key)
        if cached and cached[0] == digest:
            return cached[1]

    if not daily_by_file:
        empty_view = _empty_portfolio_view()
        with _PORTFOLIO_VIEW_CACHE_LOCK:
            _PORTFOLIO_VIEW_CACHE[cache_key] = (digest, empty_view)
        return empty_view

    returns_dict = {k: v for k, v in daily_by_file.items() if v is not None and not v.empty}
    if not returns_dict:
        empty_view = _empty_portfolio_view()
        with _PORTFOLIO_VIEW_CACHE_LOCK:
            _PORTFOLIO_VIEW_CACHE[cache_key] = (digest, empty_view)
        return empty_view

    capital_dict = {k: capital_by_file.get(k) for k in returns_dict.keys()}
    capital_dict = {k: v for k, v in capital_dict.items() if v is not None}

    (
        portfolio_daily_return_pct,
        portfolio_pct_equity_index,
        portfolio_pct_equity,
        portfolio_total_nav,
    ) = _compute_portfolio_from_maps(returns_dict, capital_dict)
    portfolio_equity_series = _compute_portfolio_equity_series(sel_key, included)

    if (
        not portfolio_daily_return_pct.empty
        and not portfolio_total_nav.empty
        and not portfolio_daily_return_pct.index.equals(portfolio_total_nav.index)
    ):
        raise ValueError("Portfolio NAV index misaligned with daily returns.")
    if not portfolio_total_nav.empty and (portfolio_total_nav < -1e-6).any():
        raise ValueError("Portfolio NAV contains negative values.")

    view = PortfolioView(
        daily_return_pct=portfolio_daily_return_pct,
        pct_equity_index=portfolio_pct_equity_index,
        pct_equity=portfolio_pct_equity,
        total_nav=portfolio_total_nav,
        portfolio_equity=portfolio_equity_series,
        capital_by_file=capital_by_file,
        daily_returns_by_file=daily_by_file,
        files_included=included,
        missing_daily=missing_daily,
        missing_capital=missing_capital,
        returns_digest=digest,
    )

    with _PORTFOLIO_VIEW_CACHE_LOCK:
        _PORTFOLIO_VIEW_CACHE[cache_key] = (digest, view)
    return view


def get_portfolio_daily_returns(sel_key) -> pd.Series:
    """Convenience accessor for cached portfolio daily percent returns."""
    view = build_portfolio_view(sel_key)
    return view.daily_return_pct


def get_portfolio_total_nav(sel_key) -> pd.Series:
    """Convenience accessor for cached portfolio total NAV series."""
    view = build_portfolio_view(sel_key)
    return view.total_nav


def get_portfolio_equity(sel_key) -> pd.Series:
    """Convenience accessor for cached portfolio cumulative dollar equity."""
    view = build_portfolio_view(sel_key)
    return view.portfolio_equity


def _build_netpos_series_for_slice(file_slice: FileSlice, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> Optional[pd.Series]:
    """Vectorized net-position series for a single file slice."""
    trades = file_slice.trades_full if file_slice.trades_full is not None else file_slice.trades
    if trades is None or trades.empty or start_dt is None or end_dt is None:
        return None

    qty = (
        pd.to_numeric(trades.get("contracts"), errors="coerce")
        .fillna(0.0)
        * float(file_slice.contracts_multiplier or 1.0)
    )
    mask = np.isfinite(qty) & (qty != 0.0)
    if not mask.any():
        return None

    qty = qty[mask]
    entry_ts = pd.to_datetime(trades.get("entry_time"), errors="coerce")[mask]
    exit_ts = pd.to_datetime(trades.get("exit_time"), errors="coerce")[mask]
    direction = (
        trades.get("direction")
        .astype(str)
        .str.strip()
        .str.title()
    )[mask]

    entry_delta = np.where(direction == "Long", qty, np.where(direction == "Short", -qty, 0.0))
    exit_delta = -entry_delta

    events = []
    if entry_ts.notna().any():
        events.append(pd.DataFrame({"ts": entry_ts, "delta": entry_delta, "kind_rank": 1}))
    if exit_ts.notna().any():
        events.append(pd.DataFrame({"ts": exit_ts, "delta": exit_delta, "kind_rank": 0}))
    if not events:
        return None

    ev = pd.concat(events, ignore_index=True)
    ev["ts"] = pd.to_datetime(ev["ts"], errors="coerce")
    ev = ev.dropna(subset=["ts"]).sort_values(["ts", "kind_rank"])
    if ev.empty:
        return None

    carry0 = float(ev.loc[ev["ts"] < start_dt, "delta"].sum())
    in_window = ev.loc[(ev["ts"] >= start_dt) & (ev["ts"] <= end_dt)]
    grouped = in_window.groupby("ts")["delta"].sum().sort_index()
    if grouped.empty and carry0 == 0.0:
        return None
    timeline = pd.Index([start_dt]).append(grouped.index).append(pd.Index([end_dt])).unique().sort_values()

    acc = carry0
    values = []
    last_t = None
    for t in timeline:
        if last_t is None:
            values.append(acc)
        else:
            acc += float(grouped.get(t, 0.0))
            values.append(acc)
        last_t = t
    return pd.Series(values, index=timeline)


@cache.memoize()
def build_core_artifact(sel_key) -> dict:
    """
    Build and cache the 'core selection artifact' for the current selection.
    Relies on existing cached per-file functions, so repeated use is fast.

    Returns a dict with:
      - 'trades_by_file': {fname: DataFrame}
      - 'equity_by_file': {fname: Series}
      - 'label_map':      {fname: str}
      - 'files':          tuple[str, ...]
    """
    # sel_key is a tuple of labeled parts, e.g. (("v", 3), ("files", (...)), ...)
    parts = dict(sel_key)
    debug_token = repr(sel_key)

    data_version = parts.get("v", 0)
    files = tuple(parts.get("files", ()))
    symbol_sel = tuple(parts.get("symbols", ()))
    interval_sel = tuple(parts.get("intervals", ()))
    strat_sel = tuple(parts.get("strategies", ()))
    direction = parts.get("direction", "All")
    start_date = parts.get("start", None)
    end_date = parts.get("end", None)

    if not start_date:
        start_date = None
    if not end_date:
        end_date = None


    contracts_kv = tuple(parts.get("contracts", ()))
    contracts_map: dict[str, Any] = {fname: mult for fname, mult in contracts_kv}

    trades_by_file: dict[str, pd.DataFrame] = {}
    equity_by_file: dict[str, pd.Series] = {}
    label_map: dict[str, str] = {}
    symbol_map: dict[str, str] = {}
    strategy_map: dict[str, str] = {}
    interval_map: dict[str, str] = {}

    netpos_for_debug: Optional[dict[str, pd.Series]] = None

    for fname in files:
        try:
            mult = float(contracts_map.get(fname, 1.0))
        except Exception:
            mult = 1.0
        if mult == 0.0:
            continue

        file_slice = get_file_slice(sel_key, fname)
        tdf = file_slice.trades
        trades_by_file[fname] = tdf

        bundle = get_equity_bundle(sel_key, fname)
        equity_by_file[fname] = bundle.equity

        usage = get_margin_usage(sel_key, fname)
        avg_contracts_abs = usage.avg_contracts_abs
        avg_contracts_signed = usage.avg_contracts_signed
        active_seconds_series = usage.active_seconds
        duty_cycle_series = usage.duty_cycle

        sleeve_returns = usage.daily_return_pct
        label_map[fname] = file_slice.label

        # Extract symbol / strategy / interval metadata for the file.
        if file_slice.symbol:
            symbol_map[fname] = file_slice.symbol
        if file_slice.strategy:
            strategy_map[fname] = file_slice.strategy
        if file_slice.interval:
            interval_map[fname] = file_slice.interval

    portfolio_view = build_portfolio_view(sel_key)
    return {
        "trades_by_file": trades_by_file,
        "equity_by_file": equity_by_file,
        "portfolio_files_included": portfolio_view.files_included,
        "portfolio_missing_daily": portfolio_view.missing_daily,
        "portfolio_missing_capital": portfolio_view.missing_capital,
        "label_map": label_map,
        "files": files,
        "symbol_map": symbol_map,
        "strategy_map": strategy_map,
        "interval_map": interval_map,
    }


# ------------------------------ Net position caching & plotting ------------------------------



@cache.memoize()
def cached_netpos_per_file(data_version: int,
                           files_key: tuple,
                           symbol_sel: tuple, interval_sel: tuple, strat_sel: tuple,
                           direction: str,
                           start_date, end_date,
                           contracts_kv: tuple) -> dict[str, pd.Series]:
    """
    Compute per-file signed net contracts step series WITH:
      • correct carry-in at start_date (sum of deltas before the window)
      • explicit start/end anchors in the timeline
    Returns {filename -> pd.Series(index=datetime, values=signed net contracts)}.
    """
    sel_key = (
        ("v", data_version),
        ("files", files_key),
        ("symbols", symbol_sel),
        ("intervals", interval_sel),
        ("strategies", strat_sel),
        ("direction", direction),
        ("start", start_date or ""),
        ("end", end_date or ""),
        ("contracts", contracts_kv),
    )

    ts_min = None
    ts_max = None
    slices: dict[str, FileSlice] = {}
    for fname in files_key:
        file_slice = get_file_slice(sel_key, fname)
        slices[str(fname)] = file_slice
        trades = file_slice.trades_full if file_slice.trades_full is not None else file_slice.trades
        if trades is None or trades.empty:
            continue
        ent = pd.to_datetime(trades.get("entry_time"), errors="coerce")
        exi = pd.to_datetime(trades.get("exit_time"), errors="coerce")
        series = pd.concat([ent.dropna(), exi.dropna()])
        if series.empty:
            continue
        ts_min = series.min() if ts_min is None else min(ts_min, series.min())
        ts_max = series.max() if ts_max is None else max(ts_max, series.max())

    start_dt, end_dt = _compute_selection_window(sel_key, fallback_min=ts_min, fallback_max=ts_max)
    if start_dt is None or end_dt is None:
        return {}

    out: dict[str, pd.Series] = {}
    for fname in files_key:
        file_slice = slices.get(str(fname)) or get_file_slice(sel_key, fname)
        series = _build_netpos_series_for_slice(file_slice, start_dt, end_dt)
        if series is None or series.empty:
            continue
        out[str(fname)] = series

    return out



# ------------------------------ Caching helpers ------------------------------


@cache.memoize()
def run_allocator_cached(
    sel_key,
    objective: str,
    leverage_cap: float,
    margin_cap: float,       # percent (e.g., 60)
    equity_val: float,       # starting balance
):

    """
    Memoized allocator runner.
    Uses equity & net positions derived from the selection key, then calls
    `find_margin_aware_weights(...)`. Returns (weights: dict, diagnostics: dict).
    """
    # 1) Pull per-file equity curves via the cached bundle
    parts = dict(sel_key)
    files = tuple(parts.get("files", ()))
    equity_by_file: dict[str, pd.Series] = {}
    for fname in files:
        bundle = get_equity_bundle(sel_key, fname)
        if bundle.equity is not None and not bundle.equity.empty:
            equity_by_file[fname] = bundle.equity

    # 2) Unpack selection parts needed for cached net positions
    data_version = parts.get("v", 0)
    files_key    = tuple(parts.get("files", ()))
    symbol_key   = tuple(parts.get("symbols", ()))
    interval_key = tuple(parts.get("intervals", ()))
    strat_key    = tuple(parts.get("strategies", ()))
    direction    = parts.get("direction", "All")
    start_date   = parts.get("start", None)
    end_date     = parts.get("end", None)
    contracts_kv = tuple(parts.get("contracts", ()))
    margins_kv   = tuple(parts.get("margins", ()))

    # 3) Net positions (memoized)
    per_file_netpos = cached_netpos_per_file(
        data_version, files_key,
        symbol_key, interval_key, strat_key,
        direction, start_date, end_date,
        contracts_kv
    )

    # 4) Run allocator
    init_cap   = _coerce_float(equity_val, DEFAULT_INITIAL_CAPITAL)
    margin_buf = max(0.0, min(1.0, float(margin_cap or 60) / 100.0))

    # Allow 0 to mean "margin-only" (no max-contracts cap)
    cap_to_pass = int(leverage_cap) if (leverage_cap is not None and float(leverage_cap) > 0) else 0

    # Build per-file margin override dict
    per_file_margin = {fname: float(m) for fname, m in (margins_kv or [])}

    weights, diag = find_margin_aware_weights(
        equity_by_file=equity_by_file,
        per_file_netpos=per_file_netpos,
        store_trades=app.server.trade_store,
        initial_capital=init_cap,
        margin_spec=MARGIN_SPEC,
        per_file_margin=per_file_margin,
        margin_buffer=margin_buf,
        objective=objective,
        step=1.0,                 # irrelevant in RP but harmless
        leverage_cap=cap_to_pass, # <-- 0 triggers margin-only path
    )
    return weights, diag




@cache.memoize()
def build_margin_series_cached(sel_key, equity_val) -> dict:
    """
    Cached builder for margin-related series.
    Returns {
        'total_init_margin': pd.Series,   # S symbols |net_sym| × IM(sym)
        'pp_series': pd.Series,           # Purchasing Power = equity_val + Portfolio P&L - Initial Margin
    }
    """
    parts = dict(sel_key)

    # Unpack selection (matches make_selection_key parts)
    data_version = parts.get("v", 0)
    files_key    = tuple(parts.get("files", ()))
    symbol_key   = tuple(parts.get("symbols", ()))
    interval_key = tuple(parts.get("intervals", ()))
    strat_key    = tuple(parts.get("strategies", ()))
    direction    = parts.get("direction", "All")
    start_date   = parts.get("start", None)
    end_date     = parts.get("end", None)
    contracts_kv = tuple(parts.get("contracts", ()))

    # 1) Net positions (memoized)
    per_file_netpos = cached_netpos_per_file(
        data_version, files_key,
        symbol_key, interval_key, strat_key,
        direction, start_date, end_date,
        contracts_kv
    )

    # 2) Aggregate by symbol
    by_symbol_net = _aggregate_netpos_per_symbol_from_series(per_file_netpos)

    # 3) Build Initial Margin series = S |net_sym| × IM(sym)
    idx_union = None
    for s in by_symbol_net.values():
        idx_union = s.index if idx_union is None else idx_union.union(s.index)
    idx_union = (idx_union.sort_values() if idx_union is not None else pd.DatetimeIndex([]))

    from src.constants import MARGIN_SPEC  # local import to keep function pure-ish
    total_init_margin = pd.Series(0.0, index=idx_union)
    for sym, s in by_symbol_net.items():
        spec = MARGIN_SPEC.get((sym or "").upper())
        if spec is None:
            continue
        init_margin = float(spec[0])
        total_init_margin = total_init_margin.add(
            s.reindex(idx_union).ffill().fillna(0.0).abs() * init_margin,
            fill_value=0.0
        )

    # Override with per-file margin overrides if provided (sum over files)
    try:
        margins_kv = tuple(parts.get("margins", ()))
        per_file_margin = {str(k): float(v) for k, v in (margins_kv or [])}
        if per_file_margin:
            idx_union2 = None
            for s in per_file_netpos.values():
                idx_union2 = s.index if idx_union2 is None else idx_union2.union(s.index)
            idx_union2 = (idx_union2.sort_values() if idx_union2 is not None else pd.DatetimeIndex([]))
            total_override = pd.Series(0.0, index=idx_union2)
            for fname, s in per_file_netpos.items():
                df_trades = app.server.trade_store.get(fname)
                sym_raw = _symbol_from_first_row(df_trades)
                sym = (sym_raw or "").upper()
                im_val = per_file_margin.get(fname)
                if im_val is None or im_val == 0:
                    spec = MARGIN_SPEC.get(sym)
                    if spec is None:
                        continue
                    im = float(spec[0])
                else:
                    im = float(im_val)
                total_override = total_override.add(
                    s.reindex(idx_union2).ffill().fillna(0.0).abs() * im,
                    fill_value=0.0
                )
            total_init_margin = total_override
    except Exception:
        pass

    # 4) Purchasing Power series using portfolio equity from the cached view
    portfolio_view = build_portfolio_view(sel_key)
    port_eq = portfolio_view.portfolio_equity if portfolio_view.portfolio_equity is not None else pd.Series(dtype=float)
    from src.margin import _purchasing_power_series
    starting_balance = _coerce_float(equity_val, DEFAULT_INITIAL_CAPITAL)
    pp_series = _purchasing_power_series(port_eq, total_init_margin, starting_balance)

    return {"total_init_margin": total_init_margin, "pp_series": pp_series}






@cache.memoize()
def build_corr_matrix_cached(
    sel_key,
    mode: str = "drawdown_pct",
    slope_window: int = 20,
    method: str = "spearman",
):
    """
    Cached correlation heatmap builder.
    Uses equity_by_file from the core selection artifact and memoizes
    by (sel_key, mode, slope_window, method).
    """
    art = build_core_artifact(sel_key)
    equity_by_file = {k: v for k, v in art.get("equity_by_file", {}).items() if v is not None and not v.empty}
    if not equity_by_file:
        return go.Figure().update_layout(margin=dict(l=10, r=10, t=30, b=10))
    label_map = art.get("label_map", {}) or {}
    return build_correlation_heatmap(
        equity_by_file,
        label_map=label_map,
        mode=mode or "drawdown_pct",
        slope_window=int(slope_window or 20),
        method=method or "spearman",
    )





@cache.memoize()
def compute_metrics_cached(sel_key, per_file_stats: bool = True, monthly_mode: str = "full_months"):
    """
    Build metrics rows once per selection via the core artifact.
    - sel_key: stable, hashable key from make_selection_key(...)
    - per_file_stats: keep True (table shows per-file rows)
    - monthly_mode: reserved for future switches; current compute_metrics uses full months already
    Returns: list[dict] of rows including a final 'Portfolio' row (if any data)
    """
    parts = dict(sel_key)
    selected_files = list(parts.get("files", ()))

    art = build_core_artifact(sel_key)
    trades_by_file = art.get("trades_by_file", {}) or {}
    portfolio_daily_pct = get_portfolio_daily_returns(sel_key)

    rows = []
    frames_for_port = []

    for f in selected_files:
        tdf = trades_by_file.get(f)
        if tdf is None or tdf.empty:
            continue
        frames_for_port.append(tdf)
        row = {
            "File": f,
            "Symbol": (tdf["Symbol"].iloc[0] if "Symbol" in tdf and not tdf.empty else ""),
            "Interval": (int(tdf["Interval"].iloc[0]) if "Interval" in tdf and not tdf.empty and not pd.isna(tdf["Interval"].iloc[0]) else None),
            "Strategy": (tdf["Strategy"].iloc[0] if "Strategy" in tdf and not tdf.empty else ""),
            # Direction is a selection attribute; mirror current table behavior
            "Direction": parts.get("direction", "All"),
        }
        usage = get_margin_usage(sel_key, f)
        daily_pct_series = _normalize_daily_returns_series(usage.daily_return_pct if usage else None)
        row.update(compute_metrics(tdf, DEFAULT_INITIAL_CAPITAL, daily_pct_series))
        rows.append(row)

    if frames_for_port:
        pf_trades = pd.concat(frames_for_port, ignore_index=True)
        pf_row = {"File": "Portfolio", "Symbol": "-", "Interval": None, "Strategy": "-", "Direction": parts.get("direction", "All")}
        pf_row.update(compute_metrics(pf_trades, DEFAULT_INITIAL_CAPITAL, portfolio_daily_pct))
        rows.append(pf_row)

    return rows


riskfolio_adapter.configure(
    artifact_fn=build_core_artifact,
    initial_capital=DEFAULT_INITIAL_CAPITAL,
    daily_returns_fn=get_daily_returns_by_file,
    daily_pnl_fn=get_daily_pnl_by_file,
)


@cache.memoize()
def cached_filtered_df(data_version: int, fname: str,
                       symbol_sel: tuple, interval_sel: tuple, strat_sel: tuple,
                       direction: str, start_date, end_date, mult: float) -> pd.DataFrame:
    """
    Return a filtered (and scaled) view of a file's DataFrame.
    Uses a version key so uploads invalidate cache.
    """
    df = app.server.trade_store.get(fname)
    if df is None or df.empty:
        return pd.DataFrame()
    out = df
    if symbol_sel:
        out = out[out["Symbol"].isin(symbol_sel)]
    if interval_sel:
        out = out[out["Interval"].isin(interval_sel)]
    if strat_sel:
        out = out[out["Strategy"].isin(strat_sel)]
    if direction in ("Long", "Short"):
        out = out[out["direction"] == direction]
    out = _within_dates(out, start_date, end_date, col="exit_time")
    if mult is not None and mult != 1.0 and not out.empty:
        for c in ["net_profit", "runup", "drawdown_trade", "gross_profit",
                  "commission", "slippage", "contracts", "CumulativePL_raw",
                  "notional_exposure"]:
            if c in out.columns:
                out = out.copy()
                out.loc[:, c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0) * float(mult)
    return out


@cache.memoize()
def cached_mtm_df(data_version: int, fname: str,
                  symbol_sel: tuple, interval_sel: tuple, strat_sel: tuple,
                  direction: str, start_date, end_date, mult: float) -> pd.DataFrame:
    """
    Return filtered mark-to-market rows for a file (Period -> Net Profit).
    Currently only supports direction='All' because MTM rows are aggregated per file.
    """
    mtm_store = getattr(app.server, "mtm_store", None)
    if not mtm_store:
        return pd.DataFrame()

    df = mtm_store.get(fname)
    if df is None or df.empty:
        return pd.DataFrame()

    # MTM data is already per-file; symbol/interval/strategy filters should either include or exclude the file.
    # Direction-specific filters cannot be separated from aggregated MTM rows -> fallback to empty (handled by caller).
    if direction in ("Long", "Short"):
        return pd.DataFrame()

    out = df.copy()
    out["mtm_date"] = pd.to_datetime(out["mtm_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["mtm_date"])

    if start_date:
        out = out[out["mtm_date"] >= pd.to_datetime(start_date)]
    if end_date:
        # ensure inclusive end date
        end_dt = pd.to_datetime(end_date)
        out = out[out["mtm_date"] <= end_dt]

    if mult is not None and mult != 1.0 and not out.empty:
        out = out.copy()
        out["mtm_net_profit"] = pd.to_numeric(out["mtm_net_profit"], errors="coerce").fillna(0.0) * float(mult)

    out = out.dropna(subset=["mtm_net_profit"])
    return out.reset_index(drop=True)


def _hybrid_equity_from_trades_and_mtm(trades_df: pd.DataFrame, mtm_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Build a cumulative equity series combining mark-to-market daily data with trade-book anchors.

    Returns (cumulative_equity, daily_pnl_series). Both indexed by normalized dates.
    """
    # Base case: no data
    if (mtm_df is None or mtm_df.empty) and (trades_df is None or trades_df.empty):
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # If no MTM, fall back to trade-based equity (no override series)
    if mtm_df is None or mtm_df.empty:
        eq = equity_from_trades_subset(trades_df)
        return eq, pd.Series(dtype=float)

    mtm_df = mtm_df.copy()
    mtm_df["mtm_date"] = pd.to_datetime(mtm_df["mtm_date"]).dt.normalize()
    mtm_df["mtm_net_profit"] = pd.to_numeric(mtm_df["mtm_net_profit"], errors="coerce").fillna(0.0)
    mtm_daily = mtm_df.groupby("mtm_date")["mtm_net_profit"].sum().sort_index()

    trades_closed = pd.DataFrame()
    if trades_df is not None and not trades_df.empty:
        trades_closed = trades_df.dropna(subset=["exit_time"]).copy()
        if not trades_closed.empty:
            trades_closed["exit_date"] = pd.to_datetime(trades_closed["exit_time"]).dt.normalize()
            trades_closed = trades_closed.sort_values("exit_time")

    # Union index across MTM days and exit dates
    idx = mtm_daily.index
    if not trades_closed.empty:
        idx = idx.union(pd.DatetimeIndex(trades_closed["exit_date"])).sort_values()
    else:
        idx = idx.sort_values()
    if idx.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Ensure daily granularity covers flat periods between trades
    idx_full = pd.date_range(start=idx.min(), end=idx.max(), freq="D")

    mtm_daily = mtm_daily.reindex(idx_full, fill_value=0.0)
    cumulative = mtm_daily.cumsum()

    tolerance = 1e-6

    if not trades_closed.empty:
        # Ensure CumulativePL_raw ready
        cum_col = pd.to_numeric(trades_closed.get("CumulativePL_raw"), errors="coerce")
        for exit_date, target in zip(trades_closed["exit_date"], cum_col):
            if pd.isna(target) or pd.isna(exit_date):
                continue
            exit_date = pd.Timestamp(exit_date)
            if exit_date not in cumulative.index:
                continue
            current_value = cumulative.loc[exit_date]
            diff = float(target) - float(current_value)
            if abs(diff) <= tolerance:
                continue
            # Adjust the daily increment on that exit date so the cumulative snaps to trade ledger
            mtm_daily.loc[exit_date] += diff
            cumulative.loc[cumulative.index >= exit_date] += diff

    # Recompute final cumulative to ensure alignment with adjusted increments
    cumulative = mtm_daily.cumsum()

    # Build open/close anchors for each day so intraday metrics have a path
    open_values = cumulative - mtm_daily  # previous day's close (0 for first day)
    eq_index: list[pd.Timestamp] = []
    eq_values: list[float] = []
    for day in cumulative.index:
        open_ts = _mtm_day_open(day)
        close_ts = _mtm_day_close(day)
        eq_index.append(open_ts)
        eq_values.append(float(open_values.loc[day]))
        eq_index.append(close_ts)
        eq_values.append(float(cumulative.loc[day]))

    equity_series = pd.Series(eq_values, index=pd.DatetimeIndex(eq_index)).sort_index()
    equity_series.name = "equity"
    daily = mtm_daily.copy()
    daily.name = "daily_pnl"
    return equity_series, daily


def _percent_equity_series_from_daily(daily_pct: pd.Series) -> pd.Series:
    """
    Convert daily returns (in decimal form) to an 'equity index' Series (1+ cumulative return)
    with explicit open/close timestamps per day.
    """
    if daily_pct is None or daily_pct.empty:
        return pd.Series(dtype=float)
    daily_pct = daily_pct.sort_index()
    cumulative = 0.0
    eq_index: list[pd.Timestamp] = []
    eq_values: list[float] = []
    for day, ret in daily_pct.items():
        open_ts = _mtm_day_open(day)
        close_ts = _mtm_day_close(day)
        eq_index.append(open_ts)
        eq_values.append(1.0 + cumulative)
        cumulative += float(ret)
        eq_index.append(close_ts)
        eq_values.append(1.0 + cumulative)
    return pd.Series(eq_values, index=pd.DatetimeIndex(eq_index)).sort_index()


def _percent_equity_with_spikes(eq_series: pd.Series, trades_df: pd.DataFrame, *, tolerance: float = 1e-9) -> pd.Series:
    """
    Inject additional percent drawdown spikes (analogous to dollar equity spikes).
    Operates on an equity index (1 + cumulative return) series.
    """
    if eq_series is None or eq_series.empty or trades_df is None or trades_df.empty:
        return eq_series

    eq_base = eq_series.sort_index()
    eq_spiked = eq_base.copy()

    trades = trades_df.dropna(subset=["exit_time"]).copy()
    if trades.empty:
        return eq_base

    trades["entry_date"] = pd.to_datetime(trades["entry_time"], errors="coerce").dt.normalize()
    trades["exit_date"] = pd.to_datetime(trades["exit_time"], errors="coerce").dt.normalize()
    offset_seconds = 0

    for _, trade in trades.iterrows():
        notional = float(trade.get("notional_exposure") or 0.0)
        drawdown_val = float(trade.get("drawdown_trade") or 0.0)
        target = abs(drawdown_val)
        if notional <= tolerance or target <= tolerance:
            continue

        entry_ts = trade.get("entry_date")
        exit_ts = trade.get("exit_date")
        if pd.isna(entry_ts) or pd.isna(exit_ts):
            continue

        entry_ts = _mtm_day_open(entry_ts)
        exit_ts = _mtm_day_close(exit_ts)

        window = eq_base.loc[(eq_base.index >= entry_ts) & (eq_base.index <= exit_ts)]
        if window.empty:
            continue

        base_value = float(window.iloc[0])
        current_min_idx = window.idxmin()
        current_min_value = float(window.loc[current_min_idx])
        actual_drop = base_value - current_min_value
        target_drop = base_value * (target / notional)
        if target_drop <= actual_drop + tolerance:
            continue

        desired_min = max(base_value - target_drop, tolerance)
        day_anchor = (pd.Timestamp(current_min_idx) - MTM_DAY_START_OFFSET).normalize()
        drop_time = _mtm_day_open(day_anchor) + MTM_SESSION_HALF + pd.Timedelta(seconds=offset_seconds)
        recover_time = drop_time + pd.Timedelta(milliseconds=1)
        offset_seconds += 2

        eq_spiked.loc[drop_time] = desired_min
        eq_spiked.loc[recover_time] = current_min_value

    return eq_spiked.sort_index()


def _percent_equity_series_to_cumulative(eq_series: pd.Series) -> pd.Series:
    """
    Collapse an equity index Series back to daily cumulative percentage returns.
    """
    if eq_series is None or eq_series.empty:
        return pd.Series(dtype=float)
    closes = eq_series.groupby(_mtm_bucketize_index(eq_series.index)).last()
    return closes - 1.0


def _common_x_range(*items) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Determine a shared x-axis range across Series/DataFrames with datetime-like indexes.
    """
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    for item in items:
        if item is None:
            continue
        if isinstance(item, pd.DataFrame):
            idx = item.index if not item.empty else None
        elif isinstance(item, pd.Series):
            idx = item.index if not item.empty else None
        else:
            continue
        if idx is None or len(idx) == 0:
            continue
        idx = pd.DatetimeIndex(idx)
        mn = idx.min()
        mx = idx.max()
        start = mn if start is None or mn < start else start
        end = mx if end is None or mx > end else end
    if start is None or end is None:
        return None
    return (pd.Timestamp(start), pd.Timestamp(end))


def _combine_percent_equity(pct_by_file: dict[str, pd.Series],
                            files: list[str],
                            portfolio_series: Optional[pd.Series]) -> pd.DataFrame:
    """
    Combine cumulative percent-return curves into a DataFrame aligned by date.
    Series should represent cumulative return (e.g., (1+r).cumprod() - 1).
    """
    cols: dict[str, pd.Series] = {}
    for fname in files:
        series = pct_by_file.get(fname)
        if series is None or series.empty:
            continue
        cols[fname] = series.sort_index()

    if portfolio_series is not None and not portfolio_series.empty:
        cols["Portfolio"] = portfolio_series.sort_index()

    if not cols:
        return pd.DataFrame()

    df = pd.concat(cols, axis=1).sort_index()
    df = df.ffill()
    df = df.fillna(0.0)
    return df


def _equity_with_spikes(equity: pd.Series, trades_df: pd.DataFrame, *, tolerance: float = 1e-6) -> pd.Series:
    """
    Overlay trade-level intraday drawdown spikes on top of the provided equity curve.
    Spikes are inserted midway through the day of the observed minimum so that charts
    show a vertical drop and recovery without disturbing close-to-close P/L.
    """
    if equity is None or equity.empty or trades_df is None or trades_df.empty:
        return equity

    eq_base = equity.sort_index()
    eq_spiked = eq_base.copy()

    trades = trades_df.dropna(subset=["exit_time"]).copy()
    if trades.empty:
        return eq_base

    trades["entry_date"] = pd.to_datetime(trades["entry_time"], errors="coerce").dt.normalize()
    trades["exit_date"] = pd.to_datetime(trades["exit_time"], errors="coerce").dt.normalize()

    offset_seconds = 0

    for _, trade in trades.iterrows():
        drawdown_val = float(trade.get("drawdown_trade") or 0.0)
        target = abs(drawdown_val)
        if target <= tolerance:
            continue

        entry_date = trade.get("entry_date")
        exit_date = trade.get("exit_date")
        if pd.isna(entry_date) or pd.isna(exit_date):
            continue

        entry_ts = _mtm_day_open(entry_date)
        exit_ts = _mtm_day_close(exit_date)
        window = eq_base.loc[(eq_base.index >= entry_ts) & (eq_base.index <= exit_ts)]
        if window.empty:
            continue

        base_value = float(window.iloc[0])

        current_min_idx = window.idxmin()
        current_min_value = float(window.loc[current_min_idx])
        actual_drop = max(0.0, base_value - current_min_value)
        if target <= actual_drop + tolerance:
            continue

        desired_min = base_value - target
        if desired_min >= current_min_value - tolerance:
            desired_min = current_min_value - tolerance

        day_anchor = (pd.Timestamp(current_min_idx) - MTM_DAY_START_OFFSET).normalize()
        drop_time = _mtm_day_open(day_anchor) + MTM_SESSION_HALF + pd.Timedelta(seconds=offset_seconds)
        recover_time = drop_time + pd.Timedelta(milliseconds=1)
        offset_seconds += 2  # ensure uniqueness

        eq_spiked.loc[drop_time] = desired_min
        eq_spiked.loc[recover_time] = current_min_value

    return eq_spiked.sort_index()
@cache.memoize()
def cached_equity_series(data_version: int, fname: str,
                         symbol_sel: tuple, interval_sel: tuple, strat_sel: tuple,
                         direction: str, start_date, end_date, mult: float) -> pd.Series:
    tdf = cached_filtered_df(data_version, fname, symbol_sel, interval_sel, strat_sel, direction, start_date, end_date, mult)
    return equity_from_trades_subset(tdf)


from dash.dependencies import ALL, MATCH

@app.callback(
    Output("file-contracts", "children"),
    Input("file-toggle", "options"),
    State("store-contracts", "data"),
    State("store-margins", "data"),
    prevent_initial_call=False
)
def _render_contract_inputs(file_options, contracts_map, margins_map):
    # Render per-file controls: Contracts and Daily Margin $/contract
    if not file_options:
        return []
    kids = []
    from src.constants import MARGIN_SPEC as _MS
    for opt in file_options:
        fname = opt["value"]
        # Default contracts
        contracts_val = float((contracts_map or {}).get(fname, 1.0))
        # Default margin from file's symbol -> MARGIN_SPEC initial margin
        df_trades = app.server.trade_store.get(fname) if hasattr(app, "server") else None
        sym_raw = _symbol_from_first_row(df_trades)
        spec = _MS.get((sym_raw or "").upper()) if sym_raw else None
        default_margin = float(spec[0]) if spec is not None else 0.0
        margin_val = float((margins_map or {}).get(fname, default_margin))

        label_style = {"fontSize": "11px", "color": "#555", "whiteSpace": "nowrap"}
        button_style = {"fontSize": "11px", "padding": "2px 8px", "cursor": "pointer", "whiteSpace": "nowrap"}

        kids.append(
            html.Div(
                [
                    html.Span(
                        fname,
                        style={
                            "fontSize": "12px",
                            "fontWeight": 600,
                            "minWidth": "160px",
                            "whiteSpace": "nowrap"
                        }
                    ),
                    html.Span("Contracts:", style=label_style),
                    dcc.Input(
                        id={"type": "contracts-input", "index": fname},
                        type="number",
                        value=contracts_val,
                        min=0,
                        step=1,
                        style={"width": "80px"}
                    ),
                    html.Span("Daily Margin $/contract:", style=label_style),
                    dcc.Input(
                        id={"type": "margin-input", "index": fname},
                        type="text",
                        value=f"{margin_val:g}",
                        style={"width": "100px"}
                    ),
                    html.Button(
                        "Use default",
                        id={"type": "margin-reset", "index": fname},
                        n_clicks=0,
                        style=button_style
                    ),
                ],
                style={
                    "border": "1px solid #eee",
                    "borderRadius": "8px",
                    "padding": "8px 12px",
                    "background": "#fafafa",
                    "display": "grid",
                    "gridTemplateColumns": "max-content auto 80px auto 110px auto",
                    "alignItems": "center",
                    "columnGap": "12px",
                    "rowGap": "6px",
                    "justifyItems": "start",
                    "width": "fit-content",
                    "justifySelf": "start",
                }
            )
        )
    return kids

# -------------------- Per-file Contracts Map (Store) --------------------

@app.callback(
    Output("store-contracts", "data"),
    Input({"type": "contracts-input", "index": ALL}, "value"),
    State({"type": "contracts-input", "index": ALL}, "id"),
    prevent_initial_call=False
)
def _collect_contracts_map(values, ids):
    # Build { filename: multiplier } from the pattern IDs and current values
    if not ids:
        return {}
    out = {}
    for val, _id in zip(values, ids):
        fname = _id.get("index")
        try:
            mul = float(val) if val is not None else 1.0
        except Exception:
            mul = 1.0
        out[str(fname)] = mul
    return out

# -------------------- Per-file Margins Map (Store) --------------------

@app.callback(
    Output("store-margins", "data"),
    Input({"type": "margin-input", "index": ALL}, "value"),
    State({"type": "margin-input", "index": ALL}, "id"),
    prevent_initial_call=False
)
def _collect_margins_map(values, ids):
    if not ids:
        return {}
    out = {}
    for val, _id in zip(values, ids):
        fname = _id.get("index")
        try:
            if val is None:
                m = 0.0
            else:
                text_val = str(val).strip()
                m = float(text_val) if text_val else 0.0
        except Exception:
            m = 0.0
        out[str(fname)] = m
    return out


@app.callback(
    Output({"type": "margin-input", "index": MATCH}, "value"),
    Input({"type": "margin-reset", "index": MATCH}, "n_clicks"),
    prevent_initial_call=True
)
def _reset_margin_to_default(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    triggered = ctx.triggered_id
    if not triggered:
        raise PreventUpdate

    fname = triggered.get("index")
    if not fname:
        raise PreventUpdate

    df_trades = app.server.trade_store.get(fname) if hasattr(app, "server") else None
    sym_raw = _symbol_from_first_row(df_trades)
    spec = MARGIN_SPEC.get((sym_raw or "").upper())
    default_margin = float(spec[0]) if spec else 0.0
    return f"{default_margin:g}"



@app.callback(
    Output("store-trades", "data"),     # tiny token: tuple[str, ...]
    Output("file-list", "children"),
    Output("file-toggle", "options"),
    Output("file-toggle", "value"),
    Output("symbol-toggle", "options"),
    Output("symbol-toggle", "value"),
    Output("interval-toggle", "options"),
    Output("interval-toggle", "value"),
    Output("strategy-toggle", "options"),
    Output("strategy-toggle", "value"),
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
    Output("store-version", "data"),    # bump to invalidate caches
    Input("upload", "contents"),
    State("upload", "filename"),
    State("store-trades", "data"),      # previously stored token: tuple[str, ...]
    State("file-toggle", "value"),
    State("symbol-toggle", "value"),
    State("interval-toggle", "value"),
    State("strategy-toggle", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    prevent_initial_call=True
)
def ingest_files(contents_list, names_list, prev_tokens, prev_file_sel, prev_sym_sel, prev_int_sel, prev_strat_sel, prev_start, prev_end):
    if not contents_list:
        raise PreventUpdate

    # Normalize to lists
    if isinstance(contents_list, str):
        contents_list = [contents_list]
        names_list = [names_list]

    # Start from previously stored token (tuple of filenames)
    tokens_prev = tuple(prev_tokens) if prev_tokens else tuple()
    existing_names = set(tokens_prev)

    # Accumulate across this upload batch
    names_accum: list[str] = list(tokens_prev)
    newly_added: list[str] = []

    # Parse and stash DataFrames server-side; store tiny tokens in client store
    for contents, fname in zip(contents_list, names_list):
        raw_bytes, disp_name = decode_upload(contents, fname)
        disp_name = _safe_unique_name(disp_name, existing_names)
        existing_names.add(disp_name)

        tdf, mtm_daily = parse_tradestation_trades(raw_bytes, disp_name)
        # Keep the real DF only on the server
        app.server.trade_store[disp_name] = tdf
        app.server.mtm_store[disp_name] = mtm_daily

        # Track new names
        names_accum.append(disp_name)
        newly_added.append(disp_name)

    # File options and selected
    file_names = sorted(set(names_accum))
    file_options = [{"label": n, "value": n} for n in file_names]
    if prev_file_sel:
        file_selected = prev_file_sel + [n for n in newly_added if n not in prev_file_sel]
    else:
        file_selected = file_names

    # Keep server-side stores in sync with active file list
    existing_keys = list(getattr(app.server, "trade_store", {}).keys())
    for key in existing_keys:
        if key not in file_names:
            app.server.trade_store.pop(key, None)
    mtm_store = getattr(app.server, "mtm_store", None)
    if mtm_store is not None:
        for key in list(mtm_store.keys()):
            if key not in file_names:
                mtm_store.pop(key, None)

    # File list UI
    file_list_view = html.Ul([html.Li(n) for n in file_names])

    # Compute global min/max dates + all symbols/intervals/strategies from server DFs
    all_min, all_max = None, None
    all_symbols: set[str] = set()
    all_intervals: set[int] = set()
    all_strategies: set[str] = set()

    for fname in file_names:
        tdf = app.server.trade_store.get(fname)
        if tdf is None or tdf.empty:
            continue
        if "Symbol" in tdf:
            all_symbols.update(tdf["Symbol"].dropna().astype(str).unique().tolist())
        if "Interval" in tdf:
            all_intervals.update(pd.to_numeric(tdf["Interval"], errors="coerce").dropna().astype(int).unique().tolist())
        if "Strategy" in tdf:
            all_strategies.update(tdf["Strategy"].dropna().astype(str).unique().tolist())

        dt = pd.to_datetime(tdf.get("exit_time"), errors="coerce").dropna()
        if dt.empty:
            continue
        mn, mx = dt.min().date(), dt.max().date()
        all_min = mn if (all_min is None or mn < all_min) else all_min
        all_max = mx if (all_max is None or mx > all_max) else all_max

    sym_options = [{"label": s, "value": s} for s in sorted(all_symbols)]
    int_options = [{"label": str(i), "value": int(i)} for i in sorted(all_intervals)]
    strat_options = [{"label": s, "value": s} for s in sorted(all_strategies)]

    sym_selected = [v for v in (prev_sym_sel or []) if v in all_symbols] or [s for s in sorted(all_symbols)]
    int_selected = [v for v in (prev_int_sel or []) if v in all_intervals] or [i for i in sorted(all_intervals)]
    strat_selected = [v for v in (prev_strat_sel or []) if v in all_strategies] or [s for s in sorted(all_strategies)]

    start_date = prev_start if prev_start else (all_min if all_min else None)
    end_date   = prev_end   if prev_end   else (all_max if all_max else None)

    version_stamp = int(datetime.now().timestamp())

    return (
        tuple(file_names),        # tiny token (tuple of filenames)
        file_list_view,
        file_options,
        file_selected,
        sym_options, sym_selected,
        int_options, int_selected,
        strat_options, strat_selected,
        start_date, end_date,
        version_stamp,
    )



# ---- Equity Tab Callback ---- #

@app.callback(
    Output("equity_tab_content", "children"),
    # Shared selection Inputs (affect equity everywhere)
    Input("store-trades", "data"),
    Input("file-toggle", "value"),
    Input("symbol-toggle", "value"),
    Input("interval-toggle", "value"),
    Input("strategy-toggle", "value"),
    Input("direction-radio", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("store-contracts", "data"),
    Input("store-margins", "data"),
    Input("store-version", "data"),
    Input("spike-toggle", "value"),
    prevent_initial_call=False
)
def update_equity_tab(store_trades, selected_files, selected_symbols, selected_intervals,
                      selected_strategies, direction, start_date, end_date,
                      contracts_map, margins_map, store_version,
                      spike_toggle):

    fig_empty = go.Figure().update_layout(margin=dict(l=10, r=10, t=30, b=10))
    if not store_trades:
        return _two_graphs_child(fig_empty, fig_empty)

    selected_files = [f for f in (selected_files or []) if store_trades and f in set(store_trades)]
    if not selected_files:
        return _two_graphs_child(fig_empty, fig_empty)

    sel_key = make_selection_key(
        selected_files, selected_symbols, selected_intervals, selected_strategies,
        direction, start_date, end_date, contracts_map, margins_map, store_version
    )
    art = build_core_artifact(sel_key)
    portfolio_view = build_portfolio_view(sel_key)
    trades_by_file = art.get("trades_by_file", {}) or {}

    equity_by_file = {k: v for k, v in art["equity_by_file"].items() if v is not None and not v.empty}
    if not equity_by_file:
        return _two_graphs_child(fig_empty, fig_empty)

    include_spikes = _spike_toggle_enabled(spike_toggle)
    equity_dollar = equity_by_file
    if include_spikes:
        equity_dollar = {
            fname: _equity_with_spikes(eq, trades_by_file.get(fname))
            for fname, eq in equity_by_file.items()
        }

    label_map = art["label_map"]
    eq_all = combine_equity(equity_dollar, list(equity_by_file.keys()))
    if eq_all is None or eq_all.empty:
        fig_dollar = go.Figure().update_layout(
            title="Equity Curves ($) unavailable for the current selection",
            margin=dict(l=10, r=10, t=40, b=10)
        )
    else:
        fig_dollar = build_equity_figure(eq_all, label_map, COLOR_PORTFOLIO)
        fig_dollar.update_layout(title=dict(text="Equity Curves ($)", y=0.97))

    pct_series_map: dict[str, pd.Series] = {}
    for fname in equity_by_file.keys():
        curve = get_percent_equity(sel_key, fname, demand_flag="equity-tab")
        idx_series = curve.index_series
        if idx_series is None or idx_series.empty:
            continue
        percent_index = idx_series
        if include_spikes:
            percent_index = _percent_equity_with_spikes(percent_index, trades_by_file.get(fname))
        pct_series_map[fname] = percent_index - 1.0

    pct_port_index = portfolio_view.pct_equity_index
    if pct_port_index is not None and not pct_port_index.empty:
        if include_spikes:
            trade_frames = [df for df in trades_by_file.values() if df is not None and not df.empty]
            all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else None
            pct_port_index = _percent_equity_with_spikes(pct_port_index, all_trades)
        pct_port_series = pct_port_index - 1.0
    else:
        pct_port_series = pd.Series(dtype=float)

    pct_df = _combine_percent_equity(pct_series_map, list(equity_by_file.keys()), pct_port_series)
    if pct_df is None or pct_df.empty:
        fig_percent = go.Figure().update_layout(
            title="Equity Curves (%) unavailable for the current selection",
            margin=dict(l=10, r=10, t=40, b=10)
        )
    else:
        fig_percent = build_equity_figure(pct_df, label_map, COLOR_PORTFOLIO, value_is_percent=True)
        fig_percent.update_layout(title=dict(text="Equity Curves (%)", y=0.97))

    x_range = _common_x_range(eq_all, pct_df)
    if x_range:
        fig_dollar.update_xaxes(range=x_range)
        fig_percent.update_xaxes(range=x_range)

    return _two_graphs_child(fig_dollar, fig_percent, ids=("equity-dollar", "equity-percent"), heights=(360, 360))



# ---- Portfolio Drawdown Tab Callback ---- #

@app.callback(
    Output("drawdown_tab_content", "children"),
    Input("store-trades", "data"),
    Input("file-toggle", "value"),
    Input("symbol-toggle", "value"),
    Input("interval-toggle", "value"),
    Input("strategy-toggle", "value"),
    Input("direction-radio", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("store-contracts", "data"),
    Input("store-margins", "data"),
    Input("store-version", "data"),
    Input("spike-toggle", "value"),
    prevent_initial_call=False
)
def update_drawdown_tab(store_trades, selected_files, selected_symbols, selected_intervals,
                        selected_strategies, direction, start_date, end_date,
                        contracts_map, margins_map, store_version,
                        spike_toggle):

    fig_empty = go.Figure().update_layout(margin=dict(l=10, r=10, t=30, b=10))
    if not store_trades:
        return _two_graphs_child(fig_empty, fig_empty)
    selected_files = [f for f in (selected_files or []) if store_trades and f in set(store_trades)]
    if not selected_files:
        return _two_graphs_child(fig_empty, fig_empty)

    sel_key = make_selection_key(selected_files, selected_symbols, selected_intervals,
                                 selected_strategies, direction, start_date, end_date,
                                 contracts_map, margins_map, store_version)
    art = build_core_artifact(sel_key)
    portfolio_view = build_portfolio_view(sel_key)
    trades_by_file = art.get("trades_by_file", {}) or {}
    equity_by_file = {k: v for k, v in art["equity_by_file"].items() if v is not None and not v.empty}
    if not equity_by_file:
        return _two_graphs_child(fig_empty, fig_empty)

    include_spikes = _spike_toggle_enabled(spike_toggle)
    equity_dollar = equity_by_file
    if include_spikes:
        equity_dollar = {
            fname: _equity_with_spikes(eq, trades_by_file.get(fname))
            for fname, eq in equity_by_file.items()
        }

    eq_all = combine_equity(equity_dollar, list(equity_by_file.keys()))
    if eq_all is None or eq_all.empty:
        fig_dollar = go.Figure().update_layout(
            title="Portfolio Drawdown ($) unavailable for the current selection",
            margin=dict(l=10, r=10, t=40, b=10)
        )
    else:
        fig_dollar = build_drawdown_figure(eq_all.get("Portfolio") if eq_all is not None else None)
        fig_dollar.update_layout(title="Portfolio Drawdown ($)")

    pct_port_index = portfolio_view.pct_equity_index

    if pct_port_index is not None and not pct_port_index.empty:
        if include_spikes:
            trade_frames = [df for df in trades_by_file.values() if df is not None and not df.empty]
            all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else None
            pct_port_index = _percent_equity_with_spikes(pct_port_index, all_trades)
        pct_port_series = pct_port_index - 1.0
    else:
        pct_port_series = pd.Series(dtype=float)

    if pct_port_series is None or pct_port_series.empty:
        fig_percent = go.Figure().update_layout(
            title="No percent-based drawdown data for the current selection",
            margin=dict(l=10, r=10, t=40, b=10)
        )
    else:
        fig_percent = build_drawdown_figure(pct_port_series, value_is_percent=True)
        fig_percent.update_layout(title="Portfolio Drawdown (%)")

    x_range = _common_x_range(eq_all, pct_port_series)
    if x_range:
        fig_dollar.update_xaxes(range=x_range)
        fig_percent.update_xaxes(range=x_range)

    return _two_graphs_child(fig_dollar, fig_percent, ids=("dd-dollar", "dd-percent"), heights=(360, 360))


# ---- Intraday Drawdown Tab Callback ---- #


@app.callback(
    Output("intraday_tab_content", "children"),
    Input("store-trades", "data"),
    Input("file-toggle", "value"),
    Input("symbol-toggle", "value"),
    Input("interval-toggle", "value"),
    Input("strategy-toggle", "value"),
    Input("direction-radio", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("store-contracts", "data"),
    Input("store-margins", "data"),
    Input("store-version", "data"),
    Input("spike-toggle", "value"),
    prevent_initial_call=False
)
def update_intraday_tab(store_trades, selected_files, selected_symbols, selected_intervals,
                        selected_strategies, direction, start_date, end_date,
                        contracts_map, margins_map, store_version,
                        spike_toggle):

    fig_empty = go.Figure().update_layout(margin=dict(l=10, r=10, t=30, b=10))
    if not store_trades:
        return _two_graphs_child(fig_empty, fig_empty)
    selected_files = [f for f in (selected_files or []) if store_trades and f in set(store_trades)]
    if not selected_files:
        return _two_graphs_child(fig_empty, fig_empty)

    sel_key = make_selection_key(selected_files, selected_symbols, selected_intervals,
                                 selected_strategies, direction, start_date, end_date,
                                 contracts_map, margins_map, store_version)
    art = build_core_artifact(sel_key)
    portfolio_view = build_portfolio_view(sel_key)
    trades_by_file = art.get("trades_by_file", {}) or {}

    equity_by_file = {k: v for k, v in art["equity_by_file"].items() if v is not None and not v.empty}
    if not equity_by_file:
        return _two_graphs_child(fig_empty, fig_empty)

    include_spikes = _spike_toggle_enabled(spike_toggle)
    equity_dollar = equity_by_file
    if include_spikes:
        equity_dollar = {
            fname: _equity_with_spikes(eq, trades_by_file.get(fname))
            for fname, eq in equity_by_file.items()
        }

    eq_all = combine_equity(equity_dollar, list(equity_by_file.keys()))
    portfolio_series = None
    if eq_all is not None and not eq_all.empty:
        portfolio_series = eq_all.get("Portfolio")

    if portfolio_series is None or portfolio_series.empty:
        fig_dollar = go.Figure().update_layout(
            title="Intraday Drawdown ($) unavailable for the current selection",
            margin=dict(l=10, r=10, t=40, b=10)
        )
    else:
        fig_dollar = build_intraday_dd_figure(portfolio_series)
        fig_dollar.update_layout(title="Intraday Drawdown ($)")

    pct_index = portfolio_view.pct_equity_index
    if pct_index is not None and not pct_index.empty:
        pct_index = pct_index.copy()
        if include_spikes:
            trade_frames = [df for df in trades_by_file.values() if df is not None and not df.empty]
            all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else None
            pct_index = _percent_equity_with_spikes(pct_index, all_trades)
        fig_percent = build_intraday_dd_figure(pct_index, value_is_percent=True)
        fig_percent.update_layout(title="Intraday Drawdown (%)")
    else:
        fig_percent = go.Figure().update_layout(
            title="Intraday Drawdown (%) unavailable for the current selection",
            margin=dict(l=10, r=10, t=40, b=10)
        )
    x_range = _common_x_range(portfolio_series, pct_index)
    if x_range:
        fig_dollar.update_xaxes(range=x_range)
        fig_percent.update_xaxes(range=x_range)

    return _two_graphs_child(fig_dollar, fig_percent, ids=("idd-dollar", "idd-percent"), heights=(340, 340))


# ---- Trade P/L Histogram Tab Callback ---- #


@app.callback(
    Output("hist_tab_content", "children"),
    Input("store-trades", "data"),
    Input("file-toggle", "value"),
    Input("symbol-toggle", "value"),
    Input("interval-toggle", "value"),
    Input("strategy-toggle", "value"),
    Input("direction-radio", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("store-contracts", "data"),
    Input("store-margins", "data"),
    Input("store-version", "data"),
    prevent_initial_call=False
)
def update_hist_tab(store_trades, selected_files, selected_symbols, selected_intervals,
                    selected_strategies, direction, start_date, end_date,
                    contracts_map, margins_map, store_version):

    fig_empty = go.Figure().update_layout(margin=dict(l=10, r=10, t=30, b=10))
    if not store_trades:
        return _single_graph_child(fig_empty)
    selected_files = [f for f in (selected_files or []) if store_trades and f in set(store_trades)]
    if not selected_files:
        return _single_graph_child(fig_empty)

    sel_key = make_selection_key(selected_files, selected_symbols, selected_intervals,
                                 selected_strategies, direction, start_date, end_date,
                                 contracts_map, margins_map, store_version)
    art = build_core_artifact(sel_key)
    trades_by_file = {k: v for k, v in art["trades_by_file"].items() if v is not None and not v.empty}
    if not trades_by_file:
        return _single_graph_child(fig_empty)

    return _single_graph_child(build_pl_histogram_figure(trades_by_file))


# ---- Margin Tab Callback ---- #


@app.callback(
    Output("margin_tab_content", "children"),
    # Shared selection Inputs
    Input("store-trades", "data"),
    Input("file-toggle", "value"),
    Input("symbol-toggle", "value"),
    Input("interval-toggle", "value"),
    Input("strategy-toggle", "value"),
    Input("direction-radio", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("store-contracts", "data"),
    Input("store-margins", "data"),
    Input("store-version", "data"),
    Input("spike-toggle", "value"),
    # Tab-specific Inputs
    Input("alloc-equity", "value"),
    prevent_initial_call=False
)
def update_margin_tab(store_trades, selected_files, selected_symbols, selected_intervals,
                      selected_strategies, direction, start_date, end_date,
                      contracts_map, margins_map, store_version,
                      spike_toggle, equity_val):

    fig_empty = go.Figure().update_layout(margin=dict(l=10, r=10, t=30, b=10))
    if not store_trades:
        return _single_graph_child(fig_empty)
    selected_files = [f for f in (selected_files or []) if store_trades and f in set(store_trades)]
    if not selected_files:
        return _single_graph_child(fig_empty)

    # Common artifact (equity + labels)
    sel_key = make_selection_key(selected_files, selected_symbols, selected_intervals,
                                 selected_strategies, direction, start_date, end_date,
                                 contracts_map, margins_map, store_version)
    art = build_core_artifact(sel_key)
    equity_by_file = {k: v for k, v in art["equity_by_file"].items() if v is not None and not v.empty}
    if not equity_by_file:
        return _single_graph_child(fig_empty)

    trades_by_file = art.get("trades_by_file", {}) or {}
    include_spikes = _spike_toggle_enabled(spike_toggle)
    if include_spikes:
        equity_by_file = {
            fname: _equity_with_spikes(eq, trades_by_file.get(fname))
            for fname, eq in equity_by_file.items()
        }

    # Cached margin-related series (reuses cached net positions under the hood)
    ms = build_margin_series_cached(sel_key, equity_val)
    total_init_margin = ms.get("total_init_margin")
    pp_series = ms.get("pp_series")

    # For the Net Contracts figure we still need per-file and by-symbol nets;
    # fetch via the memoized netpos builder (fast cache hit).
    symbol_key   = _keyify_list(selected_symbols)
    interval_key = _keyify_list(selected_intervals)
    strat_key    = _keyify_list(selected_strategies)
    contracts_kv = _keyify_contracts_map(contracts_map)
    files_key    = _files_key(selected_files)

    per_file_netpos = cached_netpos_per_file(
        store_version, files_key,
        symbol_key, interval_key, strat_key,
        direction, start_date, end_date,
        contracts_kv
    )
    by_symbol_net = _aggregate_netpos_per_symbol_from_series(per_file_netpos)


    fig_pp = go.Figure()
    if include_spikes:
        eq_all_spike = combine_equity(equity_by_file, list(equity_by_file.keys()))
        port_eq_spike = eq_all_spike.get("Portfolio") if eq_all_spike is not None else None
        starting_balance = _coerce_float(equity_val, DEFAULT_INITIAL_CAPITAL)
        pp_series = _purchasing_power_series(port_eq_spike, total_init_margin, starting_balance)
    if pp_series is not None and not pp_series.empty:
        fig_pp.add_trace(go.Scattergl(x=pp_series.index, y=pp_series, name="Purchasing Power", line=dict(width=3)))
    fig_pp.update_layout(title="Purchasing Power = Starting Balance + Portfolio P&L - Initial Margin Used",
                         xaxis_title="Date/Time", yaxis_title="Purchasing Power ($)",
                         hovermode="x unified", margin=dict(l=80, r=20, t=40, b=10))

    fig_pp_dd = go.Figure()
    pp_dd = None
    if pp_series is not None and not pp_series.empty:
        pp_dd = max_drawdown_series(pp_series)
        fig_pp_dd.add_trace(go.Scattergl(x=pp_dd.index, y=pp_dd, name="Purchasing Power Drawdown", line=dict(width=2)))
    fig_pp_dd.update_layout(title="Purchasing Power Drawdown",
                            xaxis_title="Date/Time", yaxis_title="Drawdown ($)",
                            hovermode="x unified", margin=dict(l=80, r=20, t=40, b=10))

    fig_im = go.Figure()
    if total_init_margin is not None and not total_init_margin.empty:
        fig_im.add_trace(go.Scattergl(
            x=total_init_margin.index,
            y=total_init_margin,
            name="Initial Margin Used",
            line=dict(width=2),
            line_shape="hv"
        ))
    fig_im.update_layout(title="Initial Margin Used (|net_sym| x IM(sym))",
                         xaxis_title="Date/Time", yaxis_title="Initial Margin ($)",
                         hovermode="x unified", margin=dict(l=80, r=20, t=40, b=10))

    fig_netpos = _netpos_figure_from_series(per_file_netpos, by_symbol_net)
    fig_netpos.update_layout(margin=dict(l=80, r=20, t=40, b=10))

    common_range = _common_x_range(pp_series, pp_dd, total_init_margin)
    if common_range:
        fig_pp.update_xaxes(range=common_range)
        fig_pp_dd.update_xaxes(range=common_range)
        fig_im.update_xaxes(range=common_range)
        fig_netpos.update_xaxes(range=common_range)

    return [
        dcc.Graph(figure=fig_pp, style={"height": "320px"}),
        dcc.Graph(figure=fig_pp_dd, style={"height": "260px"}),
        dcc.Graph(figure=fig_im, style={"height": "260px"}),
        dcc.Graph(figure=fig_netpos, style={"height": "520px"}),
    ]

# ---- Correlations Tab Callback ---- #


@app.callback(
    Output("corr_tab_content", "children"),
    # Shared selection Inputs
    Input("store-trades", "data"),
    Input("file-toggle", "value"),
    Input("symbol-toggle", "value"),
    Input("interval-toggle", "value"),
    Input("strategy-toggle", "value"),
    Input("direction-radio", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("store-contracts", "data"),
    Input("store-margins", "data"),
    Input("store-version", "data"),
    # Tab-specific Inputs
    Input("corr-mode", "value"),
    Input("corr-slope-window", "value"),
    prevent_initial_call=False
)
def update_corr_tab(store_trades, selected_files, selected_symbols, selected_intervals,
                    selected_strategies, direction, start_date, end_date,
                    contracts_map, margins_map, store_version,
                    corr_mode, corr_slope_window):

    fig_empty = go.Figure().update_layout(margin=dict(l=10, r=10, t=30, b=10))
    if not store_trades:
        return _single_graph_child(fig_empty)
    selected_files = [f for f in (selected_files or []) if store_trades and f in set(store_trades)]
    if not selected_files:
        return _single_graph_child(fig_empty)

    sel_key = make_selection_key(selected_files, selected_symbols, selected_intervals,
                                 selected_strategies, direction, start_date, end_date,
                                 contracts_map, margins_map, store_version)
    art = build_core_artifact(sel_key)
    equity_by_file = {k: v for k, v in art["equity_by_file"].items() if v is not None and not v.empty}
    if not equity_by_file:
        return _single_graph_child(fig_empty)

    fig_corr = build_corr_matrix_cached(
        sel_key,
        mode=corr_mode or "drawdown_pct",
        slope_window=int(corr_slope_window or 20),
        method="spearman",
    )
    
    return _single_graph_child(fig_corr)


@app.callback(
    Output("riskfolio-controls-hint", "children"),
    Output("riskfolio-btn-optimize", "disabled", allow_duplicate=True),
    Output("riskfolio-mean-risk-objective", "disabled", allow_duplicate=True),
    Output("riskfolio-mean-risk-return", "disabled", allow_duplicate=True),
    Output("riskfolio-mean-risk-rf", "disabled", allow_duplicate=True),
    Output("riskfolio-mean-risk-riskav", "disabled", allow_duplicate=True),
    Output("riskfolio-risk-measure", "disabled", allow_duplicate=True),
    Output("riskfolio-alpha", "disabled", allow_duplicate=True),
    Output("riskfolio-covariance", "disabled", allow_duplicate=True),
    Output("riskfolio-bound-min", "disabled", allow_duplicate=True),
    Output("riskfolio-bound-max", "disabled", allow_duplicate=True),
    Output("riskfolio-budget", "disabled", allow_duplicate=True),
    Output("riskfolio-group-caps", "disabled", allow_duplicate=True),
    Output("riskfolio-strategy-caps", "disabled", allow_duplicate=True),
    Output("riskfolio-turnover", "disabled", allow_duplicate=True),
    Input("store-trades", "data"),
    Input("riskfolio-subtab", "value"),
    prevent_initial_call="initial_duplicate",
)
def toggle_riskfolio_controls(store_trades, active_subtab):
    has_data = bool(store_trades)
    active_subtab = active_subtab or "mean_risk"
    is_mean_risk = active_subtab == "mean_risk"
    controls_enabled = has_data and is_mean_risk

    if not has_data:
        message = "Upload and select files to enable optimization."
    elif not is_mean_risk:
        message = "This toolbox is coming soon."
    else:
        message = "Configure the optimization model and constraints."

    disabled = not controls_enabled
    outputs = [
        message,
        disabled,  # optimize button
        disabled,  # objective
        disabled,  # return model
        disabled,  # risk-free rate
        True,      # risk aversion stays managed by its own callback
        disabled,  # risk measure
        disabled,  # alpha
        disabled,  # covariance
        disabled,  # bound min
        disabled,  # bound max
        disabled,  # budget
        disabled,  # group caps
        disabled,  # strategy caps
        True,      # turnover (reserved for future use)
    ]
    return tuple(outputs)




@app.callback(
    Output("riskfolio-mean-risk-riskav", "disabled", allow_duplicate=True),
    Input("riskfolio-mean-risk-objective", "value"),
    Input("riskfolio-subtab", "value"),
    Input("store-trades", "data"),
    prevent_initial_call=True,
)
def _toggle_risk_aversion_input(objective_value, active_subtab, store_trades):
    has_data = bool(store_trades)
    if active_subtab != "mean_risk" or not has_data:
        return True
    return objective_value != "max_utility"


@app.callback(
    Output("riskfolio-covariance", "disabled", allow_duplicate=True),
    Input("riskfolio-risk-measure", "value"),
    State("store-trades", "data"),
    prevent_initial_call="initial_duplicate",
)
def _toggle_covariance_input(risk_measure, store_trades):
    if not store_trades:
        return True
    rm = (risk_measure or "").strip().lower()
    return rm not in _COVARIANCE_DRIVEN_RISK_MEASURES

_RISKFOLIO_MODEL_LABELS = {
    "mean_risk": "Mean-Risk Optimization",
    "mean_variance": "Mean-Variance",
    "mad": "Mean Absolute Deviation",
    "cvar": "Conditional Value at Risk",
    "cdar": "Conditional Drawdown at Risk",
    "evar": "Entropic Value at Risk",
    "risk_parity": "Equal Risk Contribution",
    "hrp": "Hierarchical Risk Parity / Clustering",
}

_RISKFOLIO_RISK_LABELS = {
    "variance": "Variance",
    "semi_variance": "Semi-Variance",
    "cvar": "CVaR",
    "cdar": "CDaR",
    "evar": "EVaR",
}

_COVARIANCE_DRIVEN_RISK_MEASURES = {"variance", "semi_variance"}

_RISKFOLIO_TOTAL_STEPS = 5


def _parse_group_caps_input(value: str | None) -> Dict[str, float]:
    caps: Dict[str, float] = {}
    if not value:
        return caps
    for part in value.split(','):
        part = part.strip()
        if not part or '=' not in part:
            continue
        sym, _, cap_str = part.partition('=')
        sym = sym.strip().upper()
        try:
            caps[sym] = float(cap_str.strip())
        except Exception:
            continue
    return caps


def _build_constraint_rules(
    assets: Iterable[str],
    *,
    bound_min: float,
    bound_max: float,
    symbol_map: Dict[str, str],
    strategy_map: Dict[str, str],
    symbol_caps_text: str | None = None,
    strategy_caps_text: str | None = None,
) -> Dict[str, Any]:
    """
    Normalize UI inputs into structured constraint dictionaries ready for Riskfolio.

    Returns a dict with:
        - assets: list[str]
        - asset_bounds: {asset: {"min": float, "max": float}}
        - symbol_caps: {symbol: {"max": float}}
        - strategy_caps: {strategy: {"max": float}}
        - symbol_map / strategy_map: copies aligned to the asset list
    """
    if assets is None:
        asset_iter: List[Any] = []
    else:
        asset_iter = list(assets)
    asset_list = [str(a) for a in asset_iter]

    # Ensure min/max bounds are finite and ordered.
    try:
        min_bound = float(bound_min)
    except Exception:
        min_bound = 0.0
    try:
        max_bound = float(bound_max)
    except Exception:
        max_bound = 1.0
    if min_bound > max_bound:
        min_bound, max_bound = max_bound, min_bound

    asset_bounds = {
        asset: {"min": min_bound, "max": max_bound} for asset in asset_list
    }

    # Parse caps for symbol / strategy groupings.
    symbol_caps_raw = _parse_group_caps_input(symbol_caps_text)
    strategy_caps_raw = _parse_group_caps_input(strategy_caps_text)

    symbol_caps = {sym: {"max": float(cap)} for sym, cap in symbol_caps_raw.items()}
    strategy_caps = {strat: {"max": float(cap)} for strat, cap in strategy_caps_raw.items()}

    # Align provided maps to known assets.
    symbol_map_aligned = {asset: symbol_map.get(asset) for asset in asset_list if asset in symbol_map}
    strategy_map_aligned = {asset: strategy_map.get(asset) for asset in asset_list if asset in strategy_map}

    return {
        "assets": asset_list,
        "asset_bounds": asset_bounds,
        "symbol_caps": symbol_caps,
        "strategy_caps": strategy_caps,
        "symbol_map": symbol_map_aligned,
        "strategy_map": strategy_map_aligned,
    }


def _format_percent(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_number(value: Optional[float], digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _format_currency(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"${value:,.0f}"


def _riskfolio_error_div(message: str) -> html.Div:
    return html.Div(message, style={"color": "#b00020", "fontWeight": 600})


def _riskfolio_contracts_payload(
    weights_map: Dict[str, float],
    *,
    account_equity: float,
    contracts_map: Optional[Dict[str, Any]],
    margins_map: Optional[Dict[str, Any]],
    label_map: Optional[Dict[str, str]],
    symbol_map: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    if not weights_map:
        return {}

    files = list(weights_map.keys())
    equity_value = float(account_equity if account_equity is not None else 0.0)
    if not math.isfinite(equity_value):
        equity_value = 0.0

    def _clean_mapping(source: Optional[Dict[str, Any]], *, default: float = 0.0) -> Dict[str, float]:
        cleaned: Dict[str, float] = {}
        for key, value in (source or {}).items():
            try:
                cleaned[str(key)] = float(_coerce_float(value, default))
            except Exception:
                cleaned[str(key)] = float(default)
        return cleaned

    current_contracts = _clean_mapping(contracts_map, default=0.0)
    margins = _clean_mapping(margins_map, default=0.0)

    labels_subset = {fname: (label_map or {}).get(fname, fname) for fname in files}
    symbols_subset = {fname: (symbol_map or {}).get(fname, "") for fname in files}

    return {
        "files": files,
        "weights": {fname: float(weights_map.get(fname, 0.0)) for fname in files},
        "account_equity": equity_value,
        "current_contracts": current_contracts,
        "margins": margins,
        "labels": labels_subset,
        "symbols": symbols_subset,
    }


def _riskfolio_contract_rows(
    payload: Dict[str, Any],
    *,
    override_margins: Optional[Dict[str, Any]] = None,
    override_contracts: Optional[Dict[str, Any]] = None,
    account_equity_override: Optional[Any] = None,
) -> Tuple[list[Dict[str, Any]], str]:
    """
    Convert Riskfolio weights into suggested contracts + diagnostics rows for display.

    Returns (rows, note_text).
    """
    if not payload:
        return [], "Run an optimization to populate contract suggestions."

    files: list[str] = payload.get("files", []) or []
    if not files:
        return [], "No files available for contract suggestions."

    def _clean_map(source: Optional[Dict[str, Any]]) -> Dict[str, float]:
        cleaned: Dict[str, float] = {}
        for key, value in (source or {}).items():
            try:
                cleaned[str(key)] = float(_coerce_float(value, 0.0))
            except Exception:
                cleaned[str(key)] = 0.0
        return cleaned

    weights = {str(k): float(v) for k, v in (payload.get("weights") or {}).items()}
    labels = payload.get("labels") or {}
    symbols = payload.get("symbols") or {}
    equity = float(payload.get("account_equity") or 0.0)
    if account_equity_override is not None:
        try:
            override_equity = float(_coerce_float(account_equity_override, equity))
        except Exception:
            override_equity = equity
        if math.isfinite(override_equity) and override_equity >= 0:
            equity = override_equity

    base_margins = payload.get("margins") or {}
    base_contracts = payload.get("current_contracts") or {}
    margin_map = _clean_map(base_margins)
    contract_map = _clean_map(base_contracts)
    if override_margins:
        margin_map.update(_clean_map(override_margins))
    if override_contracts:
        contract_map.update(_clean_map(override_contracts))

    rows: list[Dict[str, Any]] = []
    sum_sugg_contracts = 0
    sum_curr_contracts = 0
    sum_sugg_margin = 0.0
    sum_curr_margin = 0.0
    max_sugg_gap = None
    max_curr_gap = None
    sugg_gap_total = 0.0
    curr_gap_total = 0.0
    active_weight_count = 0

    for fname in files:
        weight = float(weights.get(fname, 0.0))
        margin_per_contract = float(margin_map.get(fname, 0.0))
        current_contracts = float(contract_map.get(fname, 0.0))
        current_contracts_int = int(round(current_contracts))

        allocation = weight * equity
        if margin_per_contract > 0:
            suggested_contracts = math.floor(allocation / margin_per_contract)
        else:
            suggested_contracts = 0

        suggested_margin = float(suggested_contracts) * margin_per_contract
        current_margin = float(current_contracts_int) * margin_per_contract
        sum_sugg_contracts += max(suggested_contracts, 0)
        sum_curr_contracts += max(current_contracts_int, 0)
        sum_sugg_margin += max(suggested_margin, 0.0)
        sum_curr_margin += max(current_margin, 0.0)

        suggested_weight = (suggested_margin / equity) if equity > 0 else np.nan
        current_weight = (current_margin / equity) if equity > 0 else np.nan

        if np.isfinite(suggested_weight):
            sugg_gap = weight - suggested_weight
            max_sugg_gap = max(max_sugg_gap or 0.0, abs(sugg_gap))
            sugg_gap_display = _format_percent(sugg_gap)
            sugg_gap_total += abs(sugg_gap)
        else:
            sugg_gap_display = "n/a"

        if np.isfinite(current_weight):
            curr_gap = weight - current_weight
            max_curr_gap = max(max_curr_gap or 0.0, abs(curr_gap))
            curr_gap_display = _format_percent(curr_gap)
            curr_gap_total += abs(curr_gap)
        else:
            curr_gap_display = "n/a"

        if weight != 0.0:
            active_weight_count += 1

        rows.append(
            {
                "file": fname,
                "symbol": symbols.get(fname) or labels.get(fname, fname),
                "weight": _format_percent(weight),
                "suggested_gap": sugg_gap_display,
                "current_gap": curr_gap_display,
                "margin_per_contract": _format_currency(margin_per_contract),
                "suggested": int(suggested_contracts),
                "suggested_margin": _format_currency(suggested_margin),
                "current": current_contracts_int,
                "current_margin": _format_currency(current_margin),
                "delta": int(suggested_contracts - current_contracts_int),
            }
        )

    rows.append(
        {
            "file": "",
            "symbol": "",
            "weight": "",
            "suggested_gap": (
                f"Max: {_format_percent(max_sugg_gap)} / Avg: {_format_percent(sugg_gap_total / active_weight_count) if active_weight_count else 'n/a'}"
                if max_sugg_gap is not None
                else "Max: n/a"
            ),
            "current_gap": (
                f"Max: {_format_percent(max_curr_gap)} / Avg: {_format_percent(curr_gap_total / active_weight_count) if active_weight_count else 'n/a'}"
                if max_curr_gap is not None
                else "Max: n/a"
            ),
            "margin_per_contract": "Totals",
            "suggested": f"Total: {sum_sugg_contracts}",
            "suggested_margin": f"Total: {_format_currency(sum_sugg_margin)}",
            "current": f"Total: {sum_curr_contracts}",
            "current_margin": f"Total: {_format_currency(sum_curr_margin)}",
            "delta": f"Total: {sum_sugg_contracts - sum_curr_contracts}",
        }
    )

    note = (
        f"Account equity: {_format_currency(equity)}. "
        "Contracts = floor(weight × account equity ÷ the margin input on Load Trade Lists)."
    )
    return rows, note


def _execute_riskfolio_pipeline(
    set_progress: Optional[Callable[[Dict[str, Any]], None]],
    *,
    store_trades,
    selected_files,
    selected_symbols,
    selected_intervals,
    selected_strategies,
    direction,
    start_date,
    end_date,
    contracts_map,
    margins_map,
    store_version,
    model,
    risk_measure,
    alpha,
    covariance,
    bound_min,
    bound_max,
    budget,
    group_caps_text,
    strategy_caps_text,
    turnover_cap,
    alloc_equity_value,
    prev_weights_store,
    toolbox_id,
    toolbox_selections,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], Dict[str, float]]:
    total_steps = _RISKFOLIO_TOTAL_STEPS

    summary_context: Dict[str, Any] = {
        "objective_label": None,
        "return_label": None,
        "risk_free_annual_pct": None,
        "risk_aversion": None,
    }
    contracts_payload: Dict[str, Any] = {}

    def update_progress(step: int, label: str) -> None:
        if set_progress:
            pct = max(0.0, min(100.0, (step / total_steps * 100.0) if total_steps else 0.0))
            set_progress({"step": step, "total": total_steps, "label": label, "pct": pct})

    if not store_trades:
        progress = {"step": 0, "total": total_steps, "label": "Awaiting uploads", "pct": 0.0}
        update_progress(progress["step"], progress["label"])
        return _riskfolio_error_div("Upload and select at least one file to optimize."), progress, {}, {}

    selected_files = [str(f) for f in (selected_files or [])]
    if not selected_files:
        progress = {"step": 0, "total": total_steps, "label": "Awaiting selection", "pct": 0.0}
        update_progress(progress["step"], progress["label"])
        return _riskfolio_error_div("Select at least one file before optimizing."), progress, {}, {}

    update_progress(1, "Preparing selection")

    selected_symbols = list(selected_symbols or [])
    selected_intervals = list(selected_intervals or [])
    selected_strategies = list(selected_strategies or [])
    direction = direction or "All"
    contracts_map = contracts_map or {}
    margins_map = margins_map or {}

    bound_min = float(bound_min if bound_min is not None else 0.0)
    bound_max = float(bound_max if bound_max is not None else 1.0)
    if bound_min > bound_max:
        bound_min, bound_max = bound_max, bound_min

    budget_value = float(budget if budget is not None else 1.0)
    turnover_cap_value = None
    if turnover_cap not in (None, ""):
        try:
            turnover_cap_value = float(turnover_cap)
        except Exception:
            turnover_cap_value = None

    initial_equity = _coerce_float(alloc_equity_value, DEFAULT_INITIAL_CAPITAL)

    try:
        toolbox_config = build_toolbox_config(toolbox_id, toolbox_selections or {})
    except ValueError as exc:
        progress = {"step": 1, "total": total_steps, "label": "Invalid configuration", "pct": 20.0}
        update_progress(progress["step"], progress["label"])
        return _riskfolio_error_div(str(exc)), progress, {}, {}

    rf_daily = toolbox_selections.get("risk_free_rate")
    try:
        summary_context["risk_free_annual_pct"] = float(rf_daily) * 252 * 100 if rf_daily is not None else None
    except Exception:
        summary_context["risk_free_annual_pct"] = None

    if toolbox_id == "mean_risk":
        toolbox_meta = get_riskfolio_toolbox("mean_risk")
        objective_key = toolbox_selections.get("objective")
        return_key = toolbox_selections.get("return_model")
        summary_context["objective_label"] = (
            (toolbox_meta.get("objectives") or {}).get(objective_key or "", {}).get("label")
        )
        summary_context["return_label"] = (
            (toolbox_meta.get("return_models") or {}).get(return_key or "", {}).get("label")
        )
        try:
            risk_av = toolbox_selections.get("risk_aversion")
            summary_context["risk_aversion"] = float(risk_av) if risk_av is not None else None
        except Exception:
            summary_context["risk_aversion"] = None

    sel_key = make_selection_key(
        selected_files,
        selected_symbols,
        selected_intervals,
        selected_strategies,
        direction,
        start_date,
        end_date,
        contracts_map,
        margins_map,
        store_version,
    )

    update_progress(2, "Preparing returns matrix")
    returns, meta = riskfolio_adapter.prepare_returns(sel_key, {
        "scale_to_pct": True,
        "initial_capital": initial_equity,
    })

    if returns is None or returns.empty:
        progress = {"step": 2, "total": total_steps, "label": "Unable to prepare returns", "pct": 40.0}
        update_progress(progress["step"], progress["label"])
        message = meta.get("message") if isinstance(meta, dict) else "Unable to prepare returns for the selection."
        return _riskfolio_error_div(message), progress, {}, {}

    constraint_rules = _build_constraint_rules(
        returns.columns,
        bound_min=bound_min,
        bound_max=bound_max,
        symbol_map=meta.get("symbol_map", {}),
        strategy_map=meta.get("strategy_map", {}),
        symbol_caps_text=group_caps_text,
        strategy_caps_text=strategy_caps_text,
    )
    group_caps = {sym: info.get("max") for sym, info in constraint_rules.get("symbol_caps", {}).items()}
    strategy_caps = {strat: info.get("max") for strat, info in constraint_rules.get("strategy_caps", {}).items()}

    update_progress(3, "Running optimization")
    prev_weights = {}
    if isinstance(prev_weights_store, dict):
        prev_weights = {str(k): float(v) for k, v in prev_weights_store.items() if isinstance(v, (int, float))}

    config = {
        "model": model,
        "risk_measure": risk_measure,
        "alpha": float(alpha if alpha is not None else 0.95),
        "covariance": covariance,
        "bounds": (bound_min, bound_max),
        "budget": budget_value,
        "group_caps": group_caps,
        "strategy_caps": strategy_caps,
        "turnover_cap": turnover_cap_value,
        "initial_capital": initial_equity,
        "symbol_map": meta.get("symbols", {}),
        "strategy_map": meta.get("strategy_map", {}),
        "constraint_rules": constraint_rules,
        "toolbox_id": toolbox_id,
        "portfolio_kwargs": toolbox_config,
    }

    if toolbox_config.get("objective"):
        config["objective"] = toolbox_config["objective"]
    if "kelly" in toolbox_config:
        config["kelly"] = toolbox_config.get("kelly")
    if "risk_free" in toolbox_config:
        config["risk_free"] = toolbox_config["risk_free"]
    if "risk_aversion" in toolbox_config:
        config["risk_aversion"] = toolbox_config["risk_aversion"]

    result = riskfolio_adapter.run_optimization(returns, config, prev_weights=prev_weights)
    if result.get("status") != "ok":
        message = result.get("message", "Optimization failed.")
        progress = {"step": 3, "total": total_steps, "label": "Optimization failed", "pct": 60.0}
        update_progress(progress["step"], progress["label"])
        return _riskfolio_error_div(message), progress, {}, {}

    update_progress(4, "Computing diagnostics")

    weights_series = result.get("weights")
    if weights_series is None or len(weights_series) == 0:
        progress = {"step": 4, "total": total_steps, "label": "No weights returned", "pct": 80.0}
        update_progress(progress["step"], progress["label"])
        return _riskfolio_error_div("Optimization returned no weights."), progress, {}, {}
    weights_series = weights_series.astype(float)
    weights_series = weights_series.reindex(weights_series.index).fillna(0.0)
    weights_map = {k: float(v) for k, v in weights_series.items()}

    contracts_payload = _riskfolio_contracts_payload(
        weights_map,
        account_equity=initial_equity,
        contracts_map=contracts_map,
        margins_map=margins_map,
        label_map=meta.get("label_map", {}),
        symbol_map=meta.get("symbols", {}),
    )

    ex_post = riskfolio_adapter.compute_ex_post_metrics(weights_map, sel_key)
    if ex_post.get("status") == "error":
        progress = {"step": 4, "total": total_steps, "label": "Diagnostics failed", "pct": 80.0}
        update_progress(progress["step"], progress["label"])
        return (
            _riskfolio_error_div(ex_post.get("message", "Unable to compute diagnostics.")),
            progress,
            contracts_payload,
            weights_map,
        )

    metrics = ex_post.get("metrics", {})
    equity_curve = ex_post.get("equity")
    daily_returns = ex_post.get("daily_returns")

    update_progress(5, "Rendering results")

    label_map = meta.get("label_map", {})
    symbols_map = meta.get("symbols", {})
    risk_contrib_series = result.get("risk_contribution")
    if isinstance(risk_contrib_series, pd.Series):
        risk_contrib_series = risk_contrib_series.reindex(weights_series.index).fillna(0.0)
    else:
        risk_contrib_series = pd.Series(0.0, index=weights_series.index)

    returns_subset = returns[weights_series.index] if isinstance(returns, pd.DataFrame) else pd.DataFrame()

    ex_ante = result.get("ex_ante", {})
    model_label = _RISKFOLIO_MODEL_LABELS.get(model, model.replace('_', ' ').title())
    risk_label = _RISKFOLIO_RISK_LABELS.get(risk_measure, risk_measure.replace('_', ' ').title())

    table_rows = []
    rc_abs = np.abs(risk_contrib_series.values)
    rc_total = float(rc_abs.sum()) if rc_abs.size else 0.0
    for fname in weights_series.index:
        weight = float(weights_series[fname])
        rc_value = float(abs(risk_contrib_series.get(fname, 0.0)))
        rc_pct = rc_value / rc_total if rc_total > 0 else np.nan
        table_rows.append({
            "file": fname,
            "symbol": symbols_map.get(fname, "-"),
            "weight": _format_percent(weight),
            "risk": _format_percent(rc_pct),
        })

    weights_table = dash_table.DataTable(
        id="riskfolio-weights-table",
        data=table_rows,
        columns=[
            {"name": "File", "id": "file"},
            {"name": "Symbol", "id": "symbol"},
            {"name": "Weight", "id": "weight"},
            {"name": "Risk Contribution", "id": "risk"},
        ],
        style_header={"fontWeight": "bold"},
        style_cell={"padding": "6px", "whiteSpace": "normal"},
        style_table={"overflowX": "auto"},
    )

    summary_lines = [
        f"Model: {model_label}",
        f"Risk measure: {risk_label}",
    ]
    if summary_context.get("objective_label"):
        summary_lines.append(f"Objective: {summary_context['objective_label']}")
    if summary_context.get("return_label"):
        summary_lines.append(f"Return model: {summary_context['return_label']}")
    rf_annual_pct = summary_context.get("risk_free_annual_pct")
    if rf_annual_pct is not None and np.isfinite(rf_annual_pct):
        summary_lines.append(f"Risk-free (annual): {rf_annual_pct:.2f}%")
    if summary_context.get("risk_aversion") is not None:
        summary_lines.append(f"Risk aversion ?: {summary_context['risk_aversion']:.2f}")
    summary_lines.extend([
        f"Ex-ante Expected Return: {_format_percent(ex_ante.get('expected_return'))}",
        f"Ex-ante Volatility: {_format_percent(ex_ante.get('expected_volatility'))}",
        f"Ex-ante Sharpe: {_format_number(ex_ante.get('sharpe'))}",
    ])
    if np.isfinite(ex_ante.get("cvar", np.nan)):
        summary_lines.append(f"Ex-ante CVaR: {_format_percent(ex_ante.get('cvar'))}")
    summary_lines.extend([
        f"Ex-post Annual Return: {_format_percent(metrics.get('annual_return'))}",
        f"Ex-post Max Drawdown: {_format_currency(abs(metrics.get('max_drawdown')))}",
        f"Ex-post Sharpe: {_format_number(metrics.get('annual_sharpe'))}",
        f"Ex-post CVaR: {_format_percent(metrics.get('cvar'))}",
    ])

    summary_block = html.Div([
        html.H5("Summary", style={"margin": "0 0 6px"}),
        html.Ul([html.Li(line) for line in summary_lines], style={"margin": 0, "paddingLeft": "18px"}),
    ], style={"padding": "10px", "background": "#f8f9fb", "border": "1px solid #e2e4ea", "borderRadius": "8px"})

    charts = []
    frontier = result.get("frontier")
    if isinstance(frontier, pd.DataFrame) and not frontier.empty:
        columns_lower = {str(col).lower(): col for col in frontier.columns}
        risk_col = columns_lower.get("risk")
        ret_col = columns_lower.get("return")
        if risk_col and ret_col:
            fig_frontier = go.Figure()
            fig_frontier.add_trace(go.Scatter(x=frontier[risk_col], y=frontier[ret_col], mode="lines+markers", name="Frontier"))
            fig_frontier.add_trace(go.Scatter(
                x=[ex_ante.get("expected_volatility")],
                y=[ex_ante.get("expected_return")],
                mode="markers",
                name="Solution",
                marker=dict(size=10, color="#d62728"),
            ))
            fig_frontier.update_layout(title="Efficient Frontier", xaxis_title="Risk", yaxis_title="Return", hovermode="closest")
            charts.append(dcc.Graph(figure=fig_frontier, id="riskfolio-frontier"))

    if rc_total > 0:
        labels = [label_map.get(f, f) for f in weights_series.index]
        rc_pct_values = (rc_abs / rc_total) * 100.0 if rc_total else rc_abs
        fig_rc = go.Figure(go.Bar(x=labels, y=rc_pct_values, name="Risk Contribution"))
        fig_rc.update_layout(title="Risk Contribution (%)", xaxis_title="", yaxis_title="Percent", bargap=0.25)
        charts.append(dcc.Graph(figure=fig_rc, id="riskfolio-risk-contrib"))

    if isinstance(equity_curve, pd.Series) and not equity_curve.empty:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode="lines", name="Portfolio Equity"))
        fig_eq.update_layout(title="Backtested Portfolio Equity", xaxis_title="Date", yaxis_title="Cumulative P/L", hovermode="x unified")
        charts.append(dcc.Graph(figure=fig_eq, id="riskfolio-equity"))

    if isinstance(returns_subset, pd.DataFrame) and returns_subset.shape[1] >= 2:
        corr = returns_subset.corr()
        labels = [label_map.get(c, c) for c in corr.columns]
        fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=labels, y=labels, colorscale="RdBu", zmin=-1, zmax=1))
        fig_corr.update_layout(title="Asset Correlation", xaxis_title="", yaxis_title="")
        charts.append(dcc.Graph(figure=fig_corr, id="riskfolio-corr"))

    charts_block = html.Div(
        charts,
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))", "gap": "14px"}
    ) if charts else _riskfolio_error_div("No charts available for the current selection.")

    top_row = html.Div(
        [summary_block, html.Div(weights_table, style={"flex": "1", "minWidth": "280px"})],
        style={"display": "flex", "flexWrap": "wrap", "gap": "14px"}
    )

    results_children = html.Div(
        [top_row, charts_block],
        style={"display": "flex", "flexDirection": "column", "gap": "18px"}
    )

    update_progress(total_steps, "Completed")
    final_progress = {"step": total_steps, "total": total_steps, "label": "Completed", "pct": 100.0}
    return results_children, final_progress, contracts_payload, weights_map


@app.callback(
    Output("riskfolio-progress-label", "children"),
    Output("riskfolio-progress-bar", "style"),
    Input("riskfolio-progress", "data"),
    prevent_initial_call=False,
)
def update_riskfolio_progress(progress_data):
    base_style = {"width": "0%", "height": "100%", "background": "#4c8bf5", "transition": "width 0.3s"}
    if not progress_data:
        return "", base_style
    step = int(progress_data.get("step", 0))
    total = int(progress_data.get("total", _RISKFOLIO_TOTAL_STEPS))
    label = progress_data.get("label", "")
    pct = progress_data.get("pct")
    if pct is None:
        pct = (step / total * 100.0) if total else 0.0
    width = f"{max(0.0, min(100.0, pct)):.1f}%"
    base_style = base_style | {"width": width}
    label_text = f"Step {step} of {total}: {label}" if label else f"Step {step} of {total}"
    return label_text, base_style


@app.callback(
    Output("riskfolio-contracts-table", "data"),
    Output("riskfolio-contracts-note", "children"),
    Output("riskfolio-contracts-apply", "disabled"),
    Input("riskfolio-contracts-store", "data"),
    Input("store-contracts", "data"),
    Input("store-margins", "data"),
    Input("riskfolio-contract-equity", "value"),
    prevent_initial_call=False,
)
def update_riskfolio_contracts_table(payload, current_contracts_store, margins_store, contract_equity_value):
    rows, note = _riskfolio_contract_rows(
        payload or {},
        override_contracts=current_contracts_store,
        override_margins=margins_store,
        account_equity_override=contract_equity_value,
    )
    disable_apply = not rows or not payload
    return rows, note, disable_apply


@app.callback(
    Output({"type": "contracts-input", "index": ALL}, "value"),
    Output("ivp-message", "children", allow_duplicate=True),
    Output("riskfolio-contracts-note", "children", allow_duplicate=True),
    Input("ivp-btn-apply", "n_clicks"),
    Input("riskfolio-contracts-apply", "n_clicks"),
    State("ivp-store-suggested", "data"),
    State("riskfolio-contracts-store", "data"),
    State("riskfolio-contract-equity", "value"),
    State({"type": "contracts-input", "index": ALL}, "id"),
    State({"type": "contracts-input", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def apply_contract_suggestions(ivp_clicks, riskfolio_clicks, ivp_suggested, riskfolio_payload, contract_equity_value, input_ids, current_values):
    triggered = ctx.triggered_id
    if triggered is None:
        raise PreventUpdate
    if not input_ids:
        return no_update, no_update, "No contract inputs available to update."

    suggested_map: Dict[str, int] = {}
    message_ivp = no_update
    message_riskfolio = no_update

    if triggered == "ivp-btn-apply":
        if not ivp_clicks or not ivp_suggested:
            raise PreventUpdate
        suggested_map = {str(k): int(v) for k, v in (ivp_suggested or {}).items() if v is not None}
        if not suggested_map:
            raise PreventUpdate
        message_ivp = f"Applied IVP suggested contracts to {len(suggested_map)} file(s)."
    elif triggered == "riskfolio-contracts-apply":
        if not riskfolio_clicks or not riskfolio_payload:
            raise PreventUpdate
        suggested_rows, _ = _riskfolio_contract_rows(
            riskfolio_payload,
            account_equity_override=contract_equity_value,
        )
        suggested_map = {row["file"]: row["suggested"] for row in suggested_rows if row.get("file")}
        if not suggested_map:
            raise PreventUpdate
        message_riskfolio = f"Applied Riskfolio suggested contracts to {len(suggested_map)} file(s)."
    else:
        raise PreventUpdate

    new_values = []
    for idx, current in zip(input_ids, current_values or []):
        fname = str(idx.get("index"))
        suggestion = suggested_map.get(fname)
        new_values.append(int(suggestion) if suggestion is not None else current)

    return new_values, message_ivp, message_riskfolio



_riskfolio_progress_output = Output("riskfolio-progress", "data", allow_duplicate=True)
_riskfolio_contracts_output = Output("riskfolio-contracts-store", "data", allow_duplicate=True)

_RISKFOLIO_CALLBACK_OUTPUTS = (
    Output("riskfolio-results", "children", allow_duplicate=True),
    _riskfolio_progress_output,
    _riskfolio_contracts_output,
    Output("riskfolio-last-weights", "data", allow_duplicate=True),
)

_RISKFOLIO_CALLBACK_INPUTS = (Input("riskfolio-btn-optimize", "n_clicks"),)

_RISKFOLIO_CALLBACK_STATES = (
    State("store-trades", "data"),
    State("file-toggle", "value"),
    State("symbol-toggle", "value"),
    State("interval-toggle", "value"),
    State("strategy-toggle", "value"),
    State("direction-radio", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("store-contracts", "data"),
    State("store-margins", "data"),
    State("store-version", "data"),
    State("riskfolio-subtab", "value"),
    State("riskfolio-mean-risk-objective", "value"),
    State("riskfolio-mean-risk-return", "value"),
    State("riskfolio-mean-risk-rf", "value"),
    State("riskfolio-mean-risk-riskav", "value"),
    State("riskfolio-risk-measure", "value"),
    State("riskfolio-alpha", "value"),
    State("riskfolio-covariance", "value"),
    State("riskfolio-bound-min", "value"),
    State("riskfolio-bound-max", "value"),
    State("riskfolio-budget", "value"),
    State("riskfolio-group-caps", "value"),
    State("riskfolio-strategy-caps", "value"),
    State("riskfolio-turnover", "value"),
    State("alloc-equity", "value"),
    State("riskfolio-last-weights", "data"),
)

_RISKFOLIO_CALLBACK_PROGRESS = [_riskfolio_progress_output]

_RISKFOLIO_CALLBACK_RUNNING = [
    (Output("riskfolio-btn-optimize", "disabled", allow_duplicate=True), True, False),
    (Output("riskfolio-mean-risk-objective", "disabled", allow_duplicate=True), True, False),
    (Output("riskfolio-mean-risk-return", "disabled", allow_duplicate=True), True, False),
    (Output("riskfolio-mean-risk-rf", "disabled", allow_duplicate=True), True, False),
    (Output("riskfolio-mean-risk-riskav", "disabled", allow_duplicate=True), True, False),
    (Output("riskfolio-risk-measure", "disabled", allow_duplicate=True), True, False),
    (Output("riskfolio-alpha", "disabled", allow_duplicate=True), True, False),
    (Output("riskfolio-covariance", "disabled", allow_duplicate=True), True, False),
    (Output("riskfolio-bound-min", "disabled", allow_duplicate=True), True, False),
    (Output("riskfolio-bound-max", "disabled", allow_duplicate=True), True, False),
    (Output("riskfolio-budget", "disabled", allow_duplicate=True), True, False),
    (Output("riskfolio-group-caps", "disabled", allow_duplicate=True), True, False),
    (Output("riskfolio-strategy-caps", "disabled", allow_duplicate=True), True, False),
    (Output("riskfolio-turnover", "disabled", allow_duplicate=True), True, True),
]

_riskfolio_decorator = None
if _HAS_LONG_CALLBACK and hasattr(app, "long_callback"):
    _riskfolio_decorator = app.long_callback(
        outputs=_RISKFOLIO_CALLBACK_OUTPUTS,
        inputs=_RISKFOLIO_CALLBACK_INPUTS,
        state=_RISKFOLIO_CALLBACK_STATES,
        progress=_RISKFOLIO_CALLBACK_PROGRESS,
        running=_RISKFOLIO_CALLBACK_RUNNING,
        prevent_initial_call=True,
    )
elif _HAS_BACKGROUND_CALLBACK:
    _riskfolio_decorator = app.callback(
        *_RISKFOLIO_CALLBACK_OUTPUTS,
        *_RISKFOLIO_CALLBACK_INPUTS,
        *_RISKFOLIO_CALLBACK_STATES,
        background=True,
        manager=_riskfolio_manager,
        progress=_RISKFOLIO_CALLBACK_PROGRESS,
        running=_RISKFOLIO_CALLBACK_RUNNING,
        prevent_initial_call=True,
    )

if _riskfolio_decorator:

    @_riskfolio_decorator
    def run_riskfolio_long(
        set_progress,
        n_clicks,
        store_trades,
        selected_files,
        selected_symbols,
        selected_intervals,
        selected_strategies,
        direction,
        start_date,
        end_date,
        contracts_map,
        margins_map,
        store_version,
        active_subtab,
        mean_risk_objective,
        mean_risk_return,
        mean_risk_rf,
        mean_risk_riskav,
        risk_measure,
        alpha,
        covariance,
        bound_min,
        bound_max,
        budget,
        group_caps_text,
        strategy_caps_text,
        turnover_cap,
        alloc_equity_value,
        prev_weights_store,
    ):
        if not n_clicks:
            raise PreventUpdate

        toolbox_id = active_subtab or "mean_risk"
        model_value = {
            "mean_risk": "mean_variance",
            "risk_parity": "risk_parity",
            "hrp": "hrp",
        }.get(toolbox_id, toolbox_id)
        objective_value = mean_risk_objective or _MEAN_RISK_DEFAULT_OBJECTIVE
        return_value = mean_risk_return or _MEAN_RISK_DEFAULT_RETURN
        annual_rf_pct = _coerce_float(mean_risk_rf, 0.0)
        rf_value = annual_rf_pct / 100.0 / 252.0  # convert annual percent to daily decimal
        risk_av_value = None
        if objective_value == "max_utility":
            risk_av_value = _coerce_float(mean_risk_riskav, 2.0)
        toolbox_selections = {
            "objective": objective_value,
            "return_model": return_value,
            "risk_measure": _legacy_risk_measure_code(risk_measure),
            "risk_free_rate": rf_value,
            "hist": True,
        }
        if risk_av_value is not None:
            toolbox_selections["risk_aversion"] = risk_av_value

        results_children, final_progress, contracts_payload, weights_map = _execute_riskfolio_pipeline(
            set_progress,
            store_trades=store_trades,
            selected_files=selected_files,
            selected_symbols=selected_symbols,
            selected_intervals=selected_intervals,
            selected_strategies=selected_strategies,
            direction=direction,
            start_date=start_date,
            end_date=end_date,
            contracts_map=contracts_map,
            margins_map=margins_map,
            store_version=store_version,
            model=model_value,
            risk_measure=risk_measure,
            alpha=alpha,
            covariance=covariance,
            bound_min=bound_min,
            bound_max=bound_max,
            budget=budget,
            group_caps_text=group_caps_text,
            strategy_caps_text=strategy_caps_text,
            turnover_cap=turnover_cap,
            alloc_equity_value=alloc_equity_value,
            prev_weights_store=prev_weights_store,
            toolbox_id=toolbox_id,
            toolbox_selections=toolbox_selections,
        )

        if set_progress:
            set_progress(final_progress)

        return results_children, final_progress, contracts_payload, weights_map

else:  # pragma: no cover - fallback when long callback manager unavailable
    @app.callback(
        Output("riskfolio-results", "children", allow_duplicate=True),
        Output("riskfolio-progress", "data", allow_duplicate=True),
        Output("riskfolio-contracts-store", "data", allow_duplicate=True),
        Output("riskfolio-last-weights", "data", allow_duplicate=True),
        Input("riskfolio-btn-optimize", "n_clicks"),
        State("store-trades", "data"),
        State("file-toggle", "value"),
        State("symbol-toggle", "value"),
        State("interval-toggle", "value"),
        State("strategy-toggle", "value"),
        State("direction-radio", "value"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
        State("store-contracts", "data"),
        State("store-margins", "data"),
        State("store-version", "data"),
        State("riskfolio-subtab", "value"),
        State("riskfolio-mean-risk-objective", "value"),
        State("riskfolio-mean-risk-return", "value"),
        State("riskfolio-mean-risk-rf", "value"),
        State("riskfolio-mean-risk-riskav", "value"),
        State("riskfolio-risk-measure", "value"),
        State("riskfolio-alpha", "value"),
        State("riskfolio-covariance", "value"),
        State("riskfolio-bound-min", "value"),
        State("riskfolio-bound-max", "value"),
        State("riskfolio-budget", "value"),
        State("riskfolio-group-caps", "value"),
        State("riskfolio-strategy-caps", "value"),
        State("riskfolio-turnover", "value"),
        State("alloc-equity", "value"),
        State("riskfolio-last-weights", "data"),
        prevent_initial_call=True,
    )
    def run_riskfolio_standard(
        n_clicks,
        store_trades,
        selected_files,
        selected_symbols,
        selected_intervals,
        selected_strategies,
        direction,
        start_date,
        end_date,
        contracts_map,
        margins_map,
        store_version,
        active_subtab,
        mean_risk_objective,
        mean_risk_return,
        mean_risk_rf,
        mean_risk_riskav,
        risk_measure,
        alpha,
        covariance,
        bound_min,
        bound_max,
        budget,
        group_caps_text,
        strategy_caps_text,
        turnover_cap,
        alloc_equity_value,
        prev_weights_store,
    ):
        if not n_clicks:
            raise PreventUpdate

        toolbox_id = active_subtab or "mean_risk"
        model_value = {
            "mean_risk": "mean_variance",
            "risk_parity": "risk_parity",
            "hrp": "hrp",
        }.get(toolbox_id, toolbox_id)
        objective_value = mean_risk_objective or _MEAN_RISK_DEFAULT_OBJECTIVE
        return_value = mean_risk_return or _MEAN_RISK_DEFAULT_RETURN
        annual_rf_pct = _coerce_float(mean_risk_rf, 0.0)
        rf_value = annual_rf_pct / 100.0 / 252.0  # convert annual percent to daily decimal
        risk_av_value = None
        if objective_value == "max_utility":
            risk_av_value = _coerce_float(mean_risk_riskav, 2.0)
        toolbox_selections = {
            "objective": objective_value,
            "return_model": return_value,
            "risk_measure": _legacy_risk_measure_code(risk_measure),
            "risk_free_rate": rf_value,
            "hist": True,
        }
        if risk_av_value is not None:
            toolbox_selections["risk_aversion"] = risk_av_value

        results_children, final_progress, contracts_payload, weights_map = _execute_riskfolio_pipeline(
            None,
            store_trades=store_trades,
            selected_files=selected_files,
            selected_symbols=selected_symbols,
            selected_intervals=selected_intervals,
            selected_strategies=selected_strategies,
            direction=direction,
            start_date=start_date,
            end_date=end_date,
            contracts_map=contracts_map,
            margins_map=margins_map,
            store_version=store_version,
            model=model_value,
            risk_measure=risk_measure,
            alpha=alpha,
            covariance=covariance,
            bound_min=bound_min,
            bound_max=bound_max,
            budget=budget,
            group_caps_text=group_caps_text,
            strategy_caps_text=strategy_caps_text,
            turnover_cap=turnover_cap,
            alloc_equity_value=alloc_equity_value,
            prev_weights_store=prev_weights_store,
            toolbox_id=toolbox_id,
            toolbox_selections=toolbox_selections,
        )

        return results_children, final_progress, contracts_payload, weights_map




@app.callback(
    Output("riskfolio-results", "children", allow_duplicate=True),
    Output("riskfolio-progress", "data", allow_duplicate=True),
    Output("riskfolio-contracts-store", "data", allow_duplicate=True),
    Output("riskfolio-last-weights", "data", allow_duplicate=True),
    Input("store-trades", "data"),
    Input("file-toggle", "value"),
    Input("symbol-toggle", "value"),
    Input("interval-toggle", "value"),
    Input("strategy-toggle", "value"),
    Input("direction-radio", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    prevent_initial_call="initial_duplicate",
)
def reset_riskfolio_results(store_trades, selected_files, selected_symbols, selected_intervals,
                            selected_strategies, direction, start_date, end_date):
    placeholder = html.Div(
        "Run an optimization to populate results.",
        style={"color": "#666", "fontSize": "14px"}
    )
    progress = {"step": 0, "total": _RISKFOLIO_TOTAL_STEPS, "label": "Idle", "pct": 0.0}
    return placeholder, progress, {}, {}


@app.callback(
    Output("metrics-table", "data"),
    Output("metrics-table", "columns"),
    Input("store-trades", "data"),
    Input("file-toggle", "value"),
    Input("symbol-toggle", "value"),
    Input("interval-toggle", "value"),
    Input("strategy-toggle", "value"),
    Input("direction-radio", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("store-contracts", "data"),
    Input("store-margins", "data"),
    Input("store-version", "data"),  
    prevent_initial_call=True
)
def update_metrics_table(store_trades, selected_files, selected_symbols, selected_intervals,
                         selected_strategies, direction, start_date, end_date,
                         contracts_map, margins_map, store_version):

    if not store_trades or not selected_files:
        return [], []

    # Build once from the selection artifact via cached metrics
    sel_key = make_selection_key(
        selected_files, selected_symbols, selected_intervals, selected_strategies,
        direction, start_date, end_date, contracts_map, margins_map, store_version,
    )

    rows = compute_metrics_cached(sel_key, per_file_stats=True, monthly_mode="full_months")

    # Column ordering preserved from current implementation
    col_order = [
        "File", "Symbol", "Interval", "Strategy", "Direction",
        "Total Net Profit", "Gross Profit", "Gross Loss", "Profit Factor",
        "Total Number of Trades", "Percent Profitable", "Winning Trades", "Losing Trades",
        "Avg. Trade Net Profit", "Avg. Winning Trade", "Avg. Losing Trade", "Ratio Avg. Win:Avg. Loss",
        "Largest Winning Trade", "Largest Losing Trade",
        "Max. Consecutive Winning Trades", "Max. Consecutive Losing Trades",
        "Max. Shares/Contracts Held", "Total Shares/Contracts Held",
        "Account Size Required",
        "Total Commission", "Total Slippage",
        "Max. Drawdown (Trade Close to Trade Close) Value", "Max. Drawdown (Trade Close to Trade Close) Date",
        "Max. Drawdown (Intra-Day Peak to Valley) Value", "Max. Drawdown (Intra-Day Peak to Valley) Date",
        "Max. Trade Drawdown",
        "Return on Initial Capital", "Annual Rate of Return", "Return on Account",
        "Avg. Daily Return (%)", "Std. Dev. Daily Return (%)", "Daily Sharpe (252d)",
        "Avg. Monthly Return (full months)", "Std. Deviation of Monthly Return (full months)",
        "Trading Period", "Percent of Time in the Market", "Time in the Market", "Longest Flat Period",
        "Max. Equity Run-up", "Max Equity Run-up as % of Initial Capital",
    ]

    rows_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    for c in col_order:
        if c not in rows_df.columns:
            rows_df[c] = np.nan
    if not rows_df.empty:
        rows_df = rows_df[col_order]

    def _guess_type(series: pd.Series) -> str:
        return "numeric" if pd.api.types.is_numeric_dtype(series) else "text"

    cols = [{"name": c, "id": c, "type": _guess_type(rows_df[c])} for c in rows_df.columns] if not rows_df.empty else []
    return rows_df.to_dict("records"), cols


def _compute_cta_report(store_trades, selected_files, selected_symbols, selected_intervals,
                        selected_strategies, direction, start_date, end_date,
                        contracts_map, margins_map, store_version):
    """
    Build the CTA report payload for the current selection.
    Returns (CTAReportResult | None, auxiliary_label_or_message).
    """
    if not store_trades:
        return None, "Upload and select one or more files to generate the CTA report."

    selected_files = [f for f in (selected_files or []) if store_trades and f in set(store_trades)]
    if not selected_files:
        return None, "Upload and select one or more files to generate the CTA report."

    sel_key = make_selection_key(
        selected_files, selected_symbols, selected_intervals, selected_strategies,
        direction, start_date, end_date, contracts_map, margins_map, store_version,
    )
    portfolio_daily_pct = get_portfolio_daily_returns(sel_key)
    if portfolio_daily_pct is None:
        return None, "Daily percent returns unavailable for the current selection."

    series = _normalize_daily_returns_series(portfolio_daily_pct)

    if series.empty:
        return None, "Daily percent returns unavailable for the current selection."

    max_date = series.index.max()
    cutoff = max_date - pd.DateOffset(years=5)
    if series.index.min() < cutoff:
        series = series[series.index >= cutoff]

    if series.empty:
        return None, "Insufficient history (no data within the last five years)."

    report = cta_report.build_cta_report(series)
    return report, selected_files


def _format_pct(value: Optional[float], *, decimals: int = 2, blank: str = "—") -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return blank
    try:
        return f"{value * 100:.{decimals}f}%"
    except Exception:
        return blank


def _render_cta_summary(report: cta_report.CTAReportResult,
                        selected_files: list[str]) -> html.Div:
    summary = report.summary

    start_text = summary.start_date.strftime("%Y-%m-%d") if summary.start_date else "N/A"
    end_text = summary.end_date.strftime("%Y-%m-%d") if summary.end_date else "N/A"
    program_label = ", ".join(selected_files) if selected_files else "Selected Portfolio"

    largest_loss = summary.largest_monthly_loss or {}
    best_month = summary.best_month or {}
    peak_add = summary.worst_peak_to_valley_additive or {}
    peak_comp = summary.worst_peak_to_valley_compounded or {}

    info_cards = html.Div(
        [
            html.Div(
                [
                    html.H5("Program Summary", style={"marginBottom": "6px"}),
                    html.Ul(
                        [
                            html.Li(f"Program files: {program_label}"),
                            html.Li(f"Data window: {start_text} ? {end_text}"),
                            html.Li("Nominal-based returns (no reinvestment)"),
                        ],
                        style={"paddingLeft": "18px", "margin": 0},
                    ),
                ],
                style={"flex": "1 1 30%", "padding": "12px", "border": "1px solid #eee", "borderRadius": "8px"},
            ),
            html.Div(
                [
                    html.H5("Monthly Extremes", style={"marginBottom": "6px"}),
                    html.Ul(
                        [
                            html.Li(
                                f"Largest monthly loss: "
                                f"{_format_pct(largest_loss.get('value'))} in {largest_loss.get('month', 'N/A')}"
                            ),
                            html.Li(
                                f"Best month: {_format_pct(best_month.get('value'))} in {best_month.get('month', 'N/A')}"
                            ),
                        ],
                        style={"paddingLeft": "18px", "margin": 0},
                    ),
                ],
                style={"flex": "1 1 30%", "padding": "12px", "border": "1px solid #eee", "borderRadius": "8px"},
            ),
            html.Div(
                [
                    html.H5("Worst Peak-to-Valley Drawdown", style={"marginBottom": "6px"}),
                    html.Ul(
                        [
                            html.Li(
                                f"Additive: {_format_pct(peak_add.get('value'))} "
                                f"({peak_add.get('peak', 'N/A')} ? {peak_add.get('trough', 'N/A')})"
                            ),
                            html.Li(
                                f"Compounded: {_format_pct(peak_comp.get('value'))} "
                                f"({peak_comp.get('peak', 'N/A')} ? {peak_comp.get('trough', 'N/A')})"
                            ),
                            html.Li("Compounded metrics will become primary when reinvestment is enabled."),
                        ],
                        style={"paddingLeft": "18px", "margin": 0},
                    ),
                ],
                style={"flex": "1 1 30%", "padding": "12px", "border": "1px solid #eee", "borderRadius": "8px"},
            ),
        ],
        style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
    )

    # Annual ROR table (additive vs compounded)
    annual_data = []
    annual_years = sorted(set(summary.annual_ror_additive.index).union(summary.annual_ror_compounded.index))
    for yr in annual_years:
        annual_data.append(
            {
                "Year": str(yr),
                "Additive ROR": _format_pct(summary.annual_ror_additive.get(yr)),
                "Compounded ROR": _format_pct(summary.annual_ror_compounded.get(yr)),
            }
        )
    annual_data.append(
        {
            "Year": "YTD",
            "Additive ROR": _format_pct(summary.ytd_ror_additive),
            "Compounded ROR": _format_pct(summary.ytd_ror_compounded),
        }
    )
    annual_table = dash_table.DataTable(
        data=annual_data,
        columns=[
            {"name": "Year", "id": "Year"},
            {"name": "Rate of Return (Additive)", "id": "Additive ROR"},
            {"name": "Rate of Return (Compounded)", "id": "Compounded ROR"},
        ],
        style_header={"fontWeight": "bold"},
        style_cell={"padding": "6px", "textAlign": "center"},
        style_table={"maxWidth": "420px"},
        page_action="none",
    )

    supplemental = summary.supplemental_metrics
    sharpe_raw = supplemental.get("sharpe_ratio")
    sortino_raw = supplemental.get("sortino_ratio")
    sharpe_text = (
        f"Sharpe ratio (252d): {sharpe_raw:.2f}" if sharpe_raw is not None and not math.isnan(sharpe_raw)
        else "Sharpe ratio (252d): —"
    )
    sortino_text = (
        f"Sortino ratio (252d): {sortino_raw:.2f}" if sortino_raw is not None and not math.isnan(sortino_raw)
        else "Sortino ratio (252d): —"
    )
    supplemental_block = html.Div(
        [
            html.H5("Supplemental Analytics", style={"marginBottom": "6px"}),
            html.Ul(
                [
                    html.Li(f"Annualised return (daily basis): {_format_pct(supplemental.get('annualised_return_daily'))}"),
                    html.Li(f"Annualised volatility: {_format_pct(supplemental.get('annualised_volatility'))}"),
                    html.Li(sharpe_text),
                    html.Li(sortino_text),
                    html.Li(
                        f"Positive months: {supplemental.get('positive_months', 0)}/"
                        f"{supplemental.get('total_months', 0)} "
                        f"({_format_pct(supplemental.get('percent_positive_months', float('nan')))} positive)"
                    ),
                ],
                style={"paddingLeft": "18px", "margin": 0},
            ),
        ],
        style={"padding": "12px", "border": "1px solid #eee", "borderRadius": "8px"},
    )

    reinvest_note = html.Div(
        [
            html.P(
                [
                    html.Strong("Reinvestment note: "),
                    "Current metrics assume the program trades on a fixed nominal account size and do not "
                    "reinvest profits. When the walk-forward optimisation enables reinvestment, switch the CTA "
                    "report to the compounded metrics to satisfy Rule 4.35 by using the pre-computed values above.",
                ],
                style={"marginBottom": "0"},
            )
        ],
        style={"padding": "12px", "background": "#f8f9fb", "border": "1px solid #d0d7e2", "borderRadius": "8px"},
    )

    return html.Div(
        [
            html.H4("CTA Performance Capsule", style={"marginBottom": "16px"}),
            info_cards,
            html.Div(
                [
                    html.Div(annual_table, style={"flex": "0 0 320px"}),
                    html.Div(supplemental_block, style={"flex": "1 1 auto", "marginLeft": "24px"}),
                ],
                style={"display": "flex", "flexWrap": "wrap", "marginTop": "16px"},
            ),
            html.Div(reinvest_note, style={"marginTop": "16px"}),
        ]
    )


def _render_cta_monthly_table(report: cta_report.CTAReportResult) -> html.Div:
    matrix = report.monthly.matrix.copy()
    if matrix.empty:
        return html.Div(
            [
                html.H4("Monthly Performance Table"),
                html.P("Monthly data unavailable for the selected period."),
            ],
            style={"padding": "16px", "border": "1px solid #eee", "borderRadius": "8px"},
        )

    matrix_display = matrix.map(lambda v: _format_pct(v) if pd.notna(v) else "—")
    matrix_display.reset_index(inplace=True)
    matrix_display.rename(columns={"index": "Year"}, inplace=True)

    table = dash_table.DataTable(
        data=matrix_display.to_dict("records"),
        columns=[{"name": col, "id": col} for col in matrix_display.columns],
        style_header={"fontWeight": "bold", "textAlign": "center"},
        style_cell={"padding": "6px", "textAlign": "center"},
        style_table={"overflowX": "auto"},
        page_action="none",
        fixed_rows={"headers": True},
    )

    return html.Div(
        [
            html.H4("Monthly Rates of Return (nominal account basis)"),
            html.P(
                "Values reflect monthly rates of return for the last five calendar years and year-to-date, "
                "summed (not compounded) in accordance with CFTC Rule 4.35 for non-reinvesting programs.",
                style={"fontStyle": "italic", "color": "#555"},
            ),
            table,
            html.Button(
                "Download monthly ROR table (CSV)",
                id="cta-download-monthly",
                style={"marginTop": "12px"}
            ),
            dcc.Download(id="cta-download-monthly-data"),
        ],
        style={"padding": "16px", "border": "1px solid #eee", "borderRadius": "8px"},
    )


def _cta_monthly_bar_figure(report: cta_report.CTAReportResult) -> go.Figure:
    df = report.monthly.monthly_long
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Monthly Rates of Return",
            margin=dict(l=80, r=20, t=40, b=40),
            xaxis_title="Month",
            yaxis_title="Return (%)",
        )
        return fig
    fig = go.Figure()
    colors = []
    for v in df["return_additive"]:
        if pd.isna(v):
            colors.append("#8c8c8c")
        elif v >= 0:
            colors.append("#2ca02c")
        else:
            colors.append("#d62728")
    fig.add_trace(
        go.Bar(
            x=df["month_end"],
            y=df["return_additive"] * 100.0,
            name="Monthly ROR (additive)",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="Monthly Rates of Return (Additive)",
        xaxis_title="Month",
        yaxis_title="Return (%)",
        margin=dict(l=80, r=20, t=40, b=40),
        hovermode="x unified",
    )
    return fig


def _cta_nav_figure(report: cta_report.CTAReportResult) -> go.Figure:
    nav_comp = report.nav_compounded
    nav_add = report.nav_additive
    fig = go.Figure()
    if not nav_comp.empty:
        fig.add_trace(
            go.Scattergl(
                x=nav_comp.index,
                y=nav_comp.values,
                name="NAV (Compounded)",
                mode="lines",
                line=dict(width=2, color="#1f77b4"),
            )
        )
    if not nav_add.empty:
        fig.add_trace(
            go.Scattergl(
                x=nav_add.index,
                y=nav_add.values,
                name="NAV (Additive / Nominal)",
                mode="lines",
                line=dict(width=2, dash="dash", color="#ff7f0e"),
            )
        )
    fig.update_layout(
        title="NAV Path Comparison",
        xaxis_title="Date",
        yaxis_title="Index Value",
        hovermode="x unified",
        margin=dict(l=80, r=20, t=40, b=40),
    )
    return fig


def _cta_rolling_figure(report: cta_report.CTAReportResult) -> go.Figure:
    rolling = report.rolling_series
    fig = go.Figure()
    if "rolling_sum_additive" in rolling:
        fig.add_trace(
            go.Scattergl(
                x=rolling["rolling_sum_additive"].index,
                y=rolling["rolling_sum_additive"].values * 100.0,
                name="Rolling 12M Sum (Additive)",
                mode="lines",
                line=dict(width=2, color="#17becf"),
            )
        )
    if "rolling_compounded" in rolling:
        fig.add_trace(
            go.Scattergl(
                x=rolling["rolling_compounded"].index,
                y=rolling["rolling_compounded"].values * 100.0,
                name="Rolling 12M Return (Compounded)",
                mode="lines",
                line=dict(width=2, dash="dash", color="#9467bd"),
            )
        )
    fig.update_layout(
        title="Rolling 12-Month Performance",
        xaxis_title="Month",
        yaxis_title="Rolling Return (%)",
        hovermode="x unified",
        margin=dict(l=80, r=20, t=40, b=40),
    )
    return fig


def _render_cta_charts(report: cta_report.CTAReportResult) -> html.Div:
    monthly_fig = _cta_monthly_bar_figure(report)
    nav_fig = _cta_nav_figure(report)
    rolling_fig = _cta_rolling_figure(report)

    return html.Div(
        [
            html.H4("CTA Performance Charts"),
            html.Div(
                [
                    dcc.Graph(figure=monthly_fig, style={"height": "360px"}),
                    dcc.Graph(figure=nav_fig, style={"height": "360px"}),
                    dcc.Graph(figure=rolling_fig, style={"height": "320px"}),
                ],
                style={"display": "grid", "gap": "24px"},
            ),
        ]
    )


def _render_cta_disclosures() -> html.Div:
    return html.Div(
        [
            html.H4("Disclosures & Notes"),
            html.Ul(
                [
                    html.Li(html.Strong("Past performance is not necessarily indicative of future results.")),
                    html.Li(
                        "Returns are calculated on the nominal account size supplied by the strategy exports. "
                        "Account additions/withdrawals and partial funding adjustments are not currently captured "
                        "and should be disclosed separately if applicable."
                    ),
                    html.Li(
                        "Annual and drawdown percentages shown above use the additive method (sum of monthly returns) "
                        "as permitted by CFTC Regulation 4.35 for non-reinvesting programs. Compounded figures are "
                        "pre-computed and will become primary when reinvestment is implemented."
                    ),
                    html.Li(
                        "Provide manual disclosure for: total assets under management, total assets in the program, "
                        "number of accounts opened/closed with positive/negative lifetime returns, and the variability "
                        "ranges required by §4.35(a)(1)(viii)."
                    ),
                    html.Li(
                        "For partially funded accounts, disclose the conversion of nominal-based rates of return to "
                        "actual funding levels using the formula: (nominal / actual funds) × nominal ROR."
                    ),
                ],
                style={"paddingLeft": "20px"},
            ),
        ],
        style={"padding": "16px", "border": "1px solid #eee", "borderRadius": "8px"},
    )


@app.callback(
    Output("cta-summary-container", "children"),
    Output("cta-monthly-table-container", "children"),
    Output("cta-charts-container", "children"),
    Output("cta-disclosures-container", "children"),
    Input("store-trades", "data"),
    Input("file-toggle", "value"),
    Input("symbol-toggle", "value"),
    Input("interval-toggle", "value"),
    Input("strategy-toggle", "value"),
    Input("direction-radio", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("store-contracts", "data"),
    Input("store-margins", "data"),
    Input("store-version", "data"),
    Input("spike-toggle", "value"),
    prevent_initial_call=False
)
def update_cta_tab(store_trades, selected_files, selected_symbols, selected_intervals,
                   selected_strategies, direction, start_date, end_date,
                   contracts_map, margins_map, store_version,
                   spike_toggle):
    placeholder = html.Div(
        "Upload and select one or more files to generate the CTA report.",
        style={"fontStyle": "italic", "color": "#555"}
    )

    report, label = _compute_cta_report(
        store_trades, selected_files, selected_symbols, selected_intervals,
        selected_strategies, direction, start_date, end_date,
        contracts_map, margins_map, store_version,
    )

    if report is None:
        return placeholder, no_update, no_update, no_update

    selected_files = label if isinstance(label, list) else selected_files

    summary_component = _render_cta_summary(report, selected_files)
    monthly_component = _render_cta_monthly_table(report)
    charts_component = _render_cta_charts(report)
    disclosures_component = _render_cta_disclosures()

    return summary_component, monthly_component, charts_component, disclosures_component



# ------------------------------- Exports -------------------------------

@app.callback(
    Output("alloc-summary", "children"),
    Output("alloc-portfolio-graph", "figure"),
    Output("alloc-weights-table", "data"),
    Output("alloc-weights-table", "columns"),
    Input("btn-run-alloc", "n_clicks"),
    State("store-trades", "data"),
    State("file-toggle", "value"),
    State("symbol-toggle", "value"),
    State("interval-toggle", "value"),
    State("strategy-toggle", "value"),
    State("direction-radio", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("alloc-equity", "value"),
    State("alloc-margin-buffer", "value"),
    State("alloc-objective", "value"),
    State("alloc-lev-cap", "value"),
    State("store-contracts", "data"),
    State("store-margins", "data"),
    State("store-version", "data"),  
    prevent_initial_call=True
)
def run_allocator(n_clicks, store_trades, selected_files,
                  selected_symbols, selected_intervals, selected_strategies,
                  direction, start_date, end_date,
                  equity_val, margin_pct, objective, lev_cap,
                  contracts_map, margins_map, store_version):

    if not n_clicks or not store_trades or not selected_files:
        return "Upload and select files, then run.", go.Figure(), [], []

    # Build once from the selection artifact
    sel_key = make_selection_key(
        selected_files, selected_symbols, selected_intervals, selected_strategies,
        direction, start_date, end_date, contracts_map, margins_map, store_version,
    )
    art = build_core_artifact(sel_key)

    equity_by_file = {k: v for k, v in art["equity_by_file"].items() if v is not None and not v.empty}
    if not equity_by_file:
        return "No equity series for the current selection.", go.Figure(), [], []

    # Inputs for allocator
    init_cap   = float(equity_val or DEFAULT_INITIAL_CAPITAL)
    margin_buf = max(0.0, min(1.0, float(margin_pct or 60) / 100.0))
    lev_cap    = float(lev_cap) if lev_cap is not None else 0.0


    # Memoized allocator (runs on sel_key + its own inputs)
    weights, diag = run_allocator_cached(
        sel_key,
        objective=objective,
        leverage_cap=lev_cap,
        margin_cap=margin_pct,
        equity_val=init_cap,
    )

    if not weights:
        return ("No feasible allocation under margin constraints. Try raising margin headroom "
        "or lowering the maximum number of contracts."), go.Figure(), [], []


    # Portfolio equity using chosen weights
    port   = _portfolio_equity_from_weights(equity_by_file, weights)
    ann    = _annualized_return(port, init_cap)
    dd_mag = abs(_peak_drawdown_value(port))  # positive magnitude for display

    # Margin series at chosen weights (use cached net positions for summary)
    contracts_kv = _keyify_contracts_map(contracts_map)
    files_key    = _files_key(selected_files)
    symbol_key   = _keyify_list(selected_symbols)
    interval_key = _keyify_list(selected_intervals)
    strat_key    = _keyify_list(selected_strategies)

    netpos = cached_netpos_per_file(
        store_version, files_key,
        symbol_key, interval_key, strat_key,
        direction, start_date, end_date,
        contracts_kv
    )

    idx_union = None
    for ser in netpos.values():
        idx_union = ser.index if idx_union is None else idx_union.union(ser.index)
    idx_union = (idx_union.sort_values() if idx_union is not None else pd.DatetimeIndex([]))

    total_margin = pd.Series(0.0, index=idx_union)
    for f, ser in netpos.items():
        df_trades = app.server.trade_store.get(f)
        sym_raw = _symbol_from_first_row(df_trades)
        sym = (sym_raw or "").upper()
        # Use per-file override if provided; else symbol spec
        per_file_im = None
        try:
            per_file_im = float((margins_map or {}).get(f))
        except Exception:
            per_file_im = None
        if per_file_im is None or per_file_im == 0:
            spec = MARGIN_SPEC.get(sym)
            if spec is None:
                continue
            init_margin = float(spec[0])
        else:
            init_margin = float(per_file_im)
        total_margin = total_margin.add(
            ser.reindex(idx_union).ffill().fillna(0.0).abs() * init_margin * float(weights.get(f, 0.0)),
            fill_value=0.0
        )


    total_return_dollars = float(port.iloc[-1]) if (port is not None and not port.empty) else 0.0

    start_dt = port.index.min() if (port is not None and not port.empty) else None
    end_dt   = port.index.max() if (port is not None and not port.empty) else None
    years = ((end_dt - start_dt).total_seconds() / (365.2425 * 24 * 3600)) if (start_dt and end_dt and end_dt > start_dt) else None

    avg_annual_dollars = (total_return_dollars / years) if (years and years > 0) else None

    dd_dollars = dd_mag if (dd_mag is not None and np.isfinite(dd_mag)) else None

    ratio_ann_per_dd_val = (avg_annual_dollars / dd_dollars) if (
        avg_annual_dollars is not None and dd_dollars and dd_dollars > 0
    ) else None

    cagr_pct_val = (ann * 100.0) if (ann is not None and np.isfinite(ann)) else None

    totret_str = f"${total_return_dollars:,.0f}"
    avg_ann_str = f"${avg_annual_dollars:,.0f}" if avg_annual_dollars is not None else "n/a"
    dd_str = f"${dd_dollars:,.0f}" if dd_dollars is not None else "n/a"
    ratio_str = f"{ratio_ann_per_dd_val:0.2f}" if ratio_ann_per_dd_val is not None else "n/a"
    cagr_str = f"{cagr_pct_val:0.2f}%" if cagr_pct_val is not None else "n/a"

    summary = (
        f"Best allocation ({diag.get('mode','grid')}): "
        f"Total Return {totret_str} | "
        f"Avg Annual $ (non-comp) {avg_ann_str} | "
        f"Max DD {dd_str} | "
        f"Annual$/MaxDD {ratio_str} | "
        f"Max Margin Used ${total_margin.max():,.0f} "
        f"({(total_margin.max()/init_cap*100):0.1f}% of equity, cap {margin_buf*100:0.0f}%) | "
        f"CAGR {cagr_str}"
    )

    # Optional: append parity error diagnostics under the summary
    max_rem   = diag.get("max_remainder")
    avg_rem   = diag.get("avg_remainder")
    l1_err    = diag.get("l1_share_error")
    s_equiv   = diag.get("parity_scale_equiv")
    ideal_tot = diag.get("ideal_float_total")
    int_tot   = diag.get("int_total")

    parity_line = ""
    if max_rem is not None and avg_rem is not None and l1_err is not None:
        parity_line = (
            f"\nParity diagnostics — scale s˜{s_equiv:.2f}, "
            f"ideal float total {ideal_tot:.2f} vs int total {int_tot:.0f}; "
            f"max remainder {max_rem:.2f}, avg remainder {avg_rem:.2f}, "
            f"L1 share error {l1_err:.3f}"
        )
    summary = summary + parity_line


    fig = go.Figure()
    for f, s in equity_by_file.items():
        w = float(weights.get(f, 0.0))
        if w > 0 and s is not None and not s.empty:
            fig.add_trace(go.Scattergl(x=s.index, y=s * w, name=f"{f} × {int(w)}", line=dict(width=1)))
    fig.add_trace(go.Scattergl(x=port.index, y=port, name="Portfolio (alloc)", line=dict(width=3)))
    fig.update_layout(title="Allocated Portfolio Equity", xaxis_title="Date", yaxis_title="Cumulative P/L",
                      hovermode="x unified", margin=dict(l=10,r=10,t=40,b=10))

    rows = []
    for f, w in sorted(weights.items()):
        if w <= 0:
            continue
        df_trades = app.server.trade_store.get(f)
        sym_raw = _symbol_from_first_row(df_trades)
        rows.append({
            "File": f,
            "Symbol": (sym_raw or "").upper(),
            "Weight": int(float(w)),           # store as whole contracts
            "Note": "number of contracts (integer)",
        })

    cols = [
        {"name": "File", "id": "File"},
        {"name": "Symbol", "id": "Symbol"},
        {"name": "Contracts", "id": "Weight", "type": "numeric"},  # renamed header
        {"name": "Note", "id": "Note"},
    ]

    return summary, fig, rows, cols



def export_trades(n_clicks, store_trades, sel_files, sel_syms, sel_ints, sel_strats, direction,
                  start_date, end_date, contracts_map, margins_map, store_version):
    if not n_clicks or not store_trades or not sel_files:
        return None

    # Build once from the selection artifact
    sel_key = make_selection_key(
        sel_files, sel_syms, sel_ints, sel_strats,
        direction, start_date, end_date, contracts_map, margins_map, store_version,
    )
    art = build_core_artifact(sel_key)

    frames = [tdf for f, tdf in art["trades_by_file"].items() if tdf is not None and not tdf.empty and f in (sel_files or [])]

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True).sort_values(["File", "exit_time"])
    export_cols = [
        "File", "Symbol", "Interval", "Strategy", "direction",
        "entry_time", "exit_time",
        "entry_type", "entry_price", "exit_price",
        "contracts", "net_profit", "runup", "drawdown_trade",
        "commission", "slippage", "CumulativePL_raw",
    ]
    export_cols = [c for c in export_cols if c in out.columns]
    out = out[export_cols]
    return dcc.send_data_frame(out.to_csv, "filtered_trades.csv", index=False)

@app.callback(
    Output("download-trades", "data"),
    Input("btn-export-trades", "n_clicks"),
    State("store-trades", "data"),
    State("file-toggle", "value"),
    State("symbol-toggle", "value"),
    State("interval-toggle", "value"),
    State("strategy-toggle", "value"),
    State("direction-radio", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("store-contracts", "data"),
    State("store-margins", "data"),
    State("store-version", "data"),
    prevent_initial_call=True
)
def _cb_export_trades(n_clicks, store_trades, sel_files, sel_syms, sel_ints, sel_strats, direction,
                      start_date, end_date, contracts_map, margins_map, store_version):
    return export_trades(
        n_clicks, store_trades, sel_files, sel_syms, sel_ints, sel_strats, direction,
        start_date, end_date, contracts_map, margins_map, store_version,
    )


@app.callback(
    Output("download-metrics", "data"),
    Input("btn-export-metrics", "n_clicks"),
    State("metrics-table", "data"),
    prevent_initial_call=True
)
def _cb_export_metrics(n_clicks, rows):
    if not n_clicks or not rows:
        return None
    df = pd.DataFrame(rows)
    return dcc.send_data_frame(df.to_csv, "metrics.csv", index=False)

# ----------------------------- Entrypoint ------------------------------
if __name__ == "__main__":
    webbrowser.open_new("http://127.0.0.1:8050/")
    app.run(debug=True, use_reloader=False, port=8050)












