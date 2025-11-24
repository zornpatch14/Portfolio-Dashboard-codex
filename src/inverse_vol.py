from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from dash import Dash, html, dcc, dash_table, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
from dash.dependencies import ALL

from src.tabs import pretty_card
import src.riskfolio_adapter as riskfolio_adapter


TABLE_COLUMNS = [
    {"name": "File", "id": "file"},
    {"name": "Symbol", "id": "symbol"},
    {"name": "IVP Weight", "id": "weight"},
    {"name": "Daily stdev (log)", "id": "sigma"},
    {"name": "Suggested Weight Gap", "id": "suggested_gap"},
    {"name": "Current Weight Gap", "id": "current_gap"},
    {"name": "Base Contracts (min=1)", "id": "base_contracts"},
    {"name": "Suggested Contracts", "id": "suggested"},
    {"name": "Suggested Margin", "id": "suggested_margin"},
    {"name": "Current Contracts", "id": "current"},
    {"name": "Current Margin", "id": "current_margin"},
    {"name": "Delta", "id": "delta"},
]


def layout() -> html.Div:
    return html.Div(
        [
            pretty_card(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Button(
                                        "Compute IVP Suggestions",
                                        id="ivp-btn-compute",
                                        n_clicks=0,
                                        className="ivp-compute-btn",
                                        style={"padding": "6px 12px", "fontWeight": 600},
                                    ),
                                    html.Div(
                                        [
                                            html.Span("Multiplier:", style={"marginRight": "6px", "fontWeight": 500}),
                                            dcc.Input(
                                                id="ivp-multiplier",
                                                type="number",
                                                value=1.0,
                                                min=0,
                                                step=0.001,
                                                style={"width": "90px"},
                                            ),
                                        ],
                                        style={"display": "flex", "alignItems": "center", "gap": "6px"},
                                    ),
                                    html.Button(
                                        "Apply to Contracts",
                                        id="ivp-btn-apply",
                                        n_clicks=0,
                                        className="ivp-apply-btn",
                                        disabled=True,
                                        style={"padding": "6px 12px", "fontWeight": 600},
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "16px",
                                    "flexWrap": "wrap",
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(id="ivp-message", style={"color": "#1f2937", "fontSize": "13px"}),
                                    html.Div(id="ivp-multiplier-note", style={"color": "#6b7280", "fontSize": "12px"}),
                                ],
                                style={"display": "flex", "flexDirection": "column", "gap": "2px"},
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "12px"},
                    )
                ]
            ),
            html.Div(
                dash_table.DataTable(
                    id="ivp-table",
                    data=[],
                    columns=TABLE_COLUMNS,
                    style_header={"fontWeight": "bold"},
                    style_cell={
                        "padding": "6px",
                        "fontSize": "12px",
                        "whiteSpace": "normal",
                        "textAlign": "right",
                    },
                    style_cell_conditional=[
                        {"if": {"column_id": "file"}, "textAlign": "left"},
                        {"if": {"column_id": "symbol"}, "textAlign": "left"},
                    ],
                    page_size=20,
                    sort_action="native",
                ),
                style={"marginTop": "16px"},
            ),
        ],
        style={"padding": "12px 0", "display": "flex", "flexDirection": "column", "gap": "16px"},
    )


def _log_sigma(returns: pd.DataFrame) -> pd.Series:
    if returns is None or returns.empty:
        return pd.Series(dtype=float)
    clipped = returns.clip(lower=-0.999999, upper=None)
    log_ret = np.log1p(clipped).replace([np.inf, -np.inf], np.nan)
    sigma = log_ret.std(ddof=0)
    sigma = sigma.replace([np.inf, -np.inf], np.nan)
    sigma = sigma.dropna()
    return sigma[sigma > 0]


def _ivp_weights(returns: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    sigma = _log_sigma(returns)
    if sigma.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    inv = 1.0 / sigma
    inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
    if inv.empty:
        return pd.Series(dtype=float), sigma
    weights = inv / inv.sum()
    weights = weights[weights > 0]
    if weights.empty:
        return pd.Series(dtype=float), sigma
    weights = weights / weights.sum()
    return weights, sigma.loc[weights.index]


def _base_contracts(weights: pd.Series) -> pd.Series:
    if weights is None or weights.empty:
        return pd.Series(dtype=float)
    pos = weights[weights > 0]
    if pos.empty:
        return pd.Series(dtype=float)
    w_min = pos.min()
    if not np.isfinite(w_min) or w_min <= 0:
        return pd.Series(dtype=float)
    base = pos / w_min
    return base.reindex(weights.index).fillna(0.0)


def _apply_multiplier(base_contracts: pd.Series, multiplier: float) -> pd.Series:
    if base_contracts is None or base_contracts.empty:
        return pd.Series(dtype=int)
    try:
        m = float(multiplier)
    except Exception:
        m = 1.0
    if not np.isfinite(m):
        m = 1.0
    scaled = base_contracts * max(m, 0.0)
    rounded = scaled.round().clip(lower=0)
    return rounded.astype(int)


def _format_percent(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_number(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def register_callbacks(
    app: Dash,
    *,
    make_selection_key,
    coerce_float,
    default_initial_capital: float,
) -> None:
    @app.callback(
        Output("ivp-store-base", "data"),
        Output("ivp-message", "children", allow_duplicate=True),
        Input("ivp-btn-compute", "n_clicks"),
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
        State("alloc-equity", "value"),
        prevent_initial_call=True,
    )
    def compute_ivp_store(
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
        alloc_equity_value,
    ):
        if not n_clicks:
            raise PreventUpdate
        if not store_trades:
            return {}, "Upload trade files to compute inverse volatility contracts."
        selected_files = [str(f) for f in (selected_files or [])]
        if not selected_files:
            return {}, "Select at least one file to compute suggestions."

        unit_contracts = {f: 1.0 for f in selected_files}
        sel_key = make_selection_key(
            selected_files,
            selected_symbols,
            selected_intervals,
            selected_strategies,
            direction,
            start_date,
            end_date,
            unit_contracts,
            margins_map,
            store_version,
        )

        init_equity = coerce_float(alloc_equity_value, default_initial_capital)

        returns, meta = riskfolio_adapter.prepare_returns(
            sel_key,
            {
                "scale_to_pct": True,
                "initial_capital": init_equity,
            },
        )

        if returns is None or returns.empty:
            message = (
                meta.get("message")
                if isinstance(meta, dict) and meta.get("message")
                else "Unable to derive daily returns for the selected files."
            )
            return {}, message

        weights, sigma = _ivp_weights(returns)
        if weights.empty:
            return {}, "Unable to compute IVP weights (check variance and date overlap)."

        base_contracts = _base_contracts(weights)
        if base_contracts.empty:
            return {}, "Computed weights but failed to derive contract scaling."

        current_contracts = {str(k): float(v) for k, v in (contracts_map or {}).items()}
        labels = (meta or {}).get("label_map", {})
        symbols = (meta or {}).get("symbols", {})

        base_store = {
            "weights": weights.to_dict(),
            "sigma": sigma.reindex(weights.index).to_dict(),
            "base_contracts": base_contracts.to_dict(),
            "labels": labels,
            "symbols": symbols,
            "current_contracts": current_contracts,
            "files": list(weights.index),
        }

        used = len(weights)
        total_cols = returns.shape[1]
        excluded = total_cols - used
        summary_parts = [f"Computed IVP weights for {used} file{'s' if used != 1 else ''}."]
        if excluded > 0:
            summary_parts.append(f"Ignored {excluded} column(s) with insufficient variance.")
        return base_store, " ".join(summary_parts)

    @app.callback(
        Output("ivp-table", "data"),
        Output("ivp-store-suggested", "data"),
        Output("ivp-btn-apply", "disabled"),
        Output("ivp-multiplier-note", "children"),
        Input("ivp-store-base", "data"),
        Input("ivp-multiplier", "value"),
        Input("store-contracts", "data"),
        State("store-margins", "data"),
        prevent_initial_call=False,
    )
    def update_ivp_table(base_store, multiplier, contracts_map, margins_map):
        if not base_store:
            return [], {}, True, "Compute suggestions to populate the table."

        weights = pd.Series(base_store.get("weights", {}), dtype=float)
        base_contracts = pd.Series(base_store.get("base_contracts", {}), dtype=float)
        sigma = pd.Series(base_store.get("sigma", {}), dtype=float)
        labels = base_store.get("labels", {})
        symbols = base_store.get("symbols", {})
        current_contracts = {str(k): float(v) for k, v in (contracts_map or {}).items()}
        margin_map = {}
        for k, v in (margins_map or {}).items():
            try:
                margin_map[str(k)] = float(v)
            except Exception:
                margin_map[str(k)] = 0.0

        if weights.empty or base_contracts.empty:
            return [], {}, True, "No weights available for the current selection."

        suggested = _apply_multiplier(base_contracts, multiplier)
        suggested_map = {str(k): int(v) for k, v in suggested.items() if np.isfinite(v)}

        total_suggested = sum(max(int(v), 0) for v in suggested_map.values()) if suggested_map else 0
        total_current = sum(
            max(float(current_contracts.get(fname, 0.0)), 0.0) for fname in weights.index
        )

        rows = []
        sum_base_display = 0.0
        sum_suggested_display = 0
        sum_current_display = 0
        sum_suggested_margin = 0.0
        sum_current_margin = 0.0
        max_suggested_gap = None
        max_current_gap = None
        for fname in weights.index:
            weight = float(weights.get(fname, 0.0))
            base_val = float(base_contracts.get(fname, 0.0))
            sugg_val = int(suggested_map.get(fname, 0))
            current_val = float(current_contracts.get(fname, 0.0))
            sugg_amt = max(sugg_val, 0)
            curr_amt_float = max(current_val, 0.0)
            current_int = int(round(current_val))
            curr_amt_int = max(current_int, 0)
            sugg_share = (sugg_amt / total_suggested) if total_suggested > 0 else np.nan
            curr_share = (curr_amt_float / total_current) if total_current > 0 else np.nan
            sum_base_display += max(base_val, 0.0)
            sum_suggested_display += int(sugg_amt)
            sum_current_display += curr_amt_int

            margin_per_contract = float(margin_map.get(fname, 0.0))
            suggested_margin_val = margin_per_contract * sugg_amt
            current_margin_val = margin_per_contract * curr_amt_int
            if np.isfinite(suggested_margin_val):
                sum_suggested_margin += max(suggested_margin_val, 0.0)
            else:
                suggested_margin_val = np.nan
            if np.isfinite(current_margin_val):
                sum_current_margin += max(current_margin_val, 0.0)
            else:
                current_margin_val = np.nan

            if np.isfinite(sugg_share):
                gap_val = weight - sugg_share
                max_suggested_gap = max(max_suggested_gap or 0.0, abs(gap_val))
                suggested_gap_display = _format_percent(gap_val)
            else:
                suggested_gap_display = "n/a"
            if np.isfinite(curr_share):
                gap_val = weight - curr_share
                max_current_gap = max(max_current_gap or 0.0, abs(gap_val))
                current_gap_display = _format_percent(gap_val)
            else:
                current_gap_display = "n/a"
            rows.append(
                {
                    "file": fname,
                    "symbol": symbols.get(fname, labels.get(fname, "")),
                    "weight": _format_percent(weight),
                    "sigma": _format_percent(float(sigma.get(fname, np.nan))),
                    "suggested_gap": suggested_gap_display,
                    "current_gap": current_gap_display,
                    "base_contracts": _format_number(base_val, digits=2),
                    "suggested": int(sugg_amt),
                    "suggested_margin": _format_number(float(suggested_margin_val), digits=0),
                    "current": curr_amt_int,
                    "current_margin": _format_number(float(current_margin_val), digits=0),
                    "delta": int(sugg_amt - round(current_val)),
                }
            )

        rows.append(
            {
                "file": "",
                "symbol": "",
                "weight": "",
                "sigma": "",
                "suggested_gap": (
                    f"Max: {_format_percent(max_suggested_gap)}" if max_suggested_gap is not None else "Max: n/a"
                ),
                "current_gap": (
                    f"Max: {_format_percent(max_current_gap)}" if max_current_gap is not None else "Max: n/a"
                ),
                "base_contracts": f"Total: {_format_number(sum_base_display, digits=2)}",
                "suggested": f"Total: {int(sum_suggested_display)}",
                "suggested_margin": f"Total: {_format_number(sum_suggested_margin, digits=0)}",
                "current": f"Total: {int(sum_current_display)}",
                "current_margin": f"Total: {_format_number(sum_current_margin, digits=0)}",
                "delta": f"Total: {int(sum_suggested_display - sum_current_display)}",
            }
        )

        disable_apply = len(suggested_map) == 0
        mult_display = float(multiplier) if multiplier not in (None, "") else 1.0
        if not np.isfinite(mult_display):
            mult_display = 1.0
        note = f"Multiplier {mult_display:.2f} -> total suggested {sum_suggested_display} contract(s)."
        return rows, suggested_map, disable_apply, note

