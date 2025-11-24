# src/tabs.py

from __future__ import annotations
from typing import Optional, Tuple
from dash import html, dcc
import plotly.graph_objects as go

# PURPOSE
#   Pure Dash layout helpers used by multiple tabs (no callbacks, no global reads).
#   They accept Figures (or IDs/props) and return HTML components.
#
# WHAT LIVES HERE (current)
#   - _single_graph_child(fig, height=520, graph_id=None, config=None)
#   - _two_graphs_child(fig_top, fig_bottom, heights=(360, 260), ids=(None, None), config=None)
#
# DO NOT
#   - register callbacks here
#   - read app/server globals
#   - mutate figures (except small layout tweaks if desired)

def _single_graph_child(
    fig: go.Figure,
    height: int = 520,
    graph_id: Optional[str] = None,
    config: Optional[dict] = None,
):
    """
    Wrap a single Plotly figure into a Graph inside a Div so callers can drop it into any tab.
    """
    gid = graph_id or "analysis-graph"
    cfg = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]} | (config or {})
    return html.Div(
        dcc.Graph(id=gid, figure=fig, config=cfg, style={"height": f"{height}px"})
    )

def _two_graphs_child(
    fig_top: go.Figure,
    fig_bottom: go.Figure,
    heights: Tuple[int, int] = (360, 260),
    ids: Tuple[Optional[str], Optional[str]] = (None, None),
    config: Optional[dict] = None,
):
    """
    Stack two Graphs vertically (top/bottom). Handy for equity + drawdown, etc.
    """
    id_top = ids[0] or "graph-top"
    id_bottom = ids[1] or "graph-bottom"
    cfg = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]} | (config or {})
    return html.Div(
        children=[
            dcc.Graph(id=id_top, figure=fig_top, config=cfg, style={"height": f"{heights[0]}px"}),
            html.Div(style={"height": "8px"}),  # small spacer
            dcc.Graph(id=id_bottom, figure=fig_bottom, config=cfg, style={"height": f"{heights[1]}px"}),
        ]
    )

def pretty_card(children):
    return html.Div(
        children,
        style={
            "border": "1px solid #ddd",
            "borderRadius": "12px",
            "padding": "14px",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.06)",
            "background": "white",
        },
    )