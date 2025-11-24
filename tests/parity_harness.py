"""Utilities for parity/performance smoke tests."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from src.etl import parse_tradestation_trades
from src.equity import combine_equity, equity_from_trades_subset, max_drawdown_value

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BASELINE_DIR = ROOT / "baseline"


@lru_cache(maxsize=None)
def _load_trades_for_file(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    with path.open("rb") as f:
        trades, _ = parse_tradestation_trades(f.read(), path.name)
    return trades


def load_selections() -> list[dict]:
    with (BASELINE_DIR / "selections.json").open("r", encoding="utf-8") as f:
        selections = json.load(f)
    if not isinstance(selections, list) or not selections:
        raise ValueError("selections.json must be a non-empty list")
    return selections


def load_goldens() -> dict:
    with (BASELINE_DIR / "parity_goldens.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_selection_summary(selection: dict) -> dict:
    eq_by_file: Dict[str, pd.Series] = {}
    trades_frames: list[pd.DataFrame] = []

    for fname in selection.get("files", []):
        trades = _load_trades_for_file(fname).copy()
        trades_frames.append(trades)
        eq_by_file[fname] = equity_from_trades_subset(trades)

    if trades_frames:
        all_trades = pd.concat(trades_frames, ignore_index=True)
    else:
        all_trades = pd.DataFrame(columns=["File", "net_profit"])

    portfolio_eq = combine_equity(eq_by_file, selection.get("files", [])).get(
        "Portfolio", pd.Series(dtype=float)
    )

    files_summary: Dict[str, dict] = {}
    for fname, eq in eq_by_file.items():
        sub = all_trades[all_trades["File"] == fname]
        files_summary[fname] = {
            "trades": int(len(sub)),
            "net_profit": float(sub["net_profit"].sum()) if not sub.empty else 0.0,
            "max_dd": float(max_drawdown_value(eq)) if not eq.empty else 0.0,
        }

    portfolio_summary = {
        "trades": int(len(all_trades)),
        "net_profit": float(all_trades["net_profit"].sum()) if not all_trades.empty else 0.0,
        "max_dd": float(max_drawdown_value(portfolio_eq)) if not portfolio_eq.empty else 0.0,
        "equity_end": float(portfolio_eq.iloc[-1]) if not portfolio_eq.empty else 0.0,
    }

    return {"files": files_summary, "portfolio": portfolio_summary}


def get_selection_map() -> dict[str, dict]:
    return {sel["name"]: sel for sel in load_selections()}


def select_subset(names: Iterable[str]) -> list[dict]:
    mapping = get_selection_map()
    missing = [n for n in names if n not in mapping]
    if missing:
        raise KeyError(f"Missing selections: {missing}")
    return [mapping[n] for n in names]
