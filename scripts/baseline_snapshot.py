"""
Step 12 Part 1 - Baseline snapshot generator.

Run this script after installing the project dependencies. It loads a set of
representative trade files into the Dash server stores, builds the core
artifact + portfolio view, and records stable hashes for downstream consumers
(metrics, allocator, CTA report). The output JSON can be checked into the repo
to track regressions across future refactors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Portfolio_Dashboard7 import (  # noqa: E402
    app,
    make_selection_key,
    build_core_artifact,
    build_portfolio_view,
    compute_metrics_cached,
    run_allocator_cached,
    _compute_cta_report,
)
from src.etl import parse_tradestation_trades  # noqa: E402

DATA_FILES = sorted(p.name for p in ROOT.glob("tradeslist_*.xlsx"))


def _hash_series(series: pd.Series | None) -> str:
    if series is None or series.empty:
        return "empty"
    ser = pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index)
    digest = pd.util.hash_pandas_object(ser, index=True).values
    return hashlib.sha1(digest.tobytes()).hexdigest()


def _hash_frame(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "empty"
    normalized = df.reset_index(drop=True)
    digest = pd.util.hash_pandas_object(normalized, index=True).values
    return hashlib.sha1(digest.tobytes()).hexdigest()


def _hash_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _ensure_trade_store(files: List[Path]) -> None:
    trade_store: Dict[str, pd.DataFrame] = getattr(app.server, "trade_store", {})
    mtm_store: Dict[str, pd.DataFrame] = getattr(app.server, "mtm_store", {})

    for path in files:
        name = path.name
        if name in trade_store:
            continue
        if not path.exists():
            raise FileNotFoundError(f"Trade file not found: {path}")
        trades_df, mtm_df = parse_tradestation_trades(path.read_bytes(), name)
        trade_store[name] = trades_df
        mtm_store[name] = mtm_df


def _capture_selection(sel: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_trade_store([ROOT / f for f in sel["files"]])

    sel_key = make_selection_key(
        sel["files"],
        sel.get("symbols") or [],
        sel.get("intervals") or [],
        sel.get("strategies") or [],
        sel.get("direction") or "All",
        sel.get("start"),
        sel.get("end"),
        sel.get("contracts") or {},
        sel.get("margins") or {},
        sel.get("version", 0),
    )

    artifact = build_core_artifact(sel_key)
    portfolio_view = build_portfolio_view(sel_key)
    metrics_rows = compute_metrics_cached(sel_key)

    allocator_cfg = sel.get("allocator", {})
    weights, diag = run_allocator_cached(
        sel_key,
        objective=allocator_cfg.get("objective", "max_return_over_dd"),
        leverage_cap=float(allocator_cfg.get("leverage_cap", 0.0)),
        margin_cap=float(allocator_cfg.get("margin_cap", 60.0)),
        equity_val=float(allocator_cfg.get("equity", 150000.0)),
    )

    cta_report, cta_label = _compute_cta_report(
        app.server.trade_store,
        sel["files"],
        sel.get("symbols"),
        sel.get("intervals"),
        sel.get("strategies"),
        sel.get("direction"),
        sel.get("start"),
        sel.get("end"),
        sel.get("contracts"),
        sel.get("margins"),
        sel.get("version"),
    )

    selection_summary = {
        "name": sel.get("name"),
        "artifact": {
            "trades_by_file": {
                fname: _hash_frame(df) for fname, df in artifact["trades_by_file"].items()
            },
            "equity_by_file": {
                fname: _hash_series(series) for fname, series in artifact["equity_by_file"].items()
            },
        },
        "portfolio": {
            "daily_return_pct": _hash_series(portfolio_view.daily_return_pct),
            "pct_equity_index": _hash_series(portfolio_view.pct_equity_index),
            "pct_equity": _hash_series(portfolio_view.pct_equity),
            "total_nav": _hash_series(portfolio_view.total_nav),
            "files_included": portfolio_view.files_included,
        },
        "metrics_digest": _hash_json(metrics_rows),
        "allocator_digest": _hash_json({"weights": weights, "diag": diag}),
        "cta_digest": _hash_json(
            {
                "label": cta_label,
                "report": getattr(cta_report, "summary", None).__dict__ if cta_report else None,
            }
        ),
    }
    return selection_summary


def main() -> None:
    selections = [
        {
            "name": "mnq_combo_all",
            "files": DATA_FILES,
            "symbols": [],
            "intervals": [],
            "strategies": [],
            "direction": "All",
            "start": None,
            "end": None,
            "contracts": {},
            "margins": {},
            "version": 0,
            "allocator": {
                "objective": "max_return_over_dd",
                "leverage_cap": 0,
                "margin_cap": 60,
                "equity": 150000,
            },
        }
    ]

    snapshots = [_capture_selection(sel) for sel in selections]
    output_path = ROOT / "baseline_snapshot.json"
    output_path.write_text(json.dumps({"selections": snapshots}, indent=2), encoding="utf-8")
    print(f"Wrote baseline snapshot to {output_path}")


if __name__ == "__main__":
    main()
