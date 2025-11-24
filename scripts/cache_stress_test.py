"""
Step 12 Part 2 - Cache invalidation stress test.

This script runs a set of selection permutations against the core artifact and
portfolio cache helpers to ensure that caches invalidate (or stay warm) when
inputs change. Each case reports whether the resulting digest matches the
baseline selection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from Portfolio_Dashboard7_instrumented import (  # noqa: E402
        app,
        make_selection_key,
        build_core_artifact,
        build_portfolio_view,
        export_cache_stats,
    )

    INSTRUMENTED = True
except ModuleNotFoundError:
    from Portfolio_Dashboard7 import (  # type: ignore  # noqa: E402
        app,
        make_selection_key,
        build_core_artifact,
        build_portfolio_view,
    )

    INSTRUMENTED = False
    export_cache_stats = None  # type: ignore
from src.etl import parse_tradestation_trades  # noqa: E402

ALL_FILES = sorted(p.name for p in ROOT.glob("tradeslist_*.xlsx"))


def _ensure_trade_store(files: List[str]) -> None:
    trade_store: Dict[str, pd.DataFrame] = getattr(app.server, "trade_store", {})
    mtm_store: Dict[str, pd.DataFrame] = getattr(app.server, "mtm_store", {})

    for fname in files:
        if fname in trade_store:
            continue
        path = ROOT / fname
        if not path.exists():
            raise FileNotFoundError(f"Trade file not found: {path}")
        trades_df, mtm_df = parse_tradestation_trades(path.read_bytes(), fname)
        trade_store[fname] = trades_df
        mtm_store[fname] = mtm_df


def _hash_series(series: Optional[pd.Series]) -> str:
    if series is None or series.empty:
        return "empty"
    ser = pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index)
    digest = pd.util.hash_pandas_object(ser, index=True).values
    return hashlib.sha1(digest.tobytes()).hexdigest()


def _hash_frame(df: Optional[pd.DataFrame]) -> str:
    if df is None or df.empty:
        return "empty"
    normalized = df.reset_index(drop=True)
    digest = pd.util.hash_pandas_object(normalized, index=True).values
    return hashlib.sha1(digest.tobytes()).hexdigest()


def _capture_digest(sel: Dict[str, Any]) -> Dict[str, Any]:
    files = sel["files"]
    _ensure_trade_store(files)

    sel_key = make_selection_key(
        files,
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

    # Warm caches
    build_core_artifact(sel_key)
    build_portfolio_view(sel_key)

    artifact = build_core_artifact(sel_key)
    portfolio_view = build_portfolio_view(sel_key)

    digest_payload = {
        "artifact": {
            "files": tuple(sorted(files)),
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
    }
    digest_payload["overall_hash"] = hashlib.sha1(
        json.dumps(digest_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest_payload


def main() -> None:
    base_selection = {
        "files": ALL_FILES,
        "symbols": [],
        "intervals": [],
        "strategies": [],
        "direction": "All",
        "start": None,
        "end": None,
        "contracts": {},
        "margins": {},
        "version": 0,
    }

    tests: List[Dict[str, Any]] = [
        {"name": "base", "selection": base_selection},
        {"name": "repeat_base", "selection": base_selection, "expect_same_as": "base"},
        {
            "name": "subset_files",
            "selection": {**base_selection, "files": ALL_FILES[: max(1, len(ALL_FILES) // 2)]},
            "expect_same_as": "base",
            "should_match": False,
        },
        {
            "name": "date_filter",
            "selection": {**base_selection, "start": "2024-01-01", "end": "2024-12-31"},
            "expect_same_as": "base",
            "should_match": False,
        },
        {
            "name": "contracts_override",
            "selection": {
                **base_selection,
                "contracts": {ALL_FILES[0]: 2.0} if ALL_FILES else {},
            },
            "expect_same_as": "base",
            "should_match": False,
        },
        {
            "name": "margins_override",
            "selection": {
                **base_selection,
                "margins": {ALL_FILES[0]: 10000.0} if ALL_FILES else {},
            },
            "expect_same_as": "base",
            "should_match": False,
        },
    ]

    digests: Dict[str, Dict[str, Any]] = {}
    for test in tests:
        name = test["name"]
        print(f"Running selection '{name}'...")
        digest = _capture_digest(test["selection"])
        digests[name] = digest

    base_digest = digests.get("base")
    if base_digest is None:
        raise RuntimeError("Base selection did not run.")

    summary = []
    for test in tests:
        name = test["name"]
        if name == "base":
            continue
        reference = digests[test.get("expect_same_as", "base")]
        current = digests[name]
        same = current["overall_hash"] == reference["overall_hash"]
        should_match = test.get("should_match", test.get("expect_same_as") is not None)
        summary.append(
            {
                "name": name,
                "matches_reference": same,
                "expected": should_match,
            }
        )

    print("\nCache stress test summary:")
    for row in summary:
        status = "PASS" if row["matches_reference"] == row["expected"] else "FAIL"
        print(
            f" - {row['name']}: "
            f"matches_reference={row['matches_reference']} expected={row['expected']} -> {status}"
        )

    failures = [row for row in summary if row["matches_reference"] != row["expected"]]
    if failures:
        raise SystemExit(
            f"Cache invalidation failures detected: {[row['name'] for row in failures]}"
        )

    print("All cache invalidation checks passed.")

    if INSTRUMENTED and export_cache_stats:
        stats = export_cache_stats()
        if stats:
            print("\nCache instrumentation summary:")
            for name, data in stats.items():
                print(
                    f" - {name}: calls={data['calls']:.0f} hits={data['hits']:.0f} "
                    f"misses={data['misses']:.0f} miss_rate={data['miss_rate']:.2%} "
                    f"avg_miss_ms={data['avg_miss_ms']:.2f}"
                )
        else:
            print("\nCache instrumentation summary: no data recorded.")


if __name__ == "__main__":
    main()
