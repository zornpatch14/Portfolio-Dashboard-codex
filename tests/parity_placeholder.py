"""Parity smoke tests for selections defined in tests/baseline/selections.json."""

from __future__ import annotations

import json

import pytest

from tests.parity_harness import compute_selection_summary, load_goldens, load_selections


GOLDENS = load_goldens()
SELECTIONS = load_selections()


def _assert_float_close(actual: float, expected: float, rel: float = 1e-6):
    assert pytest.approx(expected, rel=rel) == actual


def _assert_summary_matches(actual: dict, expected: dict):
    assert set(actual["files"].keys()) == set(expected["files"].keys())
    for fname, metrics in expected["files"].items():
        got = actual["files"][fname]
        assert got["trades"] == metrics["trades"]
        _assert_float_close(got["net_profit"], metrics["net_profit"])
        _assert_float_close(got["max_dd"], metrics["max_dd"])

    for key in ("trades", "net_profit", "max_dd", "equity_end"):
        if key == "trades":
            assert actual["portfolio"][key] == expected["portfolio"][key]
        else:
            _assert_float_close(actual["portfolio"][key], expected["portfolio"][key])


@pytest.mark.parametrize("selection", SELECTIONS, ids=lambda s: s["name"])
def test_parity_against_baseline(selection):
    name = selection["name"]
    assert name in GOLDENS, f"Missing golden for selection {name}"
    summary = compute_selection_summary(selection)
    _assert_summary_matches(summary, GOLDENS[name])


@pytest.mark.parametrize("selection", SELECTIONS, ids=lambda s: s["name"])
def test_selection_schema_roundtrip(selection):
    # Ensure selections remain JSON-serializable and stable across runs
    as_json = json.loads(json.dumps(selection))
    assert as_json == selection
