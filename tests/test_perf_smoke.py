"""Performance-oriented smoke checks for parity selections."""

from __future__ import annotations

import time

import pytest

from tests.parity_harness import compute_selection_summary, select_subset


# Budget in seconds for each selection to stay under during CI.
PERF_BUDGETS = {
    "mnq15_single": 8.0,
    "all_files_combo": 25.0,
}


@pytest.mark.parametrize("selection", select_subset(PERF_BUDGETS.keys()), ids=lambda s: s["name"])
def test_selection_perf_smoke(selection):
    budget = PERF_BUDGETS[selection["name"]]
    start = time.perf_counter()
    summary = compute_selection_summary(selection)
    elapsed = time.perf_counter() - start

    assert summary["portfolio"]["trades"] > 0
    assert elapsed < budget, f"Selection {selection['name']} exceeded budget {budget}s (took {elapsed:.2f}s)"
