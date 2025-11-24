"""
Smoke placeholder.

Intended future flow:
- Load baseline selections from tests/baseline/selections.json.
- Call new API endpoints for each selection and compare to goldens (when available).

Current lightweight check:
- Ensure selections file parses and referenced files exist under tests/data/.
"""

import json
from pathlib import Path


def test_selections_files_exist():
    root = Path(__file__).resolve().parents[1]
    sel_path = root / "baseline" / "selections.json"
    data_dir = root / "data"

    with sel_path.open("r", encoding="utf-8") as f:
        selections = json.load(f)

    assert isinstance(selections, list) and selections, "selections.json must be a non-empty list"

    missing = []
    for sel in selections:
        for fname in sel.get("files", []):
            if not (data_dir / fname).exists():
                missing.append(fname)

    assert not missing, f"Missing files in tests/data: {missing}"
