import hashlib
import json
from datetime import date
from typing import Dict, Iterable, Optional

from fastapi import Depends, HTTPException, Query, status

from .schemas import Selection


def _normalize_list(items: Optional[Iterable[str] | str]) -> list[str]:
    if items is None:
        return []
    if isinstance(items, str):
        raw_items = [items]
    else:
        raw_items = list(items)

    normalized: list[str] = []
    for raw in raw_items:
        if raw is None:
            continue
        for part in str(raw).split(","):
            cleaned = part.strip()
            if cleaned:
                normalized.append(cleaned)
    return normalized


def _parse_mapping(entries: Optional[Iterable[str] | str], label: str) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    for item in _normalize_list(entries):
        if ":" not in item:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{label} entries must be formatted as SYMBOL:VALUE",
            )
        key, raw_val = item.split(":", 1)
        key = key.strip()
        try:
            value = float(raw_val)
        except ValueError as exc:  # pragma: no cover - defensive
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid {label} value for {key}",
            ) from exc
        if key:
            mapping[key] = value
    return mapping



class SelectionQueryParams:
    def __init__(
        self,
        files: list[str] | None = Query(default=None, description="File identifiers to include"),
        symbols: list[str] | None = Query(default=None, description="Symbols to filter"),
        intervals: list[str] | None = Query(default=None, description="Intervals to filter"),
        strategies: list[str] | None = Query(default=None, description="Strategies to filter"),
        direction: Optional[str] = Query(default=None, description="Direction filter"),
        start_date: Optional[date] = Query(default=None, description="Inclusive start date (YYYY-MM-DD)"),
        end_date: Optional[date] = Query(default=None, description="Inclusive end date (YYYY-MM-DD)"),
        contract_multipliers: list[str] | None = Query(
            default=None, description="Per-file contract multiplier overrides (FILE_ID:VALUE)"
        ),
        margin_overrides: list[str] | None = Query(
            default=None, description="Per-file margin overrides (FILE_ID:VALUE)"
        ),
        spike_flag: bool = Query(default=False, description="Whether to include spike filter"),
        data_version: Optional[str] = Query(default=None, description="Data version key"),
        account_equity: Optional[float] = Query(default=None, description="Starting account equity for equity curves"),
    ):
        if start_date and end_date and end_date < start_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="end_date must be on or after start_date",
            )

        normalized_direction = direction.lower() if direction else None

        from .constants import DEFAULT_ACCOUNT_EQUITY

        equity_value = account_equity if account_equity is not None else DEFAULT_ACCOUNT_EQUITY

        self.selection = Selection(
            files=_normalize_list(files),
            symbols=[s.upper() for s in _normalize_list(symbols)],
            intervals=_normalize_list(intervals),
            strategies=_normalize_list(strategies),
            direction=normalized_direction,
            start_date=start_date,
            end_date=end_date,
            contract_multipliers=_parse_mapping(contract_multipliers, "contract multipliers"),
            margin_overrides=_parse_mapping(margin_overrides, "margin overrides"),
            spike_flag=spike_flag,
            data_version=data_version,
            account_equity=equity_value,
        )

    def __call__(self) -> Selection:
        return self.selection


class DownsampleFlag:
    def __init__(self, downsample: bool = Query(default=True, description="Downsample series output")):
        self.downsample = downsample

    def __call__(self) -> bool:
        return self.downsample


def get_selection(params: SelectionQueryParams = Depends(SelectionQueryParams)) -> Selection:
    """Extract the Selection model from the query param helper."""
    return params.selection


def selection_hash(selection: Selection) -> str:
    """Generate a deterministic hash for a selection payload."""

    payload = selection.model_dump(exclude_none=True)
    for transient_field in ("account_equity", "start_date", "end_date"):
        payload.pop(transient_field, None)
    for key, value in list(payload.items()):
        if isinstance(value, list):
            payload[key] = sorted(value)
        if isinstance(value, dict):
            payload[key] = {k: value[k] for k in sorted(value)}

    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]
