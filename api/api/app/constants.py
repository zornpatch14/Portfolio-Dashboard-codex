"""Symbol-level contract specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ContractSpec:
    initial_margin: float
    maintenance_margin: float
    big_point_value: float
    session_start_day_offset: int = -1  # days offset relative to mtm_date (start on prior day)
    session_start_hour: int = 17
    session_start_minute: int = 0
    session_end_day_offset: int = 0    # end on mtm_date
    session_end_hour: int = 16
    session_end_minute: int = 15


# Default margin/point specs; extend as needed.
MARGIN_SPEC: Dict[str, ContractSpec] = {
    "MNQ": ContractSpec(4000.0, 4000.0, 2.0),
    "MES": ContractSpec(2300.0, 2300.0, 5.0),
    "MYM": ContractSpec(1500.0, 1500.0, 1.0),
    "M2K": ContractSpec(1200.0, 1200.0, 5.0),
    "CD": ContractSpec(1100.0, 1100.0, 100000.0),
    "JY": ContractSpec(3200.0, 3200.0, 125000.0),
    "NE1": ContractSpec(1500.0, 1500.0, 100000.0),
    "NG": ContractSpec(3800.0, 3800.0, 10000.0),
    "KW": ContractSpec(2200.0, 2200.0, 50.0),
    "SF": ContractSpec(4500.0, 4500.0, 125000.0),
}

DEFAULT_CONTRACT_MULTIPLIER = 1.0
DEFAULT_MARGIN_PER_CONTRACT = 10_000.0
DEFAULT_ACCOUNT_EQUITY = 25_000.0


def get_contract_spec(symbol: str) -> ContractSpec:
    """Return the contract spec for a symbol, falling back to a safe default."""
    sym = (symbol or "").upper()
    if sym in MARGIN_SPEC:
        return MARGIN_SPEC[sym]
    # Conservative default if unknown symbol: keep margins high, BPV 1.
    return ContractSpec(initial_margin=10000.0, maintenance_margin=10000.0, big_point_value=1.0)
