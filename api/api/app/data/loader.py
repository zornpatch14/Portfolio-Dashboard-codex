from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import polars as pl


DEFAULT_CONTRACT_MULTIPLIER = float(os.getenv("API_CONTRACT_MULTIPLIER", 5))
DEFAULT_MARGIN_PER_CONTRACT = float(os.getenv("API_MARGIN_PER_CONTRACT", 12000))


@dataclass
class LoadedTrades:
    file_id: str
    path: Path
    trades: pl.DataFrame


def _make_file_id(path: Path) -> str:
    """Create a short, deterministic id for a trades file."""
    digest = hashlib.sha256(path.name.encode()).hexdigest()
    return digest[:12]


def _pairwise(iterable: Iterable[pd.Series]) -> Iterable[tuple[pd.Series, pd.Series]]:
    iterator = iter(iterable)
    while True:
        try:
            entry = next(iterator)
            exit_row = next(iterator)
        except StopIteration:
            return
        yield entry, exit_row


def _compute_net_profit(entry_price: float, exit_price: float, direction: str, contracts: int) -> float:
    direction_lower = direction.lower()
    sign = 1 if direction_lower == "buy" else -1
    return (exit_price - entry_price) * DEFAULT_CONTRACT_MULTIPLIER * contracts * sign


def load_trade_file(path: Path) -> LoadedTrades:
    """
    Parse a TradeStation XLSX file into a normalized Polars DataFrame.

    The source sheets contain alternating entry/exit rows. We pair them and
    derive a simplified schema used by downstream caches.
    """

    df = pd.read_excel(path, skiprows=3)
    records: List[dict] = []

    for entry_row, exit_row in _pairwise(df.itertuples(index=False, name=None)):
        entry_action = str(entry_row[1])
        try:
            entry_time = pd.to_datetime(entry_row[2])
            exit_time = pd.to_datetime(exit_row[2])
        except (TypeError, ValueError):
            # skip rows that contain headers or open trades
            continue

        entry_price = float(entry_row[4])
        exit_price = float(exit_row[4])

        contracts_raw = entry_row[5]
        contracts = int(abs(contracts_raw)) if pd.notna(contracts_raw) and contracts_raw != 0 else 1

        net_profit = _compute_net_profit(entry_price, exit_price, entry_action, contracts)

        records.append(
            {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "direction": entry_action,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "contracts": contracts,
                "net_profit": net_profit,
                "margin_per_contract": DEFAULT_MARGIN_PER_CONTRACT,
            }
        )

    trades = pl.DataFrame(records)
    file_id = _make_file_id(path)
    return LoadedTrades(file_id=file_id, path=path, trades=trades)


def list_sample_trade_files() -> list[Path]:
    """Return a deterministic list of bundled sample trades files for demos/tests."""
    data_dir = Path(__file__).resolve().parents[4] / "tests" / "data"
    return sorted(data_dir.glob("tradeslist_*.xlsx"))

