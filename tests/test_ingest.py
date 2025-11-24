from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
API_SRC = ROOT / "api"
if str(API_SRC) not in sys.path:
    sys.path.append(str(API_SRC))

from api.app.ingest import IngestService, MetadataIndex, parse_filename_meta


def test_parse_filename_meta_handles_tradestation_pattern() -> None:
    symbol, interval, strategy = parse_filename_meta("tests/data/tradeslist_MNQ_15_5Xnew.xlsx")
    assert symbol == "MNQ"
    assert interval == 15
    assert strategy == "5Xnew"


def test_ingest_writes_parquet_and_metadata(tmp_path: Path) -> None:
    source = ROOT / "tests" / "data" / "tradeslist_MNQ_15_5Xnew.xlsx"
    service = IngestService(storage_root=tmp_path)

    meta = service.ingest_file(source)

    trades_path = tmp_path / "parquet" / "trades" / f"{meta.file_id}.parquet"
    assert trades_path.exists()

    trades_df = pl.read_parquet(trades_path)
    required = {
        "File",
        "Symbol",
        "Interval",
        "Strategy",
        "entry_time",
        "exit_time",
        "entry_type",
        "direction",
        "net_profit",
        "runup",
        "drawdown_trade",
        "gross_profit",
        "commission",
        "slippage",
        "contracts",
        "CumulativePL_raw",
        "pct_profit_raw",
        "notional_exposure",
    }
    assert required.issubset(set(trades_df.columns))
    assert meta.rows == trades_df.height
    assert meta.symbol == "MNQ"
    assert meta.interval == 15
    assert meta.strategy == "5Xnew"
    assert meta.date_min is not None and meta.date_max is not None
    assert meta.file_hash.startswith(meta.file_id)

    index_path = tmp_path / "metadata" / "index.json"
    assert index_path.exists()
    payload = json.loads(index_path.read_text())
    assert any(entry["file_id"] == meta.file_id for entry in payload)


def test_metadata_index_tracks_multiple_ingests(tmp_path: Path) -> None:
    service = IngestService(storage_root=tmp_path)
    file_one = ROOT / "tests" / "data" / "tradeslist_CD_30_4Xnew.xlsx"
    file_two = ROOT / "tests" / "data" / "tradeslist_MES_15_4Xnew.xlsx"

    meta1 = service.ingest_file(file_one)
    meta2 = service.ingest_file(file_two)

    index = MetadataIndex(tmp_path / "metadata" / "index.json")
    entries = index.all()
    assert len(entries) == 2
    assert {meta1.file_id, meta2.file_id} == {entry.file_id for entry in entries}
    assert index.get(meta1.file_id).symbol == "CD"
    assert index.get(meta2.file_id).interval == 15
