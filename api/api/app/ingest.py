from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import polars as pl


# --- Filename parsing helpers -------------------------------------------------


def parse_filename_meta(filename: str) -> tuple[str, int | None, str]:
    """Parse symbol/interval/strategy hints from a TradeStation filename.

    Expected form: tradeslist_<SYMBOL>_<INTERVAL>_<STRATEGY>.xlsx
    Falls back to UNKNOWN/None when the pattern is not present.
    """

    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    tokens = stem.split("_")
    if len(tokens) >= 4 and tokens[0].lower().startswith("tradeslist"):
        sym = tokens[1].upper()
        try:
            interval = int(tokens[2])
        except Exception:
            interval = None
        strat = "_".join(tokens[3:])
        return sym, interval, strat
    return "UNKNOWN", None, stem


# --- XLSX parsing helpers -----------------------------------------------------


def _find_header_row(raw_df: pd.DataFrame) -> int | None:
    """Locate the header row by the presence of '#', 'Type', and 'Date/Time'."""

    for i in range(min(len(raw_df), 50)):
        row = raw_df.iloc[i].astype(str).str.strip()
        if ("#" in row.values) and row.str.contains("Type", case=False, na=False).any() and row.str.contains(
            "Date/Time", case=False, na=False
        ).any():
            return i
    return None


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names so the parser can work across file variants."""

    col_map: dict[str, str] = {}

    def find_col(candidates: Iterable[str], required: bool = True) -> None:
        for c in df.columns:
            s = str(c).strip()
            low = s.lower()
            for cand in candidates:
                if low == cand.lower() or cand.lower() in low:
                    col_map[s] = candidates[0]
                    return
        if required:
            return

    for c in df.columns:
        s = str(c).strip()
        if s.startswith("#") or s == "#":
            col_map[s] = "#"
            break

    find_col(["Type"])
    find_col(["Date/Time", "date"])
    find_col(["Signal"], required=False)
    find_col(["Price"], required=False)
    find_col(["Shares/Ctrts - Profit/Loss", "Shares", "Ctrts", "Profit/Loss"], required=False)
    find_col(["Net Profit - Cum Net Profit", "Cum Net Profit", "Cumulative Net", "Cum P&L", "Cum Profit"])
    find_col(["% Profit", "Percent Profit", "Profit %"], required=False)
    find_col(["Run-up/Drawdown", "Run-up", "Drawdown"], required=False)
    find_col(["Comm.", "Commission"], required=False)
    find_col(["Slippage"], required=False)

    if col_map:
        df = df.rename(columns=col_map)
    return df


def _parse_mtm_daily_sheet(xls: pd.ExcelFile, filename: str) -> pd.DataFrame:
    """Extract the optional Daily MTM sheet."""

    if "Daily" not in xls.sheet_names:
        return pd.DataFrame(columns=["mtm_date", "mtm_net_profit"])

    try:
        raw = pd.read_excel(xls, sheet_name="Daily", header=None, engine="openpyxl")
    except Exception:
        return pd.DataFrame(columns=["mtm_date", "mtm_net_profit"])

    header_idx = None
    for i in range(min(len(raw), 50)):
        row = raw.iloc[i].astype(str).str.strip()
        if row.str.contains("Period", case=False, na=False).any() and row.str.contains("Net Profit", case=False, na=False).any():
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame(columns=["mtm_date", "mtm_net_profit"])

    try:
        daily = pd.read_excel(xls, sheet_name="Daily", header=header_idx, engine="openpyxl")
    except Exception:
        return pd.DataFrame(columns=["mtm_date", "mtm_net_profit"])

    daily = daily.rename(columns=lambda c: str(c).strip())
    if "Period" not in daily.columns or "Net Profit" not in daily.columns:
        return pd.DataFrame(columns=["mtm_date", "mtm_net_profit"])

    out = daily[["Period", "Net Profit"]].copy()
    out["Period"] = pd.to_datetime(out["Period"], errors="coerce").dt.normalize()
    out["Net Profit"] = pd.to_numeric(out["Net Profit"], errors="coerce")
    out = out.dropna(subset=["Period", "Net Profit"])
    if out.empty:
        return pd.DataFrame(columns=["mtm_date", "mtm_net_profit"])

    out = out.sort_values("Period").reset_index(drop=True)
    out["File"] = os.path.basename(filename)
    out = out.rename(columns={"Period": "mtm_date", "Net Profit": "mtm_net_profit"})
    return out[["File", "mtm_date", "mtm_net_profit"]]


def parse_tradestation_trades(file_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parse a TradeStation Trades List XLSX into Polars DataFrames."""

    xls = pd.ExcelFile(file_path, engine="openpyxl")
    raw = pd.read_excel(xls, header=None, engine="openpyxl")
    hdr_idx = _find_header_row(raw) or 2
    df = pd.read_excel(xls, header=hdr_idx, engine="openpyxl")

    if "Type" not in df.columns:
        matches = [c for c in df.columns if isinstance(c, str) and c.lower() == "type"]
        if matches:
            df.rename(columns={matches[0]: "Type"}, inplace=True)

    # Preserve a deterministic trade number column for downstream ordering/QA.
    df = df.dropna(subset=["Type"])
    df = df[df["Type"] != "Type"]
    df = _canonicalize_columns(df)

    if "#" in df.columns:
        df["TradeNo"] = pd.to_numeric(df["#"], errors="coerce").ffill().astype("Int64")
    else:
        df["TradeNo"] = pd.Series(np.arange(1, len(df) + 1), dtype="Int64")

    if "Date/Time" in df.columns:
        df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="coerce")

    for numcol in [
        "Price",
        "Shares/Ctrts - Profit/Loss",
        "Net Profit - Cum Net Profit",
        "Run-up/Drawdown",
        "Comm.",
        "Slippage",
    ]:
        if numcol in df.columns:
            df[numcol] = pd.to_numeric(df[numcol], errors="coerce")

    if "% Profit" in df.columns:
        pct_series = (
            df["% Profit"].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
        )
        df["% Profit"] = pd.to_numeric(pct_series, errors="coerce") / 100.0

    entry_types = {"Buy", "Sell Short"}
    exit_types = {"Sell", "Buy to Cover"}

    sym, interval, strat = parse_filename_meta(str(file_path))
    trades: list[dict[str, Any]] = []
    last_exit_cum = 0.0
    # Keep the raw cumulative/net profit values for QA/parity; recomputation happens separately.
    open_by_no: dict[int, dict[str, Any]] = {}

    for _, row in df.iterrows():
        tno = int(row["TradeNo"]) if not pd.isna(row["TradeNo"]) else None
        rtype = str(row["Type"]).strip()
        ts = row["Date/Time"] if "Date/Time" in df.columns else pd.NaT
        cum_val = float(row.get("Net Profit - Cum Net Profit", np.nan))
        if math.isnan(cum_val):
            cum_val = last_exit_cum
        raw_net_profit = float(row.get("Net Profit - Cum Net Profit", np.nan)) if "Net Profit - Cum Net Profit" in df.columns else np.nan

        if rtype in entry_types:
            open_by_no[tno] = {
                "trade_no": tno,
                "entry_time": ts,
                "entry_type": rtype,
                "entry_price": float(row.get("Price", np.nan)) if "Price" in row else np.nan,
                "contracts": float(row.get("Shares/Ctrts - Profit/Loss", np.nan))
                if "Shares/Ctrts - Profit/Loss" in row
                else np.nan,
                "runup": float(row.get("Run-up/Drawdown", 0.0)) if "Run-up/Drawdown" in row else 0.0,
                "pct_profit_raw": float(row.get("% Profit", np.nan)) if "% Profit" in row else np.nan,
                "net_profit_raw": raw_net_profit,
                "cum_net_profit_raw": cum_val,
            }
        elif rtype in exit_types:
            ent = open_by_no.pop(
                tno,
                {
                    "trade_no": tno,
                    "entry_time": pd.NaT,
                    "entry_type": None,
                    "entry_price": np.nan,
                    "contracts": np.nan,
                    "runup": 0.0,
                    "pct_profit_raw": np.nan,
                    "net_profit_raw": np.nan,
                    "cum_net_profit_raw": np.nan,
                },
            )
            netp = float(cum_val) - float(last_exit_cum)
            last_exit_cum = float(cum_val)

            exit_price = float(row.get("Price", np.nan)) if "Price" in row else np.nan
            drawdown_trade = float(row.get("Run-up/Drawdown", 0.0)) if "Run-up/Drawdown" in row else 0.0
            comm = float(row.get("Comm.", 0.0)) if "Comm." in row else 0.0
            slip = float(row.get("Slippage", 0.0)) if "Slippage" in row else 0.0
            gross_pl = float(row.get("Shares/Ctrts - Profit/Loss", np.nan)) if "Shares/Ctrts - Profit/Loss" in row else np.nan
            exit_net_profit_raw = raw_net_profit

            pct_raw = ent.get("pct_profit_raw")
            notional = np.nan
            point_value = None
            if sym:
                # Point value lookup will come later; leave placeholder for now.
                point_value = None

            if point_value is None:
                notional = np.nan
            else:
                entry_price = float(ent.get("entry_price") or 0.0)
                contracts = float(ent.get("contracts") or 0.0)
                if not (math.isfinite(entry_price) and math.isfinite(contracts)):
                    notional = np.nan
                else:
                    notional = abs(entry_price * contracts * float(point_value))

            direction = "Long" if ent.get("entry_type") == "Buy" else ("Short" if ent.get("entry_type") == "Sell Short" else "Unknown")

            trades.append(
                {
                    "File": os.path.basename(file_path),
                    "trade_no": ent.get("trade_no"),
                    "Symbol": sym,
                    "Interval": interval,
                    "Strategy": strat,
                    "entry_time": ent.get("entry_time"),
                    "exit_time": ts,
                    "entry_type": ent.get("entry_type"),
                    "direction": direction,
                    "entry_price": ent.get("entry_price"),
                    "exit_price": exit_price,
                    "contracts": ent.get("contracts"),
                    "runup": float(ent.get("runup", 0.0)),
                    "drawdown_trade": float(drawdown_trade),
                    "gross_profit": float(gross_pl) if gross_pl is not None else np.nan,
                    "commission": float(comm),
                    "slippage": float(slip),
                    "net_profit": float(netp),
                    "CumulativePL_raw": float(last_exit_cum),
                    "net_profit_raw": float(ent.get("net_profit_raw")) if not math.isnan(ent.get("net_profit_raw", math.nan)) else float(exit_net_profit_raw) if not math.isnan(exit_net_profit_raw) else np.nan,
                    "pct_profit_raw": float(pct_raw) if pct_raw is not None else np.nan,
                    "notional_exposure": float(notional) if notional is not None else np.nan,
                }
            )

    for ent in open_by_no.values():
        trades.append(
            {
                "File": os.path.basename(file_path),
                "trade_no": ent.get("trade_no"),
                "Symbol": sym,
                "Interval": interval,
                "Strategy": strat,
                "entry_time": ent.get("entry_time"),
                "exit_time": pd.NaT,
                "entry_type": ent.get("entry_type"),
                "direction": ("Long" if ent.get("entry_type") == "Buy" else ("Short" if ent.get("entry_type") == "Sell Short" else "Unknown")),
                "entry_price": ent.get("entry_price"),
                "exit_price": np.nan,
                "contracts": ent.get("contracts"),
                "runup": float(ent.get("runup", 0.0)),
                "drawdown_trade": 0.0,
                "commission": 0.0,
                "slippage": 0.0,
                "net_profit": 0.0,
                "CumulativePL_raw": float(last_exit_cum),
                "net_profit_raw": float(ent.get("net_profit_raw")) if not math.isnan(ent.get("net_profit_raw", math.nan)) else np.nan,
                "gross_profit": np.nan,
                "pct_profit_raw": float(ent.get("pct_profit_raw")) if not math.isnan(ent.get("pct_profit_raw", np.nan)) else np.nan,
                "notional_exposure": np.nan,
            }
        )

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        if "exit_time" in trades_df.columns:
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], errors="coerce")
        if "entry_time" in trades_df.columns:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], errors="coerce")

        num_cols = [
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
        ]
        for c in num_cols:
            if c in trades_df.columns:
                trades_df[c] = pd.to_numeric(trades_df[c], errors="coerce")

        if "exit_time" in trades_df.columns:
            trades_df.sort_values("exit_time", inplace=True)
            trades_df.reset_index(drop=True, inplace=True)

    mtm_daily = _parse_mtm_daily_sheet(xls, str(file_path))
    return pl.from_pandas(trades_df), pl.from_pandas(mtm_daily)


# --- Metadata index -----------------------------------------------------------


@dataclass
class TradeFileMetadata:
    file_id: str
    filename: str
    file_hash: str
    symbol: str
    interval: int | None
    strategy: str
    date_min: datetime | None
    date_max: datetime | None
    rows: int
    mtm_rows: int
    trades_path: str
    mtm_path: str | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.date_min:
            data["date_min"] = self.date_min.isoformat()
        if self.date_max:
            data["date_max"] = self.date_max.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradeFileMetadata":
        data = data.copy()
        for key in ["date_min", "date_max"]:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)


class MetadataIndex:
    """Simple JSON-backed metadata index."""

    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.entries: dict[str, TradeFileMetadata] = {}
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            self.entries = {}
            return
        with self.index_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self.entries = {item["file_id"]: TradeFileMetadata.from_dict(item) for item in raw}

    def upsert(self, meta: TradeFileMetadata) -> None:
        self.entries[meta.file_id] = meta
        self._persist()

    def _persist(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [m.to_dict() for m in self.entries.values()]
        with self.index_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def all(self) -> list[TradeFileMetadata]:
        return list(self.entries.values())

    def get(self, file_id: str) -> TradeFileMetadata | None:
        return self.entries.get(file_id)


# --- Ingest service -----------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class IngestService:
    """Ingest XLSX trades into Polars/Parquet and update metadata index."""

    def __init__(self, storage_root: str | Path | None = None):
        root = Path(storage_root or os.getenv("DATA_ROOT", "./data"))
        self.storage_root = root
        self.trades_dir = root / "parquet" / "trades"
        self.mtm_dir = root / "parquet" / "mtm"
        self.index = MetadataIndex(root / "metadata" / "index.json")

    def ingest_file(self, xlsx_path: Path) -> TradeFileMetadata:
        trades_df, mtm_df = parse_tradestation_trades(xlsx_path)

        file_hash = _sha256_file(xlsx_path)
        file_id = file_hash[:12]

        self.trades_dir.mkdir(parents=True, exist_ok=True)
        self.mtm_dir.mkdir(parents=True, exist_ok=True)

        trades_path = self.trades_dir / f"{file_id}.parquet"
        trades_df.write_parquet(trades_path)

        mtm_path = None
        if not mtm_df.is_empty():
            path = self.mtm_dir / f"{file_id}.parquet"
            mtm_df.write_parquet(path)
            mtm_path = str(path)

        date_min = None
        date_max = None
        if "exit_time" in trades_df.columns and trades_df.height > 0:
            date_min = trades_df["exit_time"].min()
            date_max = trades_df["exit_time"].max()
            if isinstance(date_min, pl.Series):
                date_min = date_min.item()
            if isinstance(date_max, pl.Series):
                date_max = date_max.item()
        if date_min is not None and not isinstance(date_min, datetime):
            date_min = pl.from_epoch(pl.Series([date_min]), time_unit="ms")[0]
        if date_max is not None and not isinstance(date_max, datetime):
            date_max = pl.from_epoch(pl.Series([date_max]), time_unit="ms")[0]

        symbol, interval, strategy = parse_filename_meta(str(xlsx_path))
        meta = TradeFileMetadata(
            file_id=file_id,
            filename=os.path.basename(xlsx_path),
            file_hash=file_hash,
            symbol=symbol,
            interval=interval,
            strategy=strategy,
            date_min=date_min,
            date_max=date_max,
            rows=trades_df.height,
            mtm_rows=mtm_df.height,
            trades_path=str(trades_path),
            mtm_path=mtm_path,
        )

        self.index.upsert(meta)
        return meta


__all__ = [
    "IngestService",
    "MetadataIndex",
    "TradeFileMetadata",
    "parse_tradestation_trades",
    "parse_filename_meta",
]

