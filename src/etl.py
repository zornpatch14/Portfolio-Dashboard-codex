# src/etl.py

from __future__ import annotations
import base64
import io
import math
import os
import numpy as np
import pandas as pd
from typing import Any, Iterable, Tuple
from .constants import SHEET_HINTS, FILENAME_RE, POINT_VALUE
from .helpers import _currency_to_float




def find_header_row(raw_df: pd.DataFrame) -> int | None:
    """Find header row by locating a row containing both '#', 'Type' and 'Date/Time'."""
    for i in range(min(50, len(raw_df))):
        row = raw_df.iloc[i].astype(str).str.strip()
        if ("#" in row.values) and row.str.contains("Type", case=False, na=False).any() \
           and row.str.contains("Date/Time", case=False, na=False).any():
            return i
    return None

def load_sheet_guess(xls: pd.ExcelFile) -> str:
    """Pick the sheet likely to be the 'Trades List'."""
    for hint in SHEET_HINTS:
        for s in xls.sheet_names:
            if hint.lower() in s.lower():
                return s
    return xls.sheet_names[0]

def parse_filename_meta(filename: str) -> tuple[str, int | None, str]:
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    m = FILENAME_RE.search(base)
    if m:
        sym = m.group("symbol").upper()
        interval = int(m.group("interval"))
        strat = m.group("strategy")
        return sym, interval, strat
    # Fallback: split on underscores
    toks = name.split("_")
    if len(toks) >= 4 and toks[0].lower().startswith("tradeslist"):
        sym = toks[1].upper()
        try:
            interval = int(toks[2])
        except Exception:
            interval = None
        strat = "_".join(toks[3:])
        return sym, interval, strat
    # Last fallback
    return "UNKNOWN", None, name

def decode_upload(contents: str, filename: str) -> tuple[bytes, str]:
    """Decode dcc.Upload content string -> raw bytes and display name."""
    _content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    display = os.path.basename(filename) or "uploaded.xlsx"
    return decoded, display


def _parse_mtm_daily_sheet(xls: pd.ExcelFile, filename: str) -> pd.DataFrame:
    """
    Extract the optional 'Daily' sheet containing mark-to-market rows.

    Expected layout:
        Row 0: "TradeStation Periodical Returns:Daily"
        Row 1: "Mark-To-Market Period Analysis:"
        Row 2+: table with 'Period', 'Net Profit', ...

    We only keep Period + Net Profit.
    """
    if "Daily" not in xls.sheet_names:
        return pd.DataFrame(columns=["mtm_date", "mtm_net_profit"])

    try:
        raw = pd.read_excel(xls, sheet_name="Daily", header=None, engine="openpyxl")
    except Exception:
        return pd.DataFrame(columns=["mtm_date", "mtm_net_profit"])

    header_idx = None
    for i in range(min(len(raw), 50)):
        row = raw.iloc[i].astype(str).str.strip()
        if row.str.contains("Period", case=False, na=False).any() and \
           row.str.contains("Net Profit", case=False, na=False).any():
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

    out["Net Profit"] = out["Net Profit"].map(_currency_to_float)
    out = out.dropna(subset=["Period", "Net Profit"])
    out["Net Profit"] = out["Net Profit"].astype(float)
    if out.empty:
        return pd.DataFrame(columns=["mtm_date", "mtm_net_profit"])

    out = out.sort_values("Period").reset_index(drop=True)
    out["File"] = os.path.basename(filename)
    out = out.rename(columns={"Period": "mtm_date", "Net Profit": "mtm_net_profit"})
    return out[["File", "mtm_date", "mtm_net_profit"]]

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names we care about (case-insensitive).
    Expected columns (or close variants):
      '#', 'Type', 'Date/Time', 'Signal', 'Price',
      'Shares/Ctrts - Profit/Loss',
      'Net Profit - Cum Net Profit',
      'Run-up/Drawdown', 'Comm.', 'Slippage'
    """
    col_map: dict[str, str] = {}

    def find_col(candidates: Iterable[str], required: bool = True) -> str | None:
        for c in df.columns:
            s = str(c).strip()
            low = s.lower()
            for cand in candidates:
                if low == cand.lower() or cand.lower() in low:
                    col_map[s] = candidates[0]  # map to primary name
                    return s
        if required:
            return None
        return None

    # map pound column
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

def parse_tradestation_trades(file_bytes: bytes, filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Parse a TradeStation 'Trades List' Excel into one-row-per-trade with metadata.
    Returns (trades_df, mtm_daily_df).

    trades_df columns:
      File, Symbol, Interval, Strategy,
      entry_time, exit_time, entry_type, direction,
      net_profit, runup, drawdown_trade, contracts,
      commission, slippage, CumulativePL_raw
    """
    bio = io.BytesIO(file_bytes)
    xls = pd.ExcelFile(bio, engine="openpyxl")
    sheet = load_sheet_guess(xls)
    raw = pd.read_excel(xls, sheet_name=sheet, header=None, engine="openpyxl")

    hdr_idx = find_header_row(raw)
    if hdr_idx is None:
        hdr_idx = 2  # common case

    df = pd.read_excel(xls, sheet_name=sheet, header=hdr_idx, engine="openpyxl")

    # Fix Type column if case mismatch
    if "Type" not in df.columns:
        matches = [c for c in df.columns if isinstance(c, str) and c.lower() == "type"]
        if matches:
            df.rename(columns={matches[0]: "Type"}, inplace=True)

    # Drop repeated header rows, blanks
    df = df.dropna(subset=["Type"])
    df = df[df["Type"] != "Type"]

    # Canonicalize
    df = canonicalize_columns(df)

    # Trade number
    if "#" in df.columns:
        df["TradeNo"] = pd.to_numeric(df["#"], errors="coerce").ffill().astype("Int64")
    else:
        df["TradeNo"] = pd.Series(np.arange(1, len(df) + 1), dtype="Int64")

    # Parse datetimes & numerics
    if "Date/Time" in df.columns:
        df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="coerce")

    for numcol in ["Price", "Shares/Ctrts - Profit/Loss", "Net Profit - Cum Net Profit",
                   "Run-up/Drawdown", "Comm.", "Slippage"]:
        if numcol in df.columns:
            df[numcol] = pd.to_numeric(df[numcol], errors="coerce")

    if "% Profit" in df.columns:
        pct_series = (
            df["% Profit"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df["% Profit"] = pd.to_numeric(pct_series, errors="coerce") / 100.0

    # Pair entries/exits by TradeNo
    entry_types = {"Buy", "Sell Short"}
    exit_types = {"Sell", "Buy to Cover"}

    sym, interval, strat = parse_filename_meta(filename)
    trades: list[dict[str, Any]] = []
    last_exit_cum = 0.0
    open_by_no: dict[int, dict[str, Any]] = {}

    for _, row in df.iterrows():
        tno = int(row["TradeNo"]) if not pd.isna(row["TradeNo"]) else None
        rtype = str(row["Type"]).strip()
        ts = row["Date/Time"] if "Date/Time" in df.columns else pd.NaT
        cum_val = float(row.get("Net Profit - Cum Net Profit", np.nan))
        if np.isnan(cum_val):
            cum_val = last_exit_cum

        if rtype in entry_types:
            open_by_no[tno] = {
                "entry_time": ts,
                "entry_type": rtype,
                "entry_price": float(row.get("Price", np.nan)) if "Price" in row else np.nan,
                "contracts": float(row.get("Shares/Ctrts - Profit/Loss", np.nan))
                             if "Shares/Ctrts - Profit/Loss" in row else np.nan,
                "runup": float(row.get("Run-up/Drawdown", 0.0)) if "Run-up/Drawdown" in row else 0.0,
                "pct_profit_raw": float(row.get("% Profit", np.nan)) if "% Profit" in row else np.nan,
            }
        elif rtype in exit_types:
            ent = open_by_no.pop(tno, {
                "entry_time": pd.NaT, "entry_type": None,
                "entry_price": np.nan, "contracts": np.nan, "runup": 0.0,
                "pct_profit_raw": np.nan,
            })
            netp = float(cum_val) - float(last_exit_cum)
            last_exit_cum = float(cum_val)

            exit_price = float(row.get("Price", np.nan)) if "Price" in row else np.nan
            drawdown_trade = float(row.get("Run-up/Drawdown", 0.0)) if "Run-up/Drawdown" in row else 0.0
            comm = float(row.get("Comm.", 0.0)) if "Comm." in row else 0.0
            slip = float(row.get("Slippage", 0.0)) if "Slippage" in row else 0.0
            gross_pl = float(row.get("Shares/Ctrts - Profit/Loss", np.nan)) if "Shares/Ctrts - Profit/Loss" in row else np.nan

            pct_raw = ent.get("pct_profit_raw")
            notional = np.nan
            pct_net = np.nan
            point_value = POINT_VALUE.get((sym or '').upper())
            if point_value is None:
                notional = np.nan
            else:
                entry_price = float(ent.get('entry_price') or 0.0)
                contracts = float(ent.get('contracts') or 0.0)
                if not (math.isfinite(entry_price) and math.isfinite(contracts)):
                    notional = np.nan
                else:
                    notional = abs(entry_price * contracts * float(point_value))

            direction = "Long" if ent.get("entry_type") == "Buy" else ("Short" if ent.get("entry_type") == "Sell Short" else "Unknown")

            trades.append({
                "File": os.path.basename(filename),
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
                "pct_profit_raw": float(pct_raw) if pct_raw is not None else np.nan,
                "notional_exposure": float(notional) if notional is not None else np.nan,
            })

    # Include open trades (0 P&L at last known equity)
    for ent in open_by_no.values():
        trades.append({
            "File": os.path.basename(filename),
            "Symbol": sym,
            "Interval": interval,
            "Strategy": strat,

            "entry_time": ent.get("entry_time"),
            "exit_time": pd.NaT,
            "entry_type": ent.get("entry_type"),
            "direction": ("Long" if ent.get("entry_type") == "Buy"
                          else ("Short" if ent.get("entry_type") == "Sell Short" else "Unknown")),

            "entry_price": ent.get("entry_price"),
            "exit_price": np.nan,
            "contracts": ent.get("contracts"),
            "runup": float(ent.get("runup", 0.0)),
            "drawdown_trade": 0.0,
            "commission": 0.0,
            "slippage": 0.0,
            "net_profit": 0.0,
            "CumulativePL_raw": float(last_exit_cum),
            "gross_profit": np.nan,
            "pct_profit_raw": float(ent.get("pct_profit_raw")) if not np.isnan(ent.get("pct_profit_raw", np.nan)) else np.nan,
            "pct_profit_net": np.nan,
            "notional_exposure": np.nan,
        })

    out = pd.DataFrame(trades)

    # dtypes & ordering
    if not out.empty:
        if "exit_time" in out.columns:
            out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
        if "entry_time" in out.columns:
            out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")

        num_cols = ["net_profit", "runup", "drawdown_trade", "gross_profit",
                    "commission", "slippage", "contracts", "CumulativePL_raw",
                    "pct_profit_raw", "notional_exposure"]
        for c in num_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        cat_cols = ["Symbol", "Strategy", "direction", "entry_type"]
        for c in cat_cols:
            if c in out.columns:
                out[c] = out[c].astype("category")

        if "exit_time" in out.columns:
            out.sort_values("exit_time", inplace=True)
            out.reset_index(drop=True, inplace=True)

    mtm_daily = _parse_mtm_daily_sheet(xls, filename)

    return out, mtm_daily






