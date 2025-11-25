from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import polars as pl

from api.app.data.loader import LoadedTrades, load_trade_file


@dataclass
class SeriesBundle:
    equity: pl.DataFrame
    percent_equity: pl.DataFrame
    daily_returns: pl.DataFrame
    net_position: pl.DataFrame
    margin: pl.DataFrame


class PerFileCache:
    """Lightweight per-file cache that stores computed series as Parquet files."""

    def __init__(
        self,
        storage_dir: Path | str | None = None,
        starting_equity: float = 100_000.0,
        margin_per_contract: float | None = None,
    ) -> None:
        base_dir = Path(storage_dir) if storage_dir else Path(os.getenv("API_CACHE_DIR", ".cache"))
        self.storage_dir = base_dir / "per_file"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.starting_equity = starting_equity
        self.margin_per_contract = margin_per_contract
        self.default_contract_multiplier = 1.0

    # ---------- public API ----------
    def load_trades(self, path: Path) -> LoadedTrades:
        return load_trade_file(path)

    def load_mtm(self, trades_path: Path, file_id: str) -> pl.DataFrame:
        """Load MTM parquet if available (derived from trades path)."""

        # Expect layout: .../parquet/trades/<file_id>.parquet -> mtm at .../parquet/mtm/<file_id>.parquet
        mtm_dir = trades_path.parent.parent / "mtm"
        mtm_path = mtm_dir / f"{file_id}.parquet"
        if mtm_path.exists():
            try:
                return pl.read_parquet(mtm_path)
            except Exception:
                return pl.DataFrame({"mtm_date": [], "mtm_net_profit": [], "mtm_session_start": [], "mtm_session_end": []})
        return pl.DataFrame({"mtm_date": [], "mtm_net_profit": [], "mtm_session_start": [], "mtm_session_end": []})

    def equity_curve(self, path: Path) -> pl.DataFrame:
        return self._get_or_build(path, "equity", self._compute_equity)

    def percent_equity(self, path: Path) -> pl.DataFrame:
        return self._get_or_build(path, "percent_equity", self._compute_percent_equity)

    def daily_returns(self, path: Path) -> pl.DataFrame:
        return self._get_or_build(path, "daily_returns", self._compute_daily_returns)

    def net_position(self, path: Path) -> pl.DataFrame:
        return self._get_or_build(path, "netpos", self._compute_net_positions)

    def margin_usage(self, path: Path) -> pl.DataFrame:
        return self._get_or_build(path, "margin", self._compute_margin)

    def bundle(self, path: Path) -> SeriesBundle:
        return SeriesBundle(
            equity=self.equity_curve(path),
            percent_equity=self.percent_equity(path),
            daily_returns=self.daily_returns(path),
            net_position=self.net_position(path),
            margin=self.margin_usage(path),
        )

    # ---------- cache helpers ----------
    def _artifact_path(self, file_id: str, artifact: str) -> Path:
        return self.storage_dir / f"{file_id}_{artifact}.parquet"

    def _get_or_build(self, path: Path, artifact: str, builder: Callable[[LoadedTrades], pl.DataFrame]) -> pl.DataFrame:
        loaded = self.load_trades(path)
        target = self._artifact_path(loaded.file_id, artifact)
        if target.exists():
            return pl.read_parquet(target)

        df = builder(loaded)
        df.write_parquet(target)
        return df

    # ---------- scaling helpers ----------
    @staticmethod
    def _apply_contract_multiplier(trades: pl.DataFrame, multiplier: float) -> pl.DataFrame:
        """Scale contracts and P&L-related fields by a contract multiplier.

        Keeps raw trades intact when multiplier==1. Intended to be applied at compute time
        (not persisted) so overrides don't poison cached artifacts.
        """

        if multiplier == 1 or multiplier == 1.0:
            return trades

        factor = float(multiplier)
        cols_to_scale = {
            "contracts": "contracts",
            "net_profit": "net_profit",
            "runup": "runup",
            "drawdown_trade": "drawdown_trade",
            "commission": "commission",
            "slippage": "slippage",
            "gross_profit": "gross_profit",
            "net_profit_raw": "net_profit_raw",
        }
        updates = []
        for col, alias in cols_to_scale.items():
            if col in trades.columns:
                updates.append((pl.col(col) * factor).alias(alias))
        if not updates:
            return trades
        return trades.with_columns(*updates)

    # ---------- computations ----------
    def _compute_equity(self, loaded: LoadedTrades) -> pl.DataFrame:
        # Try MTM-driven equity first
        mtm_df = self.load_mtm(loaded.path, loaded.file_id)
        if not mtm_df.is_empty():
            # Stepwise equity: open at session start, close at session end with MTM P&L applied.
            mtm_df = mtm_df.sort("mtm_date")
            equity_points = []
            running = self.starting_equity
            for row in mtm_df.iter_rows(named=True):
                session_start = row.get("mtm_session_start") or row["mtm_date"]
                session_end = row.get("mtm_session_end") or row["mtm_date"]
                equity_points.append({"timestamp": session_start, "equity": running})
                running = running + float(row["mtm_net_profit"])
                equity_points.append({"timestamp": session_end, "equity": running})

            equity_df = pl.DataFrame(equity_points).sort("timestamp")

            # Optional snap to trade exits: align cumulative trade P&L on exit sessions.
            trades = loaded.trades.sort(["exit_time", "entry_time", "trade_no"])
            if not trades.is_empty():
                cum_trade = trades["net_profit"].cum_sum()
                trades = trades.with_columns(cum_trade.alias("cum_trade_pl"))
                # Snap by matching exit_date to mtm_date.
                for row in trades.iter_rows(named=True):
                    exit_time = row["exit_time"]
                    if exit_time is None:
                        continue
                    exit_date = pl.Series([exit_time]).dt.date()[0]
                    match = mtm_df.filter(pl.col("mtm_date") == exit_date)
                    if match.is_empty():
                        continue
                    # Find the session end point index
                    session_end = match["mtm_session_end"][0]
                    idx = equity_df.get_column("timestamp").to_list().index(session_end) if session_end in equity_df["timestamp"].to_list() else None
                    if idx is not None:
                        equity_df = equity_df.with_row_count()
                        target_idx = equity_df.filter(pl.col("timestamp") == session_end)["row_nr"][0]
                        delta = row["cum_trade_pl"] + self.starting_equity - equity_df.filter(pl.col("timestamp") == session_end)["equity"][0]
                        if delta != 0:
                            equity_df = equity_df.with_columns(
                                pl.when(pl.col("row_nr") >= target_idx)
                                .then(pl.col("equity") + delta)
                                .otherwise(pl.col("equity"))
                                .alias("equity")
                            ).drop("row_nr")
                        else:
                            equity_df = equity_df.drop("row_nr")
                    else:
                        continue

            return equity_df

        # Fallback: trade-based equity
        trades = loaded.trades.sort(["exit_time", "entry_time", "trade_no"])
        cumulative = trades["net_profit"].cum_sum()
        equity = cumulative + self.starting_equity
        return pl.DataFrame({"timestamp": trades["exit_time"], "equity": equity})

    def _compute_percent_equity(self, loaded: LoadedTrades) -> pl.DataFrame:
        equity_df = self._compute_equity(loaded)
        percent = equity_df["equity"] / self.starting_equity * 100
        return pl.DataFrame({"timestamp": equity_df["timestamp"], "percent_equity": percent})

    def _compute_daily_returns(self, loaded: LoadedTrades) -> pl.DataFrame:
        mtm_df = self.load_mtm(loaded.path, loaded.file_id)

        # Use MTM sessions when available.
        if not mtm_df.is_empty():
            mtm_df = mtm_df.sort("mtm_date")
            _, intervals = self._netpos_intervals(loaded)
            rows = []
            for row in mtm_df.iter_rows(named=True):
                session_start = row.get("mtm_session_start") or row["mtm_date"]
                session_end = row.get("mtm_session_end") or row["mtm_date"]
                pnl = float(row["mtm_net_profit"])
                # Overlap intervals with session to compute avg open-position margin.
                overlap = intervals.filter(
                    (pl.col("start") < session_end) & (pl.col("end") > session_start)
                )
                if overlap.is_empty():
                    avg_capital = 0.0
                else:
                    caps = []
                    mins = []
                    for iv in overlap.iter_rows(named=True):
                        s = max(iv["start"], session_start)
                        e = min(iv["end"], session_end)
                        if s >= e:
                            continue
                        minutes = (e - s).total_seconds() / 60.0
                        if iv["net_position"] == 0:
                            continue
                        caps.append(iv["margin_used"])
                        mins.append(minutes)
                    if caps and mins and sum(mins) > 0:
                        # Weighted average by minutes open.
                        avg_capital = sum(c * m for c, m in zip(caps, mins)) / sum(mins)
                    else:
                        avg_capital = 0.0
                daily_ret = (pnl / avg_capital) if avg_capital > 0 else 0.0
                rows.append(
                    {
                        "date": row["mtm_date"],
                        "pnl": pnl,
                        "capital": avg_capital,
                        "daily_return": daily_ret,
                    }
                )
            return pl.DataFrame(rows) if rows else pl.DataFrame({"date": [], "pnl": [], "capital": [], "daily_return": []})

        # Fallback: group by exit date if MTM absent.
        trades = loaded.trades.with_columns(pl.col("exit_time").dt.date().alias("date"))
        margin_value = self.margin_per_contract or float(trades["margin_per_contract"].max())
        grouped = trades.group_by("date").agg(
            pnl=pl.col("net_profit").sum(),
            contracts=pl.col("contracts").sum(),
        )
        capital = grouped["contracts"].abs() * margin_value
        daily_return = pl.when(capital > 0).then(grouped["pnl"] / capital).otherwise(0)
        return grouped.drop("contracts").with_columns(capital=capital, daily_return=daily_return)

    def _compute_net_positions(self, loaded: LoadedTrades) -> pl.DataFrame:
        return self._netpos_intervals(loaded)[0]

    def _compute_margin(self, loaded: LoadedTrades) -> pl.DataFrame:
        _, intervals = self._netpos_intervals(loaded)
        return intervals.select(
            pl.col("start").alias("timestamp"),
            pl.col("margin_used"),
        )

    # ---------- interval builders ----------
    def _netpos_intervals(self, loaded: LoadedTrades) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Build per-file net position and margin intervals.

        Returns:
            points_df: point-in-time net position series (timestamp, net_position)
            intervals_df: piecewise-constant intervals with start/end, symbol, net_position, margin_used
        """

        trades = loaded.trades
        if trades.is_empty():
            empty_points = pl.DataFrame({"timestamp": [], "net_position": []})
            empty_intervals = pl.DataFrame({"start": [], "end": [], "symbol": [], "net_position": [], "margin_used": []})
            return empty_points, empty_intervals

        # Signed events from entry/exit.
        events = []
        for row in trades.iter_rows(named=True):
            direction = str(row.get("direction", "")).lower()
            contracts = float(row["contracts"])
            is_long = direction in {"buy", "long", "buy to open", "buy to cover"}
            signed = contracts if is_long else -contracts
            events.append((row["entry_time"], signed, row.get("Symbol") or row.get("symbol")))
            events.append((row["exit_time"], -signed, row.get("Symbol") or row.get("symbol")))

        if not events:
            empty_points = pl.DataFrame({"timestamp": [], "net_position": []})
            empty_intervals = pl.DataFrame({"start": [], "end": [], "symbol": [], "net_position": [], "margin_used": []})
            return empty_points, empty_intervals

        # Sort by time, then stabilize by trade number if needed.
        events.sort(key=lambda tup: (tup[0],))
        timestamps, deltas, symbols = zip(*events)
        cumulative = pl.Series(deltas).cum_sum()
        points_df = pl.DataFrame({"timestamp": pl.Series(timestamps), "net_position": cumulative})

        # Build intervals as piecewise constant segments between change points.
        starts = list(timestamps)
        ends = list(timestamps[1:]) + [timestamps[-1]]
        netpos_values = list(cumulative)
        sym = symbols[-1] if symbols else None

        # Margin per contract: prefer override; else from trades; else 0.
        margin_value = self.margin_per_contract or float(trades["margin_per_contract"].max())
        interval_df = pl.DataFrame(
            {
                "start": starts,
                "end": ends,
                "symbol": [sym] * len(starts),
                "net_position": netpos_values,
                "margin_used": [abs(v) * margin_value for v in netpos_values],
            }
        )
        return points_df, interval_df

    @staticmethod
    def expand_intervals_to_minutes(intervals: pl.DataFrame, start: str | None = None, end: str | None = None) -> pl.DataFrame:
        """Expand piecewise-constant intervals to per-minute samples within an optional window."""

        if intervals.is_empty():
            return pl.DataFrame({"timestamp": [], "net_position": [], "margin_used": [], "symbol": []})

        samples = []
        for row in intervals.iter_rows(named=True):
            s = row["start"]
            e = row["end"]
            if start and s < start:
                s = start
            if end and e > end:
                e = end
            if s >= e:
                continue
            rng = pl.datetime_range(start=s, end=e, interval="1m", eager=True, closed="left")
            samples.append(
                pl.DataFrame(
                    {
                        "timestamp": rng,
                        "net_position": [row["net_position"]] * len(rng),
                        "margin_used": [row["margin_used"]] * len(rng),
                        "symbol": [row.get("symbol")] * len(rng),
                    }
                )
            )
        if not samples:
            return pl.DataFrame({"timestamp": [], "net_position": [], "margin_used": [], "symbol": []})
        return pl.concat(samples).sort("timestamp")

