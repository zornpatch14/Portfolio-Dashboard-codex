from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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

    # ---------- public API ----------
    def load_trades(self, path: Path) -> LoadedTrades:
        return load_trade_file(path)

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

    # ---------- computations ----------
    def _compute_equity(self, loaded: LoadedTrades) -> pl.DataFrame:
        trades = loaded.trades.sort("exit_time")
        cumulative = trades["net_profit"].cum_sum()
        equity = cumulative + self.starting_equity
        return pl.DataFrame({"timestamp": trades["exit_time"], "equity": equity})

    def _compute_percent_equity(self, loaded: LoadedTrades) -> pl.DataFrame:
        equity_df = self._compute_equity(loaded)
        percent = equity_df["equity"] / self.starting_equity * 100
        return pl.DataFrame({"timestamp": equity_df["timestamp"], "percent_equity": percent})

    def _compute_daily_returns(self, loaded: LoadedTrades) -> pl.DataFrame:
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
        events = []
        for row in loaded.trades.iter_rows(named=True):
            direction = row["direction"].lower()
            contracts = int(row["contracts"])
            signed = contracts if direction == "buy" else -contracts
            events.append((row["entry_time"], signed))
            events.append((row["exit_time"], -signed))

        if not events:
            return pl.DataFrame({"timestamp": [], "net_position": []})

        events.sort(key=lambda tup: tup[0])
        timestamps, position_deltas = zip(*events)
        cumulative = pl.Series(position_deltas).cum_sum()
        return pl.DataFrame({"timestamp": pl.Series(timestamps), "net_position": cumulative})

    def _compute_margin(self, loaded: LoadedTrades) -> pl.DataFrame:
        netpos = self._compute_net_positions(loaded)
        margin_value = self.margin_per_contract or float(loaded.trades["margin_per_contract"].max())
        return netpos.with_columns((pl.col("net_position").abs() * margin_value).alias("margin_used"))

