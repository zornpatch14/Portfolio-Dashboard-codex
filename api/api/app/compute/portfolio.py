from __future__ import annotations



from dataclasses import dataclass

from pathlib import Path

from typing import Iterable, Optional



import polars as pl



from api.app.compute.caches import PerFileCache, SeriesBundle, percent_from_equity
from api.app.data.loader import LoadedTrades

from api.app.compute.downsampling import DownsampleResult, downsample_timeseries

from api.app.ingest import TradeFileMetadata





_EMPTY_SPIKES = pl.DataFrame({"timestamp": [], "marker_value": [], "drawdown": [], "runup": [], "trade_no": []})





@dataclass

class ContributorSeries:

    file_id: str

    path: Path

    bundle: SeriesBundle

    label: str

    symbol: str | None = None

    interval: str | None = None

    strategy: str | None = None





@dataclass
class PortfolioView:
    equity: pl.DataFrame
    percent_equity: pl.DataFrame
    daily_percent_portfolio: pl.DataFrame
    daily_returns: pl.DataFrame
    net_position: pl.DataFrame
    margin: pl.DataFrame
    contributors: list[ContributorSeries]
    spikes: pl.DataFrame | None = None



class PortfolioAggregator:

    def __init__(self, cache: PerFileCache) -> None:

        self.cache = cache



    def aggregate(

        self,

        files: Iterable[Path],

        metas: Optional[Iterable[TradeFileMetadata]] = None,

        contract_multipliers: Optional[dict[str, float]] = None,

        margin_overrides: Optional[dict[str, float]] = None,

        direction: Optional[str] = None,

        include_spikes: bool = False,

        loaded_trades: Optional[dict[str, LoadedTrades]] = None,

    ) -> PortfolioView:

        files = list(files)

        if not files:
            empty_equity = pl.DataFrame({"timestamp": [], "equity": []})
            empty_percent = pl.DataFrame({"timestamp": [], "percent_equity": []})
            empty_daily = pl.DataFrame({"date": [], "pnl": [], "capital": [], "daily_return": []})
            empty_netpos = pl.DataFrame({"timestamp": [], "net_position": []})
            empty_margin = pl.DataFrame({"timestamp": [], "margin_used": []})
            return PortfolioView(
                equity=empty_equity,
                percent_equity=empty_percent,
                daily_percent_portfolio=pl.DataFrame(
                    {"date": [], "pnl": [], "capital": [], "daily_pct": [], "cum_pct": []}
                ),
                daily_returns=empty_daily,
                net_position=empty_netpos,
                margin=empty_margin,
                contributors=[],
                spikes=pl.DataFrame({"timestamp": [], "marker_value": [], "drawdown": [], "runup": [], "trade_no": []})
                if include_spikes
                else None,
            )



        contributors = self.build_contributors(

            files,

            metas=metas,

            contract_multipliers=contract_multipliers,

            margin_overrides=margin_overrides,

            direction=direction,

            include_spikes=include_spikes,

            loaded_trades=loaded_trades,

        )



        daily_frames = [c.bundle.daily_returns for c in contributors]

        netpos_frames = [c.bundle.net_position for c in contributors]

        margin_frames = [c.bundle.margin for c in contributors]



        combined_daily = pl.concat(daily_frames).group_by("date").agg(

            pnl=pl.col("pnl").sum(),

            capital=pl.col("capital").sum(),

        )

        combined_daily = combined_daily.with_columns(

            pl.when(pl.col("capital") > 0)

            .then(pl.col("pnl") / pl.col("capital"))

            .otherwise(0)

            .alias("daily_return")

        ).sort("date")



        starting_equity = self.cache.starting_equity

        equity_curve = combined_daily.select(

            pl.col("date").alias("timestamp"),

            (pl.col("pnl").cum_sum() + starting_equity).alias("equity"),

        )
        percent_equity = percent_from_equity(equity_curve)



        portfolio_percent = _combine_daily_percent(contributors)
        netpos = _combine_timeseries(netpos_frames, "net_position")

        margin = _combine_timeseries(margin_frames, "margin_used")

        spikes_df = None

        if include_spikes:

            spike_frames = [c.bundle.spikes for c in contributors if len(c.bundle.spikes)]

            spikes_df = pl.concat(spike_frames).sort("timestamp") if spike_frames else None



        return PortfolioView(
            equity=equity_curve,
            percent_equity=percent_equity,
            daily_percent_portfolio=portfolio_percent,
            daily_returns=combined_daily,
            net_position=netpos,
            margin=margin,
            contributors=contributors,
            spikes=spikes_df,
        )



    def build_contributors(

        self,

        files: Iterable[Path],

        metas: Optional[Iterable[TradeFileMetadata]] = None,

        contract_multipliers: Optional[dict[str, float]] = None,

        margin_overrides: Optional[dict[str, float]] = None,

        direction: Optional[str] = None,

        include_spikes: bool = False,

        loaded_trades: Optional[dict[str, LoadedTrades]] = None,

    ) -> list[ContributorSeries]:

        meta_map = {m.file_id: m for m in metas} if metas else {}

        if metas:

            for meta in metas:

                path_stem = Path(meta.trades_path).stem

                if path_stem not in meta_map:

                    meta_map[path_stem] = meta

        contract_multipliers = contract_multipliers or {}

        margin_overrides = margin_overrides or {}



        def overrides_for(path: Path) -> tuple[float | None, float | None, TradeFileMetadata | None, str]:

            file_id = path.stem

            meta = meta_map.get(file_id)

            if meta and meta.file_id:

                file_id = meta.file_id



            key_candidates = [file_id]

            if file_id != path.stem:

                key_candidates.append(path.stem)



            cmult = None

            margin = None

            for key in key_candidates:

                if cmult is None and key in contract_multipliers:

                    cmult = contract_multipliers[key]

                if margin is None and key in margin_overrides:

                    margin = margin_overrides[key]

                if cmult is not None and margin is not None:

                    break



            return cmult, margin, meta, file_id



        contributors: list[ContributorSeries] = []

        for path in files:

            cmult, marg, meta, file_id = overrides_for(path)

            loaded = loaded_trades.get(file_id) if loaded_trades else None

            equity = self.cache.equity_curve(path, contract_multiplier=cmult, margin_override=marg, direction=direction, loaded=loaded, file_id=file_id)

            percent_equity = self.cache.percent_equity_curve(path, contract_multiplier=cmult, margin_override=marg, direction=direction, loaded=loaded, file_id=file_id)

            daily = self.cache.daily_returns(path, contract_multiplier=cmult, margin_override=marg, direction=direction, loaded=loaded, file_id=file_id)

            netpos = self.cache.net_position(path, contract_multiplier=cmult, margin_override=marg, direction=direction, loaded=loaded, file_id=file_id)

            margin_usage = self.cache.margin_usage(path, contract_multiplier=cmult, margin_override=marg, direction=direction, loaded=loaded, file_id=file_id)

            spikes_df = (

                self.cache.spike_overlay(path, contract_multiplier=cmult, direction=direction, loaded=loaded, file_id=file_id)

                if include_spikes

                else _EMPTY_SPIKES

            )

            label = (meta.original_filename or meta.filename) if meta else path.name

            daily_percent = self.cache.daily_percent_returns(
                path, contract_multiplier=cmult, margin_override=marg, direction=direction, loaded=loaded, file_id=file_id
            )
            contributors.append(
                ContributorSeries(
                    file_id=file_id,
                    path=path,
                    bundle=SeriesBundle(
                        equity=equity,
                        percent_equity=percent_equity,
                        daily_returns=daily,
                        daily_percent=daily_percent,
                        net_position=netpos,
                        margin=margin_usage,
                        spikes=spikes_df,
                    ),
                    label=label,
                    symbol=(meta.symbol or "").upper() if meta and meta.symbol else None,
                    interval=str(meta.interval) if meta and meta.interval is not None else None,
                    strategy=meta.strategy or None,
                )
            )

        return contributors



    def downsample_equity(self, view: PortfolioView, target_points: int = 2000) -> DownsampleResult:

        return downsample_timeseries(view.equity, "timestamp", "equity", target_points=target_points)





def _combine_timeseries(frames: list[pl.DataFrame], value_col: str) -> pl.DataFrame:

    if not frames:

        return pl.DataFrame({"timestamp": [], value_col: []})



    stacked = pl.concat(frames).sort("timestamp")

    combined = stacked.group_by("timestamp").agg(pl.col(value_col).sum()).sort("timestamp")

    return combined



def _combine_daily_percent(contributors: list[ContributorSeries]) -> pl.DataFrame:
    if not contributors:
        return pl.DataFrame({"date": [], "pnl": [], "capital": [], "daily_pct": [], "cum_pct": []})

    rows: list[pl.DataFrame] = []
    for contributor in contributors:
        frame = contributor.bundle.daily_percent
        if frame.is_empty():
            continue
        rows.append(frame.select("date", "pnl", "capital", "daily_return"))
    if not rows:
        return pl.DataFrame({"date": [], "pnl": [], "capital": [], "daily_pct": [], "cum_pct": []})

    stacked = pl.concat(rows)
    grouped = stacked.group_by("date").agg(
        pnl=pl.col("pnl").sum(),
        capital=pl.col("capital").sum(),
    ).sort("date")
    daily_pct = pl.when(pl.col("capital") > 0).then(pl.col("pnl") / pl.col("capital")).otherwise(0.0)
    result = grouped.with_columns(daily_pct.alias("daily_pct"))
    result = result.with_columns(pl.col("daily_pct").cumsum().alias("cum_pct"))
    return result

