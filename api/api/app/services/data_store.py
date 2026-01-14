from __future__ import annotations



import tempfile

import os

import logging

import math

from datetime import date, datetime

from pathlib import Path

from typing import Iterable, List

from uuid import uuid4



import numpy as np

import pandas as pd
import polars as pl

from fastapi import HTTPException, UploadFile, status



from api.app.compute.caches import PerFileCache

from api.app.compute.downsampling import downsample_timeseries
from api.app.compute.exposure import (
    build_per_file_stepwise,
    build_symbol_positions,
    build_symbol_stepwise,
    build_portfolio_positions,
    build_portfolio_stepwise,
    build_portfolio_daily_max,
    build_step_points_from_positions,
)

from api.app.compute.portfolio import ContributorSeries, PortfolioAggregator, PortfolioView

from api.app.constants import get_contract_spec

from api.app.data.loader import LoadedTrades

from api.app.ingest import IngestService, TradeFileMetadata

from api.app.dependencies import selection_hash

from api.app.services.cache import build_selection_key, get_cache_backend

from api.app.schemas import (
    CTAMonthlyRow,
    CTAResponse,
    FileMetadata,
    FileUploadResponse,
    HistogramBucket,
    HistogramResponse,
    MetricsBlock,
    MetricsResponse,
    Selection,
    SelectionMeta,
    SeriesPoint,
    SeriesContributor,
    SeriesResponse,
)



logger = logging.getLogger(__name__)





def _meta_to_schema(meta: TradeFileMetadata) -> FileMetadata:

    spec = get_contract_spec(meta.symbol)

    intervals: list[str] = []

    if meta.interval is not None:

        intervals = [str(meta.interval)]

    display_name = meta.original_filename or meta.filename

    return FileMetadata(

        file_id=meta.file_id,

        filename=display_name,

        symbols=[meta.symbol] if meta.symbol else [],

        intervals=intervals,

        strategies=[meta.strategy] if meta.strategy else [],

        date_min=meta.date_min.date() if isinstance(meta.date_min, datetime) else meta.date_min,

        date_max=meta.date_max.date() if isinstance(meta.date_max, datetime) else meta.date_max,

        mtm_available=meta.mtm_rows > 0,

        margin_per_contract=spec.initial_margin if spec else None,

        big_point_value=spec.big_point_value if spec else None,

    )





def _drawdown_from_equity(equity) -> list[tuple[datetime, float]]:

    values = []

    running_max = float("-inf")

    for ts, eq in zip(equity["timestamp"], equity["equity"]):

        eq_val = float(eq)

        running_max = max(running_max, eq_val)

        values.append((ts, eq_val - running_max))

    return values





def _max_drawdown(equity) -> float:

    dd = _drawdown_from_equity(equity)

    if not dd:

        return 0.0

    return float(min(val for _, val in dd))


LEGACY_METRIC_LABELS: dict[str, str] = {
    "total_net_profit": "Total Net Profit",
    "gross_profit": "Gross Profit",
    "gross_loss": "Gross Loss",
    "profit_factor": "Profit Factor",
    "total_trades": "Total Number of Trades",
    "percent_profitable": "Percent Profitable",
    "winning_trades": "Winning Trades",
    "losing_trades": "Losing Trades",
    "avg_trade_net_profit": "Avg. Trade Net Profit",
    "avg_winning_trade": "Avg. Winning Trade",
    "avg_losing_trade": "Avg. Losing Trade",
    "ratio_avg_win_loss": "Ratio Avg. Win:Avg. Loss",
    "largest_winning_trade": "Largest Winning Trade",
    "largest_losing_trade": "Largest Losing Trade",
    "max_consecutive_wins": "Max. Consecutive Winning Trades",
    "max_consecutive_losses": "Max. Consecutive Losing Trades",
    "max_contracts_held": "Max. Shares/Contracts Held",
    "total_contracts_held": "Total Shares/Contracts Held",
    "account_size_required": "Account Size Required",
    "total_commission": "Total Commission",
    "total_slippage": "Total Slippage",
    "close_to_close_drawdown_value": "Max. Drawdown (Trade Close to Trade Close) Value",
    "close_to_close_drawdown_date": "Max. Drawdown (Trade Close to Trade Close) Date",
    "max_trade_drawdown": "Max. Trade Drawdown",
    "intraday_peak_to_valley_drawdown_value": "Max. Drawdown (Intra-Day Peak to Valley) Value",
    "intraday_peak_to_valley_drawdown_date": "Max. Drawdown (Intra-Day Peak to Valley) Date",
}


def _timestamp_to_iso(value: datetime | None) -> str | None:

    if value is None:

        return None

    if isinstance(value, datetime):

        return value.isoformat()

    return str(value)


def _max_consecutive_runs(pnl: list[float]) -> tuple[int, int]:

    max_w = max_l = cur_w = cur_l = 0

    for val in pnl:

        if val > 0:

            cur_w += 1

            cur_l = 0

        elif val < 0:

            cur_l += 1

            cur_w = 0

        else:

            cur_w = 0

            cur_l = 0

        max_w = max(max_w, cur_w)

        max_l = max(max_l, cur_l)

    return max_w, max_l


def _compute_closed_trade_drawdown(trades: pl.DataFrame) -> tuple[float, datetime | None]:

    if trades.is_empty() or "exit_time" not in trades.columns:

        return 0.0, None

    closed = trades.drop_nulls("exit_time").sort("exit_time")

    if closed.is_empty():

        return 0.0, None

    running = 0.0

    peak = float("-inf")

    max_dd = 0.0

    max_dd_time: datetime | None = None

    for row in closed.iter_rows(named=True):

        running += float(row.get("net_profit") or 0.0)

        peak = running if peak == float("-inf") else max(peak, running)

        dd = peak - running

        if dd > max_dd:

            max_dd = dd

            max_dd_time = row.get("exit_time")

    return float(max_dd), max_dd_time


def _compute_intraday_p2v(trades: pl.DataFrame) -> tuple[float, datetime | None]:

    if trades.is_empty() or "exit_time" not in trades.columns:

        return 0.0, None

    ordered = trades.drop_nulls("exit_time").sort("exit_time")

    if ordered.is_empty():

        return 0.0, None

    base = 0.0

    high_water = 0.0

    max_dd = 0.0

    max_dd_time: datetime | None = None

    for row in ordered.iter_rows(named=True):

        runup = float(row.get("runup") or 0.0)

        drawdown_trade = float(row.get("drawdown_trade") or 0.0)

        pnl = float(row.get("net_profit") or 0.0)

        peak_candidate = base + max(0.0, runup, pnl, 0.0)

        high_water = max(high_water, base, peak_candidate)

        valley_candidate = base + min(0.0, drawdown_trade, pnl, 0.0)

        dd = high_water - valley_candidate

        if dd > max_dd:

            max_dd = dd

            max_dd_time = row.get("exit_time")

        base += pnl

    return float(max_dd), max_dd_time


def _netpos_stats_from_intervals(intervals: pl.DataFrame) -> tuple[int, int]:

    if intervals.is_empty() or "net_position" not in intervals.columns:

        return 0, 0

    positions = intervals["net_position"].to_list()

    prev = 0.0

    max_abs = 0.0

    delta_sum = 0.0

    for value in positions:

        current = float(value or 0.0)

        max_abs = max(max_abs, abs(current))

        delta = current - prev

        delta_sum += abs(delta)

        prev = current

    total_contracts = int(delta_sum / 2.0)

    return int(max_abs), total_contracts


def _netpos_stats_from_trades(trades: pl.DataFrame) -> tuple[int, int]:

    if trades.is_empty():

        return 0, 0

    events: list[tuple[datetime, float, int]] = []

    for idx, row in enumerate(trades.iter_rows(named=True)):

        entry_time = row.get("entry_time")

        exit_time = row.get("exit_time")

        contracts = float(row.get("contracts") or 0.0)

        if contracts == 0.0:

            continue

        direction_raw = str(row.get("direction") or "").lower()

        is_long = direction_raw in {"long", "buy", "buy to open", "buy to cover"}

        signed = contracts if is_long else -contracts

        if entry_time is not None:

            events.append((entry_time, signed, idx))

        if exit_time is not None:

            events.append((exit_time, -signed, idx))

    if not events:

        return 0, 0

    events.sort(key=lambda item: (item[0], item[2]))

    max_abs = 0.0

    total_delta = 0.0

    current = 0.0

    for _, delta, _ in events:

        current += float(delta)

        max_abs = max(max_abs, abs(current))

        total_delta += abs(delta)

    return int(max_abs), int(total_delta / 2.0)


def _build_metrics_dict(

    trades: pl.DataFrame,

    max_contracts: int,

    total_contracts: int,

    close_dd_value: float,

    close_dd_time: datetime | None,

    intraday_value: float,

    intraday_time: datetime | None,

) -> dict[str, float | int | str | None]:

    metrics: dict[str, float | int | str | None] = {key: 0.0 for key in LEGACY_METRIC_LABELS}

    pnl_values = trades["net_profit"].to_list() if "net_profit" in trades.columns else []

    total_trades = len(pnl_values)

    wins = len([val for val in pnl_values if val > 0])

    losses = len([val for val in pnl_values if val < 0])

    gross_profit = float(sum(val for val in pnl_values if val > 0))

    gross_loss = float(sum(val for val in pnl_values if val < 0))

    net_profit = gross_profit + gross_loss

    avg_trade = float(net_profit / total_trades) if total_trades else 0.0

    avg_win = float(gross_profit / wins) if wins else 0.0

    avg_loss = float(gross_loss / losses) if losses else 0.0

    ratio_aw_al = (avg_win / abs(avg_loss)) if losses and avg_loss else math.nan

    largest_win = float(max(pnl_values)) if pnl_values else 0.0

    largest_loss = float(min(pnl_values)) if pnl_values else 0.0

    max_wins, max_losses = _max_consecutive_runs(pnl_values)

    commission = float(trades["commission"].fill_null(0).sum()) if "commission" in trades.columns else 0.0

    slippage = float(trades["slippage"].fill_null(0).sum()) if "slippage" in trades.columns else 0.0

    drawdown_trade = float(trades["drawdown_trade"].min()) if "drawdown_trade" in trades.columns and not trades.is_empty() else 0.0

    percent_profitable = (wins / total_trades * 100.0) if total_trades else 0.0

    if gross_loss < 0:

        profit_factor = gross_profit / abs(gross_loss) if abs(gross_loss) > 0 else 0.0

    elif gross_profit > 0:

        profit_factor = float("inf")

    else:

        profit_factor = 0.0

    metrics.update(

        {

            "total_net_profit": net_profit,

            "gross_profit": gross_profit,

            "gross_loss": gross_loss,

            "profit_factor": profit_factor,

            "total_trades": float(total_trades),

            "percent_profitable": percent_profitable,

            "winning_trades": float(wins),

            "losing_trades": float(losses),

            "avg_trade_net_profit": avg_trade,

            "avg_winning_trade": avg_win,

            "avg_losing_trade": avg_loss,

            "ratio_avg_win_loss": ratio_aw_al,

            "largest_winning_trade": largest_win,

            "largest_losing_trade": largest_loss,

            "max_consecutive_wins": float(max_wins),

            "max_consecutive_losses": float(max_losses),

            "max_contracts_held": float(max_contracts),

            "total_contracts_held": float(total_contracts),

            "account_size_required": close_dd_value,

            "total_commission": commission,

            "total_slippage": slippage,

            "close_to_close_drawdown_value": -close_dd_value,

            "close_to_close_drawdown_date": _timestamp_to_iso(close_dd_time),

            "max_trade_drawdown": drawdown_trade,

            "intraday_peak_to_valley_drawdown_value": -intraday_value,

            "intraday_peak_to_valley_drawdown_date": _timestamp_to_iso(intraday_time),

        }

    )

    return metrics





class DataStore:

    """File-backed store that wraps ingest + basic compute."""



    def __init__(self, storage_root: str | Path | None = None) -> None:

        root = Path(storage_root or os.getenv("DATA_ROOT", "./data"))

        self.storage_root = root

        self.ingest_service = IngestService(storage_root=root)

        self.cache = PerFileCache(storage_dir=root / ".cache")

        self.aggregator = PortfolioAggregator(self.cache)

        self._portfolio_cache: dict[str, PortfolioView] = {}
        self._portfolio_exposure_cache: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {}

        self._backend = get_cache_backend()

        try:

            ttl_env = os.getenv("PORTFOLIO_CACHE_TTL_SECONDS")

            self._ttl_seconds = int(ttl_env) if ttl_env else 21_600  # default 6 hours

        except ValueError:

            self._ttl_seconds = 21_600

    def _clone_view(self, view: PortfolioView) -> PortfolioView:
        contributors: list[ContributorSeries] = []
        for contributor in view.contributors:
            bundle = contributor.bundle
            cloned_bundle = type(bundle)(
                equity=bundle.equity.clone(),
                daily_returns=bundle.daily_returns.clone(),
                daily_percent=bundle.daily_percent.clone(),
                spikes=bundle.spikes.clone(),
            )
            contributors.append(
                ContributorSeries(
                    file_id=contributor.file_id,
                    path=contributor.path,
                    bundle=cloned_bundle,
                    label=contributor.label,
                    symbol=contributor.symbol,
                    interval=contributor.interval,
                    strategy=contributor.strategy,
                )
            )
        return PortfolioView(
            equity=view.equity.clone(),
            daily_percent_portfolio=view.daily_percent_portfolio.clone(),
            daily_returns=view.daily_returns.clone(),
            contributors=contributors,
            spikes=view.spikes.clone() if view.spikes is not None else None,
        )



    # ---------------- ingest & metadata ----------------

    def ingest(self, files: Iterable[UploadFile]) -> FileUploadResponse:

        if not files:

            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")



        uploaded: List[FileMetadata] = []

        for upload in files:

            with tempfile.NamedTemporaryFile(suffix=f"_{upload.filename}", delete=False) as tmp:

                content = upload.file.read()

                tmp.write(content)

                tmp_path = Path(tmp.name)



            try:

                meta = self.ingest_service.ingest_file(tmp_path, original_filename=upload.filename)

            except ValueError as exc:

                logger.warning("Ingest failed for %s: %s", upload.filename, exc)

                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

            except Exception as exc:

                logger.error("Unexpected ingest failure for %s: %s", upload.filename, exc)

                raise HTTPException(

                    status_code=status.HTTP_400_BAD_REQUEST,

                    detail=f"Failed to ingest {upload.filename}: {exc}",

                ) from exc

            uploaded.append(_meta_to_schema(meta))



        job_id = str(uuid4())

        return FileUploadResponse(job_id=job_id, files=uploaded, message="ingest completed")



    def list_files(self) -> list[FileMetadata]:

        return [_meta_to_schema(m) for m in self.ingest_service.index.all()]



    def get_file(self, file_id: str) -> FileMetadata:

        meta = self.ingest_service.index.get(file_id)

        if not meta:

            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

        return _meta_to_schema(meta)



    def selection_meta(self) -> SelectionMeta:

        files = self.list_files()

        symbols = sorted({s for f in files for s in f.symbols})

        intervals = sorted({i for f in files for i in f.intervals})

        strategies = sorted({s for f in files for s in f.strategies})

        date_min = min((f.date_min for f in files if f.date_min), default=None)

        date_max = max((f.date_max for f in files if f.date_max), default=None)

        from api.app.constants import DEFAULT_ACCOUNT_EQUITY

        return SelectionMeta(
            symbols=symbols,
            intervals=intervals,
            strategies=strategies,
            date_min=date_min,
            date_max=date_max,
            files=files,
            account_equity=DEFAULT_ACCOUNT_EQUITY,
        )


    # ---------------- helpers ----------------

    def _paths_for_selection(self, selection: Selection) -> list[Path]:

        """Filter available files using selection filters and return Parquet paths."""



        filtered = self._metas_for_selection(selection)

        return [Path(m.trades_path) for m in filtered]



    def _metas_for_selection(self, selection: Selection) -> list[TradeFileMetadata]:

        """Filter available files using selection filters and return metadata."""



        all_meta = self.ingest_service.index.all()

        if selection.files:

            # explicit ids take precedence

            filtered = []

            for fid in selection.files:

                meta = self.ingest_service.index.get(fid)

                if not meta:

                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File id {fid} not found")

                filtered.append(meta)

        else:

            filtered = all_meta



        symbols = set(selection.symbols or [])

        intervals = set(selection.intervals or [])

        strategies = set(selection.strategies or [])

        start_date = selection.start_date

        end_date = selection.end_date



        symbol_dimension_active = any(m.symbol for m in filtered)

        interval_dimension_active = any(m.interval is not None for m in filtered)

        strategy_dimension_active = any(m.strategy for m in filtered)



        if symbol_dimension_active and not symbols:

            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No files match selection filters")

        if interval_dimension_active and not intervals:

            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No files match selection filters")

        if strategy_dimension_active and not strategies:

            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No files match selection filters")



        def keep(meta: TradeFileMetadata) -> bool:

            if symbol_dimension_active:

                if not meta.symbol or meta.symbol not in symbols:

                    return False

            if interval_dimension_active:

                interval_val = str(meta.interval) if meta.interval is not None else None

                if not interval_val or interval_val not in intervals:

                    return False

            if strategy_dimension_active:

                if not meta.strategy or meta.strategy not in strategies:

                    return False

            if start_date and meta.date_max and meta.date_max.date() < start_date:

                return False

            if end_date and meta.date_min and meta.date_min.date() > end_date:

                return False

            return True



        filtered = [m for m in filtered if keep(m)]

        if not filtered:

            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No files match selection filters")



        return filtered



    def _load_trades_map(self, metas: list[TradeFileMetadata]) -> dict[str, LoadedTrades]:

        loaded: dict[str, LoadedTrades] = {}

        for meta in metas:

            path = Path(meta.trades_path)

            loaded[meta.file_id] = self.cache.load_trades(path)

        return loaded



    def _to_points(self, rows, ts_key: str, value_key: str) -> list[SeriesPoint]:

        points: list[SeriesPoint] = []

        for row in rows:

            ts = row[ts_key]

            if isinstance(ts, datetime):

                ts_val = ts

            else:

                # Polars date/datetime come through as Python types already

                ts_val = ts

            points.append(SeriesPoint(timestamp=ts_val, value=float(row[value_key])))

        return points



    def _view_for_selection(

        self,

        selection: Selection,

        metas: list[TradeFileMetadata] | None = None,

        loaded_trades: dict[str, LoadedTrades] | None = None,

    ) -> PortfolioView:

        key = selection_hash(selection)

        cache_key = build_selection_key(key, selection.data_version, "portfolio")

        base_view = self._portfolio_cache.get(key)

        if base_view is None:
            metas = metas or self._metas_for_selection(selection)
            paths = [Path(m.trades_path) for m in metas]

            cached_view = self._load_portfolio_from_cache(cache_key)

            if cached_view:
                base_view = cached_view
            else:
                trades_map = loaded_trades or self._load_trades_map(metas)
                built = self.aggregator.aggregate(
                    paths,
                    metas=metas,
                    contract_multipliers=selection.contract_multipliers,
                    margin_overrides=selection.margin_overrides,
                    direction=selection.direction,
                    include_spikes=selection.spike_flag,
                    loaded_trades=trades_map,
                )
                base_view = PortfolioView(
                    equity=built.equity.clone(),
                    daily_percent_portfolio=built.daily_percent_portfolio.clone(),
                    daily_returns=built.daily_returns.clone(),
                    contributors=list(built.contributors),
                    spikes=built.spikes.clone() if built.spikes is not None else None,
                )
                exposure_netpos, exposure_margin = self._build_portfolio_daily_exposure_frames(
                    selection, metas, trades_map
                )

                self._portfolio_exposure_cache[cache_key] = (exposure_netpos, exposure_margin)
                self._persist_portfolio_to_cache(cache_key, base_view, exposure_netpos, exposure_margin)

            if not base_view.contributors:
                trades_map = loaded_trades or self._load_trades_map(metas)
                self._hydrate_contributors(base_view, metas, selection, paths, trades_map)

            self._portfolio_cache[key] = base_view

        view = self._clone_view(base_view)
        view = self._apply_account_equity(view, selection.account_equity)
        return self._filter_view_by_date(view, selection)


    def _persist_portfolio_to_cache(
        self,
        cache_key: str,
        view: PortfolioView,
        net_position: pl.DataFrame,
        margin: pl.DataFrame,
    ) -> None:

        try:

            import io

            import base64

            import json



            def to_bytes(frame: pl.DataFrame) -> bytes:

                buf = io.BytesIO()

                frame.write_parquet(buf)

                return buf.getvalue()



            payload = {
                "equity": to_bytes(view.equity),
                "daily_percent": to_bytes(view.daily_percent_portfolio),
                "daily_returns": to_bytes(view.daily_returns),
                "net_position": to_bytes(net_position),
                "margin": to_bytes(margin),
                "spikes": to_bytes(view.spikes) if view.spikes is not None else None,
            }

            # Parquet bytes are not JSON serializable; store as a dict of base64-encoded bytes.
            encoded = {
                "equity": base64.b64encode(payload["equity"]).decode(),
                "daily_percent": base64.b64encode(payload["daily_percent"]).decode(),
                "daily_returns": base64.b64encode(payload["daily_returns"]).decode(),
                "net_position": base64.b64encode(payload["net_position"]).decode(),
                "margin": base64.b64encode(payload["margin"]).decode(),
                "spikes": base64.b64encode(payload["spikes"]).decode() if payload["spikes"] is not None else None,
            }

            self._backend.set(cache_key, json.dumps(encoded).encode(), ttl_seconds=self._ttl_seconds)

        except Exception:

            logger.warning("Failed to persist portfolio cache for key %s", cache_key, exc_info=True)



    def _load_portfolio_from_cache(self, cache_key: str) -> PortfolioView | None:

        raw = self._backend.get(cache_key)

        if raw is None:

            return None

        try:

            import json

            import base64



            encoded = json.loads(raw.decode())



            def decode_frame(b64val):

                if b64val is None:

                    return None

                data = base64.b64decode(b64val)

                return pl.read_parquet(data)



            equity = decode_frame(encoded.get("equity"))
            daily_percent = decode_frame(encoded.get("daily_percent"))
            daily = decode_frame(encoded.get("daily_returns"))
            netpos = decode_frame(encoded.get("net_position"))
            margin = decode_frame(encoded.get("margin"))
            spikes = decode_frame(encoded.get("spikes"))
            if (
                equity is None
                or daily_percent is None
                or daily is None
                or netpos is None
                or margin is None
            ):
                return None
            self._portfolio_exposure_cache[cache_key] = (netpos, margin)
            return PortfolioView(
                equity=equity,
                daily_percent_portfolio=daily_percent,
                daily_returns=daily,
                contributors=[],
                spikes=spikes,
            )

        except Exception:

            logger.warning("Failed to load portfolio cache for key %s", cache_key, exc_info=True)

            return None



    def _filter_view_by_date(self, view: PortfolioView, selection: Selection) -> PortfolioView:

        start_dt = selection.start_date

        end_dt = selection.end_date

        if not start_dt and not end_dt:

            return view



        start_ts = datetime.combine(start_dt, datetime.min.time()) if start_dt else None

        end_ts = datetime.combine(end_dt, datetime.max.time()) if end_dt else None



        def filter_frame(frame: pl.DataFrame, column: str, is_date: bool = False) -> pl.DataFrame:

            out = frame

            start_bound = start_dt if is_date else start_ts

            end_bound = end_dt if is_date else end_ts

            if start_bound:

                out = out.filter(pl.col(column) >= start_bound)

            if end_bound:

                out = out.filter(pl.col(column) <= end_bound)

            return out



        view.equity = filter_frame(view.equity, "timestamp")
        view.daily_percent_portfolio = filter_frame(view.daily_percent_portfolio, "date", is_date=True)

        if view.spikes is not None:

            view.spikes = filter_frame(view.spikes, "timestamp")

        if start_dt or end_dt:

            view.daily_returns = filter_frame(view.daily_returns, "date", is_date=True)

        for contributor in view.contributors:

            contributor.bundle.equity = filter_frame(contributor.bundle.equity, "timestamp")

            contributor.bundle.spikes = filter_frame(contributor.bundle.spikes, "timestamp")

            if start_dt or end_dt:

                contributor.bundle.daily_returns = filter_frame(contributor.bundle.daily_returns, "date", is_date=True)
                contributor.bundle.daily_percent = filter_frame(contributor.bundle.daily_percent, "date", is_date=True)

        return view



    def _hydrate_contributors(
        self,
        view: PortfolioView,
        metas: list[TradeFileMetadata],
        selection: Selection,
        paths: list[Path] | None = None,

        loaded_trades: dict[str, LoadedTrades] | None = None,

    ) -> None:

        if view.contributors:

            return

        paths = paths or [Path(m.trades_path) for m in metas]

        contributors = self.aggregator.build_contributors(
            paths,
            metas=metas,
            contract_multipliers=selection.contract_multipliers,
            margin_overrides=selection.margin_overrides,
            direction=selection.direction,
            include_spikes=selection.spike_flag,
            loaded_trades=loaded_trades,
        )
        view.contributors = contributors

    def _apply_account_equity(self, view: PortfolioView, account_equity: float | None) -> PortfolioView:
        if account_equity is None:
            return view
        delta = account_equity - self.cache.starting_equity
        if delta == 0:
            return view
        view.equity = view.equity.with_columns((pl.col("equity") + delta).alias("equity"))
        for contributor in view.contributors:
            contributor.bundle.equity = contributor.bundle.equity.with_columns(
                (pl.col("equity") + delta).alias("equity")
            )
        return view


    def _downsample_frame_for_series(

        self,

        frame: pl.DataFrame,

        timestamp_col: str,

        value_col: str,

        downsample: bool,

    ) -> tuple[pl.DataFrame, int, int]:

        if frame.is_empty():

            return frame, 0, 0

        if not downsample:

            return frame, len(frame), len(frame)

        result = downsample_timeseries(frame, timestamp_col, value_col, target_points=2000)

        return result.downsampled, result.raw_count, result.downsampled_count



    def _frame_from_contributor(

        self,

        series_name: str,

        contributor: ContributorSeries,

    ) -> tuple[pl.DataFrame, str] | None:

        bundle = contributor.bundle

        if series_name == "equity":

            return bundle.equity, "equity"

        if series_name in {"equity_percent", "equity-percent"}:
            percent_frame = bundle.daily_percent.select(
                pl.col("date").alias("timestamp"),
                pl.col("cum_pct").alias("value"),
            )
            return percent_frame, "value"

        if series_name in {"drawdown", "intraday_drawdown"}:

            equity = bundle.equity

            dd_values = _drawdown_from_equity(equity)

            frame = pl.DataFrame(dd_values, schema=["timestamp", "drawdown"]) if dd_values else pl.DataFrame(

                {"timestamp": [], "drawdown": []}

            )

            return frame, "drawdown"

        return None



    def _build_per_file_lines(self, series_name: str, view: PortfolioView, downsample: bool) -> list[SeriesContributor]:

        lines: list[SeriesContributor] = []

        for contributor in view.contributors:

            frame_info = self._frame_from_contributor(series_name, contributor)

            if frame_info is None:

                continue

            frame, value_col = frame_info

            downsampled, _, _ = self._downsample_frame_for_series(frame, "timestamp", value_col, downsample)

            points = self._to_points(downsampled.iter_rows(named=True), "timestamp", value_col)

            label = contributor.label or contributor.file_id

            lines.append(

                SeriesContributor(

                    contributor_id=contributor.file_id,

                    label=label,

                    symbol=contributor.symbol,

                    interval=contributor.interval,

                    strategy=contributor.strategy,

                    points=points,

                )

            )

        return lines



    # ---------------- series & metrics ----------------

    def series(
        self,
        series_name: str,
        selection: Selection,
        downsample: bool,
        exposure_view: str = "portfolio_daily",
    ) -> SeriesResponse:

        if series_name in {"netpos", "margin"}:
            return self._exposure_series(series_name, selection, downsample, exposure_view)

        view = self._view_for_selection(selection)



        frame: pl.DataFrame | None = None

        value_col = "value"

        label = series_name

        if series_name == "equity":

            frame = view.equity

            value_col = "equity"

            label = "equity"



        elif series_name in {"equity_percent", "equity-percent"}:

            frame = view.daily_percent_portfolio.select(
                pl.col("date").alias("timestamp"),
                pl.col("cum_pct").alias("value"),
            )
            value_col = "value"
            label = "equity_percent"



        elif series_name in {"drawdown", "intraday_drawdown"}:

            dd_values = _drawdown_from_equity(view.equity)

            frame = pl.DataFrame(dd_values, schema=["timestamp", "drawdown"]) if dd_values else pl.DataFrame(

                {"timestamp": [], "drawdown": []}

            )

            value_col = "drawdown"

            label = "drawdown"



        else:

            raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Series not implemented")



        downsampled_frame, raw, sampled = self._downsample_frame_for_series(frame, "timestamp", value_col, downsample)

        data = self._to_points(downsampled_frame.iter_rows(named=True), "timestamp", value_col)

        per_file_lines = self._build_per_file_lines(series_name, view, downsample)

        return SeriesResponse(

            series=label,

            selection=selection,

            downsampled=downsample,

            raw_count=raw,

            downsampled_count=sampled,

            portfolio=data,

            per_file=per_file_lines,

        )

    def _exposure_series(
        self,
        series_name: str,
        selection: Selection,
        downsample: bool,
        exposure_view: str,
    ) -> SeriesResponse:
        if exposure_view not in {"portfolio_daily", "portfolio_step", "per_symbol", "per_file"}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid exposure_view")

        def filter_points(frame: pl.DataFrame, column: str) -> pl.DataFrame:
            start_dt = selection.start_date
            end_dt = selection.end_date
            if not start_dt and not end_dt:
                return frame
            if frame.is_empty():
                return frame
            start_bound = datetime.combine(start_dt, datetime.min.time()) if start_dt else None
            end_bound = datetime.combine(end_dt, datetime.max.time()) if end_dt else None
            out = frame
            if start_bound:
                out = out.filter(pl.col(column) >= start_bound)
            if end_bound:
                out = out.filter(pl.col(column) <= end_bound)
            return out

        if exposure_view == "portfolio_daily":
            cache_key = build_selection_key(selection_hash(selection), selection.data_version, "portfolio")
            cached = self._portfolio_exposure_cache.get(cache_key)
            if cached is None:
                _ = self._load_portfolio_from_cache(cache_key)
                cached = self._portfolio_exposure_cache.get(cache_key)
            if cached is not None:
                cached_frame = cached[0] if series_name == "netpos" else cached[1]
                if not cached_frame.is_empty():
                    value_col = "net_position" if series_name == "netpos" else "margin_used"
                    cached_frame = filter_points(cached_frame, "timestamp")
                    downsampled_frame, raw, sampled = self._downsample_frame_for_series(
                        cached_frame, "timestamp", value_col, downsample
                    )
                    portfolio_points = self._to_points(
                        downsampled_frame.iter_rows(named=True), "timestamp", value_col
                    )
                    label = "netpos" if series_name == "netpos" else "margin"
                    return SeriesResponse(
                        series=label,
                        selection=selection,
                        downsampled=downsample,
                        raw_count=raw,
                        downsampled_count=sampled,
                        portfolio=portfolio_points,
                        per_file=[],
                    )

        metas = self._metas_for_selection(selection)
        loaded_trades = self._load_trades_map(metas)
        intervals_frames: list[pl.DataFrame] = []
        for meta in metas:
            path = Path(meta.trades_path)
            cmult = selection.contract_multipliers.get(meta.file_id)
            marg = selection.margin_overrides.get(meta.file_id)
            loaded = loaded_trades.get(meta.file_id)
            intervals = self.cache.net_position_intervals(
                path,
                contract_multiplier=cmult,
                margin_override=marg,
                direction=selection.direction,
                loaded=loaded,
                file_id=meta.file_id,
            )
            if intervals.is_empty():
                continue
            label = meta.original_filename or meta.filename or path.stem
            intervals = intervals.with_columns(
                pl.lit(meta.file_id).alias("file_id"),
                pl.lit(label).alias("label"),
                pl.lit(meta.symbol.upper() if meta.symbol else None).alias("symbol"),
                pl.lit(str(meta.interval) if meta.interval is not None else None).alias("interval"),
                pl.lit(meta.strategy or None).alias("strategy"),
            )
            intervals_frames.append(intervals)

        if intervals_frames:
            intervals = pl.concat(intervals_frames)
        else:
            intervals = pl.DataFrame(
                {
                    "start": [],
                    "end": [],
                    "symbol": [],
                    "net_position": [],
                    "margin_used": [],
                    "file_id": [],
                    "label": [],
                    "interval": [],
                    "strategy": [],
                }
            )

        label = "netpos" if series_name == "netpos" else "margin"
        if intervals.is_empty():
            return SeriesResponse(
                series=label,
                selection=selection,
                downsampled=downsample,
                raw_count=0,
                downsampled_count=0,
                portfolio=[],
                per_file=[],
            )

        end_time = None
        if not intervals.is_empty():
            end_time = intervals["end"].max()

        value_col = "net_position" if series_name == "netpos" else "margin_used"

        per_file_points = build_per_file_stepwise(
            intervals,
            file_col="file_id",
            value_col=value_col,
        )

        symbol_positions = build_symbol_positions(intervals, file_col="file_id", value_col="net_position")
        per_symbol_points: pl.DataFrame

        if series_name == "margin":
            margin_per_contract = (
                intervals.filter(pl.col("net_position") != 0)
                .with_columns(
                    (pl.col("margin_used") / pl.col("net_position").abs()).alias("margin_per_contract")
                )
                .group_by("symbol")
                .agg(pl.col("margin_per_contract").max().alias("margin_per_contract"))
            )
            symbol_margin_positions = (
                symbol_positions.join(margin_per_contract, on="symbol", how="left")
                .with_columns(
                    (pl.col("net_position").abs() * pl.col("margin_per_contract").fill_null(0.0)).alias(
                        "margin_used"
                    )
                )
                .select("symbol", "timestamp", "margin_used")
            )
            end_times = intervals.group_by("symbol").agg(pl.col("end").max().alias("end"))
            per_symbol_points = build_step_points_from_positions(
                symbol_margin_positions,
                end_times=end_times,
                group_cols=["symbol"],
                timestamp_col="timestamp",
                value_col="margin_used",
                end_col="end",
            )
            portfolio_positions = build_portfolio_positions(symbol_margin_positions, value_col="margin_used")
            portfolio_step = build_step_points_from_positions(
                portfolio_positions,
                end_times=end_time,
                group_cols=None,
                timestamp_col="timestamp",
                value_col="margin_used",
                end_col="end",
            )
            portfolio_daily = build_portfolio_daily_max(
                symbol_margin_positions,
                end_time=end_time,
                value_col="margin_used",
            )
        else:
            per_symbol_points = build_symbol_stepwise(intervals, file_col="file_id", value_col="net_position")
            portfolio_step = build_portfolio_stepwise(symbol_positions, end_time=end_time, value_col="net_position")
            portfolio_daily = build_portfolio_daily_max(symbol_positions, end_time=end_time, value_col="net_position")

        portfolio_frame = portfolio_daily if exposure_view == "portfolio_daily" else portfolio_step
        if "date" in portfolio_frame.columns:
            portfolio_frame = portfolio_frame.with_columns(
                pl.col("date").cast(pl.Datetime).alias("timestamp")
            ).select("timestamp", value_col)

        per_file_points = filter_points(per_file_points, "timestamp")
        per_symbol_points = filter_points(per_symbol_points, "timestamp")
        portfolio_frame = filter_points(portfolio_frame, "timestamp")

        if exposure_view == "portfolio_step":
            per_file_lines: list[SeriesContributor] = []
            per_symbol_lines: list[SeriesContributor] = []
        elif exposure_view == "per_symbol":
            per_file_lines = []
            per_symbol_lines = self._exposure_symbol_lines(per_symbol_points, value_col, downsample)
        elif exposure_view == "per_file":
            per_file_lines = self._exposure_file_lines(per_file_points, intervals, value_col, downsample)
            per_symbol_lines = []
        else:
            per_file_lines = []
            per_symbol_lines = []

        downsampled_frame, raw, sampled = self._downsample_frame_for_series(
            portfolio_frame, "timestamp", value_col, downsample
        )
        portfolio_points = self._to_points(downsampled_frame.iter_rows(named=True), "timestamp", value_col)

        per_file = per_file_lines or per_symbol_lines
        return SeriesResponse(
            series=label,
            selection=selection,
            downsampled=downsample,
            raw_count=raw,
            downsampled_count=sampled,
            portfolio=portfolio_points,
            per_file=per_file,
        )

    def _exposure_file_lines(
        self,
        points: pl.DataFrame,
        intervals: pl.DataFrame,
        value_col: str,
        downsample: bool,
    ) -> list[SeriesContributor]:
        if points.is_empty():
            return []

        meta = intervals.select("file_id", "label", "symbol", "interval", "strategy").unique()
        groups = points.partition_by("file_id", as_dict=True)
        lines: list[SeriesContributor] = []
        for file_id, group in groups.items():
            file_key = file_id[0] if isinstance(file_id, tuple) and len(file_id) == 1 else file_id
            frame = group.select(pl.col("timestamp"), pl.col(value_col))
            downsampled, _, _ = self._downsample_frame_for_series(frame, "timestamp", value_col, downsample)
            meta_row = meta.filter(pl.col("file_id") == file_key)
            if meta_row.is_empty():
                label = file_key
                symbol = None
                interval = None
                strategy = None
            else:
                row = meta_row.to_dicts()[0]
                label = row.get("label") or file_key
                symbol = row.get("symbol")
                interval = row.get("interval")
                strategy = row.get("strategy")
            points_out = self._to_points(downsampled.iter_rows(named=True), "timestamp", value_col)
            lines.append(
                SeriesContributor(
                    contributor_id=str(file_key),
                    label=label,
                    symbol=symbol,
                    interval=interval,
                    strategy=strategy,
                    points=points_out,
                )
            )
        return lines

    def _exposure_symbol_lines(
        self,
        points: pl.DataFrame,
        value_col: str,
        downsample: bool,
    ) -> list[SeriesContributor]:
        if points.is_empty():
            return []

        groups = points.partition_by("symbol", as_dict=True)
        lines: list[SeriesContributor] = []
        for symbol, group in groups.items():
            symbol_key = symbol[0] if isinstance(symbol, tuple) and len(symbol) == 1 else symbol
            frame = group.select(pl.col("timestamp"), pl.col(value_col))
            downsampled, _, _ = self._downsample_frame_for_series(frame, "timestamp", value_col, downsample)
            points_out = self._to_points(downsampled.iter_rows(named=True), "timestamp", value_col)
            label = str(symbol_key) if symbol_key is not None else "unknown"
            lines.append(
                SeriesContributor(
                    contributor_id=label,
                    label=label,
                    symbol=symbol_key,
                    interval=None,
                    strategy=None,
                    points=points_out,
                )
            )
        return lines

    def _build_exposure_intervals(
        self,
        selection: Selection,
        metas: list[TradeFileMetadata],
        loaded_trades: dict[str, LoadedTrades],
    ) -> pl.DataFrame:
        intervals_frames: list[pl.DataFrame] = []
        for meta in metas:
            path = Path(meta.trades_path)
            cmult = selection.contract_multipliers.get(meta.file_id)
            marg = selection.margin_overrides.get(meta.file_id)
            loaded = loaded_trades.get(meta.file_id)
            intervals = self.cache.net_position_intervals(
                path,
                contract_multiplier=cmult,
                margin_override=marg,
                direction=selection.direction,
                loaded=loaded,
                file_id=meta.file_id,
            )
            if intervals.is_empty():
                continue
            intervals = intervals.with_columns(
                pl.lit(meta.file_id).alias("file_id"),
                pl.lit(meta.symbol.upper() if meta.symbol else None).alias("symbol"),
            )
            intervals_frames.append(intervals)
        if intervals_frames:
            return pl.concat(intervals_frames)
        return pl.DataFrame(
            {
                "start": [],
                "end": [],
                "symbol": [],
                "net_position": [],
                "margin_used": [],
                "file_id": [],
            }
        )

    def _build_portfolio_daily_exposure_frames(
        self,
        selection: Selection,
        metas: list[TradeFileMetadata],
        loaded_trades: dict[str, LoadedTrades],
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        intervals = self._build_exposure_intervals(selection, metas, loaded_trades)
        if intervals.is_empty():
            empty_netpos = pl.DataFrame({"timestamp": [], "net_position": []})
            empty_margin = pl.DataFrame({"timestamp": [], "margin_used": []})
            return empty_netpos, empty_margin

        end_time = intervals["end"].max()
        symbol_positions = build_symbol_positions(intervals, file_col="file_id", value_col="net_position")

        netpos_daily = build_portfolio_daily_max(
            symbol_positions, end_time=end_time, value_col="net_position"
        )
        netpos_frame = netpos_daily.with_columns(
            pl.col("date").cast(pl.Datetime).alias("timestamp")
        ).select("timestamp", "net_position")

        margin_per_contract = (
            intervals.filter(pl.col("net_position") != 0)
            .with_columns(
                (pl.col("margin_used") / pl.col("net_position").abs()).alias("margin_per_contract")
            )
            .group_by("symbol")
            .agg(pl.col("margin_per_contract").max().alias("margin_per_contract"))
        )
        symbol_margin_positions = (
            symbol_positions.join(margin_per_contract, on="symbol", how="left")
            .with_columns(
                (pl.col("net_position").abs() * pl.col("margin_per_contract").fill_null(0.0)).alias(
                    "margin_used"
                )
            )
            .select("symbol", "timestamp", "margin_used")
        )
        margin_daily = build_portfolio_daily_max(
            symbol_margin_positions, end_time=end_time, value_col="margin_used"
        )
        margin_frame = margin_daily.with_columns(
            pl.col("date").cast(pl.Datetime).alias("timestamp")
        ).select("timestamp", "margin_used")

        return netpos_frame, margin_frame



    def histogram(self, selection: Selection, bins: int = 16) -> HistogramResponse:

        view = self._view_for_selection(selection)

        returns = view.daily_returns



        if returns.is_empty():

            return HistogramResponse(label="return_distribution", selection=selection, buckets=[])



        values = returns["pnl"].to_numpy()

        max_abs = float(np.max(np.abs(values)))
        if max_abs == 0:
            edges = np.linspace(-1, 1, bins + 1)
        else:
            edges = np.linspace(-max_abs, max_abs, bins + 1)
        hist, _ = np.histogram(values, bins=edges)

        def _format_range(start: float, end: float) -> str:

            return f"${start:,.2f} to ${end:,.2f}"

        buckets: list[HistogramBucket] = []

        for count, start, end in zip(hist, edges[:-1], edges[1:]):

            label = _format_range(float(start), float(end))

            buckets.append(

                HistogramBucket(

                    bucket=label,

                    count=int(count),

                    start_value=float(start),

                    end_value=float(end),

                )

            )



        return HistogramResponse(label="return_distribution", selection=selection, buckets=buckets)


    def cta(self, selection: Selection) -> CTAResponse:
        view = self._view_for_selection(selection)
        daily = view.daily_returns

        if daily.is_empty():
            return CTAResponse(selection=selection)

        daily = (
            daily.sort("date")
            .with_columns(pl.col("date").dt.truncate("1mo").alias("month_start"))
            .with_columns(pl.col("pnl").cum_sum().over("month_start").alias("month_cum_pnl"))
        )

        monthly = (
            daily.group_by("month_start")
            .agg(
                total_pnl=pl.col("pnl").sum(),
                drawdown=pl.col("month_cum_pnl").min(),
                median_daily_pnl=pl.col("pnl").median(),
                mean_return=pl.col("daily_return").mean(),
                std_return=pl.col("daily_return").std(),
                count=pl.col("daily_return").count(),
            )
            .sort("month_start")
            .with_columns(
                pl.col("mean_return").fill_null(0.0),
                pl.col("std_return").fill_null(0.0),
            )
            .with_columns(
                pl.when(pl.col("drawdown") < 0)
                .then(pl.col("drawdown"))
                .otherwise(0.0)
                .alias("drawdown"),
                pl.when((pl.col("std_return") > 0) & (pl.col("count") > 1))
                .then(
                    (pl.col("mean_return") / pl.col("std_return"))
                    * pl.col("count").cast(pl.Float64).sqrt()
                )
                .otherwise(0.0)
                .alias("sharpe"),
                pl.col("month_start").dt.strftime("%B %Y").alias("label"),
            )
        )

        account_equity = float(selection.account_equity or 0.0)
        monthly_rows: list[CTAMonthlyRow] = []
        monthly_pnl_points: list[SeriesPoint] = []
        monthly_return_points: list[SeriesPoint] = []

        month_starts: list[date] = []
        monthly_pnls: list[float] = []
        monthly_returns: list[float] = []

        for row in monthly.iter_rows(named=True):
            month_start = row["month_start"]
            total_pnl = float(row["total_pnl"]) if row["total_pnl"] is not None else 0.0
            drawdown = float(row["drawdown"]) if row["drawdown"] is not None else 0.0
            median_daily_pnl = (
                float(row["median_daily_pnl"]) if row["median_daily_pnl"] is not None else 0.0
            )
            sharpe = float(row["sharpe"]) if row["sharpe"] is not None else 0.0
            label = row["label"] or month_start.strftime("%B %Y")

            monthly_rows.append(
                CTAMonthlyRow(
                    month_start=month_start,
                    label=label,
                    total_pnl=total_pnl,
                    drawdown=drawdown,
                    median_daily_pnl=median_daily_pnl,
                    sharpe=sharpe,
                )
            )

            timestamp = datetime.combine(month_start, datetime.min.time())
            monthly_pnl_points.append(SeriesPoint(timestamp=timestamp, value=total_pnl))
            monthly_return = total_pnl / account_equity if account_equity > 0 else 0.0
            monthly_return_points.append(SeriesPoint(timestamp=timestamp, value=monthly_return))

            month_starts.append(month_start)
            monthly_pnls.append(total_pnl)
            monthly_returns.append(monthly_return)

        rolling_pnl_points: list[SeriesPoint] = []
        rolling_return_points: list[SeriesPoint] = []
        window = 12
        for idx in range(len(month_starts)):
            if idx + 1 < window:
                continue
            start_idx = idx + 1 - window
            rolling_pnl = float(sum(monthly_pnls[start_idx : idx + 1]))
            rolling_return = float(sum(monthly_returns[start_idx : idx + 1]))
            timestamp = datetime.combine(month_starts[idx], datetime.min.time())
            rolling_pnl_points.append(SeriesPoint(timestamp=timestamp, value=rolling_pnl))
            rolling_return_points.append(SeriesPoint(timestamp=timestamp, value=rolling_return))

        return CTAResponse(
            selection=selection,
            monthly=monthly_rows,
            monthly_pnl=monthly_pnl_points,
            monthly_return=monthly_return_points,
            rolling_pnl=rolling_pnl_points,
            rolling_return=rolling_return_points,
        )


    def metrics(self, selection: Selection) -> MetricsResponse:
        metas = self._metas_for_selection(selection)
        loaded_trades = self._load_trades_map(metas)
        direction = selection.direction
        file_blocks: list[MetricsBlock] = []
        combined_trades: list[pl.DataFrame] = []

        for meta in metas:
            path = Path(meta.trades_path)
            cmult = selection.contract_multipliers.get(meta.file_id)
            marg = selection.margin_overrides.get(meta.file_id)
            loaded = loaded_trades.get(meta.file_id)
            if loaded is None:
                loaded = self.cache.load_trades(path)
                loaded_trades[meta.file_id] = loaded

            trades_df = self.cache._filter_by_direction(loaded.trades, direction)
            if cmult is not None:
                trades_df = self.cache._apply_contract_multiplier(trades_df, float(cmult))
            combined_trades.append(trades_df)

            intervals = self.cache.net_position_intervals(
                path,
                contract_multiplier=cmult,
                margin_override=marg,
                direction=direction,
                loaded=loaded,
                file_id=meta.file_id,
            )
            max_contracts, total_contracts = _netpos_stats_from_intervals(intervals)
            close_dd_value, close_dd_time = _compute_closed_trade_drawdown(trades_df)
            intraday_value, intraday_time = _compute_intraday_p2v(trades_df)
            metrics_map = _build_metrics_dict(
                trades_df,
                max_contracts=max_contracts,
                total_contracts=total_contracts,
                close_dd_value=close_dd_value,
                close_dd_time=close_dd_time,
                intraday_value=intraday_value,
                intraday_time=intraday_time,
            )
            label = meta.original_filename or meta.filename or path.stem
            file_blocks.append(
                MetricsBlock(
                    key=meta.file_id,
                    file_id=meta.file_id,
                    label=label,
                    metrics=metrics_map,
                )
            )

        if combined_trades:
            combined = pl.concat(combined_trades, how="diagonal")
        else:
            combined = pl.DataFrame()
        portfolio_max, portfolio_total = _netpos_stats_from_trades(combined)
        portfolio_close_dd, portfolio_close_time = _compute_closed_trade_drawdown(combined)
        portfolio_intraday_dd, portfolio_intraday_time = _compute_intraday_p2v(combined)
        portfolio_metrics = _build_metrics_dict(
            combined,
            max_contracts=portfolio_max,
            total_contracts=portfolio_total,
            close_dd_value=portfolio_close_dd,
            close_dd_time=portfolio_close_time,
            intraday_value=portfolio_intraday_dd,
            intraday_time=portfolio_intraday_time,
        )

        portfolio_block = MetricsBlock(key="portfolio", label="Portfolio", metrics=portfolio_metrics)
        return MetricsResponse(selection=selection, portfolio=portfolio_block, files=file_blocks)

    # ---------------- optimizer helpers ----------------

    def returns_frame(self, selection: Selection) -> tuple[pd.DataFrame, list[TradeFileMetadata]]:
        """
        Assemble a pandas DataFrame of per-file daily returns suitable for Riskfolio.

        Columns map to file ids; index is the trading date. MTM ingestion densities each
        series across its active range, so the resulting frame is already aligned on the
        common calendar without padding or synthetic fills.
        """
        metas = self._metas_for_selection(selection)
        if not metas:
            return pd.DataFrame(), []

        start_date = selection.start_date
        end_date = selection.end_date

        loaded_trades = self._load_trades_map(metas)
        series: list[pd.DataFrame] = []

        for meta in metas:
            path = Path(meta.trades_path)
            multiplier = selection.contract_multipliers.get(meta.file_id)
            margin_override = selection.margin_overrides.get(meta.file_id)
            loaded = loaded_trades.get(meta.file_id)

            daily = self.cache.daily_percent_returns(
                path,
                contract_multiplier=multiplier,
                margin_override=margin_override,
                direction=selection.direction,
                loaded=loaded,
                file_id=meta.file_id,
            )
            if daily.is_empty():
                continue

            frame = daily.select(["date", "daily_return"]).to_pandas()
            frame["date"] = pd.to_datetime(frame["date"])
            if start_date:
                frame = frame[frame["date"] >= pd.Timestamp(start_date)]
            if end_date:
                frame = frame[frame["date"] <= pd.Timestamp(end_date)]
            if frame.empty:
                continue

            frame = frame.set_index("date").sort_index()
            frame.rename(columns={"daily_return": meta.file_id}, inplace=True)
            series.append(frame)

        if not series:
            return pd.DataFrame(), metas

        combined = pd.concat(series, axis=1, join="outer").sort_index()
        return combined, metas

store = DataStore()
