from __future__ import annotations



import tempfile

import os

import logging

from datetime import datetime

from pathlib import Path

from typing import Iterable, List

from uuid import uuid4



import numpy as np

import polars as pl

from fastapi import HTTPException, UploadFile, status



from api.app.compute.caches import PerFileCache

from api.app.compute.downsampling import downsample_timeseries

from api.app.compute.portfolio import ContributorSeries, PortfolioAggregator, PortfolioView

from api.app.constants import get_contract_spec

from api.app.data.loader import LoadedTrades

from api.app.ingest import IngestService, TradeFileMetadata

from api.app.dependencies import selection_hash

from api.app.services.cache import build_selection_key, get_cache_backend

from api.app.schemas import (

    FileMetadata,

    FileUploadResponse,

    HistogramBucket,

    HistogramResponse,

    MetricsResponse,

    MetricsRow,

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





class DataStore:

    """File-backed store that wraps ingest + basic compute."""



    def __init__(self, storage_root: str | Path | None = None) -> None:

        root = Path(storage_root or os.getenv("DATA_ROOT", "./data"))

        self.storage_root = root

        self.ingest_service = IngestService(storage_root=root)

        self.cache = PerFileCache(storage_dir=root / ".cache")

        self.aggregator = PortfolioAggregator(self.cache)

        self._portfolio_cache: dict[str, PortfolioView] = {}

        self._backend = get_cache_backend()

        try:

            ttl_env = os.getenv("PORTFOLIO_CACHE_TTL_SECONDS")

            self._ttl_seconds = int(ttl_env) if ttl_env else 21_600  # default 6 hours

        except ValueError:

            self._ttl_seconds = 21_600



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



        metas = metas or self._metas_for_selection(selection)

        paths = [Path(m.trades_path) for m in metas]

        trades_map = loaded_trades or self._load_trades_map(metas)

        cached_view = self._load_portfolio_from_cache(cache_key)

        if cached_view:

            view = cached_view

        else:

            built = self.aggregator.aggregate(

                paths,

                metas=metas,

                contract_multipliers=selection.contract_multipliers,

                margin_overrides=selection.margin_overrides,

                direction=selection.direction,

                include_spikes=selection.spike_flag,

                loaded_trades=trades_map,

            )

            view = PortfolioView(
                equity=built.equity.clone(),
                percent_equity=built.percent_equity.clone(),
                daily_percent_portfolio=built.daily_percent_portfolio.clone(),
                daily_returns=built.daily_returns.clone(),
                net_position=built.net_position.clone(),
                margin=built.margin.clone(),
                contributors=list(built.contributors),
                spikes=built.spikes.clone() if built.spikes is not None else None,
            )

            self._portfolio_cache[key] = view

            self._persist_portfolio_to_cache(cache_key, view)



        self._hydrate_contributors(view, metas, selection, paths, trades_map)
        view = self._apply_account_equity(view, selection.account_equity)
        return self._filter_view_by_date(view, selection)


    def _persist_portfolio_to_cache(self, cache_key: str, view: PortfolioView) -> None:

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
                "percent_equity": to_bytes(view.percent_equity),
                "daily_percent": to_bytes(view.daily_percent_portfolio),
                "daily_returns": to_bytes(view.daily_returns),
                "net_position": to_bytes(view.net_position),
                "margin": to_bytes(view.margin),
                "spikes": to_bytes(view.spikes) if view.spikes is not None else None,
            }

            # Parquet bytes are not JSON serializable; store as a dict of base64-encoded bytes.
            encoded = {
                "equity": base64.b64encode(payload["equity"]).decode(),
                "percent_equity": base64.b64encode(payload["percent_equity"]).decode(),
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
            percent_equity = decode_frame(encoded.get("percent_equity"))
            daily_percent = decode_frame(encoded.get("daily_percent"))
            daily = decode_frame(encoded.get("daily_returns"))
            netpos = decode_frame(encoded.get("net_position"))
            margin = decode_frame(encoded.get("margin"))
            spikes = decode_frame(encoded.get("spikes"))
            if (
                equity is None
                or percent_equity is None
                or daily_percent is None
                or daily is None
                or netpos is None
                or margin is None
            ):
                return None
            return PortfolioView(
                equity=equity,
                percent_equity=percent_equity,
                daily_percent_portfolio=daily_percent,
                daily_returns=daily,
                net_position=netpos,
                margin=margin,
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
        view.percent_equity = filter_frame(view.percent_equity, "timestamp")
        view.daily_percent_portfolio = filter_frame(view.daily_percent_portfolio, "date", is_date=True)

        view.net_position = filter_frame(view.net_position, "timestamp")

        view.margin = filter_frame(view.margin, "timestamp")

        if view.spikes is not None:

            view.spikes = filter_frame(view.spikes, "timestamp")

        if start_dt or end_dt:

            view.daily_returns = filter_frame(view.daily_returns, "date", is_date=True)

        for contributor in view.contributors:

            contributor.bundle.equity = filter_frame(contributor.bundle.equity, "timestamp")
            contributor.bundle.percent_equity = filter_frame(contributor.bundle.percent_equity, "timestamp")

            contributor.bundle.net_position = filter_frame(contributor.bundle.net_position, "timestamp")

            contributor.bundle.margin = filter_frame(contributor.bundle.margin, "timestamp")

            contributor.bundle.spikes = filter_frame(contributor.bundle.spikes, "timestamp")

            if start_dt or end_dt:

                contributor.bundle.daily_returns = filter_frame(contributor.bundle.daily_returns, "date", is_date=True)

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

            return bundle.percent_equity, "percent_equity"

        if series_name in {"drawdown", "intraday_drawdown"}:

            equity = bundle.equity

            dd_values = _drawdown_from_equity(equity)

            frame = pl.DataFrame(dd_values, schema=["timestamp", "drawdown"]) if dd_values else pl.DataFrame(

                {"timestamp": [], "drawdown": []}

            )

            return frame, "drawdown"

        if series_name == "netpos":

            return bundle.net_position, "net_position"

        if series_name == "margin":

            margin_frame = bundle.margin.select(pl.col("timestamp"), pl.col("margin_used"))

            return margin_frame, "margin_used"

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

    def series(self, series_name: str, selection: Selection, downsample: bool) -> SeriesResponse:

        view = self._view_for_selection(selection)



        frame: pl.DataFrame | None = None

        value_col = "value"

        label = series_name

        if series_name == "equity":

            frame = view.equity

            value_col = "equity"

            label = "equity"



        elif series_name in {"equity_percent", "equity-percent"}:

            frame = view.percent_equity
            value_col = "percent_equity"
            label = "equity_percent"



        elif series_name in {"drawdown", "intraday_drawdown"}:

            dd_values = _drawdown_from_equity(view.equity)

            frame = pl.DataFrame(dd_values, schema=["timestamp", "drawdown"]) if dd_values else pl.DataFrame(

                {"timestamp": [], "drawdown": []}

            )

            value_col = "drawdown"

            label = "drawdown"



        elif series_name == "netpos":

            frame = view.net_position

            value_col = "net_position"

            label = "netpos"



        elif series_name == "margin":

            frame = view.margin

            value_col = "margin_used"

            label = "margin"



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



    def histogram(self, selection: Selection, bins: int = 20) -> HistogramResponse:

        view = self._view_for_selection(selection)

        returns = view.daily_returns



        if returns.is_empty():

            return HistogramResponse(label="return_distribution", selection=selection, buckets=[])



        values = returns["daily_return"].to_numpy()

        hist, edges = np.histogram(values, bins=bins)



        buckets: list[HistogramBucket] = []

        for count, start, end in zip(hist, edges[:-1], edges[1:]):

            label = f"{start * 100:.2f}% to {end * 100:.2f}%"

            buckets.append(HistogramBucket(bucket=label, count=int(count)))



        return HistogramResponse(label="return_distribution", selection=selection, buckets=buckets)



    def metrics(self, selection: Selection) -> MetricsResponse:
        metas = self._metas_for_selection(selection)
        paths = [Path(m.trades_path) for m in metas]
        loaded_trades = self._load_trades_map(metas)
        rows: list[MetricsRow] = []
        direction = selection.direction
        account_equity = selection.account_equity if selection.account_equity is not None else self.cache.starting_equity
        equity_delta = account_equity - self.cache.starting_equity

        def _filter_direction(df: pl.DataFrame) -> pl.DataFrame:
            if not direction or "direction" not in df.columns:
                return df
            dir_norm = direction.lower()

            dir_col = pl.col("direction").cast(pl.Utf8).str.to_lowercase()

            if dir_norm == "long":

                mask = dir_col.is_in(["long", "buy", "buy to open", "buy to cover"])

            elif dir_norm == "short":

                mask = dir_col.is_in(["short", "sell short", "sell", "sell to open", "sell to cover"])

            else:

                return df

            return df.filter(mask)



        # Per-file metrics

        for meta in metas:
            path = Path(meta.trades_path)
            cmult = selection.contract_multipliers.get(meta.file_id)
            marg = selection.margin_overrides.get(meta.file_id)
            loaded = loaded_trades.get(meta.file_id)
            if loaded is None:

                loaded = self.cache.load_trades(path)

                loaded_trades[meta.file_id] = loaded



            equity = self.cache.equity_curve(path, contract_multiplier=cmult, margin_override=marg, direction=direction, loaded=loaded, file_id=meta.file_id)
            if equity_delta != 0:
                equity = equity.with_columns((pl.col("equity") + equity_delta).alias("equity"))
            trades_df = _filter_direction(loaded.trades)
            daily = self.cache.daily_returns(path, contract_multiplier=cmult, margin_override=marg, direction=direction, loaded=loaded, file_id=meta.file_id)

            net_profit = float(equity["equity"][-1] - account_equity) if len(equity) else 0.0
            trades = len(trades_df)

            wins = int((trades_df["net_profit"] > 0).sum()) if trades else 0

            win_rate = (wins / trades * 100.0) if trades else 0.0

            expectancy = (net_profit / trades) if trades else 0.0

            max_dd = _max_drawdown(equity)

            avg_daily = float(daily["daily_return"].mean()) if len(daily) else 0.0



            def add(metric: str, value: float) -> None:

                rows.append(MetricsRow(file_id=path.stem, metric=metric, value=value, level="file"))



            add("net_profit", net_profit)

            add("trades", float(trades))

            add("max_drawdown", max_dd)

            add("win_rate", win_rate)

            add("expectancy", expectancy)

            add("avg_daily_return", avg_daily)



        # Portfolio metrics

        view = self._view_for_selection(selection, metas=metas, loaded_trades=loaded_trades)
        if len(view.equity):
            portfolio_net = float(view.equity["equity"][-1] - account_equity)
            port_dd = _max_drawdown(view.equity)

            port_daily = float(view.daily_returns["daily_return"].mean()) if len(view.daily_returns) else 0.0

            dir_trades = sum(len(_filter_direction(loaded_trades[p.stem].trades)) for p in paths)

            def addp(metric: str, value: float) -> None:

                rows.append(MetricsRow(file_id="portfolio", metric=metric, value=value, level="portfolio"))



            addp("net_profit", portfolio_net)

            addp("trades", float(dir_trades))

            addp("max_drawdown", port_dd)

            addp("avg_daily_return", port_daily)



        return MetricsResponse(selection=selection, rows=rows)

store = DataStore()

