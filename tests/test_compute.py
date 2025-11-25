from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "api"))

from api.app.compute.caches import PerFileCache  # type: ignore  # noqa: E402
from api.app.compute.downsampling import downsample_timeseries  # type: ignore  # noqa: E402
from api.app.compute.portfolio import PortfolioAggregator  # type: ignore  # noqa: E402
from api.app.data.loader import list_sample_trade_files  # type: ignore  # noqa: E402


def _temp_cache(tmp_path: Path) -> PerFileCache:
    return PerFileCache(storage_dir=tmp_path)


def test_per_file_caches_roundtrip(tmp_path: Path) -> None:
    files = list_sample_trade_files()
    cache = _temp_cache(tmp_path)

    bundle = cache.bundle(files[0])
    assert len(bundle.equity) > 0
    assert set(bundle.daily_returns.columns) == {"date", "pnl", "capital", "daily_return"}

    # ensure cache hit path works
    equity_again = cache.equity_curve(files[0])
    assert equity_again.equals(bundle.equity)


def test_portfolio_aggregation(tmp_path: Path) -> None:
    files = list_sample_trade_files()[:2]
    cache = _temp_cache(tmp_path)
    aggregator = PortfolioAggregator(cache)

    view = aggregator.aggregate(files)
    assert len(view.daily_returns) > 0
    assert len(view.net_position) > 0
    assert len(view.margin) == len(view.net_position)


def test_downsampling() -> None:
    frame = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(start=pl.datetime(2020, 1, 1), end=pl.datetime(2020, 1, 10), interval="1h", eager=True),
            "equity": list(range(217)),
        }
    )

    result = downsample_timeseries(frame, "timestamp", "equity", target_points=10)
    assert result.raw_count == len(frame)
    assert result.downsampled_count <= 10
