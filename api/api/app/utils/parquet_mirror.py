from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

import polars as pl

logger = logging.getLogger(__name__)


class MirrorPair(NamedTuple):
    source: Path
    destination: Path


@dataclass
class MirrorStats:
    converted: int = 0
    skipped: int = 0
    failed: int = 0

    def __iadd__(self, other: "MirrorStats") -> "MirrorStats":
        self.converted += other.converted
        self.skipped += other.skipped
        self.failed += other.failed
        return self


def build_pairs(root: Path) -> Iterable[MirrorPair]:
    mapping = [
        (Path("parquet") / "mtm", Path("CSV tests") / "mtm"),
        (Path("parquet") / "trades", Path("CSV tests") / "trades"),
        (Path(".cache") / "per_file", Path("CSV tests") / "per_file"),
    ]
    for src_rel, dst_rel in mapping:
        yield MirrorPair(source=root / src_rel, destination=root / dst_rel)


def mirror_directory(pair: MirrorPair, *, force: bool = False) -> MirrorStats:
    src_dir, dst_dir = pair
    stats = MirrorStats()

    if not src_dir.exists():
        logger.debug("CSV mirror skip: source missing -> %s", src_dir)
        return stats

    parquet_files = sorted(src_dir.rglob("*.parquet"))
    if not parquet_files:
        logger.debug("CSV mirror: no parquet files under %s", src_dir)
        return stats

    for parquet_path in parquet_files:
        rel = parquet_path.relative_to(src_dir)
        csv_path = dst_dir / rel.with_suffix(".csv")
        try:
            if (
                not force
                and csv_path.exists()
                and csv_path.stat().st_mtime >= parquet_path.stat().st_mtime
            ):
                stats.skipped += 1
                continue

            csv_path.parent.mkdir(parents=True, exist_ok=True)
            frame = pl.read_parquet(parquet_path)
            frame.write_csv(csv_path)
            stats.converted += 1
            logger.debug("CSV mirror: %s -> %s", parquet_path, csv_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            stats.failed += 1
            logger.warning("CSV mirror failed for %s: %s", parquet_path, exc)

    return stats


def mirror_storage(
    storage_root: Path,
    *,
    force: bool = False,
) -> MirrorStats:
    stats = MirrorStats()
    for pair in build_pairs(storage_root):
        stats += mirror_directory(pair, force=force)
    return stats
