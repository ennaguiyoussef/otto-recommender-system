"""Session-level subsampling of raw OTTO Parquet datasets using Polars."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_make_sample(config: dict[str, Any], mode: str = "sample") -> None:
    """Create a lightweight deterministic session sample from raw Parquet files.

    Args:
        config: Loaded pipeline configuration with resolved Drive paths.
        mode: Execution mode ('sample' or 'full').
    """
    if mode == "full":
        logger.info(
            "[sample] Mode is 'full' (sample_frac=1.0); skipping subsample generation."
        )
        return

    import polars as pl

    raw_dir = Path(config["paths"]["raw_data"])
    sample_dir = Path(config["paths"]["sample_data"])
    sample_frac = config["sample_frac"]

    logger.info(
        "[sample] Preparing %.1f%% session sample from '%s' -> '%s'",
        sample_frac * 100,
        raw_dir,
        sample_dir,
    )
    if not 0 <= sample_frac <= 1:
        raise ValueError("sample_frac must be between 0 and 1")

    sources = sorted(raw_dir.rglob("*.parquet"))
    if not sources:
        raise FileNotFoundError(f"No Parquet files found in {raw_dir}")

    # A shared hash threshold keeps every occurrence of a session across files.
    seed = config["pipeline"]["seed"]
    if sample_frac == 1:
        keep_session = pl.lit(True)
    else:
        threshold = pl.lit(int(sample_frac * (1 << 64)), dtype=pl.UInt64)
        keep_session = pl.col("session").hash(seed=seed) < threshold

    for source in sources:
        destination = sample_dir / source.relative_to(raw_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pl.scan_parquet(source).filter(keep_session).sink_parquet(destination)
    logger.info("[sample] Wrote %d sampled Parquet files to %s", len(sources), sample_dir)
