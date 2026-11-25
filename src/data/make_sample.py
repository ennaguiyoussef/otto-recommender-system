"""Session-level subsampling of raw OTTO Parquet datasets using Polars."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.utils.checkpoint import atomic_temp_path, commit_atomic, is_valid_checkpoint

logger = logging.getLogger(__name__)


def run_make_sample(config: dict[str, Any], mode: str = "sample") -> None:
    """Create a lightweight deterministic session sample from raw Parquet files."""
    if mode == "full":
        logger.info(
            "[sample] Mode is 'full' (sample_frac=1.0); skipping subsample generation."
        )
        return

    import polars as pl

    raw_dir = Path(config["paths"]["raw_data"])
    sample_dir = Path(config["paths"]["sample_data"])
    sample_frac = config["sample_frac"]
    force = bool(config.get("force_recompute", False))
    history_corpus_frac = float(
        config.get("modes", {}).get(mode, {}).get("history_corpus_frac", 1.0)
    )

    if not 0 <= sample_frac <= 1:
        raise ValueError("sample_frac must be between 0 and 1")

    sources = sorted(raw_dir.rglob("*.parquet"))
    if not sources:
        raise FileNotFoundError(f"No Parquet files found in {raw_dir}")

    if history_corpus_frac >= 1.0:
        test_sources = [
            p
            for p in sources
            if "test" in p.name.lower() or "test" in p.parent.name.lower()
        ]
        if test_sources:
            sources = test_sources

    destinations = [sample_dir / s.relative_to(raw_dir) for s in sources]
    if all(is_valid_checkpoint(dst, force=force) for dst in destinations):
        logger.info(
            "[checkpoint] Sampled Parquet file(s) already exist in '%s'; skipping.",
            sample_dir,
        )
        return

    logger.info(
        "[sample] Preparing %.1f%% session sample (history_corpus_frac=%.2f) from '%s' -> '%s'",
        sample_frac * 100,
        history_corpus_frac,
        raw_dir,
        sample_dir,
    )

    seed = config["pipeline"]["seed"]
    if sample_frac == 1:
        keep_session = pl.lit(True)
    else:
        threshold = pl.lit(int(sample_frac * (1 << 64)), dtype=pl.UInt64)
        keep_session = pl.col("session").hash(seed=seed) < threshold

    for source, destination in zip(sources, destinations):
        if is_valid_checkpoint(destination, force=force):
            continue
        tmp_dst = atomic_temp_path(destination)
        pl.scan_parquet(source).filter(keep_session).sink_parquet(tmp_dst)
        commit_atomic(tmp_dst, destination)

    logger.info("[sample] Ready (%d file(s) in %s).", len(sources), sample_dir)
