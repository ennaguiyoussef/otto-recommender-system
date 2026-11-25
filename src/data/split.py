"""Temporal train/validation split and session truncation for local validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

TYPE_TO_CODE: dict[str, int] = {
    "clicks": 0,
    "carts": 1,
    "orders": 2,
}


def _discover_split_files(source_dir: Path) -> tuple[list[Path], list[Path]]:
    """Locate train and test Parquet files inside source_dir."""
    all_parquets = sorted(source_dir.rglob("*.parquet"))
    if not all_parquets:
        raise FileNotFoundError(f"No Parquet files found in '{source_dir}'.")

    train_files = [
        p for p in all_parquets if "train" in p.name.lower() or "train" in p.parent.name.lower()
    ]
    test_files = [
        p for p in all_parquets if "test" in p.name.lower() or "test" in p.parent.name.lower()
    ]

    if not train_files:
        # Fallback if a single unnamed parquet was supplied
        train_files = [p for p in all_parquets if p not in test_files]
    if not train_files:
        raise FileNotFoundError(
            f"Could not identify train Parquet file(s) inside '{source_dir}'."
        )
    return train_files, test_files


def _normalize_events(files: list[Path]) -> pl.LazyFrame:
    """Scan Parquet files and standardize OTTO columns and compact dtypes."""
    lf = pl.scan_parquet(files)
    schema = lf.collect_schema()

    if schema["type"].is_integer():
        type_expr = pl.col("type").cast(pl.UInt8)
    else:
        type_expr = (
            pl.col("type")
            .cast(pl.Utf8)
            .replace_strict(TYPE_TO_CODE, return_dtype=pl.UInt8)
        )

    return lf.select(
        pl.col("session").cast(pl.Int32).alias("session"),
        pl.col("aid").cast(pl.Int32).alias("aid"),
        pl.col("ts").cast(pl.Int64).alias("ts"),
        type_expr.alias("type"),
    )


def _extract_ground_truth_labels(suffix_events: pl.LazyFrame) -> pl.LazyFrame:
    """Extract official OTTO ground-truth targets from held-out session suffixes.

    Official OTTO rules:
      - clicks (0): single next clicked aid after truncation.
      - carts  (1): all unique carted aids after truncation.
      - orders (2): all unique ordered aids after truncation.
    """
    ordered_suffix = suffix_events.sort(["session", "ts"])

    next_clicks = (
        ordered_suffix.filter(pl.col("type") == 0)
        .group_by("session")
        .agg(pl.col("aid").first().alias("aid"))
        .with_columns(pl.lit(0, dtype=pl.UInt8).alias("type"))
        .select("session", "aid", "type")
    )

    carts_and_orders = (
        ordered_suffix.filter(pl.col("type").is_in([1, 2]))
        .select("session", "aid", "type")
        .unique()
    )

    return pl.concat([next_clicks, carts_and_orders])


def run_split(config: dict[str, Any], mode: str = "sample") -> None:
    """Generate leak-free temporal train/validation splits and ground-truth labels.

    Outputs written to `<splits>/<mode>/`:
      - `train.parquet`: Historical events strictly before the validation cutoff.
      - `history.parquet`: Observed prefixes of validation sessions.
      - `labels.parquet`: Held-out ground-truth targets `(session, aid, type)`.
      - `test_history.parquet`: Standardized Kaggle test events (when test Parquet exists).

    Args:
        config: Loaded pipeline configuration with resolved Drive paths.
        mode: Execution mode ('sample' or 'full').
    """
    source_dir = Path(
        config["paths"]["sample_data"]
        if mode == "sample"
        else config["paths"]["raw_data"]
    )
    out_dir = Path(config["paths"]["splits"]) / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    val_days = int(config["pipeline"]["validation_days"])
    seed = int(config["pipeline"]["seed"])

    logger.info(
        "[split] Splitting sessions (%d-day validation window) from '%s' -> '%s'",
        val_days,
        source_dir,
        out_dir,
    )

    train_files, test_files = _discover_split_files(source_dir)
    events = _normalize_events(train_files)

    ts_stats = events.select(
        pl.col("ts").min().alias("min_ts"),
        pl.col("ts").max().alias("max_ts"),
    ).collect()
    min_ts = int(ts_stats["min_ts"][0])
    max_ts = int(ts_stats["max_ts"][0])

    # Auto-detect milliseconds vs seconds (radek1 dataset stores seconds as Int32)
    ts_per_sec = 1000 if max_ts > 10_000_000_000 else 1
    val_window = val_days * 86_400 * ts_per_sec

    if (max_ts - min_ts) > val_window:
        cutoff_ts = max_ts - val_window
    else:
        # Fallback for short synthetic/test spans: split on upper 25% of timeline
        cutoff_ts = min_ts + int((max_ts - min_ts) * 0.75)

    logger.info(
        "[split] Timeline [%d .. %d] (scale=%ds) -> cutoff_ts=%d",
        min_ts,
        max_ts,
        ts_per_sec,
        cutoff_ts,
    )

    train_path = out_dir / "train.parquet"
    history_path = out_dir / "history.parquet"
    labels_path = out_dir / "labels.parquet"

    # 1. Historical training events before the validation window
    events.filter(pl.col("ts") < cutoff_ts).sink_parquet(train_path)

    # 2. Validation sessions: sessions active in [cutoff_ts, max_ts] with >= 2 events
    val_events = (
        events.filter(pl.col("ts") >= cutoff_ts)
        .sort(["session", "ts"])
        .with_columns(
            pl.len().over("session").alias("_sess_len"),
            pl.int_range(0, pl.len()).over("session").alias("_row_idx"),
        )
        .filter(pl.col("_sess_len") >= 2)
        .with_columns(
            (
                1
                + (
                    pl.col("session").hash(seed=seed)
                    % (pl.col("_sess_len") - 1).cast(pl.UInt64)
                ).cast(pl.Int64)
            ).alias("_split_idx")
        )
    )

    val_events.filter(pl.col("_row_idx") < pl.col("_split_idx")).select(
        "session", "aid", "ts", "type"
    ).sink_parquet(history_path)

    suffix_events = val_events.filter(pl.col("_row_idx") >= pl.col("_split_idx")).select(
        "session", "aid", "ts", "type"
    )
    _extract_ground_truth_labels(suffix_events).sink_parquet(labels_path)

    # 3. Kaggle test events (if test.parquet is present)
    if test_files:
        test_out = out_dir / "test_history.parquet"
        _normalize_events(test_files).sort(["session", "ts"]).sink_parquet(test_out)
        logger.info("[split] Wrote standardized test history -> %s", test_out)
    else:
        logger.warning("[split] No test Parquet found in '%s'; skipping test_history.parquet.", source_dir)

    logger.info(
        "[split] Completed splits in '%s' (train, history, labels).",
        out_dir,
    )
