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


def _discover_train_files(source_dir: Path) -> list[Path]:
    """Locate train Parquet file(s) inside source_dir."""
    all_parquets = sorted(source_dir.rglob("*.parquet"))
    if not all_parquets:
        raise FileNotFoundError(f"No Parquet files found in '{source_dir}'.")

    train_files = [
        p
        for p in all_parquets
        if "train" in p.name.lower() or "train" in p.parent.name.lower()
    ]
    if not train_files:
        test_files = _discover_test_files(source_dir)
        train_files = [p for p in all_parquets if p not in test_files]
    if not train_files:
        raise FileNotFoundError(
            f"Could not identify train Parquet file(s) inside '{source_dir}'."
        )
    return train_files


def _discover_test_files(source_dir: Path) -> list[Path]:
    """Locate test Parquet file(s) inside source_dir (returns empty list if absent)."""
    if not source_dir.exists():
        return []
    return [
        p
        for p in sorted(source_dir.rglob("*.parquet"))
        if "test" in p.name.lower() or "test" in p.parent.name.lower()
    ]


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


def _session_sample_expr(frac: float, seed: int) -> pl.Expr:
    """Deterministic 64-bit hash filter over `session`."""
    if frac >= 1.0:
        return pl.lit(True)
    threshold = pl.lit(int(frac * (1 << 64)), dtype=pl.UInt64)
    return pl.col("session").hash(seed=seed) < threshold


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

    Mirrors the exact Kaggle OTTO test setup:
      - All historical events (`ts < cutoff_ts`) PLUS the strictly observed prefixes
        (`_row_idx < _split_idx`) of validation-window sessions form the observed
        interaction corpus (`train.parquet`), while 100% of held-out suffixes
        (`_row_idx >= _split_idx`) are strictly withheld from the corpus.
      - Query validation sessions (`history.parquet`, `labels.parquet`) are sampled
        at `sample_frac` (5% in `sample` mode, up to `max_val_sessions` in `full`).
    """
    raw_dir = Path(config["paths"]["raw_data"])
    sample_dir = Path(config["paths"]["sample_data"])
    out_dir = Path(config["paths"]["splits"]) / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_cfg = config.get("modes", {}).get(mode, {})
    history_corpus_frac = float(mode_cfg.get("history_corpus_frac", 1.0))
    sample_frac = float(config["sample_frac"])
    val_days = int(config["pipeline"]["validation_days"])
    max_val_sessions = int(config["pipeline"].get("max_val_sessions", 300_000))
    seed = int(config["pipeline"]["seed"])

    use_raw_for_history = (
        mode == "full"
        or (history_corpus_frac >= 1.0 and any(raw_dir.rglob("*.parquet")))
    )
    train_source_dir = raw_dir if use_raw_for_history else sample_dir
    test_source_dir = (
        sample_dir
        if (mode == "sample" and any(sample_dir.rglob("*test*.parquet")))
        else train_source_dir
    )

    logger.info(
        "[split] Splitting sessions (mode=%s, val_days=%d, query_frac=%.2f, corpus_frac=%.2f) from '%s' -> '%s'",
        mode,
        val_days,
        sample_frac,
        history_corpus_frac,
        train_source_dir,
        out_dir,
    )

    train_files = _discover_train_files(train_source_dir)
    test_files = _discover_test_files(test_source_dir)
    events = _normalize_events(train_files)

    ts_stats = events.select(
        pl.col("ts").min().alias("min_ts"),
        pl.col("ts").max().alias("max_ts"),
    ).collect()
    min_ts = int(ts_stats["min_ts"][0])
    max_ts = int(ts_stats["max_ts"][0])

    ts_per_sec = 1000 if max_ts > 10_000_000_000 else 1
    val_window = val_days * 86_400 * ts_per_sec
    cutoff_ts = (
        max_ts - val_window
        if (max_ts - min_ts) > val_window
        else min_ts + int((max_ts - min_ts) * 0.75)
    )

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

    # Identify all sessions starting in the validation window with >= 2 events
    strict_val_sessions_df = (
        events.group_by("session")
        .agg(
            pl.col("ts").min().alias("_s_min"),
            pl.len().alias("_s_len"),
        )
        .filter((pl.col("_s_min") >= cutoff_ts) & (pl.col("_s_len") >= 2))
        .select("session")
        .collect()
    )
    if len(strict_val_sessions_df) == 0:
        strict_val_sessions_df = (
            events.filter(pl.col("ts") >= cutoff_ts)
            .group_by("session")
            .len()
            .filter(pl.col("len") >= 2)
            .select("session")
            .collect()
        )

    # Deterministically truncate ALL validation-window sessions into observed prefix vs held-out suffix
    all_val_truncated = (
        events.filter(pl.col("ts") >= cutoff_ts)
        .join(strict_val_sessions_df.lazy(), on="session", how="semi")
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

    # Select query validation sessions (5% in sample mode, capped in full mode)
    query_val_sessions = strict_val_sessions_df
    if use_raw_for_history and sample_frac < 1.0:
        query_val_sessions = query_val_sessions.filter(
            _session_sample_expr(sample_frac, seed)
        )
    if len(query_val_sessions) > max_val_sessions:
        query_val_sessions = query_val_sessions.sample(n=max_val_sessions, seed=seed)

    logger.info(
        "[split] Selected %d/%d validation query sessions (100%% suffixes held out).",
        len(query_val_sessions),
        len(strict_val_sessions_df),
    )

    # Non-query validation sessions' observed prefixes join `train.parquet` (0 label leakage, mirrors Kaggle test.parquet!)
    non_query_observed_prefixes = (
        all_val_truncated.join(query_val_sessions.lazy(), on="session", how="anti")
        .filter(pl.col("_row_idx") < pl.col("_split_idx"))
        .select("session", "aid", "ts", "type")
    )

    hist_corpus = events.filter(pl.col("ts") < cutoff_ts)
    if history_corpus_frac < 1.0 and use_raw_for_history:
        hist_corpus = hist_corpus.filter(_session_sample_expr(history_corpus_frac, seed))

    pl.concat([hist_corpus, non_query_observed_prefixes]).sink_parquet(train_path)

    # Query validation sessions -> history.parquet (observed prefix) & labels.parquet (held-out suffix)
    query_val_events = all_val_truncated.join(
        query_val_sessions.lazy(), on="session", how="semi"
    )
    query_val_events.filter(pl.col("_row_idx") < pl.col("_split_idx")).select(
        "session", "aid", "ts", "type"
    ).sink_parquet(history_path)

    suffix_events = query_val_events.filter(
        pl.col("_row_idx") >= pl.col("_split_idx")
    ).select("session", "aid", "ts", "type")
    _extract_ground_truth_labels(suffix_events).sink_parquet(labels_path)

    # Kaggle test events
    if test_files:
        test_out = out_dir / "test_history.parquet"
        test_lf = _normalize_events(test_files)
        if mode == "sample" and test_source_dir == raw_dir and sample_frac < 1.0:
            test_lf = test_lf.filter(_session_sample_expr(sample_frac, seed))
        test_lf.sort(["session", "ts"]).sink_parquet(test_out)
        logger.info("[split] Wrote standardized test history -> %s", test_out)
    else:
        logger.warning(
            "[split] No test Parquet found in '%s'; skipping test_history.parquet.",
            test_source_dir,
        )

    logger.info("[split] Completed leak-free splits in '%s'.", out_dir)
