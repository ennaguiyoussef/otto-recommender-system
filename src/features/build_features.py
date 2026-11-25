"""Session, item, and session-item interaction feature engineering (Stage 2)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

EVENTS: tuple[str, str, str] = ("clicks", "carts", "orders")


def _type_flags(frame: pl.LazyFrame) -> list[pl.Expr]:
    """Return boolean expressions for clicks (0), carts (1), and orders (2)."""
    numeric = frame.collect_schema()["type"].is_integer()
    return [
        (pl.col("type") == (code if numeric else event)).alias(event)
        for code, event in enumerate(EVENTS)
    ]


def _build_item_features(item_corpus: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate global item popularity and conversion rate features."""
    items = item_corpus.group_by("aid").agg(
        pl.len().alias("item_events"),
        pl.col("session").n_unique().alias("item_sessions"),
        *[
            flag.sum().alias(f"item_{event}")
            for event, flag in zip(EVENTS, _type_flags(item_corpus))
        ],
    )
    for numerator, denominator in (("carts", "clicks"), ("orders", "carts")):
        base = pl.col(f"item_{denominator}")
        items = items.with_columns(
            pl.when(base > 0)
            .then(pl.col(f"item_{numerator}") / base)
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias(f"item_{denominator}_to_{numerator}")
        )
    return items


def _build_feature_table(
    history_path: Path,
    candidates_path: Path,
    covis_path: Path,
    items: pl.LazyFrame,
    out_path: Path,
    labels_path: Path | None = None,
) -> None:
    """Build and stream a candidate feature matrix for validation or test sessions."""
    history = pl.scan_parquet(history_path)
    candidates = pl.scan_parquet(candidates_path)

    keys = ["session", "aid"]
    ts = pl.col("ts").cast(pl.Int64)
    hist_flags = _type_flags(history)

    sessions = history.group_by("session").agg(
        pl.len().alias("session_events"),
        pl.col("aid").n_unique().alias("session_unique_items"),
        ts.max().alias("session_end"),
        (ts.max() - ts.min()).alias("session_duration"),
    )

    interactions = (
        history.sort(["session", "ts"])
        .group_by(keys, maintain_order=True)
        .agg(
            pl.len().alias("pair_events"),
            ts.max().alias("pair_last_ts"),
            *[
                flag.sum().alias(f"pair_{event}")
                for event, flag in zip(EVENTS, hist_flags)
            ],
        )
    )

    cand_schema = candidates.collect_schema()
    if "covisitation_score" not in cand_schema:
        edges = pl.scan_parquet(covis_path)
        sources = interactions.select("session", pl.col("aid").alias("source_aid"))
        scores = (
            sources.join(edges, on="source_aid")
            .join(candidates.select(keys), on=keys, how="semi")
            .group_by(keys)
            .agg(pl.col("score").sum().cast(pl.Float32).alias("covisitation_score"))
        )
        candidates = candidates.join(scores, on=keys, how="left")

    if "cand_rank" not in candidates.collect_schema():
        candidates = (
            candidates.sort(["session", "covisitation_score"], descending=[False, True])
            .with_columns(
                pl.int_range(1, pl.len() + 1).over("session").cast(pl.UInt16).alias("cand_rank")
            )
        )

    features = (
        candidates.join(sessions, on="session", how="left")
        .join(interactions, on=keys, how="left")
        .join(items, on="aid", how="left")
    )

    if labels_path is not None:
        labels = pl.scan_parquet(labels_path)
        targets = labels.group_by(keys).agg(
            *[
                flag.any().cast(pl.UInt8).alias(f"label_{event}")
                for event, flag in zip(EVENTS, _type_flags(labels))
            ]
        )
        features = features.join(targets, on=keys, how="left")

    fill_exprs: list[pl.Expr] = [
        (pl.col("session_end") - pl.col("pair_last_ts")).alias("pair_recency"),
        pl.col("^item_.*$").fill_null(0),
        pl.col(
            "pair_events",
            "pair_clicks",
            "pair_carts",
            "pair_orders",
            "session_events",
            "session_unique_items",
            "session_duration",
            "covisitation_score",
        ).fill_null(0),
        (pl.col("pair_events").fill_null(0) > 0).cast(pl.UInt8).alias("in_session"),
    ]
    if labels_path is not None:
        fill_exprs.append(pl.col("^label_.*$").fill_null(0))

    features = features.with_columns(*fill_exprs).drop("session_end", "pair_last_ts")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.sink_parquet(out_path)
    logger.info("[features] Wrote %s", out_path)


def run_build_features(config: dict[str, Any], mode: str = "sample") -> None:
    """Construct user/session, item, and interaction features for candidate pairs.

    Writes `<features>/<mode>/features.parquet` for validation (with `label_*` columns)
    and `<features>/<mode>/test_features.parquet` for Kaggle test sessions when present.

    Args:
        config: Loaded pipeline configuration with resolved Drive paths.
        mode: Execution mode ('sample' or 'full').
    """
    candidates_dir = config["paths"]["candidates"]
    features_dir = config["paths"]["features"]

    logger.info(
        "[features] Building reranker feature tables (mode=%s) from '%s' -> '%s'",
        mode,
        candidates_dir,
        features_dir,
    )

    splits = Path(config["paths"]["splits"]) / mode
    retrieval = Path(candidates_dir) / mode
    output = Path(features_dir) / mode

    val_item_corpus = pl.concat(
        [
            pl.scan_parquet(splits / "train.parquet"),
            pl.scan_parquet(splits / "history.parquet"),
        ]
    )
    val_items = _build_item_features(val_item_corpus)

    _build_feature_table(
        history_path=splits / "history.parquet",
        candidates_path=retrieval / "candidates.parquet",
        covis_path=retrieval / "covisitation.parquet",
        items=val_items,
        out_path=output / "features.parquet",
        labels_path=splits / "labels.parquet",
    )

    test_history_path = splits / "test_history.parquet"
    test_cands_path = retrieval / "test_candidates.parquet"
    if test_history_path.exists() and test_cands_path.exists():
        test_item_corpus = pl.concat(
            [val_item_corpus, pl.scan_parquet(test_history_path)]
        )
        test_items = _build_item_features(test_item_corpus)
        _build_feature_table(
            history_path=test_history_path,
            candidates_path=test_cands_path,
            covis_path=retrieval / "covisitation.parquet",
            items=test_items,
            out_path=output / "test_features.parquet",
            labels_path=None,
        )
