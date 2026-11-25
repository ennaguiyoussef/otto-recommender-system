"""Session, item, and session-item interaction feature engineering (Stage 2)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

EVENTS: tuple[str, str, str] = ("clicks", "carts", "orders")
EVENT_WEIGHTS: dict[int, float] = {0: 1.0, 1: 4.0, 2: 3.0}


def _type_flags(frame: pl.LazyFrame) -> list[pl.Expr]:
    """Return boolean expressions for clicks (0), carts (1), and orders (2)."""
    numeric = frame.collect_schema()["type"].is_integer()
    return [
        (pl.col("type") == (code if numeric else event)).alias(event)
        for code, event in enumerate(EVENTS)
    ]


def _type_code_expr(frame: pl.LazyFrame) -> pl.Expr:
    """Return integer code 0/1/2 expression for event type."""
    if frame.collect_schema()["type"].is_integer():
        return pl.col("type").cast(pl.Int8)
    return (
        pl.col("type")
        .cast(pl.Utf8)
        .replace_strict({"clicks": 0, "carts": 1, "orders": 2}, return_dtype=pl.Int8)
    )


def _build_item_features(item_corpus: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate global item popularity, conversion rates, and 7d/2d trend features."""
    max_ts = item_corpus.select(pl.col("ts").max()).collect().item()
    ts_per_sec = 1000 if (max_ts is not None and int(max_ts) > 10_000_000_000) else 1
    recent_7d = (int(max_ts) - 7 * 86_400 * ts_per_sec) if max_ts is not None else 0
    recent_2d = (int(max_ts) - 2 * 86_400 * ts_per_sec) if max_ts is not None else 0

    items = item_corpus.group_by("aid").agg(
        pl.len().cast(pl.Int32).alias("item_events"),
        pl.col("session").n_unique().cast(pl.Int32).alias("item_sessions"),
        *[
            flag.sum().cast(pl.Int32).alias(f"item_{event}")
            for event, flag in zip(EVENTS, _type_flags(item_corpus))
        ],
        (pl.col("ts") >= recent_7d).sum().cast(pl.Int32).alias("item_recent_7d_events"),
        (pl.col("ts") >= recent_2d).sum().cast(pl.Int32).alias("item_recent_2d_events"),
    )
    items = items.with_columns(
        pl.when(pl.col("item_clicks") > 0)
        .then(pl.col("item_carts").cast(pl.Float32) / pl.col("item_clicks").cast(pl.Float32))
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias("item_clicks_to_carts"),
        pl.when(pl.col("item_carts") > 0)
        .then(pl.col("item_orders").cast(pl.Float32) / pl.col("item_carts").cast(pl.Float32))
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias("item_carts_to_orders"),
        (
            pl.col("item_recent_7d_events").cast(pl.Float32)
            / (pl.col("item_events").cast(pl.Float32) + 1.0)
        )
        .cast(pl.Float32)
        .alias("item_recent_share"),
        (
            pl.col("item_recent_2d_events").cast(pl.Float32)
            / (pl.col("item_recent_7d_events").cast(pl.Float32) + 1.0)
        )
        .cast(pl.Float32)
        .alias("item_2d_spike"),
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
    """Build and stream a 40-feature candidate table for validation or test sessions."""
    history = pl.scan_parquet(history_path)
    candidates = pl.scan_parquet(candidates_path)

    keys = ["session", "aid"]
    ts = pl.col("ts").cast(pl.Int64)
    hist_flags = _type_flags(history)
    type_code = _type_code_expr(history)

    sessions = (
        history.group_by("session")
        .agg(
            pl.len().cast(pl.Int32).alias("session_events"),
            pl.col("aid").n_unique().cast(pl.Int32).alias("session_unique_items"),
            ts.max().alias("session_end"),
            (ts.max() - ts.min()).cast(pl.Int32).alias("session_duration"),
            *[
                flag.sum().cast(pl.Int16).alias(f"session_{event}")
                for event, flag in zip(EVENTS, hist_flags)
            ],
        )
        .with_columns(
            (
                pl.col("session_events").cast(pl.Float32)
                / pl.col("session_unique_items").clip(lower_bound=1).cast(pl.Float32)
            )
            .cast(pl.Float32)
            .alias("session_repeat_ratio")
        )
    )

    ordered_hist = (
        history.with_columns(type_code.alias("_tcode"))
        .sort(["session", "ts"])
        .with_columns(
            pl.len().over("session").alias("_slen"),
            pl.int_range(0, pl.len()).over("session").alias("_fpos"),
        )
        .with_columns(
            (pl.col("_slen") - 1 - pl.col("_fpos")).cast(pl.Int16).alias("_rpos"),
            (
                (
                    pl.lit(2.0, dtype=pl.Float32).pow(
                        0.1
                        + 0.9
                        * pl.col("_fpos").cast(pl.Float32)
                        / (pl.col("_slen") - 1).clip(lower_bound=1).cast(pl.Float32)
                    )
                    - 1.0
                )
                * pl.col("_tcode").replace_strict(
                    EVENT_WEIGHTS, default=1.0, return_dtype=pl.Float32
                )
            )
            .cast(pl.Float32)
            .alias("_log_rec"),
        )
    )

    interactions = ordered_hist.group_by(keys).agg(
        pl.len().cast(pl.Int16).alias("pair_events"),
        ts.max().alias("pair_last_ts"),
        (pl.col("_tcode") == 0).sum().cast(pl.Int16).alias("pair_clicks"),
        (pl.col("_tcode") == 1).sum().cast(pl.Int16).alias("pair_carts"),
        (pl.col("_tcode") == 2).sum().cast(pl.Int16).alias("pair_orders"),
        pl.col("_tcode").last().cast(pl.Int8).alias("pair_last_type"),
        pl.col("_rpos").min().cast(pl.Int16).alias("pair_reverse_pos"),
        pl.col("_log_rec").sum().cast(pl.Float32).alias("pair_log_recency"),
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

    cand_schema = candidates.collect_schema()
    missing_cand_cols: list[pl.Expr] = []
    for col_name in (
        "score_clicks",
        "score_carts_orders",
        "score_buy2buy",
        "last_aid_covis",
        "raw_covis_sum",
    ):
        if col_name not in cand_schema:
            missing_cand_cols.append(pl.lit(0.0, dtype=pl.Float32).alias(col_name))
    if "covis_support" not in cand_schema:
        missing_cand_cols.append(pl.lit(0, dtype=pl.UInt16).alias("covis_support"))
    if missing_cand_cols:
        candidates = candidates.with_columns(*missing_cand_cols)

    if "cand_rank" not in candidates.collect_schema():
        candidates = candidates.sort(
            ["session", "covisitation_score"], descending=[False, True]
        ).with_columns(
            pl.int_range(1, pl.len() + 1).over("session").cast(pl.UInt16).alias("cand_rank")
        )

    candidates = candidates.with_columns(
        (
            pl.col("covisitation_score").fill_null(0.0)
            / (pl.col("covisitation_score").max().over("session").fill_null(0.0) + 1e-5)
        )
        .cast(pl.Float32)
        .alias("score_rel_max"),
        (
            pl.col("score_clicks").fill_null(0.0)
            / (pl.col("score_clicks").max().over("session").fill_null(0.0) + 1e-5)
        )
        .cast(pl.Float32)
        .alias("clicks_rel_max"),
        (
            pl.col("score_carts_orders").fill_null(0.0)
            / (pl.col("score_carts_orders").max().over("session").fill_null(0.0) + 1e-5)
        )
        .cast(pl.Float32)
        .alias("carts_rel_max"),
        (
            pl.col("score_buy2buy").fill_null(0.0)
            / (pl.col("score_buy2buy").max().over("session").fill_null(0.0) + 1e-5)
        )
        .cast(pl.Float32)
        .alias("b2b_rel_max"),
        (
            pl.col("last_aid_covis").fill_null(0.0)
            / (pl.col("last_aid_covis").max().over("session").fill_null(0.0) + 1e-5)
        )
        .cast(pl.Float32)
        .alias("last_aid_rel_max"),
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
        pl.when(pl.col("pair_last_ts").is_not_null())
        .then((pl.col("session_end") - pl.col("pair_last_ts")).cast(pl.Int32))
        .otherwise(pl.lit(-1, dtype=pl.Int32))
        .alias("pair_recency"),
        pl.col("pair_last_type").fill_null(-1).cast(pl.Int8),
        pl.col("pair_reverse_pos").fill_null(999).cast(pl.Int16),
        pl.col("pair_log_recency").fill_null(0.0).cast(pl.Float32),
        pl.col("^item_.*$").fill_null(0),
        pl.col(
            "pair_events",
            "pair_clicks",
            "pair_carts",
            "pair_orders",
            "session_events",
            "session_unique_items",
            "session_duration",
            "session_clicks",
            "session_carts",
            "session_orders",
            "session_repeat_ratio",
            "covisitation_score",
            "score_clicks",
            "score_carts_orders",
            "score_buy2buy",
            "last_aid_covis",
            "raw_covis_sum",
            "covis_support",
        ).fill_null(0),
        (pl.col("pair_events").fill_null(0) > 0).cast(pl.UInt8).alias("in_session"),
        (
            pl.col("pair_events").fill_null(0).cast(pl.Float32)
            / pl.col("session_events").fill_null(1).clip(lower_bound=1).cast(pl.Float32)
        )
        .cast(pl.Float32)
        .alias("pair_event_share"),
    ]
    if labels_path is not None:
        fill_exprs.append(pl.col("^label_.*$").fill_null(0))

    features = features.with_columns(*fill_exprs).drop("session_end", "pair_last_ts")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.sink_parquet(out_path)
    logger.info("[features] Wrote %s", out_path)


def run_build_features(config: dict[str, Any], mode: str = "sample") -> None:
    """Construct user/session, item, and interaction features for candidate pairs."""
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
