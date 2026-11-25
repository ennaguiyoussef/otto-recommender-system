"""Session, item, and interaction feature engineering with sharded Drive checkpoints."""

from __future__ import annotations

import gc
import logging
import shutil
from pathlib import Path
from typing import Any

import polars as pl

from src.utils.checkpoint import atomic_temp_path, commit_atomic, is_valid_checkpoint

logger = logging.getLogger(__name__)

EVENTS: tuple[str, str, str] = ("clicks", "carts", "orders")
EVENT_WEIGHTS: dict[int, float] = {0: 1.0, 1: 4.0, 2: 3.0}
FULL_TEST_SHARDS = 16


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


def _ensure_item_features_cached(
    item_corpus: pl.LazyFrame, cache_path: Path, force: bool = False
) -> pl.LazyFrame:
    """Materialize item feature lookup table on Drive so shard resumes are instant."""
    if is_valid_checkpoint(cache_path, force=force):
        return pl.scan_parquet(cache_path)

    max_ts = item_corpus.select(pl.col("ts").max()).collect().item()
    ts_per_sec = 1000 if (max_ts is not None and int(max_ts) > 10_000_000_000) else 1
    recent_7d = (int(max_ts) - 7 * 86_400 * ts_per_sec) if max_ts is not None else 0
    recent_2d = (int(max_ts) - 2 * 86_400 * ts_per_sec) if max_ts is not None else 0

    items = (
        item_corpus.group_by("aid")
        .agg(
            pl.len().cast(pl.Int32).alias("item_events"),
            pl.col("session").n_unique().cast(pl.Int32).alias("item_sessions"),
            *[
                flag.sum().cast(pl.Int32).alias(f"item_{event}")
                for event, flag in zip(EVENTS, _type_flags(item_corpus))
            ],
            (pl.col("ts") >= recent_7d).sum().cast(pl.Int32).alias("item_recent_7d_events"),
            (pl.col("ts") >= recent_2d).sum().cast(pl.Int32).alias("item_recent_2d_events"),
        )
        .with_columns(
            pl.when(pl.col("item_clicks") > 0)
            .then(
                pl.col("item_carts").cast(pl.Float32)
                / pl.col("item_clicks").cast(pl.Float32)
            )
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("item_clicks_to_carts"),
            pl.when(pl.col("item_carts") > 0)
            .then(
                pl.col("item_orders").cast(pl.Float32)
                / pl.col("item_carts").cast(pl.Float32)
            )
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
    )
    tmp_it = atomic_temp_path(cache_path)
    items.sink_parquet(tmp_it)
    commit_atomic(tmp_it, cache_path)
    return pl.scan_parquet(cache_path)


def _build_feature_lazy(
    history: pl.LazyFrame,
    candidates: pl.LazyFrame,
    covis_path: Path,
    items: pl.LazyFrame,
    labels_path: Path | None = None,
) -> pl.LazyFrame:
    """Construct the 41-feature LazyFrame for a given (history, candidates) slice."""
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

    return features.with_columns(*fill_exprs).drop("session_end", "pair_last_ts")


def run_build_features(config: dict[str, Any], mode: str = "sample") -> None:
    """Construct user/session, item, and interaction features with Drive checkpointing."""
    candidates_dir = config["paths"]["candidates"]
    features_dir = config["paths"]["features"]
    raw_dir = Path(config["paths"]["raw_data"])
    force = bool(config.get("force_recompute", False))

    splits = Path(config["paths"]["splits"]) / mode
    retrieval = Path(candidates_dir) / mode
    output = Path(features_dir) / mode
    output.mkdir(parents=True, exist_ok=True)

    val_out = output / "features.parquet"
    val_items_cache = output / "_val_item_features.parquet"

    if is_valid_checkpoint(val_out, force=force):
        logger.info("[checkpoint] Validation features already exist -> %s", val_out)
    else:
        logger.info("[features] Building validation features -> '%s'", val_out)
        val_item_corpus = pl.concat(
            [
                pl.scan_parquet(splits / "train.parquet"),
                pl.scan_parquet(splits / "history.parquet"),
            ]
        )
        val_items = _ensure_item_features_cached(
            val_item_corpus, val_items_cache, force=force
        )
        lf_val = _build_feature_lazy(
            history=pl.scan_parquet(splits / "history.parquet"),
            candidates=pl.scan_parquet(retrieval / "candidates.parquet"),
            covis_path=retrieval / "covisitation.parquet",
            items=val_items,
            labels_path=splits / "labels.parquet",
        )
        tmp_val = atomic_temp_path(val_out)
        lf_val.sink_parquet(tmp_val)
        commit_atomic(tmp_val, val_out)
        logger.info("[features] Wrote %s", val_out)

    test_history_path = splits / "test_history.parquet"
    test_cands_path = retrieval / "test_candidates.parquet"
    if not (test_history_path.exists() and test_cands_path.exists()):
        return

    if mode == "sample":
        test_out = output / "test_features.parquet"
        if is_valid_checkpoint(test_out, force=force):
            logger.info("[checkpoint] Test features already exist -> %s", test_out)
            return

        test_item_corpus = pl.concat(
            [
                pl.scan_parquet(splits / "train.parquet"),
                pl.scan_parquet(splits / "history.parquet"),
                pl.scan_parquet(test_history_path),
            ]
        )
        test_items = _ensure_item_features_cached(
            test_item_corpus, output / "_test_item_features.parquet", force=force
        )
        lf_test = _build_feature_lazy(
            history=pl.scan_parquet(test_history_path),
            candidates=pl.scan_parquet(test_cands_path),
            covis_path=retrieval / "covisitation.parquet",
            items=test_items,
            labels_path=None,
        )
        tmp_test = atomic_temp_path(test_out)
        lf_test.sink_parquet(tmp_test)
        commit_atomic(tmp_test, test_out)
        logger.info("[features] Wrote %s", test_out)
    else:
        # Full mode: shard 1.67M test sessions into 16 bounded, individually checkpointed shards!
        shards_dir = output / "test_features_shards"
        complete_flag = shards_dir / "_COMPLETE.flag"
        if force and shards_dir.exists():
            shutil.rmtree(shards_dir, ignore_errors=True)
        shards_dir.mkdir(parents=True, exist_ok=True)

        if not force and complete_flag.exists():
            logger.info(
                "[checkpoint] All %d full test feature shards already exist in '%s'; skipping.",
                FULL_TEST_SHARDS,
                shards_dir,
            )
            return

        raw_train_files = [
            p
            for p in sorted(raw_dir.rglob("*.parquet"))
            if "train" in p.name.lower() or "train" in p.parent.name.lower()
        ]
        if raw_train_files:
            raw_lf = pl.scan_parquet(raw_train_files).select(
                pl.col("session").cast(pl.Int32).alias("session"),
                pl.col("aid").cast(pl.Int32).alias("aid"),
                pl.col("ts").cast(pl.Int64).alias("ts"),
                _type_code_expr(pl.scan_parquet(raw_train_files))
                .cast(pl.UInt8)
                .alias("type"),
            )
            test_item_corpus = pl.concat([raw_lf, pl.scan_parquet(test_history_path)])
        else:
            test_item_corpus = pl.concat(
                [
                    pl.scan_parquet(splits / "train.parquet"),
                    pl.scan_parquet(splits / "history.parquet"),
                    pl.scan_parquet(test_history_path),
                ]
            )

        test_items = _ensure_item_features_cached(
            test_item_corpus, output / "_test_item_features.parquet", force=force
        )

        for shard_id in range(FULL_TEST_SHARDS):
            shard_path = shards_dir / f"shard_{shard_id:02d}.parquet"
            if is_valid_checkpoint(shard_path, force=False):
                logger.info(
                    "[checkpoint] Test feature shard %d/%d already exists; skipping.",
                    shard_id + 1,
                    FULL_TEST_SHARDS,
                )
                continue

            shard_filter = (
                pl.col("session").hash(seed=99) % FULL_TEST_SHARDS
            ) == shard_id
            lf_shard = _build_feature_lazy(
                history=pl.scan_parquet(test_history_path).filter(shard_filter),
                candidates=pl.scan_parquet(test_cands_path).filter(shard_filter),
                covis_path=retrieval / "covisitation.parquet",
                items=test_items,
                labels_path=None,
            )
            tmp_s = atomic_temp_path(shard_path)
            lf_shard.sink_parquet(tmp_s)
            commit_atomic(tmp_s, shard_path)
            gc.collect()
            logger.info(
                "[features] Checkpointed test feature shard %d/%d -> %s",
                shard_id + 1,
                FULL_TEST_SHARDS,
                shard_path,
            )

        complete_flag.write_text("ok", encoding="utf-8")
        logger.info("[features] Completed all %d test feature shards.", FULL_TEST_SHARDS)
