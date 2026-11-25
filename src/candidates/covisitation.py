"""Out-of-core 3-Matrix Co-visitation & Retrieval with Bucket-Level Drive Checkpoints.

Every sub-step (per-bucket edge extraction, matrix normalization, popular items, and
per-chunk session candidate retrieval) checkpoints atomically on Google Drive so an
interrupted Colab session resumes from the exact unfinished bucket/chunk.
"""

from __future__ import annotations

import gc
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

from src.metrics.recall import CODE_TO_TYPE, compute_recall_at_k
from src.utils.checkpoint import atomic_temp_path, commit_atomic, is_valid_checkpoint

logger = logging.getLogger(__name__)

MAX_SESSION_TAIL = 30
MAX_BUY_TAIL = 20
MAX_SOURCE_ITEMS = 15
SLICE_ROWS = 15_000_000
FINALIZE_PARTITIONS = 4

EVENT_WEIGHTS: dict[int, float] = {
    0: 1.0,  # clicks
    1: 6.0,  # carts
    2: 3.0,  # orders
}

TYPE_STR_TO_CODE: dict[str, int] = {
    "clicks": 0,
    "carts": 1,
    "orders": 2,
}


def _type_uint8_expr(lf: pl.LazyFrame) -> pl.Expr:
    """Normalize integer or string `type` column to `UInt8` codes 0/1/2."""
    if lf.collect_schema()["type"].is_integer():
        return pl.col("type").cast(pl.UInt8)
    return (
        pl.col("type")
        .cast(pl.Utf8)
        .replace_strict(TYPE_STR_TO_CODE, return_dtype=pl.UInt8)
    )


def _detect_ts_per_sec(parquet_paths: list[Path]) -> int:
    """Detect whether timestamps are in milliseconds or seconds across heterogeneous files."""
    max_vals = [
        pl.scan_parquet(p).select(pl.col("ts").cast(pl.Int64).max()).collect().item()
        for p in parquet_paths
    ]
    valid = [int(v) for v in max_vals if v is not None]
    if not valid:
        return 1
    return 1000 if max(valid) > 10_000_000_000 else 1


def _scatter_pending_buckets(
    parquet_paths: list[Path],
    num_buckets: int,
    pending_buckets: set[int],
    bucket_root: Path,
) -> None:
    """Stream input Parquet files in 15M-row slices into local shards for pending buckets only."""
    for b in pending_buckets:
        (bucket_root / f"b_{b}").mkdir(parents=True, exist_ok=True)

    pending_list = sorted(pending_buckets)
    shard_idx = 0
    for pq_path in parquet_paths:
        lf_full = pl.scan_parquet(pq_path)
        type_expr = _type_uint8_expr(lf_full)
        total_rows = lf_full.select(pl.len()).collect().item()
        logger.info(
            "[candidates] Streaming '%s' (%d rows) for %d/%d pending buckets...",
            pq_path.name,
            total_rows,
            len(pending_buckets),
            num_buckets,
        )
        for offset in range(0, total_rows, SLICE_ROWS):
            chunk = (
                pl.scan_parquet(pq_path)
                .slice(offset, SLICE_ROWS)
                .select(
                    pl.col("session").cast(pl.Int32).alias("session"),
                    pl.col("aid").cast(pl.Int32).alias("aid"),
                    pl.col("ts").cast(pl.Int64).alias("ts"),
                    type_expr.alias("type"),
                )
                .with_columns(
                    (pl.col("session").hash(seed=0) % num_buckets)
                    .cast(pl.UInt8)
                    .alias("_b")
                )
                .filter(pl.col("_b").is_in(pending_list))
                .collect()
            )
            if len(chunk) > 0:
                parts = chunk.partition_by("_b", as_dict=True, include_key=False)
                for b_key, b_df in parts.items():
                    b_id = int(b_key[0]) if isinstance(b_key, tuple) else int(b_key)
                    b_df.write_parquet(
                        bucket_root / f"b_{b_id}" / f"s_{shard_idx}.parquet"
                    )
                del parts
            del chunk
            shard_idx += 1
            gc.collect()


def _finalize_partitioned_edges(
    edge_dir: Path,
    top_k: int,
    out_path: Path,
    label: str,
    force: bool = False,
) -> None:
    """Aggregate partitioned edge shards into a degree-normalized co-visitation Parquet."""
    if is_valid_checkpoint(out_path, force=force):
        logger.info("[checkpoint] '%s' matrix already exists -> %s", label, out_path)
        return

    empty_schema = {
        "source_aid": pl.Int32,
        "aid": pl.Int32,
        "score": pl.Float32,
        "norm_score": pl.Float32,
    }
    norm_chunks: list[pl.DataFrame] = []

    for p_id in range(FINALIZE_PARTITIONS):
        p_files = sorted((edge_dir / f"p_{p_id}").glob("*.parquet"))
        if not p_files:
            continue

        part_norm = (
            pl.scan_parquet(p_files)
            .group_by(["source_aid", "aid"])
            .agg(pl.col("score").sum().cast(pl.Float32).alias("score"))
            .sort(["source_aid", "score"], descending=[False, True])
            .group_by("source_aid", maintain_order=True)
            .agg(
                pl.col("aid").head(top_k),
                pl.col("score").head(top_k),
            )
            .with_columns(
                pl.col("score").list.max().clip(lower_bound=1e-6).alias("_smax"),
                pl.col("score").list.sum().clip(lower_bound=1e-6).alias("_ssum"),
                pl.int_ranges(1, pl.col("aid").list.len() + 1, dtype=pl.UInt16).alias(
                    "_rk"
                ),
            )
            .explode(["aid", "score", "_rk"])
            .with_columns(
                (
                    (
                        0.45 * (pl.col("score") / pl.col("_smax"))
                        + 0.35 * (1.0 / (pl.col("_rk").cast(pl.Float32) + 2.0) * 3.0)
                        + 0.20 * (pl.col("score") / pl.col("_ssum") * 5.0)
                    )
                    * (1.0 + 0.04 * pl.col("score").log1p())
                )
                .cast(pl.Float32)
                .alias("norm_score")
            )
            .select("source_aid", "aid", "score", "norm_score")
            .collect()
        )
        if len(part_norm) > 0:
            norm_chunks.append(part_norm)
        gc.collect()

    final_df = (
        pl.concat(norm_chunks) if norm_chunks else pl.DataFrame(schema=empty_schema)
    )
    tmp_out = atomic_temp_path(out_path)
    final_df.write_parquet(tmp_out)
    commit_atomic(tmp_out, out_path)
    logger.info(
        "[candidates] Saved %s co-visitation (%d normalized edges) -> %s",
        label,
        len(final_df),
        out_path,
    )


def _build_all_three_matrices_out_of_core(
    parquet_paths: list[Path],
    clicks_top_k: int,
    carts_top_k: int,
    b2b_top_k: int,
    clicks_win_hours: float,
    b2b_win_days: float,
    num_buckets: int,
    clicks_out: Path,
    carts_out: Path,
    b2b_out: Path,
    ckpt_dir: Path,
    legacy_out: Path | None = None,
    force: bool = False,
) -> None:
    """Build clicks, carts_orders, and buy2buy matrices with per-bucket Drive checkpoints."""
    targets = [clicks_out, carts_out, b2b_out]
    if legacy_out is not None:
        targets.append(legacy_out)

    if all(is_valid_checkpoint(p, force=force) for p in targets):
        logger.info(
            "[checkpoint] All 3 co-visitation matrices already exist in '%s'; skipping.",
            clicks_out.parent,
        )
        return

    if force and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    clicks_edge_dir = ckpt_dir / "edges_clicks"
    carts_edge_dir = ckpt_dir / "edges_carts"
    b2b_edge_dir = ckpt_dir / "edges_b2b"
    flags_dir = ckpt_dir / "done_flags"
    flags_dir.mkdir(parents=True, exist_ok=True)
    for ed in (clicks_edge_dir, carts_edge_dir, b2b_edge_dir):
        for p_id in range(FINALIZE_PARTITIONS):
            (ed / f"p_{p_id}").mkdir(parents=True, exist_ok=True)

    pending_buckets = {
        b_id
        for b_id in range(num_buckets)
        if not (flags_dir / f"b_{b_id}.done").exists()
    }

    if pending_buckets:
        if len(pending_buckets) < num_buckets:
            logger.info(
                "[checkpoint] Resuming co-visitation matrices: %d/%d buckets already completed on Drive!",
                num_buckets - len(pending_buckets),
                num_buckets,
            )

        ts_per_sec = _detect_ts_per_sec(parquet_paths)
        win_24h = int(clicks_win_hours * 3600 * ts_per_sec)
        inv_win_24h = 1.0 / max(win_24h, 1)
        win_14d = int(b2b_win_days * 86_400 * ts_per_sec)

        tmp_scatter = Path(tempfile.mkdtemp(prefix="otto_covis_"))
        try:
            _scatter_pending_buckets(
                parquet_paths, num_buckets, pending_buckets, tmp_scatter
            )

            for b_id in sorted(pending_buckets):
                b_files = sorted((tmp_scatter / f"b_{b_id}").glob("*.parquet"))
                if not b_files:
                    (flags_dir / f"b_{b_id}.done").write_text("ok", encoding="utf-8")
                    continue

                raw_bucket = pl.read_parquet(b_files).sort(["session", "ts"])
                shutil.rmtree(tmp_scatter / f"b_{b_id}", ignore_errors=True)

                tail_lf = (
                    raw_bucket.lazy()
                    .group_by("session", maintain_order=True)
                    .tail(MAX_SESSION_TAIL)
                    .with_columns(
                        pl.col("type")
                        .replace_strict(
                            EVENT_WEIGHTS, default=1.0, return_dtype=pl.Float32
                        )
                        .alias("type_w")
                    )
                )
                right_lf = tail_lf.select(
                    pl.col("session"),
                    pl.col("aid").alias("aid_right"),
                    pl.col("ts").alias("ts_right"),
                    pl.col("type_w").alias("type_w_right"),
                )

                pairs = (
                    tail_lf.join(right_lf, on="session", how="inner")
                    .filter(
                        (pl.col("aid") != pl.col("aid_right"))
                        & ((pl.col("ts") - pl.col("ts_right")).abs() <= win_24h)
                    )
                    .with_columns(
                        (
                            (pl.col("ts") - pl.col("ts_right")).abs().cast(pl.Float32)
                            * inv_win_24h
                        ).alias("_rel_dt"),
                        pl.when(pl.col("ts_right") >= pl.col("ts"))
                        .then(pl.lit(1.6, dtype=pl.Float32))
                        .otherwise(pl.lit(0.9, dtype=pl.Float32))
                        .alias("_dir_w"),
                    )
                    .with_columns(
                        (
                            pl.col("_dir_w")
                            * (1.0 + 3.0 * (1.0 - pl.col("_rel_dt")).pow(2))
                        )
                        .cast(pl.Float32)
                        .alias("w_click"),
                        (
                            pl.col("type_w_right")
                            * pl.col("_dir_w")
                            * (1.0 + 2.0 * (1.0 - pl.col("_rel_dt")))
                        )
                        .cast(pl.Float32)
                        .alias("w_cart"),
                    )
                    .group_by(["session", "aid", "aid_right"])
                    .agg(
                        pl.col("w_click").max().alias("w_click"),
                        pl.col("w_cart").max().alias("w_cart"),
                    )
                    .group_by(
                        [
                            pl.col("aid").alias("source_aid"),
                            pl.col("aid_right").alias("aid"),
                        ]
                    )
                    .agg(
                        pl.col("w_click").sum().cast(pl.Float32).alias("score_clicks"),
                        pl.col("w_cart").sum().cast(pl.Float32).alias("score_carts"),
                    )
                    .collect()
                )

                if len(pairs) > 0:
                    c_top = (
                        pairs.lazy()
                        .select(
                            "source_aid",
                            "aid",
                            pl.col("score_clicks").alias("score"),
                            (pl.col("source_aid").hash(seed=3) % FINALIZE_PARTITIONS)
                            .cast(pl.UInt8)
                            .alias("_p"),
                        )
                        .sort(["source_aid", "score"], descending=[False, True])
                        .group_by("source_aid", maintain_order=True)
                        .head(clicks_top_k + 20)
                        .collect()
                    )
                    for p_key, p_df in c_top.partition_by(
                        "_p", as_dict=True, include_key=False
                    ).items():
                        p_id = int(p_key[0]) if isinstance(p_key, tuple) else int(p_key)
                        dst = clicks_edge_dir / f"p_{p_id}" / f"b_{b_id}.parquet"
                        tmp_d = atomic_temp_path(dst)
                        p_df.write_parquet(tmp_d)
                        commit_atomic(tmp_d, dst)
                    del c_top

                    o_top = (
                        pairs.lazy()
                        .select(
                            "source_aid",
                            "aid",
                            pl.col("score_carts").alias("score"),
                            (pl.col("source_aid").hash(seed=3) % FINALIZE_PARTITIONS)
                            .cast(pl.UInt8)
                            .alias("_p"),
                        )
                        .sort(["source_aid", "score"], descending=[False, True])
                        .group_by("source_aid", maintain_order=True)
                        .head(carts_top_k + 20)
                        .collect()
                    )
                    for p_key, p_df in o_top.partition_by(
                        "_p", as_dict=True, include_key=False
                    ).items():
                        p_id = int(p_key[0]) if isinstance(p_key, tuple) else int(p_key)
                        dst = carts_edge_dir / f"p_{p_id}" / f"b_{b_id}.parquet"
                        tmp_d = atomic_temp_path(dst)
                        p_df.write_parquet(tmp_d)
                        commit_atomic(tmp_d, dst)
                    del o_top

                del pairs

                buy_lf = (
                    raw_bucket.lazy()
                    .filter(pl.col("type").is_in([1, 2]))
                    .group_by("session", maintain_order=True)
                    .tail(MAX_BUY_TAIL)
                    .with_columns(
                        pl.when(pl.col("type") == 2)
                        .then(pl.lit(1.5, dtype=pl.Float32))
                        .otherwise(pl.lit(1.0, dtype=pl.Float32))
                        .alias("buy_w")
                    )
                )
                buy_r = buy_lf.select(
                    pl.col("session"),
                    pl.col("aid").alias("aid_right"),
                    pl.col("ts").alias("ts_right"),
                    pl.col("buy_w").alias("buy_w_right"),
                )
                b_pairs = (
                    buy_lf.join(buy_r, on="session", how="inner")
                    .filter(
                        (pl.col("aid") != pl.col("aid_right"))
                        & ((pl.col("ts") - pl.col("ts_right")).abs() <= win_14d)
                    )
                    .group_by(["session", "aid", "aid_right"])
                    .agg(pl.col("buy_w_right").max().alias("w"))
                    .group_by(
                        [
                            pl.col("aid").alias("source_aid"),
                            pl.col("aid_right").alias("aid"),
                        ]
                    )
                    .agg(pl.col("w").sum().cast(pl.Float32).alias("score"))
                    .with_columns(
                        (pl.col("source_aid").hash(seed=3) % FINALIZE_PARTITIONS)
                        .cast(pl.UInt8)
                        .alias("_p")
                    )
                    .collect()
                )
                if len(b_pairs) > 0:
                    for p_key, p_df in b_pairs.partition_by(
                        "_p", as_dict=True, include_key=False
                    ).items():
                        p_id = int(p_key[0]) if isinstance(p_key, tuple) else int(p_key)
                        dst = b2b_edge_dir / f"p_{p_id}" / f"b_{b_id}.parquet"
                        tmp_d = atomic_temp_path(dst)
                        p_df.write_parquet(tmp_d)
                        commit_atomic(tmp_d, dst)
                del b_pairs, raw_bucket
                gc.collect()

                # Mark bucket completed on Google Drive
                (flags_dir / f"b_{b_id}.done").write_text("ok", encoding="utf-8")
                if (b_id + 1) % 4 == 0 or (b_id + 1) == num_buckets:
                    logger.info(
                        "[candidates] Checkpointed session bucket %d/%d on Drive",
                        b_id + 1,
                        num_buckets,
                    )
        finally:
            shutil.rmtree(tmp_scatter, ignore_errors=True)

    _finalize_partitioned_edges(
        clicks_edge_dir, clicks_top_k, clicks_out, "clicks", force=force
    )
    _finalize_partitioned_edges(
        carts_edge_dir, carts_top_k, carts_out, "carts/orders", force=force
    )
    if legacy_out is not None and not is_valid_checkpoint(legacy_out, force=force):
        tmp_leg = atomic_temp_path(legacy_out)
        shutil.copy2(carts_out, tmp_leg)
        commit_atomic(tmp_leg, legacy_out)
    _finalize_partitioned_edges(
        b2b_edge_dir, b2b_top_k, b2b_out, "buy2buy", force=force
    )

    # Clean up bucket edge checkpoints once all final matrices exist
    shutil.rmtree(ckpt_dir, ignore_errors=True)


def _save_popular_items(
    parquet_paths: list[Path], top_k: int, out_path: Path, force: bool = False
) -> list[int]:
    """Compute or load top popular items in the recent 7-day window."""
    if is_valid_checkpoint(out_path, force=force):
        pop_df = pl.read_parquet(out_path)
        return pop_df.filter(pl.col("type") == 255)["aid"].to_list()

    events_list = [
        pl.scan_parquet(p).select(
            pl.col("aid").cast(pl.Int32).alias("aid"),
            pl.col("ts").cast(pl.Int64).alias("ts"),
            _type_uint8_expr(pl.scan_parquet(p)).alias("type"),
        )
        for p in parquet_paths
    ]
    events = pl.concat(events_list)
    max_ts = events.select(pl.col("ts").max()).collect().item()
    ts_per_sec = 1000 if (max_ts is not None and int(max_ts) > 10_000_000_000) else 1
    recent_cutoff = (int(max_ts) - 7 * 86_400 * ts_per_sec) if max_ts is not None else 0

    recent_events = events.filter(pl.col("ts") >= recent_cutoff)
    if recent_events.limit(1).collect().height == 0:
        recent_events = events

    by_type = (
        recent_events.group_by(["type", "aid"])
        .agg(pl.len().alias("pop_count"))
        .sort(["type", "pop_count"], descending=[False, True])
        .group_by("type", maintain_order=True)
        .head(top_k)
        .collect()
    )
    overall = (
        recent_events.group_by("aid")
        .agg(pl.len().alias("pop_count"))
        .sort("pop_count", descending=True)
        .head(top_k)
        .with_columns(pl.lit(255, dtype=pl.UInt8).alias("type"))
        .select("type", "aid", "pop_count")
        .collect()
    )
    tmp_pop = atomic_temp_path(out_path)
    pl.concat([by_type.select("type", "aid", "pop_count"), overall]).write_parquet(
        tmp_pop
    )
    commit_atomic(tmp_pop, out_path)
    return overall["aid"].to_list()


def _retrieve_candidates_for_sessions(
    session_history: pl.LazyFrame,
    clicks_covis_path: Path,
    carts_covis_path: Path,
    buy2buy_covis_path: Path,
    fallback_aids: list[int],
    top_k: int,
    num_buckets: int,
    out_path: Path,
    ckpt_dir: Path,
    force: bool = False,
) -> None:
    """Retrieve multi-matrix candidates per session with chunk-level Drive checkpointing."""
    if is_valid_checkpoint(out_path, force=force):
        logger.info("[checkpoint] Candidate table already exists -> %s", out_path)
        return

    if force and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    covis_clicks = pl.scan_parquet(clicks_covis_path).select(
        "source_aid", "aid", "norm_score", pl.col("score").alias("raw_score")
    )
    covis_carts = pl.scan_parquet(carts_covis_path).select(
        "source_aid", "aid", "norm_score", pl.col("score").alias("raw_score")
    )
    covis_b2b = pl.scan_parquet(buy2buy_covis_path).select(
        "source_aid", "aid", "norm_score"
    )

    hist_df = (
        session_history.sort(["session", "ts"])
        .with_columns(
            pl.len().over("session").alias("_slen"),
            pl.int_range(0, pl.len()).over("session").alias("_fpos"),
            pl.col("type")
            .replace_strict(EVENT_WEIGHTS, default=1.0, return_dtype=pl.Float32)
            .alias("_tw"),
        )
        .with_columns(
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
                * pl.col("_tw")
            )
            .cast(pl.Float32)
            .alias("_log_w")
        )
        .group_by(["session", "aid"])
        .agg(
            pl.col("ts").max().alias("last_ts"),
            pl.col("type").max().alias("max_type"),
            pl.col("_log_w").sum().cast(pl.Float32).alias("revisit_score"),
        )
        .sort(["session", "last_ts"], descending=[False, True])
        .with_columns(
            pl.int_range(0, pl.len()).over("session").alias("_urev"),
            (pl.col("session").hash(seed=11) % num_buckets).cast(pl.UInt8).alias("_b"),
        )
        .collect()
    )

    fallback_df = pl.DataFrame(
        {
            "aid": pl.Series(fallback_aids[: min(20, top_k)], dtype=pl.Int32),
            "_fb_score": pl.Series(
                [1e-5 * (20 - i) for i in range(min(len(fallback_aids), 20, top_k))],
                dtype=pl.Float32,
            ),
        }
    ).lazy()

    raw_buckets = hist_df.partition_by("_b", as_dict=True, include_key=False)
    buckets: dict[int, pl.DataFrame] = {
        (int(k[0]) if isinstance(k, tuple) else int(k)): v
        for k, v in raw_buckets.items()
    }
    del hist_df, raw_buckets
    gc.collect()

    written_parts: list[Path] = []
    for b_id in sorted(buckets.keys()):
        part_df = buckets.pop(b_id)
        chunk_file = ckpt_dir / f"c_{b_id:03d}.parquet"
        if is_valid_checkpoint(chunk_file, force=False):
            written_parts.append(chunk_file)
            del part_df
            continue

        hb = part_df.lazy()
        recent_src = (
            hb.filter(pl.col("_urev") < MAX_SOURCE_ITEMS)
            .with_columns(
                pl.lit(0.58, dtype=pl.Float32)
                .pow(pl.col("_urev"))
                .cast(pl.Float32)
                .alias("_click_src_w"),
                (
                    pl.lit(0.75, dtype=pl.Float32).pow(pl.col("_urev"))
                    * pl.when(pl.col("max_type") >= 1)
                    .then(pl.lit(2.2, dtype=pl.Float32))
                    .otherwise(pl.lit(1.0, dtype=pl.Float32))
                )
                .cast(pl.Float32)
                .alias("_cart_src_w"),
                pl.when(pl.col("max_type") >= 1)
                .then(pl.lit(2.5, dtype=pl.Float32))
                .otherwise(pl.lit(0.4, dtype=pl.Float32))
                .alias("_b2b_mult"),
                (pl.col("_urev") == 0).alias("_is_last"),
            )
            .select(
                pl.col("session"),
                pl.col("aid").alias("source_aid"),
                pl.col("_click_src_w"),
                pl.col("_cart_src_w"),
                pl.col("_b2b_mult"),
                pl.col("_is_last"),
            )
        )

        revisit = hb.filter(pl.col("_urev") < top_k).select(
            pl.col("session"),
            pl.col("aid"),
            (1000.0 + pl.col("revisit_score")).cast(pl.Float32).alias("_revisit_prio"),
            pl.lit(0.0, dtype=pl.Float32).alias("score_clicks"),
            pl.lit(0.0, dtype=pl.Float32).alias("score_carts_orders"),
            pl.lit(0.0, dtype=pl.Float32).alias("score_buy2buy"),
            pl.lit(0.0, dtype=pl.Float32).alias("last_aid_covis"),
            pl.lit(0.0, dtype=pl.Float32).alias("raw_covis_sum"),
            pl.lit(0, dtype=pl.UInt16).alias("covis_support"),
        )

        exp_clicks = (
            recent_src.join(covis_clicks, on="source_aid", how="inner")
            .group_by(["session", "aid"])
            .agg(
                pl.lit(0.0, dtype=pl.Float32).alias("_revisit_prio"),
                (pl.col("norm_score") * pl.col("_click_src_w"))
                .sum()
                .cast(pl.Float32)
                .alias("score_clicks"),
                pl.lit(0.0, dtype=pl.Float32).alias("score_carts_orders"),
                pl.lit(0.0, dtype=pl.Float32).alias("score_buy2buy"),
                pl.when(pl.col("_is_last"))
                .then(pl.col("norm_score"))
                .otherwise(0.0)
                .max()
                .cast(pl.Float32)
                .alias("last_aid_covis"),
                pl.col("raw_score").sum().log1p().cast(pl.Float32).alias("raw_covis_sum"),
                pl.len().cast(pl.UInt16).alias("covis_support"),
            )
        )

        exp_carts = (
            recent_src.join(covis_carts, on="source_aid", how="inner")
            .group_by(["session", "aid"])
            .agg(
                pl.lit(0.0, dtype=pl.Float32).alias("_revisit_prio"),
                pl.lit(0.0, dtype=pl.Float32).alias("score_clicks"),
                (pl.col("norm_score") * pl.col("_cart_src_w"))
                .sum()
                .cast(pl.Float32)
                .alias("score_carts_orders"),
                pl.lit(0.0, dtype=pl.Float32).alias("score_buy2buy"),
                pl.when(pl.col("_is_last"))
                .then(pl.col("norm_score") * 0.8)
                .otherwise(0.0)
                .max()
                .cast(pl.Float32)
                .alias("last_aid_covis"),
                pl.col("raw_score").sum().log1p().cast(pl.Float32).alias("raw_covis_sum"),
                pl.len().cast(pl.UInt16).alias("covis_support"),
            )
        )

        exp_b2b = (
            recent_src.join(covis_b2b, on="source_aid", how="inner")
            .group_by(["session", "aid"])
            .agg(
                pl.lit(0.0, dtype=pl.Float32).alias("_revisit_prio"),
                pl.lit(0.0, dtype=pl.Float32).alias("score_clicks"),
                pl.lit(0.0, dtype=pl.Float32).alias("score_carts_orders"),
                (pl.col("norm_score") * pl.col("_cart_src_w") * pl.col("_b2b_mult"))
                .sum()
                .cast(pl.Float32)
                .alias("score_buy2buy"),
                pl.lit(0.0, dtype=pl.Float32).alias("last_aid_covis"),
                pl.lit(0.0, dtype=pl.Float32).alias("raw_covis_sum"),
                pl.len().cast(pl.UInt16).alias("covis_support"),
            )
        )

        merged = (
            pl.concat([revisit, exp_clicks, exp_carts, exp_b2b])
            .group_by(["session", "aid"])
            .agg(
                pl.col("_revisit_prio").max().alias("_revisit_prio"),
                pl.col("score_clicks").sum().cast(pl.Float32).alias("score_clicks"),
                pl.col("score_carts_orders")
                .sum()
                .cast(pl.Float32)
                .alias("score_carts_orders"),
                pl.col("score_buy2buy").sum().cast(pl.Float32).alias("score_buy2buy"),
                pl.col("last_aid_covis").max().cast(pl.Float32).alias("last_aid_covis"),
                pl.col("raw_covis_sum").max().cast(pl.Float32).alias("raw_covis_sum"),
                pl.col("covis_support").sum().cast(pl.UInt16).alias("covis_support"),
            )
            .with_columns(
                (
                    pl.col("score_clicks") * 1.2
                    + pl.col("score_carts_orders") * 1.5
                    + pl.col("score_buy2buy") * 1.3
                    + pl.col("last_aid_covis") * 1.5
                )
                .cast(pl.Float32)
                .alias("covisitation_score"),
                (
                    pl.col("_revisit_prio")
                    + pl.col("score_clicks") * 1.2
                    + pl.col("score_carts_orders") * 1.5
                    + pl.col("score_buy2buy") * 1.3
                    + pl.col("last_aid_covis") * 1.5
                )
                .cast(pl.Float32)
                .alias("_pool_rank_score"),
            )
            .drop("_revisit_prio")
        )

        min_needed = min(20, top_k)
        underfilled = (
            hb.select("session")
            .unique()
            .join(
                merged.group_by("session")
                .agg(pl.len().alias("_cnt"))
                .filter(pl.col("_cnt") >= min_needed),
                on="session",
                how="anti",
            )
        )
        padded = underfilled.join(fallback_df, how="cross").select(
            pl.col("session"),
            pl.col("aid"),
            pl.lit(0.0, dtype=pl.Float32).alias("score_clicks"),
            pl.lit(0.0, dtype=pl.Float32).alias("score_carts_orders"),
            pl.lit(0.0, dtype=pl.Float32).alias("score_buy2buy"),
            pl.lit(0.0, dtype=pl.Float32).alias("last_aid_covis"),
            pl.lit(0.0, dtype=pl.Float32).alias("raw_covis_sum"),
            pl.lit(0, dtype=pl.UInt16).alias("covis_support"),
            pl.col("_fb_score").alias("covisitation_score"),
            pl.col("_fb_score").alias("_pool_rank_score"),
        )

        top_cands = (
            pl.concat([merged, padded])
            .group_by(["session", "aid"])
            .agg(
                pl.col("score_clicks").max().alias("score_clicks"),
                pl.col("score_carts_orders").max().alias("score_carts_orders"),
                pl.col("score_buy2buy").max().alias("score_buy2buy"),
                pl.col("last_aid_covis").max().alias("last_aid_covis"),
                pl.col("raw_covis_sum").max().alias("raw_covis_sum"),
                pl.col("covis_support").max().alias("covis_support"),
                pl.col("covisitation_score").max().alias("covisitation_score"),
                pl.col("_pool_rank_score").max().alias("_pool_rank_score"),
            )
            .sort(["session", "_pool_rank_score"], descending=[False, True])
            .group_by("session", maintain_order=True)
            .agg(
                pl.col("aid").head(top_k),
                pl.col("score_clicks").head(top_k),
                pl.col("score_carts_orders").head(top_k),
                pl.col("score_buy2buy").head(top_k),
                pl.col("last_aid_covis").head(top_k),
                pl.col("raw_covis_sum").head(top_k),
                pl.col("covis_support").head(top_k),
                pl.col("covisitation_score").head(top_k),
            )
            .with_columns(
                pl.int_ranges(1, pl.col("aid").list.len() + 1, dtype=pl.UInt16).alias(
                    "cand_rank"
                )
            )
            .explode(
                [
                    "aid",
                    "score_clicks",
                    "score_carts_orders",
                    "score_buy2buy",
                    "last_aid_covis",
                    "raw_covis_sum",
                    "covis_support",
                    "covisitation_score",
                    "cand_rank",
                ]
            )
            .collect()
        )
        del part_df
        if len(top_cands) > 0:
            tmp_c = atomic_temp_path(chunk_file)
            top_cands.write_parquet(tmp_c)
            commit_atomic(tmp_c, chunk_file)
            written_parts.append(chunk_file)
        del top_cands
        gc.collect()

    tmp_final = atomic_temp_path(out_path)
    if written_parts:
        total_written = pl.scan_parquet(written_parts).select(pl.len()).collect().item()
        pl.scan_parquet(written_parts).sink_parquet(tmp_final)
    else:
        total_written = 0
        pl.DataFrame(
            schema={
                "session": pl.Int32,
                "aid": pl.Int32,
                "score_clicks": pl.Float32,
                "score_carts_orders": pl.Float32,
                "score_buy2buy": pl.Float32,
                "last_aid_covis": pl.Float32,
                "raw_covis_sum": pl.Float32,
                "covis_support": pl.UInt16,
                "covisitation_score": pl.Float32,
                "cand_rank": pl.UInt16,
            }
        ).write_parquet(tmp_final)
    commit_atomic(tmp_final, out_path)
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    logger.info("[candidates] Wrote %d candidate pairs -> %s", total_written, out_path)


def _log_stage1_recall(
    val_cands_path: Path,
    labels_path: Path,
    top_k: int,
    weights: dict[str, float],
) -> None:
    """Log Stage-1 Unreranked Baseline Recall@20 and Candidate Pool Ceiling Recall@K."""
    if not labels_path.exists() or not val_cands_path.exists():
        return

    gt = (
        pl.scan_parquet(labels_path)
        .with_columns(
            pl.col("aid").cast(pl.Int32),
            pl.col("type").replace_strict(CODE_TO_TYPE, default="clicks").alias("type"),
        )
        .group_by(["session", "type"])
        .agg(pl.col("aid").unique().alias("ground_truth"))
        .collect()
    )

    cands_grouped = (
        pl.scan_parquet(val_cands_path)
        .sort(["session", "cand_rank"])
        .group_by("session", maintain_order=True)
        .agg(pl.col("aid").alias("labels"))
        .collect()
    )

    preds_all = pl.concat(
        [
            cands_grouped.with_columns(pl.lit(ev).alias("type")).select(
                "session", "type", "labels"
            )
            for ev in ("clicks", "carts", "orders")
        ]
    )

    r20 = compute_recall_at_k(preds_all, gt, weights=weights, k=20)
    rk = compute_recall_at_k(preds_all, gt, weights=weights, k=top_k)
    logger.info(
        "[candidates] Stage-1 Baseline Recall@20 (before GBDT) | clicks=%.5f | carts=%.5f | orders=%.5f | OVERALL=%.5f",
        r20["clicks"],
        r20["carts"],
        r20["orders"],
        r20["overall"],
    )
    logger.info(
        "[candidates] Stage-1 Pool Ceiling Recall@%d          | clicks=%.5f | carts=%.5f | orders=%.5f | OVERALL=%.5f",
        top_k,
        rk["clicks"],
        rk["carts"],
        rk["orders"],
        rk["overall"],
    )


def run_candidates(config: dict[str, Any], mode: str = "sample") -> None:
    """Build 3 degree-normalized co-visitation matrices and retrieve candidates with Drive resume."""
    splits_dir = Path(config["paths"]["splits"]) / mode
    raw_dir = Path(config["paths"]["raw_data"])
    out_dir = Path(config["paths"]["candidates"]) / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    force = bool(config.get("force_recompute", False))

    top_k = int(config["pipeline"]["top_k_candidates"])
    covis_cfg = config.get("covisitation", {})
    clicks_top_k = int(covis_cfg.get("clicks_top_k", 50))
    carts_top_k = int(covis_cfg.get("carts_orders_top_k", 50))
    b2b_top_k = int(covis_cfg.get("buy2buy_top_k", 50))
    clicks_win_h = float(covis_cfg.get("clicks_window_hours", 24))
    b2b_win_d = float(covis_cfg.get("buy2buy_window_days", 14))
    num_buckets = 16

    train_path = splits_dir / "train.parquet"
    history_path = splits_dir / "history.parquet"
    labels_path = splits_dir / "labels.parquet"
    if not train_path.exists() or not history_path.exists():
        raise FileNotFoundError(
            f"Missing split files in '{splits_dir}'. Run '--stage split' first."
        )

    val_corpus_paths = [train_path, history_path]
    clicks_covis_path = out_dir / "covis_clicks.parquet"
    carts_covis_path = out_dir / "covis_carts_orders.parquet"
    b2b_covis_path = out_dir / "covis_buy2buy.parquet"
    legacy_covis_path = out_dir / "covisitation.parquet"
    pop_path = out_dir / "popular_items.parquet"
    val_cands_path = out_dir / "candidates.parquet"

    _build_all_three_matrices_out_of_core(
        parquet_paths=val_corpus_paths,
        clicks_top_k=clicks_top_k,
        carts_top_k=carts_top_k,
        b2b_top_k=b2b_top_k,
        clicks_win_hours=clicks_win_h,
        b2b_win_days=b2b_win_d,
        num_buckets=num_buckets,
        clicks_out=clicks_covis_path,
        carts_out=carts_covis_path,
        b2b_out=b2b_covis_path,
        ckpt_dir=out_dir / "_ckpt_val_matrices",
        legacy_out=legacy_covis_path,
        force=force,
    )

    fallback_aids = _save_popular_items(
        parquet_paths=val_corpus_paths,
        top_k=max(top_k, int(config["pipeline"]["top_k_predict"])),
        out_path=pop_path,
        force=force,
    )

    val_already_existed = is_valid_checkpoint(val_cands_path, force=force)
    _retrieve_candidates_for_sessions(
        session_history=pl.scan_parquet(history_path),
        clicks_covis_path=clicks_covis_path,
        carts_covis_path=carts_covis_path,
        buy2buy_covis_path=b2b_covis_path,
        fallback_aids=fallback_aids,
        top_k=top_k,
        num_buckets=8 if mode == "sample" else 20,
        out_path=val_cands_path,
        ckpt_dir=out_dir / "_ckpt_val_cands",
        force=force,
    )
    if not val_already_existed:
        _log_stage1_recall(
            val_cands_path=val_cands_path,
            labels_path=labels_path,
            top_k=top_k,
            weights=dict(config["pipeline"]["metric_weights"]),
        )

    test_history_path = splits_dir / "test_history.parquet"
    if test_history_path.exists():
        test_cands_path = out_dir / "test_candidates.parquet"
        # In full mode, enrich test co-visitation with 100% of raw train (Weeks 1-4) + Week-5 test prefixes!
        raw_train_files = [
            p
            for p in sorted(raw_dir.rglob("*.parquet"))
            if "train" in p.name.lower() or "train" in p.parent.name.lower()
        ]
        if mode == "full" and raw_train_files:
            t_clicks = out_dir / "test_covis_clicks.parquet"
            t_carts = out_dir / "test_covis_carts_orders.parquet"
            t_b2b = out_dir / "test_covis_buy2buy.parquet"
            t_pop = out_dir / "test_popular_items.parquet"
            test_corpus = [*raw_train_files, test_history_path]

            _build_all_three_matrices_out_of_core(
                parquet_paths=test_corpus,
                clicks_top_k=clicks_top_k,
                carts_top_k=carts_top_k,
                b2b_top_k=b2b_top_k,
                clicks_win_hours=clicks_win_h,
                b2b_win_days=b2b_win_d,
                num_buckets=num_buckets,
                clicks_out=t_clicks,
                carts_out=t_carts,
                b2b_out=t_b2b,
                ckpt_dir=out_dir / "_ckpt_test_matrices",
                legacy_out=None,
                force=force,
            )
            test_fb = _save_popular_items(
                parquet_paths=test_corpus,
                top_k=max(top_k, int(config["pipeline"]["top_k_predict"])),
                out_path=t_pop,
                force=force,
            )
        else:
            t_clicks, t_carts, t_b2b, test_fb = (
                clicks_covis_path,
                carts_covis_path,
                b2b_covis_path,
                fallback_aids,
            )

        _retrieve_candidates_for_sessions(
            session_history=pl.scan_parquet(test_history_path),
            clicks_covis_path=t_clicks,
            carts_covis_path=t_carts,
            buy2buy_covis_path=t_b2b,
            fallback_aids=test_fb,
            top_k=top_k,
            num_buckets=8 if mode == "sample" else 24,
            out_path=test_cands_path,
            ckpt_dir=out_dir / "_ckpt_test_cands",
            force=force,
        )
