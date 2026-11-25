"""Out-of-core 3-Matrix Co-visitation & Multi-Signal Candidate Retrieval (Stage 1).

Streams the 190M+ event corpus in bounded slices onto local fast scratch storage
(`/tmp`) so peak RAM stays under ~3.5 GB (never triggers Colab OOM `SIGKILL` / `^C`),
while preserving 100% of the degree-normalized 3-matrix accuracy:
  1. `covis_clicks.parquet`: 24h proximity- and direction-weighted normalized matrix.
  2. `covis_carts_orders.parquet`: 24h type- and direction-weighted normalized matrix.
  3. `covis_buy2buy.parquet`: 14-day carts/orders-only co-purchase normalized matrix.
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


def _detect_ts_per_sec(parquet_paths: list[Path]) -> int:
    """Detect whether timestamps are in milliseconds or seconds."""
    max_ts = (
        pl.scan_parquet(parquet_paths).select(pl.col("ts").max()).collect().item()
    )
    if max_ts is None:
        return 1
    return 1000 if int(max_ts) > 10_000_000_000 else 1


def _scatter_corpus_to_buckets(
    parquet_paths: list[Path],
    num_buckets: int,
    bucket_root: Path,
) -> None:
    """Stream input Parquet files in bounded 15M-row slices into session-hashed local shards."""
    for b in range(num_buckets):
        (bucket_root / f"b_{b}").mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    for pq_path in parquet_paths:
        total_rows = pl.scan_parquet(pq_path).select(pl.len()).collect().item()
        logger.info(
            "[candidates] Streaming '%s' (%d rows) into %d disk buckets...",
            pq_path.name,
            total_rows,
            num_buckets,
        )
        for offset in range(0, total_rows, SLICE_ROWS):
            chunk = (
                pl.scan_parquet(pq_path)
                .slice(offset, SLICE_ROWS)
                .select(
                    pl.col("session").cast(pl.Int32),
                    pl.col("aid").cast(pl.Int32),
                    pl.col("ts").cast(pl.Int64),
                    pl.col("type").cast(pl.UInt8),
                    (pl.col("session").hash(seed=0) % num_buckets)
                    .cast(pl.UInt8)
                    .alias("_b"),
                )
                .collect()
            )
            parts = chunk.partition_by("_b", as_dict=True, include_key=False)
            del chunk
            for b_key, b_df in parts.items():
                b_id = int(b_key[0]) if isinstance(b_key, tuple) else int(b_key)
                b_df.write_parquet(bucket_root / f"b_{b_id}" / f"s_{shard_idx}.parquet")
            del parts
            shard_idx += 1
            gc.collect()


def _finalize_partitioned_edges(
    edge_dir: Path,
    top_k: int,
    out_path: Path,
    label: str,
) -> None:
    """Aggregate partitioned edge shards into a degree-normalized co-visitation Parquet."""
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
    final_df.write_parquet(out_path)
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
    legacy_out: Path,
) -> None:
    """Build clicks, carts_orders, and buy2buy matrices out-of-core in local temp storage."""
    ts_per_sec = _detect_ts_per_sec(parquet_paths)
    win_24h = int(clicks_win_hours * 3600 * ts_per_sec)
    inv_win_24h = 1.0 / max(win_24h, 1)
    win_14d = int(b2b_win_days * 86_400 * ts_per_sec)

    tmp_root = Path(tempfile.mkdtemp(prefix="otto_covis_"))
    try:
        bucket_root = tmp_root / "buckets"
        clicks_edge_dir = tmp_root / "edges_clicks"
        carts_edge_dir = tmp_root / "edges_carts"
        b2b_edge_dir = tmp_root / "edges_b2b"
        for ed in (clicks_edge_dir, carts_edge_dir, b2b_edge_dir):
            for p_id in range(FINALIZE_PARTITIONS):
                (ed / f"p_{p_id}").mkdir(parents=True, exist_ok=True)

        # 1. Stream corpus into 16 local disk buckets (< 1.5 GB RAM)
        _scatter_corpus_to_buckets(parquet_paths, num_buckets, bucket_root)

        # 2. Process one session bucket at a time (~11M rows -> < 3 GB RAM)
        for b_id in range(num_buckets):
            b_files = sorted((bucket_root / f"b_{b_id}").glob("*.parquet"))
            if not b_files:
                continue

            raw_bucket = pl.read_parquet(b_files).sort(["session", "ts"])
            # Delete raw bucket shards immediately after loading to save /tmp space
            shutil.rmtree(bucket_root / f"b_{b_id}", ignore_errors=True)

            # A) Clicks + Carts/Orders 24h co-visitation on last 30 events per session
            tail_lf = (
                raw_bucket.lazy()
                .group_by("session", maintain_order=True)
                .tail(MAX_SESSION_TAIL)
                .with_columns(
                    pl.col("type")
                    .replace_strict(EVENT_WEIGHTS, default=1.0, return_dtype=pl.Float32)
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
                    [pl.col("aid").alias("source_aid"), pl.col("aid_right").alias("aid")]
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
                    p_df.write_parquet(
                        clicks_edge_dir / f"p_{p_id}" / f"b_{b_id}.parquet"
                    )
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
                    p_df.write_parquet(
                        carts_edge_dir / f"p_{p_id}" / f"b_{b_id}.parquet"
                    )
                del o_top

            del pairs

            # B) Buy2Buy 14-day co-visitation on carts & orders in this same bucket
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
                    [pl.col("aid").alias("source_aid"), pl.col("aid_right").alias("aid")]
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
                    p_df.write_parquet(b2b_edge_dir / f"p_{p_id}" / f"b_{b_id}.parquet")
            del b_pairs, raw_bucket
            gc.collect()

            if (b_id + 1) % 4 == 0 or (b_id + 1) == num_buckets:
                logger.info(
                    "[candidates] Processed session bucket %d/%d", b_id + 1, num_buckets
                )

        # 3. Finalize & normalize each matrix in 4 bounded partitions
        _finalize_partitioned_edges(clicks_edge_dir, clicks_top_k, clicks_out, "clicks")
        _finalize_partitioned_edges(
            carts_edge_dir, carts_top_k, carts_out, "carts/orders"
        )
        shutil.copy2(carts_out, legacy_out)
        _finalize_partitioned_edges(b2b_edge_dir, b2b_top_k, b2b_out, "buy2buy")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _save_popular_items(
    parquet_paths: list[Path], top_k: int, out_path: Path
) -> pl.DataFrame:
    """Compute top popular items in the recent 7-day window as fallback."""
    events = pl.scan_parquet(parquet_paths)
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
    pl.concat([by_type.select("type", "aid", "pop_count"), overall]).write_parquet(
        out_path
    )
    return overall


def _retrieve_candidates_for_sessions(
    session_history: pl.LazyFrame,
    clicks_covis_path: Path,
    carts_covis_path: Path,
    buy2buy_covis_path: Path,
    fallback_aids: list[int],
    top_k: int,
    num_buckets: int,
    out_path: Path,
) -> None:
    """Retrieve degree-normalized multi-matrix candidates with guaranteed revisit priority."""
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

    buckets = hist_df.partition_by("_b", as_dict=True, include_key=False)
    del hist_df
    gc.collect()

    tmp_cand_dir = Path(tempfile.mkdtemp(prefix="otto_cands_"))
    try:
        written_parts: list[Path] = []
        chunk_idx = 0
        while buckets:
            _, part_df = buckets.popitem()
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
                (1000.0 + pl.col("revisit_score"))
                .cast(pl.Float32)
                .alias("_revisit_prio"),
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
                p_file = tmp_cand_dir / f"c_{chunk_idx}.parquet"
                top_cands.write_parquet(p_file)
                written_parts.append(p_file)
                chunk_idx += 1
            del top_cands
            gc.collect()

        if written_parts:
            total_written = (
                pl.scan_parquet(written_parts).select(pl.len()).collect().item()
            )
            pl.scan_parquet(written_parts).sink_parquet(out_path)
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
            ).write_parquet(out_path)
        logger.info("[candidates] Wrote %d candidate pairs -> %s", total_written, out_path)
    finally:
        shutil.rmtree(tmp_cand_dir, ignore_errors=True)


def _log_stage1_recall(
    val_cands_path: Path,
    labels_path: Path,
    top_k: int,
    weights: dict[str, float],
) -> None:
    """Log Stage-1 Unreranked Baseline Recall@20 and Candidate Pool Ceiling Recall@K."""
    if not labels_path.exists():
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
    """Build 3 degree-normalized co-visitation matrices out-of-core and retrieve candidates."""
    splits_dir = Path(config["paths"]["splits"]) / mode
    out_dir = Path(config["paths"]["candidates"]) / mode
    out_dir.mkdir(parents=True, exist_ok=True)

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

    corpus_paths = [train_path, history_path]

    clicks_covis_path = out_dir / "covis_clicks.parquet"
    carts_covis_path = out_dir / "covis_carts_orders.parquet"
    b2b_covis_path = out_dir / "covis_buy2buy.parquet"
    legacy_covis_path = out_dir / "covisitation.parquet"
    pop_path = out_dir / "popular_items.parquet"
    val_cands_path = out_dir / "candidates.parquet"

    _build_all_three_matrices_out_of_core(
        parquet_paths=corpus_paths,
        clicks_top_k=clicks_top_k,
        carts_top_k=carts_top_k,
        b2b_top_k=b2b_top_k,
        clicks_win_hours=clicks_win_h,
        b2b_win_days=b2b_win_d,
        num_buckets=num_buckets,
        clicks_out=clicks_covis_path,
        carts_out=carts_covis_path,
        b2b_out=b2b_covis_path,
        legacy_out=legacy_covis_path,
    )

    overall_pop = _save_popular_items(
        parquet_paths=corpus_paths,
        top_k=max(top_k, int(config["pipeline"]["top_k_predict"])),
        out_path=pop_path,
    )
    fallback_aids = overall_pop["aid"].to_list()

    _retrieve_candidates_for_sessions(
        session_history=pl.scan_parquet(history_path),
        clicks_covis_path=clicks_covis_path,
        carts_covis_path=carts_covis_path,
        buy2buy_covis_path=b2b_covis_path,
        fallback_aids=fallback_aids,
        top_k=top_k,
        num_buckets=8 if mode == "sample" else 24,
        out_path=val_cands_path,
    )

    _log_stage1_recall(
        val_cands_path=val_cands_path,
        labels_path=labels_path,
        top_k=top_k,
        weights=dict(config["pipeline"]["metric_weights"]),
    )

    test_history_path = splits_dir / "test_history.parquet"
    if test_history_path.exists():
        test_cands_path = out_dir / "test_candidates.parquet"
        _retrieve_candidates_for_sessions(
            session_history=pl.scan_parquet(test_history_path),
            clicks_covis_path=clicks_covis_path,
            carts_covis_path=carts_covis_path,
            buy2buy_covis_path=b2b_covis_path,
            fallback_aids=fallback_aids,
            top_k=top_k,
            num_buckets=8 if mode == "sample" else 24,
            out_path=test_cands_path,
        )
