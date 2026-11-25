"""Degree-normalized 3-Matrix Co-visitation & Multi-Signal Candidate Retrieval (Stage 1).

Key architectural principles for >= 0.56–0.58+ Recall@20 on OTTO:
  1. Per-session pair deduplication so bot/repeat sessions get at most 1 vote per item pair.
  2. Conditional probability normalization `P(aid | source_aid)` + reciprocal rank weighting
     on all 3 matrices (`clicks`, `carts_orders`, `buy2buy`) to eliminate popular-hub bias.
  3. Guaranteed top priority for in-session history items + steep recency decay (`0.60^rank`)
     and dedicated last-item (`_urev == 0`) transition scores.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import polars as pl

from src.metrics.recall import CODE_TO_TYPE, compute_recall_at_k

logger = logging.getLogger(__name__)

MAX_SESSION_TAIL = 30
MAX_BUY_TAIL = 20
MAX_SOURCE_ITEMS = 15

EVENT_WEIGHTS: dict[int, float] = {
    0: 1.0,  # clicks
    1: 6.0,  # carts
    2: 3.0,  # orders
}


def _detect_ts_per_sec(events: pl.LazyFrame) -> int:
    """Detect whether timestamps are in milliseconds or seconds."""
    max_ts = events.select(pl.col("ts").max()).collect().item()
    if max_ts is None:
        return 1
    return 1000 if int(max_ts) > 10_000_000_000 else 1


def _compact_edge_parts(parts: list[pl.DataFrame], keep_k: int) -> list[pl.DataFrame]:
    """Merge accumulated bucket edge frames in-place to bound peak RAM."""
    if len(parts) <= 1:
        return parts
    merged = (
        pl.concat(parts)
        .lazy()
        .group_by(["source_aid", "aid"])
        .agg(pl.col("score").sum().cast(pl.Float32).alias("score"))
        .sort(["source_aid", "score"], descending=[False, True])
        .group_by("source_aid", maintain_order=True)
        .head(keep_k)
        .collect()
    )
    parts.clear()
    gc.collect()
    return [merged]


def _finalize_normalized_matrix(
    parts: list[pl.DataFrame],
    top_k: int,
    out_path: Path,
    label: str,
) -> None:
    """Aggregate bucket edges and compute degree-normalized `norm_score` without `.over()` windows."""
    empty_schema = {
        "source_aid": pl.Int32,
        "aid": pl.Int32,
        "score": pl.Float32,
        "norm_score": pl.Float32,
    }
    if not parts:
        pl.DataFrame(schema=empty_schema).write_parquet(out_path)
        return

    logger.info("[candidates] Finalizing & normalizing '%s' co-visitation matrix...", label)
    df = (
        pl.concat(parts)
        .lazy()
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
            pl.int_ranges(1, pl.col("aid").list.len() + 1, dtype=pl.UInt16).alias("_rk"),
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
    parts.clear()
    gc.collect()
    df.write_parquet(out_path)
    logger.info(
        "[candidates] Saved %s co-visitation (%d normalized edges) -> %s",
        label,
        len(df),
        out_path,
    )


def _build_clicks_and_carts_matrices(
    events: pl.LazyFrame,
    clicks_top_k: int,
    carts_top_k: int,
    window_hours: float,
    num_buckets: int,
    clicks_out: Path,
    carts_out: Path,
    legacy_out: Path,
) -> None:
    """Build `covis_clicks` and `covis_carts_orders` in one pass with per-session dedup."""
    ts_per_sec = _detect_ts_per_sec(events)
    window_units = int(window_hours * 3600 * ts_per_sec)
    inv_window = 1.0 / max(window_units, 1)

    logger.info(
        "[candidates] Scanning corpus once & partitioning into %d RAM buckets...",
        num_buckets,
    )
    tail_df = (
        events.sort(["session", "ts"])
        .group_by("session", maintain_order=True)
        .tail(MAX_SESSION_TAIL)
        .select(
            pl.col("session").cast(pl.Int32),
            pl.col("aid").cast(pl.Int32),
            pl.col("ts").cast(pl.Int64),
            pl.col("type")
            .replace_strict(EVENT_WEIGHTS, default=1.0, return_dtype=pl.Float32)
            .alias("type_w"),
            (pl.col("session").hash(seed=0) % num_buckets).cast(pl.UInt8).alias("_b"),
        )
        .collect()
    )

    partitions = tail_df.partition_by("_b", as_dict=True, include_key=False)
    del tail_df
    gc.collect()

    total_buckets = len(partitions)
    click_parts: list[pl.DataFrame] = []
    cart_parts: list[pl.DataFrame] = []

    for idx in range(1, total_buckets + 1):
        _, chunk_df = partitions.popitem()
        chunk = chunk_df.lazy()
        right = chunk.select(
            pl.col("session"),
            pl.col("aid").alias("aid_right"),
            pl.col("ts").alias("ts_right"),
            pl.col("type_w").alias("type_w_right"),
        )

        pairs = (
            chunk.join(right, on="session", how="inner")
            .filter(
                (pl.col("aid") != pl.col("aid_right"))
                & ((pl.col("ts") - pl.col("ts_right")).abs() <= window_units)
            )
            .with_columns(
                ((pl.col("ts") - pl.col("ts_right")).abs().cast(pl.Float32) * inv_window).alias(
                    "_rel_dt"
                ),
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
            .group_by([pl.col("aid").alias("source_aid"), pl.col("aid_right").alias("aid")])
            .agg(
                pl.col("w_click").sum().cast(pl.Float32).alias("score_clicks"),
                pl.col("w_cart").sum().cast(pl.Float32).alias("score_carts"),
            )
            .collect()
        )
        del chunk_df

        if len(pairs) > 0:
            c_top = (
                pairs.lazy()
                .select("source_aid", "aid", pl.col("score_clicks").alias("score"))
                .sort(["source_aid", "score"], descending=[False, True])
                .group_by("source_aid", maintain_order=True)
                .head(clicks_top_k + 20)
                .collect()
            )
            o_top = (
                pairs.lazy()
                .select("source_aid", "aid", pl.col("score_carts").alias("score"))
                .sort(["source_aid", "score"], descending=[False, True])
                .group_by("source_aid", maintain_order=True)
                .head(carts_top_k + 20)
                .collect()
            )
            click_parts.append(c_top)
            cart_parts.append(o_top)

        del pairs
        if idx % 4 == 0:
            click_parts = _compact_edge_parts(click_parts, clicks_top_k + 25)
            cart_parts = _compact_edge_parts(cart_parts, carts_top_k + 25)
            logger.info("[candidates] Processed & compacted bucket %d/%d", idx, total_buckets)

    del partitions
    gc.collect()

    _finalize_normalized_matrix(click_parts, clicks_top_k, clicks_out, "clicks")
    del click_parts
    gc.collect()

    _finalize_normalized_matrix(cart_parts, carts_top_k, carts_out, "carts/orders")
    shutil_copy = pl.read_parquet(carts_out)
    shutil_copy.write_parquet(legacy_out)
    del cart_parts, shutil_copy
    gc.collect()


def _build_buy2buy_matrix(
    events: pl.LazyFrame,
    top_k: int,
    window_days: float,
    out_path: Path,
) -> None:
    """Build 14-day carts/orders -> carts/orders (`buy2buy`) normalized matrix."""
    ts_per_sec = _detect_ts_per_sec(events)
    window_units = int(window_days * 86_400 * ts_per_sec)

    buys = (
        events.filter(pl.col("type").is_in([1, 2]))
        .sort(["session", "ts"])
        .group_by("session", maintain_order=True)
        .tail(MAX_BUY_TAIL)
        .select(
            pl.col("session").cast(pl.Int32),
            pl.col("aid").cast(pl.Int32),
            pl.col("ts").cast(pl.Int64),
            pl.when(pl.col("type") == 2)
            .then(pl.lit(1.5, dtype=pl.Float32))
            .otherwise(pl.lit(1.0, dtype=pl.Float32))
            .alias("buy_w"),
            (pl.col("session").hash(seed=7) % 4).cast(pl.UInt8).alias("_b"),
        )
        .collect()
    )

    if len(buys) == 0:
        _finalize_normalized_matrix([], top_k, out_path, "buy2buy")
        return

    parts = buys.partition_by("_b", as_dict=True, include_key=False)
    del buys
    buy_edges: list[pl.DataFrame] = []

    for part_df in parts.values():
        lf = part_df.lazy()
        right = lf.select(
            pl.col("session"),
            pl.col("aid").alias("aid_right"),
            pl.col("ts").alias("ts_right"),
            pl.col("buy_w").alias("buy_w_right"),
        )
        e = (
            lf.join(right, on="session", how="inner")
            .filter(
                (pl.col("aid") != pl.col("aid_right"))
                & ((pl.col("ts") - pl.col("ts_right")).abs() <= window_units)
            )
            .group_by(["session", "aid", "aid_right"])
            .agg(pl.col("buy_w_right").max().alias("w"))
            .group_by([pl.col("aid").alias("source_aid"), pl.col("aid_right").alias("aid")])
            .agg(pl.col("w").sum().cast(pl.Float32).alias("score"))
            .collect()
        )
        if len(e) > 0:
            buy_edges.append(e)

    _finalize_normalized_matrix(buy_edges, top_k, out_path, "buy2buy")


def _save_popular_items(events: pl.LazyFrame, top_k: int, out_path: Path) -> pl.DataFrame:
    """Compute top popular items in the recent 7-day window as fallback."""
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
    pl.concat([by_type.select("type", "aid", "pop_count"), overall]).write_parquet(out_path)
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

    out_chunks: list[pl.DataFrame] = []
    for part_df in buckets.values():
        hb = part_df.lazy()
        recent_src = (
            hb.filter(pl.col("_urev") < MAX_SOURCE_ITEMS)
            .with_columns(
                # Steep recency decay (0.58^rank) for immediate next-click transitions
                pl.lit(0.58, dtype=pl.Float32).pow(pl.col("_urev")).cast(pl.Float32).alias("_click_src_w"),
                # Moderate recency decay (0.75^rank) * intent multiplier for carts/orders
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

        # 1. In-session unique items: offset +1000 guarantees 100% of recent session items stay in pool
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

        # 2. Normalized Clicks co-visitation + exact transition from the session's VERY LAST item
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

        # 3. Normalized Carts/Orders co-visitation
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

        # 4. Normalized Buy2Buy 14-day co-visitation
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
                pl.col("score_carts_orders").sum().cast(pl.Float32).alias("score_carts_orders"),
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
            .head(top_k)
            .drop("_pool_rank_score")
            .with_columns(
                pl.int_range(1, pl.len() + 1).over("session").cast(pl.UInt16).alias("cand_rank")
            )
            .collect()
        )
        if len(top_cands) > 0:
            out_chunks.append(top_cands)

    final_df = (
        pl.concat(out_chunks)
        if out_chunks
        else pl.DataFrame(
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
        )
    )
    final_df.write_parquet(out_path)
    logger.info("[candidates] Wrote %d candidate pairs -> %s", len(final_df), out_path)


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
    """Build 3 degree-normalized co-visitation matrices and retrieve top-K candidates."""
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
    num_buckets = 12

    train_path = splits_dir / "train.parquet"
    history_path = splits_dir / "history.parquet"
    labels_path = splits_dir / "labels.parquet"
    if not train_path.exists() or not history_path.exists():
        raise FileNotFoundError(
            f"Missing split files in '{splits_dir}'. Run '--stage split' first."
        )

    observed_val_corpus = pl.concat(
        [pl.scan_parquet(train_path), pl.scan_parquet(history_path)]
    )

    clicks_covis_path = out_dir / "covis_clicks.parquet"
    carts_covis_path = out_dir / "covis_carts_orders.parquet"
    b2b_covis_path = out_dir / "covis_buy2buy.parquet"
    legacy_covis_path = out_dir / "covisitation.parquet"
    pop_path = out_dir / "popular_items.parquet"
    val_cands_path = out_dir / "candidates.parquet"

    _build_clicks_and_carts_matrices(
        events=observed_val_corpus,
        clicks_top_k=clicks_top_k,
        carts_top_k=carts_top_k,
        window_hours=clicks_win_h,
        num_buckets=num_buckets,
        clicks_out=clicks_covis_path,
        carts_out=carts_covis_path,
        legacy_out=legacy_covis_path,
    )
    _build_buy2buy_matrix(
        events=observed_val_corpus,
        top_k=b2b_top_k,
        window_days=b2b_win_d,
        out_path=b2b_covis_path,
    )

    overall_pop = _save_popular_items(
        events=observed_val_corpus,
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
        num_buckets=4 if mode == "sample" else 16,
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
            num_buckets=4 if mode == "sample" else 16,
            out_path=test_cands_path,
        )
