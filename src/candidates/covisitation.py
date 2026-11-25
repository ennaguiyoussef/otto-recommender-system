"""3-Matrix Co-visitation construction and candidate retrieval (Stage 1).

Implements the high-performing OTTO 3-matrix co-visitation architecture:
  1. `covis_clicks.parquet`: 24h proximity-weighted item-to-item matrix (deduplicated per session).
  2. `covis_carts_orders.parquet`: 24h type- and direction-weighted matrix (`carts=6, orders=3`).
  3. `covis_buy2buy.parquet`: 14-day carts/orders-only co-purchase matrix.
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
MAX_SOURCE_ITEMS = 20

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
    """Build `covis_clicks` and `covis_carts_orders` in a single scan with per-session dedup."""
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

    click_parts: list[pl.DataFrame] = []
    cart_parts: list[pl.DataFrame] = []

    for idx, chunk_df in enumerate(partitions.values(), start=1):
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
                .then(pl.lit(1.5, dtype=pl.Float32))
                .otherwise(pl.lit(1.0, dtype=pl.Float32))
                .alias("_dir_w"),
            )
            .with_columns(
                (1.0 + 3.0 * (1.0 - pl.col("_rel_dt"))).cast(pl.Float32).alias("w_click"),
                (
                    pl.col("type_w_right")
                    * pl.col("_dir_w")
                    * (1.0 + 2.0 * (1.0 - pl.col("_rel_dt")))
                )
                .cast(pl.Float32)
                .alias("w_cart"),
            )
            # Deduplicate per session so a single bot/repeat session gets at most 1 vote per pair
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

        if len(pairs) > 0:
            c_top = (
                pairs.lazy()
                .select("source_aid", "aid", pl.col("score_clicks").alias("score"))
                .sort(["source_aid", "score"], descending=[False, True])
                .group_by("source_aid", maintain_order=True)
                .head(clicks_top_k * 2)
                .collect()
            )
            o_top = (
                pairs.lazy()
                .select("source_aid", "aid", pl.col("score_carts").alias("score"))
                .sort(["source_aid", "score"], descending=[False, True])
                .group_by("source_aid", maintain_order=True)
                .head(carts_top_k * 2)
                .collect()
            )
            click_parts.append(c_top)
            cart_parts.append(o_top)

        del pairs
        if idx % 4 == 0:
            logger.info("[candidates] Processed bucket %d/%d", idx, len(partitions))

    del partitions
    gc.collect()

    empty_schema = {"source_aid": pl.Int32, "aid": pl.Int32, "score": pl.Float32}

    clicks_df = (
        pl.concat(click_parts)
        .lazy()
        .group_by(["source_aid", "aid"])
        .agg(pl.col("score").sum().cast(pl.Float32).alias("score"))
        .sort(["source_aid", "score"], descending=[False, True])
        .group_by("source_aid", maintain_order=True)
        .head(clicks_top_k)
        .collect()
        if click_parts
        else pl.DataFrame(schema=empty_schema)
    )
    clicks_df.write_parquet(clicks_out)
    logger.info(
        "[candidates] Saved clicks co-visitation (%d edges) -> %s",
        len(clicks_df),
        clicks_out,
    )
    del click_parts, clicks_df
    gc.collect()

    carts_df = (
        pl.concat(cart_parts)
        .lazy()
        .group_by(["source_aid", "aid"])
        .agg(pl.col("score").sum().cast(pl.Float32).alias("score"))
        .sort(["source_aid", "score"], descending=[False, True])
        .group_by("source_aid", maintain_order=True)
        .head(carts_top_k)
        .collect()
        if cart_parts
        else pl.DataFrame(schema=empty_schema)
    )
    carts_df.write_parquet(carts_out)
    carts_df.write_parquet(legacy_out)
    logger.info(
        "[candidates] Saved carts/orders co-visitation (%d edges) -> %s",
        len(carts_df),
        carts_out,
    )


def _build_buy2buy_matrix(
    events: pl.LazyFrame,
    top_k: int,
    window_days: float,
    out_path: Path,
) -> None:
    """Build 14-day carts/orders -> carts/orders (`buy2buy`) matrix."""
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
        pl.DataFrame(
            schema={"source_aid": pl.Int32, "aid": pl.Int32, "score": pl.Float32}
        ).write_parquet(out_path)
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

    buy_df = (
        pl.concat(buy_edges)
        .lazy()
        .group_by(["source_aid", "aid"])
        .agg(pl.col("score").sum().cast(pl.Float32).alias("score"))
        .sort(["source_aid", "score"], descending=[False, True])
        .group_by("source_aid", maintain_order=True)
        .head(top_k)
        .collect()
        if buy_edges
        else pl.DataFrame(
            schema={"source_aid": pl.Int32, "aid": pl.Int32, "score": pl.Float32}
        )
    )
    buy_df.write_parquet(out_path)
    logger.info(
        "[candidates] Saved buy2buy co-visitation (%d edges) -> %s",
        len(buy_df),
        out_path,
    )


def _save_popular_items(events: pl.LazyFrame, top_k: int, out_path: Path) -> pl.DataFrame:
    """Compute top popular items (preferring recent 7-day window) as fallback."""
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
    """Retrieve multi-matrix top-K candidates per session with separate matrix scores."""
    covis_clicks = pl.scan_parquet(clicks_covis_path)
    covis_carts = pl.scan_parquet(carts_covis_path)
    covis_b2b = pl.scan_parquet(buy2buy_covis_path)

    # Deduplicate session history keeping each item's latest interaction and total strength
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
                [1e-4 * (20 - i) for i in range(min(len(fallback_aids), 20, top_k))],
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
                (
                    pl.lit(0.80, dtype=pl.Float32).pow(pl.col("_urev"))
                    * pl.when(pl.col("max_type") >= 1)
                    .then(pl.lit(2.0, dtype=pl.Float32))
                    .otherwise(pl.lit(1.0, dtype=pl.Float32))
                )
                .cast(pl.Float32)
                .alias("_src_w"),
                pl.when(pl.col("max_type") >= 1)
                .then(pl.lit(3.0, dtype=pl.Float32))
                .otherwise(pl.lit(0.5, dtype=pl.Float32))
                .alias("_b2b_mult"),
            )
            .select(
                pl.col("session"),
                pl.col("aid").alias("source_aid"),
                pl.col("_src_w"),
                pl.col("_b2b_mult"),
            )
        )

        # 1. In-session history items (guaranteed strong priority)
        revisit = hb.select(
            pl.col("session"),
            pl.col("aid"),
            (pl.col("revisit_score") * 50.0).cast(pl.Float32).alias("_revisit_boost"),
            pl.lit(0.0, dtype=pl.Float32).alias("score_clicks"),
            pl.lit(0.0, dtype=pl.Float32).alias("score_carts_orders"),
            pl.lit(0.0, dtype=pl.Float32).alias("score_buy2buy"),
            pl.lit(0, dtype=pl.UInt16).alias("covis_support"),
        )

        # 2. Clicks co-visitation candidates
        exp_clicks = (
            recent_src.join(covis_clicks, on="source_aid", how="inner")
            .group_by(["session", "aid"])
            .agg(
                pl.lit(0.0, dtype=pl.Float32).alias("_revisit_boost"),
                (pl.col("score") * pl.col("_src_w")).sum().cast(pl.Float32).alias("score_clicks"),
                pl.lit(0.0, dtype=pl.Float32).alias("score_carts_orders"),
                pl.lit(0.0, dtype=pl.Float32).alias("score_buy2buy"),
                pl.len().cast(pl.UInt16).alias("covis_support"),
            )
        )

        # 3. Carts/Orders type-weighted co-visitation candidates
        exp_carts = (
            recent_src.join(covis_carts, on="source_aid", how="inner")
            .group_by(["session", "aid"])
            .agg(
                pl.lit(0.0, dtype=pl.Float32).alias("_revisit_boost"),
                pl.lit(0.0, dtype=pl.Float32).alias("score_clicks"),
                (pl.col("score") * pl.col("_src_w"))
                .sum()
                .cast(pl.Float32)
                .alias("score_carts_orders"),
                pl.lit(0.0, dtype=pl.Float32).alias("score_buy2buy"),
                pl.len().cast(pl.UInt16).alias("covis_support"),
            )
        )

        # 4. Buy2Buy 14-day co-visitation candidates
        exp_b2b = (
            recent_src.join(covis_b2b, on="source_aid", how="inner")
            .group_by(["session", "aid"])
            .agg(
                pl.lit(0.0, dtype=pl.Float32).alias("_revisit_boost"),
                pl.lit(0.0, dtype=pl.Float32).alias("score_clicks"),
                pl.lit(0.0, dtype=pl.Float32).alias("score_carts_orders"),
                (pl.col("score") * pl.col("_src_w") * pl.col("_b2b_mult"))
                .sum()
                .cast(pl.Float32)
                .alias("score_buy2buy"),
                pl.len().cast(pl.UInt16).alias("covis_support"),
            )
        )

        merged = (
            pl.concat([revisit, exp_clicks, exp_carts, exp_b2b])
            .group_by(["session", "aid"])
            .agg(
                pl.col("_revisit_boost").sum().alias("_revisit_boost"),
                pl.col("score_clicks").sum().cast(pl.Float32).alias("score_clicks"),
                pl.col("score_carts_orders").sum().cast(pl.Float32).alias("score_carts_orders"),
                pl.col("score_buy2buy").sum().cast(pl.Float32).alias("score_buy2buy"),
                pl.col("covis_support").sum().cast(pl.UInt16).alias("covis_support"),
            )
            .with_columns(
                (
                    pl.col("_revisit_boost")
                    + pl.col("score_clicks") * 1.0
                    + pl.col("score_carts_orders") * 0.6
                    + pl.col("score_buy2buy") * 1.5
                )
                .cast(pl.Float32)
                .alias("covisitation_score")
            )
            .drop("_revisit_boost")
        )

        # Pad underfilled sessions (< 20 candidates) with popular items
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
            pl.lit(0, dtype=pl.UInt16).alias("covis_support"),
            pl.col("_fb_score").alias("covisitation_score"),
        )

        top_cands = (
            pl.concat([merged, padded])
            .group_by(["session", "aid"])
            .agg(
                pl.col("score_clicks").max().alias("score_clicks"),
                pl.col("score_carts_orders").max().alias("score_carts_orders"),
                pl.col("score_buy2buy").max().alias("score_buy2buy"),
                pl.col("covis_support").max().alias("covis_support"),
                pl.col("covisitation_score").max().alias("covisitation_score"),
            )
            .sort(["session", "covisitation_score"], descending=[False, True])
            .group_by("session", maintain_order=True)
            .head(top_k)
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
    """Log Stage-1 Candidate Pool Recall@K and Unreranked Baseline Recall@20."""
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
    """Build 3 co-visitation matrices and retrieve top-K item candidates per session."""
    splits_dir = Path(config["paths"]["splits"]) / mode
    out_dir = Path(config["paths"]["candidates"]) / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    top_k = int(config["pipeline"]["top_k_candidates"])
    covis_cfg = config.get("covisitation", {})
    clicks_top_k = int(covis_cfg.get("clicks_top_k", 40))
    carts_top_k = int(covis_cfg.get("carts_orders_top_k", 40))
    b2b_top_k = int(covis_cfg.get("buy2buy_top_k", 40))
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
