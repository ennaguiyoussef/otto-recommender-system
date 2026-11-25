"""Co-visitation matrix construction and candidate retrieval (Stage 1)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

# Tail cap per session prevents O(L^2) self-join blowup on scraper/bot sessions
MAX_SESSION_TAIL = 30
MAX_SOURCE_HISTORY = 20

# Event-type weights inspired by top OTTO co-visitation solutions
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


def _build_covisitation_matrix(
    events: pl.LazyFrame,
    top_k: int,
    time_window_hours: float,
    num_buckets: int,
    out_path: Path,
) -> None:
    """Build a time- and type-weighted directed co-visitation table in hash buckets."""
    ts_per_sec = _detect_ts_per_sec(events)
    window_units = int(time_window_hours * 3600 * ts_per_sec)
    inv_window = 1.0 / max(window_units, 1)

    prepared = (
        events.sort(["session", "ts"])
        .group_by("session", maintain_order=True)
        .tail(MAX_SESSION_TAIL)
        .with_columns(
            pl.col("type")
            .replace_strict(EVENT_WEIGHTS, default=1.0, return_dtype=pl.Float32)
            .alias("type_w"),
            (pl.col("session").hash(seed=0) % num_buckets).alias("_bucket"),
        )
    )

    bucket_frames: list[pl.DataFrame] = []
    for bucket_id in range(num_buckets):
        chunk = prepared.filter(pl.col("_bucket") == bucket_id).drop("_bucket")
        right = chunk.select(
            pl.col("session"),
            pl.col("aid").alias("aid_right"),
            pl.col("ts").alias("ts_right"),
            pl.col("type_w").alias("type_w_right"),
        )

        edges = (
            chunk.join(right, on="session", how="inner")
            .filter(
                (pl.col("aid") != pl.col("aid_right"))
                & ((pl.col("ts") - pl.col("ts_right")).abs() <= window_units)
            )
            .with_columns(
                (
                    pl.col("type_w_right")
                    * (
                        1.0
                        + 3.0
                        * (
                            1.0
                            - (pl.col("ts") - pl.col("ts_right")).abs().cast(pl.Float32)
                            * inv_window
                        )
                    )
                ).alias("score")
            )
            .group_by([pl.col("aid").alias("source_aid"), pl.col("aid_right").alias("aid")])
            .agg(pl.col("score").sum().cast(pl.Float32).alias("score"))
            .sort(["source_aid", "score"], descending=[False, True])
            .group_by("source_aid", maintain_order=True)
            .head(top_k * 2)
            .collect()
        )
        if len(edges) > 0:
            bucket_frames.append(edges)

    if bucket_frames:
        combined = (
            pl.concat(bucket_frames)
            .lazy()
            .group_by(["source_aid", "aid"])
            .agg(pl.col("score").sum().cast(pl.Float32).alias("score"))
            .sort(["source_aid", "score"], descending=[False, True])
            .group_by("source_aid", maintain_order=True)
            .head(top_k)
            .collect()
        )
    else:
        combined = pl.DataFrame(
            schema={"source_aid": pl.Int32, "aid": pl.Int32, "score": pl.Float32}
        )

    combined.write_parquet(out_path)
    logger.info(
        "[candidates] Saved co-visitation matrix (%d directed edges) -> %s",
        len(combined),
        out_path,
    )


def _save_popular_items(events: pl.LazyFrame, top_k: int, out_path: Path) -> pl.DataFrame:
    """Compute top popular items overall and per event type as fallback candidates."""
    by_type = (
        events.group_by(["type", "aid"])
        .agg(pl.len().alias("pop_count"))
        .sort(["type", "pop_count"], descending=[False, True])
        .group_by("type", maintain_order=True)
        .head(top_k)
        .collect()
    )
    overall = (
        events.group_by("aid")
        .agg(pl.len().alias("pop_count"))
        .sort("pop_count", descending=True)
        .head(top_k)
        .with_columns(pl.lit(255, dtype=pl.UInt8).alias("type"))
        .select("type", "aid", "pop_count")
        .collect()
    )
    pop_df = pl.concat([by_type.select("type", "aid", "pop_count"), overall])
    pop_df.write_parquet(out_path)
    return overall


def _retrieve_candidates_for_sessions(
    session_history: pl.LazyFrame,
    covis_path: Path,
    fallback_aids: list[int],
    top_k: int,
    num_buckets: int,
    out_path: Path,
) -> None:
    """Retrieve top-K candidates per session combining history, co-visitation, and popularity."""
    covis = pl.scan_parquet(covis_path)

    ranked_history = (
        session_history.sort(["session", "ts"], descending=[False, True])
        .with_columns(
            pl.int_range(0, pl.len()).over("session").alias("_recency_rank"),
            pl.col("type")
            .replace_strict(EVENT_WEIGHTS, default=1.0, return_dtype=pl.Float32)
            .alias("_type_w"),
            (pl.col("session").hash(seed=1) % num_buckets).alias("_bucket"),
        )
        .filter(pl.col("_recency_rank") < MAX_SOURCE_HISTORY)
        .with_columns(
            (pl.lit(0.85, dtype=pl.Float32).pow(pl.col("_recency_rank")) * pl.col("_type_w"))
            .cast(pl.Float32)
            .alias("_pos_w")
        )
    )

    fallback_df = pl.DataFrame(
        {
            "aid": pl.Series(fallback_aids[:top_k], dtype=pl.Int32),
            "_fb_score": pl.Series(
                [1e-4 * (top_k - i) for i in range(min(len(fallback_aids), top_k))],
                dtype=pl.Float32,
            ),
        }
    ).lazy()

    chunks: list[pl.DataFrame] = []
    for bucket_id in range(num_buckets):
        hist_b = ranked_history.filter(pl.col("_bucket") == bucket_id).drop("_bucket")

        # 1. Re-visitation candidates (items already interacted with in the session)
        revisit = hist_b.group_by(["session", "aid"]).agg(
            (pl.col("_pos_w").sum() * 20.0).cast(pl.Float32).alias("covisitation_score")
        )

        # 2. Co-visitation expansion from recent session items
        expanded = (
            hist_b.select(
                pl.col("session"),
                pl.col("aid").alias("source_aid"),
                pl.col("_pos_w"),
            )
            .join(covis, on="source_aid", how="inner")
            .group_by(["session", "aid"])
            .agg((pl.col("score") * pl.col("_pos_w")).sum().cast(pl.Float32).alias("covisitation_score"))
        )

        merged = (
            pl.concat([revisit, expanded])
            .group_by(["session", "aid"])
            .agg(pl.col("covisitation_score").sum().cast(pl.Float32).alias("covisitation_score"))
        )

        # 3. Pad sessions that still have fewer than min(20, top_k) candidates using popular items
        min_needed = min(20, top_k)
        underfilled_sessions = (
            hist_b.select("session")
            .unique()
            .join(
                merged.group_by("session").agg(pl.len().alias("_cnt")).filter(pl.col("_cnt") >= min_needed),
                on="session",
                how="anti",
            )
        )
        padded = underfilled_sessions.join(fallback_df, how="cross").select(
            pl.col("session"),
            pl.col("aid"),
            pl.col("_fb_score").alias("covisitation_score"),
        )

        top_cands = (
            pl.concat([merged, padded])
            .group_by(["session", "aid"])
            .agg(pl.col("covisitation_score").max().cast(pl.Float32).alias("covisitation_score"))
            .sort(["session", "covisitation_score"], descending=[False, True])
            .group_by("session", maintain_order=True)
            .head(top_k)
            .with_columns(
                pl.int_range(1, pl.len() + 1).over("session").cast(pl.UInt16).alias("cand_rank")
            )
            .collect()
        )
        if len(top_cands) > 0:
            chunks.append(top_cands)

    final_df = (
        pl.concat(chunks)
        if chunks
        else pl.DataFrame(
            schema={
                "session": pl.Int32,
                "aid": pl.Int32,
                "covisitation_score": pl.Float32,
                "cand_rank": pl.UInt16,
            }
        )
    )
    final_df.write_parquet(out_path)
    logger.info("[candidates] Wrote %d candidate pairs -> %s", len(final_df), out_path)


def run_candidates(config: dict[str, Any], mode: str = "sample") -> None:
    """Build co-visitation matrices and retrieve top-K item candidates per session.

    Outputs written to `<candidates>/<mode>/`:
      - `covisitation.parquet`: Directed edges `(source_aid, aid, score)`.
      - `popular_items.parquet`: Global & per-type popular items fallback.
      - `candidates.parquet`: Validation candidate pairs `(session, aid, covisitation_score, cand_rank)`.
      - `test_candidates.parquet`: Test candidate pairs (when `test_history.parquet` exists).

    Args:
        config: Loaded pipeline configuration with resolved Drive paths.
        mode: Execution mode ('sample' or 'full').
    """
    splits_dir = Path(config["paths"]["splits"]) / mode
    out_dir = Path(config["paths"]["candidates"]) / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    top_k = int(config["pipeline"]["top_k_candidates"])
    covis_cfg = config.get("covisitation", {})
    time_window_hours = float(covis_cfg.get("time_window_hours", 24))
    num_buckets = 2 if mode == "sample" else 16

    logger.info(
        "[candidates] Generating top-%d co-visitation candidates (mode=%s, buckets=%d) from '%s' -> '%s'",
        top_k,
        mode,
        num_buckets,
        splits_dir,
        out_dir,
    )

    train_path = splits_dir / "train.parquet"
    history_path = splits_dir / "history.parquet"
    if not train_path.exists() or not history_path.exists():
        raise FileNotFoundError(
            f"Missing split files in '{splits_dir}'. Run '--stage split' first."
        )

    observed_val_corpus = pl.concat(
        [pl.scan_parquet(train_path), pl.scan_parquet(history_path)]
    )

    covis_path = out_dir / "covisitation.parquet"
    pop_path = out_dir / "popular_items.parquet"
    val_cands_path = out_dir / "candidates.parquet"

    _build_covisitation_matrix(
        events=observed_val_corpus,
        top_k=top_k,
        time_window_hours=time_window_hours,
        num_buckets=num_buckets,
        out_path=covis_path,
    )
    overall_pop = _save_popular_items(
        events=observed_val_corpus,
        top_k=max(top_k, int(config["pipeline"]["top_k_predict"])),
        out_path=pop_path,
    )
    fallback_aids = overall_pop["aid"].to_list()

    _retrieve_candidates_for_sessions(
        session_history=pl.scan_parquet(history_path),
        covis_path=covis_path,
        fallback_aids=fallback_aids,
        top_k=top_k,
        num_buckets=max(1, num_buckets // 2),
        out_path=val_cands_path,
    )

    test_history_path = splits_dir / "test_history.parquet"
    if test_history_path.exists():
        test_cands_path = out_dir / "test_candidates.parquet"
        _retrieve_candidates_for_sessions(
            session_history=pl.scan_parquet(test_history_path),
            covis_path=covis_path,
            fallback_aids=fallback_aids,
            top_k=top_k,
            num_buckets=max(1, num_buckets // 2),
            out_path=test_cands_path,
        )
