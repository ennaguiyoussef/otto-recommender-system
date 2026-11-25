"""Official OTTO weighted Recall@20 evaluation metric."""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: dict[str, float] = {
    "clicks": 0.10,
    "carts": 0.30,
    "orders": 0.60,
}

CODE_TO_TYPE: dict[int, str] = {
    0: "clicks",
    1: "carts",
    2: "orders",
}


def _normalize_type_expr(frame: pl.LazyFrame) -> pl.Expr:
    """Ensure the 'type' column is represented as string event names."""
    dtype = frame.collect_schema()["type"]
    if dtype.is_integer():
        return pl.col("type").replace_strict(CODE_TO_TYPE, default="clicks")
    return pl.col("type").cast(pl.Utf8)


def compute_recall_at_k(
    predictions: pl.DataFrame,
    ground_truth: pl.DataFrame,
    weights: dict[str, float] | None = None,
    k: int = 20,
) -> dict[str, float]:
    """Compute per-objective Recall@K and the weighted OTTO competition score.

    Args:
        predictions: Polars DataFrame with columns ['session', 'type', 'labels'].
        ground_truth: Polars DataFrame with columns ['session', 'type', 'ground_truth'].
        weights: Objective weights mapping ('clicks': 0.1, 'carts': 0.3, 'orders': 0.6).
        k: Cutoff rank for recall calculation (default: 20).

    Returns:
        Dictionary with recall scores for 'clicks', 'carts', 'orders', and 'overall'.
    """
    active_weights = weights or DEFAULT_WEIGHTS
    logger.info("[metrics] Computing weighted Recall@%d across %s", k, list(active_weights))

    gt_lazy = ground_truth.lazy().with_columns(
        _normalize_type_expr(ground_truth.lazy()).alias("type"),
        pl.col("ground_truth").list.unique().alias("ground_truth"),
    )
    pred_lazy = predictions.lazy().select(
        pl.col("session"),
        _normalize_type_expr(predictions.lazy()).alias("type"),
        pl.col("labels").list.unique(maintain_order=True).list.head(k).alias("labels"),
    )

    hits_expr = (
        pl.when(pl.col("labels").is_null())
        .then(pl.lit(0, dtype=pl.UInt64))
        .otherwise(
            pl.col("labels")
            .list.set_intersection(pl.col("ground_truth"))
            .list.len()
            .cast(pl.UInt64)
        )
    )
    possible_expr = (
        pl.col("ground_truth").list.len().clip(upper_bound=k).cast(pl.UInt64)
    )

    totals = (
        gt_lazy.join(pred_lazy, on=["session", "type"], how="left")
        .group_by("type")
        .agg(
            hits_expr.sum().alias("hits"),
            possible_expr.sum().alias("possible"),
        )
        .collect()
    )

    scores: dict[str, float] = dict.fromkeys(DEFAULT_WEIGHTS, 0.0)
    for event_type, hits, possible in totals.iter_rows():
        scores[str(event_type)] = float(hits) / float(possible) if possible else 0.0

    scores["overall"] = sum(
        weight * scores.get(event, 0.0) for event, weight in active_weights.items()
    )
    return scores
