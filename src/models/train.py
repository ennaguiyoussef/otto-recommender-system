"""LightGBM reranker model training for clicks, carts, and orders."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from src.metrics.recall import CODE_TO_TYPE, compute_recall_at_k

logger = logging.getLogger(__name__)

NON_FEATURE_COLS = {
    "session",
    "aid",
    "label_clicks",
    "label_carts",
    "label_orders",
}

EVENT_TO_CODE: dict[str, int] = {
    "clicks": 0,
    "carts": 1,
    "orders": 2,
}


def _load_fallback_by_event(candidates_dir: Path, top_k: int) -> dict[str, list[int]]:
    """Load top popular items per event type to pad short candidate lists."""
    pop_path = candidates_dir / "popular_items.parquet"
    if not pop_path.exists():
        return {event: [] for event in EVENT_TO_CODE}

    pop_df = pl.read_parquet(pop_path)
    overall = pop_df.filter(pl.col("type") == 255)["aid"].to_list()
    result: dict[str, list[int]] = {}
    for event, code in EVENT_TO_CODE.items():
        typed = pop_df.filter(pl.col("type") == code)["aid"].to_list()
        seen = set(typed)
        combined = typed + [a for a in overall if a not in seen]
        result[event] = combined[:top_k]
    return result


def _pad_predictions(
    ranked: pl.DataFrame,
    fallback: list[int],
    k: int,
) -> pl.DataFrame:
    """Ensure each session's predicted label list has up to `k` unique items."""
    if not fallback:
        return ranked.with_columns(pl.col("labels").list.head(k))

    padded_lists: list[list[int]] = []
    for row_labels in ranked["labels"].to_list():
        items = list(row_labels)
        if len(items) < k:
            existing = set(items)
            for fb_aid in fallback:
                if fb_aid not in existing:
                    items.append(fb_aid)
                    existing.add(fb_aid)
                    if len(items) >= k:
                        break
        padded_lists.append(items[:k])

    return ranked.with_columns(pl.Series("labels", padded_lists, dtype=pl.List(pl.Int32)))


def _prepare_ground_truth(labels_path: Path, eval_sessions: pl.DataFrame) -> pl.DataFrame:
    """Aggregate `(session, aid, type)` rows into `(session, type, ground_truth)` lists."""
    lf = pl.scan_parquet(labels_path)
    schema = lf.collect_schema()
    if schema["type"].is_integer():
        type_expr = pl.col("type").replace_strict(CODE_TO_TYPE, default="clicks")
    else:
        type_expr = pl.col("type").cast(pl.Utf8)

    return (
        lf.join(eval_sessions.lazy(), on="session", how="semi")
        .with_columns(
            pl.col("aid").cast(pl.Int32),
            type_expr.alias("type"),
        )
        .group_by(["session", "type"])
        .agg(pl.col("aid").unique().alias("ground_truth"))
        .collect()
    )


def run_train(config: dict[str, Any], mode: str = "sample") -> None:
    """Train per-objective LightGBM rankers and evaluate local validation Recall@20.

    Saves `<models>/<mode>/lgbm_<event>.txt` and `<models>/<mode>/model_meta.json`.

    Args:
        config: Loaded pipeline configuration with resolved Drive paths.
        mode: Execution mode ('sample' or 'full').
    """
    features_path = Path(config["paths"]["features"]) / mode / "features.parquet"
    labels_path = Path(config["paths"]["splits"]) / mode / "labels.parquet"
    candidates_dir = Path(config["paths"]["candidates"]) / mode
    models_dir = Path(config["paths"]["models"]) / mode
    models_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features table not found at '{features_path}'. Run '--stage features' first."
        )

    event_types: list[str] = list(config["pipeline"]["event_types"])
    top_k_predict = int(config["pipeline"]["top_k_predict"])
    metric_weights: dict[str, float] = dict(config["pipeline"]["metric_weights"])
    seed = int(config["pipeline"]["seed"])

    raw_lgb_params: dict[str, Any] = dict(config["lightgbm"])
    early_stopping_rounds = int(raw_lgb_params.pop("early_stopping_rounds", 50))

    logger.info(
        "[train] Loading validation candidate features from '%s'",
        features_path,
    )
    df = pl.read_parquet(features_path).sort(["session", "cand_rank"])
    if len(df) == 0:
        raise ValueError(f"Feature table '{features_path}' is empty.")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    logger.info(
        "[train] Loaded %d rows across %d sessions with %d features: %s",
        len(df),
        df["session"].n_unique(),
        len(feature_cols),
        feature_cols,
    )

    # Deterministic 80/20 session split for ranker fitting vs out-of-sample Recall@20 eval
    unique_sessions = df.select("session").unique(maintain_order=True)
    if len(unique_sessions) >= 5:
        val_mask = (pl.col("session").hash(seed=seed) % 5) == 0
        train_df = df.filter(~val_mask).sort("session")
        eval_df = df.filter(val_mask).sort("session")
        if len(train_df) == 0 or len(eval_df) == 0:
            cut = max(1, int(len(unique_sessions) * 0.8))
            train_sess = unique_sessions.head(cut)
            train_df = df.join(train_sess, on="session", how="semi").sort("session")
            eval_df = df.join(train_sess, on="session", how="anti").sort("session")
    else:
        cut = max(1, len(unique_sessions) - 1)
        train_sess = unique_sessions.head(cut)
        train_df = df.join(train_sess, on="session", how="semi").sort("session")
        eval_df = (
            df.join(train_sess, on="session", how="anti").sort("session")
            if len(unique_sessions) > 1
            else train_df
        )

    train_groups = (
        train_df.group_by("session", maintain_order=True).len()["len"].to_numpy()
    )
    eval_groups = (
        eval_df.group_by("session", maintain_order=True).len()["len"].to_numpy()
    )

    x_train = train_df.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    x_eval = eval_df.select(feature_cols).to_numpy().astype(np.float32, copy=False)

    fallbacks = _load_fallback_by_event(candidates_dir, top_k_predict)
    pred_frames: list[pl.DataFrame] = []

    for event in event_types:
        label_col = f"label_{event}"
        y_train = train_df[label_col].to_numpy()
        y_eval = eval_df[label_col].to_numpy()

        params = dict(raw_lgb_params)
        if len(train_df) < int(params.get("min_child_samples", 50)) * 2:
            params["min_child_samples"] = max(1, len(train_df) // 4)

        logger.info(
            "[train] Fitting LightGBM ranker for '%s' (positives: train=%d, eval=%d)...",
            event,
            int(y_train.sum()),
            int(y_eval.sum()),
        )

        ranker = lgb.LGBMRanker(**params)
        ranker.fit(
            x_train,
            y_train,
            group=train_groups,
            eval_set=[(x_eval, y_eval)],
            eval_group=[eval_groups],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        model_path = models_dir / f"lgbm_{event}.txt"
        ranker.booster_.save_model(str(model_path))
        logger.info(
            "[train] Saved '%s' model (best_iteration=%s) -> %s",
            event,
            ranker.best_iteration_,
            model_path,
        )

        scores = ranker.predict(x_eval)
        ranked = (
            eval_df.select(
                pl.col("session").cast(pl.Int32),
                pl.col("aid").cast(pl.Int32),
            )
            .with_columns(pl.Series("_score", scores, dtype=pl.Float64))
            .sort(["session", "_score"], descending=[False, True])
            .group_by("session", maintain_order=True)
            .agg(pl.col("aid").head(top_k_predict).alias("labels"))
            .with_columns(pl.lit(event).alias("type"))
        )
        ranked = _pad_predictions(ranked, fallbacks.get(event, []), top_k_predict)
        pred_frames.append(ranked.select("session", "type", "labels"))

    predictions_df = pl.concat(pred_frames)
    eval_sessions = eval_df.select("session").unique()
    ground_truth_df = _prepare_ground_truth(labels_path, eval_sessions)

    recall_scores = compute_recall_at_k(
        predictions=predictions_df,
        ground_truth=ground_truth_df,
        weights=metric_weights,
        k=top_k_predict,
    )
    logger.info(
        "[train] Validation Recall@%d | clicks=%.5f | carts=%.5f | orders=%.5f | OVERALL=%.5f",
        top_k_predict,
        recall_scores["clicks"],
        recall_scores["carts"],
        recall_scores["orders"],
        recall_scores["overall"],
    )

    meta_path = models_dir / "model_meta.json"
    meta_payload = {
        "mode": mode,
        "feature_cols": feature_cols,
        "event_types": event_types,
        "top_k_predict": top_k_predict,
        "validation_recall": recall_scores,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    logger.info("[train] Saved model metadata -> %s", meta_path)
