"""Vectorized LightGBM + GPU XGBoost reranker inference and submission export."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from src.models.train import (
    _detect_cuda_gpu,
    _load_fallback_by_event,
    _pad_predictions_vectorized,
    _zscore_per_session,
)

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

BATCH_SESSIONS = 150_000


def run_predict(config: dict[str, Any], mode: str = "sample") -> None:
    """Score test candidates with trained models and write Kaggle submission CSV."""
    features_dir = Path(config["paths"]["features"]) / mode
    candidates_dir = Path(config["paths"]["candidates"]) / mode
    models_dir = Path(config["paths"]["models"]) / mode
    submissions_dir = Path(config["paths"]["submissions"])
    submissions_dir.mkdir(parents=True, exist_ok=True)

    top_k_predict = int(config["pipeline"]["top_k_predict"])
    meta_path = models_dir / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found at '{meta_path}'. Run '--stage train' first."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols: list[str] = meta["feature_cols"]
    event_types: list[str] = meta["event_types"]
    blend_weights: dict[str, dict[str, float]] = meta.get("blend_weights", {})
    use_xgb = bool(meta.get("xgb_enabled", False)) and (xgb is not None)
    has_cuda = _detect_cuda_gpu()

    lgb_boosters: dict[str, lgb.Booster] = {}
    xgb_boosters: dict[str, Any] = {}

    for event in event_types:
        lgb_file = models_dir / f"lgbm_{event}.txt"
        if not lgb_file.exists():
            raise FileNotFoundError(f"Missing trained LightGBM model: {lgb_file}")
        lgb_boosters[event] = lgb.Booster(model_file=str(lgb_file))

        xgb_file = models_dir / f"xgb_{event}.json"
        if use_xgb and xgb_file.exists() and xgb is not None:
            booster = xgb.Booster()
            booster.load_model(str(xgb_file))
            booster.set_param({"device": "cuda" if has_cuda else "cpu"})
            xgb_boosters[event] = booster

    test_features_path = features_dir / "test_features.parquet"
    val_features_path = features_dir / "features.parquet"
    if test_features_path.exists():
        active_features_path = test_features_path
    elif val_features_path.exists():
        logger.warning(
            "[predict] 'test_features.parquet' not found; scoring validation 'features.parquet' instead."
        )
        active_features_path = val_features_path
    else:
        raise FileNotFoundError(
            f"No feature Parquet found in '{features_dir}'. Run '--stage features' first."
        )

    fallbacks = _load_fallback_by_event(candidates_dir, top_k_predict)
    out_csv = submissions_dir / f"submission_{mode}.csv"

    logger.info(
        "[predict] Scoring '%s' (LightGBM=%d, XGBoost=%d, cuda=%s) -> '%s'",
        active_features_path,
        len(lgb_boosters),
        len(xgb_boosters),
        has_cuda,
        out_csv,
    )

    lf = pl.scan_parquet(active_features_path)
    all_sessions = (
        lf.select("session").unique().sort("session").collect()["session"].to_list()
    )
    total_sessions = len(all_sessions)

    with out_csv.open("wb") as out_f:
        out_f.write(b"session_type,labels\n")

        for start in range(0, total_sessions, BATCH_SESSIONS):
            batch_sess = pl.Series(
                "session", all_sessions[start : start + BATCH_SESSIONS], dtype=pl.Int32
            )
            batch_df = (
                lf.join(batch_sess.to_frame().lazy(), on="session", how="semi")
                .select(["session", "aid", *feature_cols])
                .sort("session")
                .collect()
            )
            x_batch = (
                batch_df.select(feature_cols).to_numpy().astype(np.float32, copy=False)
            )
            dmatrix_batch = (
                xgb.DMatrix(x_batch) if (xgb_boosters and xgb is not None) else None
            )
            base_ids = batch_df.select(
                pl.col("session").cast(pl.Int32),
                pl.col("aid").cast(pl.Int32),
            )

            sub_chunks: list[pl.DataFrame] = []
            for event in event_types:
                lgb_scores = lgb_boosters[event].predict(x_batch)
                w = blend_weights.get(event, {"lgbm": 1.0, "xgb": 0.0})

                if (
                    event in xgb_boosters
                    and dmatrix_batch is not None
                    and w.get("xgb", 0.0) > 0.0
                ):
                    xgb_scores = xgb_boosters[event].predict(dmatrix_batch)
                    if w.get("lgbm", 0.0) > 0.0:
                        final_scores = float(w["lgbm"]) * _zscore_per_session(
                            base_ids, lgb_scores
                        ) + float(w["xgb"]) * _zscore_per_session(base_ids, xgb_scores)
                    else:
                        final_scores = xgb_scores
                else:
                    final_scores = lgb_scores

                ranked = (
                    base_ids.with_columns(pl.Series("_score", final_scores, dtype=pl.Float32))
                    .sort(["session", "_score"], descending=[False, True])
                    .group_by("session", maintain_order=True)
                    .agg(pl.col("aid").head(top_k_predict).alias("labels"))
                )
                ranked = _pad_predictions_vectorized(
                    ranked, fallbacks.get(event, []), top_k_predict
                )
                formatted = ranked.select(
                    (pl.col("session").cast(pl.Utf8) + pl.lit(f"_{event}")).alias(
                        "session_type"
                    ),
                    pl.col("labels")
                    .cast(pl.List(pl.Utf8))
                    .list.join(" ")
                    .alias("labels"),
                )
                sub_chunks.append(formatted)

            merged_sub = pl.concat(sub_chunks).sort("session_type")
            merged_sub.write_csv(out_f, include_header=False)

    logger.info(
        "[predict] Wrote %d submission rows (%d sessions x %d types) -> %s",
        total_sessions * len(event_types),
        total_sessions,
        len(event_types),
        out_csv,
    )
