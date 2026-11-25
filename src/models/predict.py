"""LightGBM reranker inference and Kaggle submission generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from src.models.train import _load_fallback_by_event, _pad_predictions

logger = logging.getLogger(__name__)

BATCH_SESSIONS = 100_000


def run_predict(config: dict[str, Any], mode: str = "sample") -> None:
    """Score test candidates with trained LightGBM models and write submission CSV.

    Writes `<submissions>/submission_<mode>.csv` with columns `session_type,labels`.

    Args:
        config: Loaded pipeline configuration with resolved Drive paths.
        mode: Execution mode ('sample' or 'full').
    """
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

    boosters: dict[str, lgb.Booster] = {}
    for event in event_types:
        model_file = models_dir / f"lgbm_{event}.txt"
        if not model_file.exists():
            raise FileNotFoundError(f"Missing trained model file: {model_file}")
        boosters[event] = lgb.Booster(model_file=str(model_file))

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
        "[predict] Scoring '%s' with %d LightGBM models -> '%s'",
        active_features_path,
        len(boosters),
        out_csv,
    )

    lf = pl.scan_parquet(active_features_path)
    all_sessions = (
        lf.select("session").unique().sort("session").collect()["session"].to_list()
    )
    total_sessions = len(all_sessions)

    with out_csv.open("w", encoding="utf-8", newline="\n") as out_f:
        out_f.write("session_type,labels\n")

        for start in range(0, total_sessions, BATCH_SESSIONS):
            batch_sess = pl.Series(
                "session", all_sessions[start : start + BATCH_SESSIONS], dtype=pl.Int32
            )
            batch_df = (
                lf.join(batch_sess.to_frame().lazy(), on="session", how="semi")
                .select(["session", "aid", *feature_cols])
                .collect()
            )
            x_batch = (
                batch_df.select(feature_cols).to_numpy().astype(np.float32, copy=False)
            )
            base_ids = batch_df.select(
                pl.col("session").cast(pl.Int32),
                pl.col("aid").cast(pl.Int32),
            )

            sub_chunks: list[pl.DataFrame] = []
            for event, booster in boosters.items():
                scores = booster.predict(x_batch)
                ranked = (
                    base_ids.with_columns(pl.Series("_score", scores, dtype=pl.Float64))
                    .sort(["session", "_score"], descending=[False, True])
                    .group_by("session", maintain_order=True)
                    .agg(pl.col("aid").head(top_k_predict).alias("labels"))
                )
                ranked = _pad_predictions(
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
            for session_type, label_str in merged_sub.iter_rows():
                out_f.write(f"{session_type},{label_str}\n")

    logger.info(
        "[predict] Wrote %d submission rows (%d sessions x %d types) -> %s",
        total_sessions * len(event_types),
        total_sessions,
        len(event_types),
        out_csv,
    )
