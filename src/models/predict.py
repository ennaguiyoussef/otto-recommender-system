"""Fault-tolerant 3-Way Ensemble inference with batch-level Google Drive checkpoints."""

from __future__ import annotations

import gc
import json
import logging
import shutil
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
    compute_stage1_priors,
)
from src.utils.checkpoint import atomic_temp_path, commit_atomic, is_valid_checkpoint

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

BATCH_SESSIONS = 150_000


def _score_dataframe_batch(
    batch_df: pl.DataFrame,
    feature_cols: list[str],
    event_types: list[str],
    lgb_boosters: dict[str, lgb.Booster],
    xgb_boosters: dict[str, Any],
    blend_weights: dict[str, dict[str, float]],
    fallbacks: dict[str, list[int]],
    top_k_predict: int,
    out_part_csv: Path,
) -> int:
    """Score one session batch across all objectives and atomically save its CSV shard."""
    x_batch = batch_df.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    dmatrix_batch = (
        xgb.DMatrix(x_batch) if (xgb_boosters and xgb is not None) else None
    )
    base_ids = batch_df.select(
        pl.col("session").cast(pl.Int32),
        pl.col("aid").cast(pl.Int32),
    )

    sub_chunks: list[pl.DataFrame] = []
    for event in event_types:
        w = blend_weights.get(
            event, {"lgbm": 1.0, "xgb": 0.0, "s1_rank": 0.0, "s1_obj": 0.0}
        )
        w_lgb = float(w.get("lgbm", 1.0))
        w_xgb = float(w.get("xgb", 0.0))
        w_rk = float(w.get("s1_rank", 0.0))
        w_obj = float(w.get("s1_obj", 0.0))

        final_scores = np.zeros(len(batch_df), dtype=np.float32)

        if w_lgb > 0.0:
            lgb_s = lgb_boosters[event].predict(x_batch)
            final_scores += w_lgb * _zscore_per_session(base_ids, lgb_s)

        if w_xgb > 0.0 and event in xgb_boosters and dmatrix_batch is not None:
            xgb_s = xgb_boosters[event].predict(dmatrix_batch)
            final_scores += w_xgb * _zscore_per_session(base_ids, xgb_s)
        elif w_xgb > 0.0 and w_lgb == 0.0:
            lgb_s = lgb_boosters[event].predict(x_batch)
            final_scores += w_xgb * _zscore_per_session(base_ids, lgb_s)

        if w_rk > 0.0 or w_obj > 0.0:
            rk_raw, obj_raw = compute_stage1_priors(batch_df, event)
            if w_rk > 0.0:
                final_scores += w_rk * _zscore_per_session(base_ids, rk_raw)
            if w_obj > 0.0:
                final_scores += w_obj * _zscore_per_session(base_ids, obj_raw)

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
            pl.col("labels").cast(pl.List(pl.Utf8)).list.join(" ").alias("labels"),
        )
        sub_chunks.append(formatted)

    merged_sub = pl.concat(sub_chunks).sort("session_type")
    tmp_part = atomic_temp_path(out_part_csv)
    merged_sub.write_csv(tmp_part, include_header=False)
    commit_atomic(tmp_part, out_part_csv)
    return len(merged_sub)


def run_predict(config: dict[str, Any], mode: str = "sample") -> None:
    """Score test candidates with 3-way ensemble and batch-level Drive checkpointing."""
    features_dir = Path(config["paths"]["features"]) / mode
    candidates_dir = Path(config["paths"]["candidates"]) / mode
    models_dir = Path(config["paths"]["models"]) / mode
    submissions_dir = Path(config["paths"]["submissions"])
    submissions_dir.mkdir(parents=True, exist_ok=True)
    force = bool(config.get("force_recompute", False))

    out_csv = submissions_dir / f"submission_{mode}.csv"
    if is_valid_checkpoint(out_csv, force=force):
        logger.info(
            "[checkpoint] Submission file already exists -> '%s'; skipping (--force to regenerate).",
            out_csv,
        )
        return

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

    test_pop_path = candidates_dir / "test_popular_items.parquet"
    if test_pop_path.exists():
        pop_df = pl.read_parquet(test_pop_path)
        overall = pop_df.filter(pl.col("type") == 255)["aid"].to_list()
        fallbacks: dict[str, list[int]] = {}
        for ev, code in (("clicks", 0), ("carts", 1), ("orders", 2)):
            typed = pop_df.filter(pl.col("type") == code)["aid"].to_list()
            seen = set(typed)
            fallbacks[ev] = (typed + [a for a in overall if a not in seen])[:top_k_predict]
    else:
        fallbacks = _load_fallback_by_event(candidates_dir, top_k_predict)

    ckpt_dir = submissions_dir / f"_ckpt_predict_{mode}"
    if force and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    shards_dir = features_dir / "test_features_shards"
    test_features_path = features_dir / "test_features.parquet"
    val_features_path = features_dir / "features.parquet"

    part_csvs: list[Path] = []
    needed_cols = list(dict.fromkeys(["session", "aid", *feature_cols]))

    if (shards_dir / "_COMPLETE.flag").exists():
        shard_files = sorted(shards_dir.glob("shard_*.parquet"))
        logger.info(
            "[predict] Scoring %d test feature shards (cuda=%s) -> '%s'",
            len(shard_files),
            has_cuda,
            out_csv,
        )
        for idx, shard_file in enumerate(shard_files):
            part_csv = ckpt_dir / f"part_{idx:04d}.csv"
            if is_valid_checkpoint(part_csv, force=False):
                logger.info(
                    "[checkpoint] Prediction shard %d/%d already scored; skipping.",
                    idx + 1,
                    len(shard_files),
                )
                part_csvs.append(part_csv)
                continue

            batch_df = pl.read_parquet(shard_file, columns=needed_cols).sort("session")
            if len(batch_df) > 0:
                _score_dataframe_batch(
                    batch_df=batch_df,
                    feature_cols=feature_cols,
                    event_types=event_types,
                    lgb_boosters=lgb_boosters,
                    xgb_boosters=xgb_boosters,
                    blend_weights=blend_weights,
                    fallbacks=fallbacks,
                    top_k_predict=top_k_predict,
                    out_part_csv=part_csv,
                )
                part_csvs.append(part_csv)
            del batch_df
            gc.collect()
            logger.info(
                "[predict] Checkpointed prediction shard %d/%d",
                idx + 1,
                len(shard_files),
            )
    else:
        active_path = (
            test_features_path if test_features_path.exists() else val_features_path
        )
        if not active_path.exists():
            raise FileNotFoundError(
                f"No feature Parquet found in '{features_dir}'. Run '--stage features' first."
            )
        logger.info(
            "[predict] Scoring '%s' (cuda=%s) -> '%s'",
            active_path,
            has_cuda,
            out_csv,
        )
        lf = pl.scan_parquet(active_path)
        all_sessions = (
            lf.select("session").unique().sort("session").collect()["session"].to_list()
        )
        for batch_idx, start in enumerate(range(0, len(all_sessions), BATCH_SESSIONS)):
            part_csv = ckpt_dir / f"part_{batch_idx:04d}.csv"
            if is_valid_checkpoint(part_csv, force=False):
                part_csvs.append(part_csv)
                continue

            batch_sess = pl.Series(
                "session", all_sessions[start : start + BATCH_SESSIONS], dtype=pl.Int32
            )
            batch_df = (
                lf.join(batch_sess.to_frame().lazy(), on="session", how="semi")
                .select(needed_cols)
                .sort("session")
                .collect()
            )
            if len(batch_df) > 0:
                _score_dataframe_batch(
                    batch_df=batch_df,
                    feature_cols=feature_cols,
                    event_types=event_types,
                    lgb_boosters=lgb_boosters,
                    xgb_boosters=xgb_boosters,
                    blend_weights=blend_weights,
                    fallbacks=fallbacks,
                    top_k_predict=top_k_predict,
                    out_part_csv=part_csv,
                )
                part_csvs.append(part_csv)
            del batch_df
            gc.collect()

    # Assemble final submission CSV atomically
    tmp_sub = atomic_temp_path(out_csv)
    total_rows = 0
    with tmp_sub.open("wb") as out_f:
        out_f.write(b"session_type,labels\n")
        for p_csv in part_csvs:
            data = p_csv.read_bytes()
            total_rows += data.count(b"\n")
            out_f.write(data)
    commit_atomic(tmp_sub, out_csv)
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    logger.info("[predict] Wrote %d submission rows -> %s", total_rows, out_csv)
