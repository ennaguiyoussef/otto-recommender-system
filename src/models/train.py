"""Stage-2 Reranker training: Positive-filtered LightGBM + CUDA GPU XGBoost Ensemble."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from src.metrics.recall import CODE_TO_TYPE, compute_recall_at_k

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]

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


def _detect_cuda_gpu() -> bool:
    """Return True when an NVIDIA CUDA GPU is available on the host."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        res = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0 and "GPU" in res.stdout:
            logger.info("[train] Detected NVIDIA GPU: %s", res.stdout.strip().splitlines()[0])
            return True
    except Exception:
        pass
    return False


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


def _pad_predictions_vectorized(
    ranked: pl.DataFrame,
    fallback: list[int],
    k: int,
) -> pl.DataFrame:
    """Vectorized Polars padding so every session has `k` unique predicted items."""
    if not fallback:
        return ranked.with_columns(
            pl.col("labels").list.unique(maintain_order=True).list.head(k)
        )

    fb_lit = pl.lit(fallback[:k], dtype=pl.List(pl.Int32))
    return ranked.with_columns(
        pl.when(pl.col("labels").list.len() >= k)
        .then(pl.col("labels").list.head(k))
        .otherwise(
            pl.col("labels")
            .list.concat(fb_lit)
            .list.unique(maintain_order=True)
            .list.head(k)
        )
        .alias("labels")
    )


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


def _rank_scores_to_preds(
    eval_ids: pl.DataFrame,
    scores: np.ndarray,
    event: str,
    fallback: list[int],
    k: int,
) -> pl.DataFrame:
    """Rank candidates by score per session and pad to top-`k`."""
    ranked = (
        eval_ids.with_columns(pl.Series("_score", scores, dtype=pl.Float32))
        .sort(["session", "_score"], descending=[False, True])
        .group_by("session", maintain_order=True)
        .agg(pl.col("aid").head(k).alias("labels"))
        .with_columns(pl.lit(event).alias("type"))
    )
    return _pad_predictions_vectorized(ranked, fallback, k).select("session", "type", "labels")


def _zscore_per_session(eval_ids: pl.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Standardize raw ranker margins within each session before ensembling."""
    normed = eval_ids.with_columns(pl.Series("_s", scores, dtype=pl.Float32)).select(
        (
            (pl.col("_s") - pl.col("_s").mean().over("session"))
            / (pl.col("_s").std().over("session").fill_null(1.0) + 1e-6)
        )
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias("_z")
    )
    return normed["_z"].to_numpy()


def run_train(config: dict[str, Any], mode: str = "sample") -> None:
    """Train per-objective LightGBM + CUDA XGBoost rankers and evaluate Recall@20."""
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
    lgb_early_stop = int(raw_lgb_params.pop("early_stopping_rounds", 50))
    raw_lgb_params.pop("ndcg_eval_at", None)

    xgb_cfg: dict[str, Any] = dict(config.get("xgboost", {}))
    xgb_enabled = bool(xgb_cfg.pop("enabled", True)) and (xgb is not None)
    xgb_early_stop = int(xgb_cfg.pop("early_stopping_rounds", 50))
    has_cuda = _detect_cuda_gpu()
    if xgb_enabled:
        requested_dev = str(xgb_cfg.get("device", "auto"))
        xgb_cfg["device"] = "cuda" if (requested_dev in {"auto", "cuda"} and has_cuda) else "cpu"
        logger.info("[train] XGBoost Ranker enabled on device='%s'", xgb_cfg["device"])

    logger.info("[train] Loading validation candidate features from '%s'", features_path)
    df = pl.read_parquet(features_path).sort(["session", "cand_rank"])
    if len(df) == 0:
        raise ValueError(f"Feature table '{features_path}' is empty.")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    logger.info(
        "[train] Loaded %d rows across %d sessions with %d features.",
        len(df),
        df["session"].n_unique(),
        len(feature_cols),
    )

    # Deterministic 80/20 session split: 80% ranker fit, 20% unbiased Recall@20 evaluation
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

    eval_ids = eval_df.select(
        pl.col("session").cast(pl.Int32),
        pl.col("aid").cast(pl.Int32),
    )
    eval_groups = (
        eval_df.group_by("session", maintain_order=True).len()["len"].to_numpy()
    )
    x_eval = eval_df.select(feature_cols).to_numpy().astype(np.float32, copy=False)

    # Also filter eval_set for early stopping to sessions with >=1 positive for fast metric calc
    fallbacks = _load_fallback_by_event(candidates_dir, top_k_predict)
    lgb_pred_frames: list[pl.DataFrame] = []
    xgb_pred_frames: list[pl.DataFrame] = []
    ens_pred_frames: list[pl.DataFrame] = []
    blend_weights: dict[str, dict[str, float]] = {}

    for event in event_types:
        label_col = f"label_{event}"

        # Filter training sessions to groups with >=1 positive label (identical LambdaRank gradient, ~10x faster)
        pos_train_sessions = (
            train_df.group_by("session")
            .agg(pl.col(label_col).sum().alias("_pos"))
            .filter(pl.col("_pos") > 0)
            .select("session")
        )
        obj_train_df = train_df.join(pos_train_sessions, on="session", how="semi").sort(
            "session"
        )
        if len(obj_train_df) == 0:
            obj_train_df = train_df

        pos_eval_sessions = (
            eval_df.group_by("session")
            .agg(pl.col(label_col).sum().alias("_pos"))
            .filter(pl.col("_pos") > 0)
            .select("session")
        )
        obj_es_df = eval_df.join(pos_eval_sessions, on="session", how="semi").sort("session")
        if len(obj_es_df) == 0:
            obj_es_df = eval_df

        x_tr = obj_train_df.select(feature_cols).to_numpy().astype(np.float32, copy=False)
        y_tr = obj_train_df[label_col].to_numpy()
        tr_groups = (
            obj_train_df.group_by("session", maintain_order=True).len()["len"].to_numpy()
        )

        x_es = obj_es_df.select(feature_cols).to_numpy().astype(np.float32, copy=False)
        y_es = obj_es_df[label_col].to_numpy()
        es_groups = (
            obj_es_df.group_by("session", maintain_order=True).len()["len"].to_numpy()
        )

        logger.info(
            "[train] '%s' | Filtered positive groups: %d/%d sessions (%d rows, %d positives)",
            event,
            len(tr_groups),
            train_df["session"].n_unique(),
            len(x_tr),
            int(y_tr.sum()),
        )

        lgb_params = dict(raw_lgb_params)
        if len(obj_train_df) < int(lgb_params.get("min_child_samples", 50)) * 2:
            lgb_params["min_child_samples"] = max(1, len(obj_train_df) // 4)

        ranker = lgb.LGBMRanker(**lgb_params)
        ranker.fit(
            x_tr,
            y_tr,
            group=tr_groups,
            eval_set=[(x_es, y_es)],
            eval_group=[es_groups],
            eval_at=[top_k_predict],
            callbacks=[
                lgb.early_stopping(stopping_rounds=lgb_early_stop, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        lgb_model_path = models_dir / f"lgbm_{event}.txt"
        ranker.booster_.save_model(str(lgb_model_path))
        lgb_scores = ranker.booster_.predict(x_eval)
        lgb_pred_frames.append(
            _rank_scores_to_preds(
                eval_ids, lgb_scores, event, fallbacks.get(event, []), top_k_predict
            )
        )

        if xgb_enabled and xgb is not None:
            cur_xgb_params = dict(xgb_cfg)
            cur_xgb_params["early_stopping_rounds"] = xgb_early_stop
            xgb_ranker = xgb.XGBRanker(**cur_xgb_params)
            xgb_ranker.fit(
                x_tr,
                y_tr,
                group=tr_groups,
                eval_set=[(x_es, y_es)],
                eval_group=[es_groups],
                verbose=100,
            )
            xgb_model_path = models_dir / f"xgb_{event}.json"
            xgb_ranker.get_booster().save_model(str(xgb_model_path))
            xgb_scores = xgb_ranker.get_booster().inplace_predict(x_eval)
            xgb_pred_frames.append(
                _rank_scores_to_preds(
                    eval_ids, xgb_scores, event, fallbacks.get(event, []), top_k_predict
                )
            )

            blended_scores = 0.5 * _zscore_per_session(
                eval_ids, lgb_scores
            ) + 0.5 * _zscore_per_session(eval_ids, xgb_scores)
            ens_pred_frames.append(
                _rank_scores_to_preds(
                    eval_ids,
                    blended_scores,
                    event,
                    fallbacks.get(event, []),
                    top_k_predict,
                )
            )
            blend_weights[event] = {"lgbm": 0.5, "xgb": 0.5}
        else:
            ens_pred_frames.append(lgb_pred_frames[-1])
            blend_weights[event] = {"lgbm": 1.0, "xgb": 0.0}

    eval_sessions = eval_df.select("session").unique()
    ground_truth_df = _prepare_ground_truth(labels_path, eval_sessions)

    lgb_recall = compute_recall_at_k(
        pl.concat(lgb_pred_frames), ground_truth_df, weights=metric_weights, k=top_k_predict
    )
    logger.info(
        "[train] LightGBM Recall@%d | clicks=%.5f | carts=%.5f | orders=%.5f | OVERALL=%.5f",
        top_k_predict,
        lgb_recall["clicks"],
        lgb_recall["carts"],
        lgb_recall["orders"],
        lgb_recall["overall"],
    )

    final_recall = lgb_recall
    if xgb_enabled and xgb_pred_frames:
        xgb_recall = compute_recall_at_k(
            pl.concat(xgb_pred_frames),
            ground_truth_df,
            weights=metric_weights,
            k=top_k_predict,
        )
        ens_recall = compute_recall_at_k(
            pl.concat(ens_pred_frames),
            ground_truth_df,
            weights=metric_weights,
            k=top_k_predict,
        )
        logger.info(
            "[train] XGBoost  Recall@%d | clicks=%.5f | carts=%.5f | orders=%.5f | OVERALL=%.5f",
            top_k_predict,
            xgb_recall["clicks"],
            xgb_recall["carts"],
            xgb_recall["orders"],
            xgb_recall["overall"],
        )
        logger.info(
            "[train] ENSEMBLE Recall@%d | clicks=%.5f | carts=%.5f | orders=%.5f | OVERALL=%.5f",
            top_k_predict,
            ens_recall["clicks"],
            ens_recall["carts"],
            ens_recall["orders"],
            ens_recall["overall"],
        )
        final_recall = ens_recall

    meta_path = models_dir / "model_meta.json"
    meta_payload = {
        "mode": mode,
        "feature_cols": feature_cols,
        "event_types": event_types,
        "top_k_predict": top_k_predict,
        "xgb_enabled": xgb_enabled,
        "blend_weights": blend_weights,
        "validation_recall": final_recall,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    logger.info("[train] Saved model metadata -> %s", meta_path)
