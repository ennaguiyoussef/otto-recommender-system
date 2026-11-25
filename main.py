"""CLI entrypoint for the OTTO 2-Stage Recommender System pipeline."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Callable

from src.candidates.covisitation import run_candidates
from src.data.make_sample import run_make_sample
from src.data.split import run_split
from src.features.build_features import run_build_features
from src.models.predict import run_predict
from src.models.train import run_train
from src.utils.config_loader import load_config

StageFn = Callable[[dict[str, Any], str], None]

STAGE_REGISTRY: dict[str, StageFn] = {
    "sample": run_make_sample,
    "split": run_split,
    "candidates": run_candidates,
    "features": run_build_features,
    "train": run_train,
    "predict": run_predict,
}

PIPELINE_ORDER: list[str] = [
    "sample",
    "split",
    "candidates",
    "features",
    "train",
    "predict",
]


def setup_logging() -> None:
    """Configure structured console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="OTTO Multi-Objective Recommender System (2-Stage Pipeline)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample", "full"],
        default="sample",
        help="Execution mode: 'sample' (5%% fast test) or 'full' (100%% dataset).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="Pipeline stage ('all', single stage, or comma-separated e.g. 'candidates,features,train,predict').",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the requested OTTO pipeline stage(s)."""
    setup_logging()
    logger = logging.getLogger("otto.main")
    args = parse_args()

    logger.info(
        "Starting OTTO pipeline | mode='%s' | stage='%s' | config='%s'",
        args.mode,
        args.stage,
        args.config,
    )

    config = load_config(config_path=args.config, mode=args.mode, create_dirs=True)
    if args.stage == "all":
        stages_to_run = PIPELINE_ORDER
    else:
        stages_to_run = [s.strip() for s in args.stage.split(",") if s.strip()]
        invalid = [s for s in stages_to_run if s not in STAGE_REGISTRY]
        if invalid:
            raise ValueError(
                f"Invalid stage(s) {invalid}. Choose from {['all', *PIPELINE_ORDER]}."
            )

    pipeline_start = time.perf_counter()
    for stage_name in stages_to_run:
        stage_start = time.perf_counter()
        logger.info("=== Running stage: %s ===", stage_name.upper())
        STAGE_REGISTRY[stage_name](config, args.mode)
        elapsed = time.perf_counter() - stage_start
        logger.info("=== Finished stage: %s (%.2fs) ===", stage_name.upper(), elapsed)

    total_elapsed = time.perf_counter() - pipeline_start
    logger.info("Pipeline execution completed in %.2fs.", total_elapsed)


if __name__ == "__main__":
    main()
