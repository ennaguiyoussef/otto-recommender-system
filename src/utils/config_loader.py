"""Configuration loader and Google Drive directory initializer."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

VALID_MODES = {"sample", "full"}


def load_config(
    config_path: str | Path = "configs/config.yaml",
    mode: str = "sample",
    create_dirs: bool = True,
) -> dict[str, Any]:
    """Load YAML configuration, resolve storage paths, and ensure Drive directories exist.

    Args:
        config_path: Path to the YAML configuration file.
        mode: Execution mode ('sample' or 'full').
        create_dirs: If True, create base_dir and all configured subdirectories.

    Returns:
        Dictionary containing parsed config, active mode settings, and resolved paths.

    Raises:
        FileNotFoundError: If `config_path` does not exist.
        ValueError: If `mode` is not one of the configured modes.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Expected one of {sorted(VALID_MODES)}.")

    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {cfg_file.resolve()}")

    with cfg_file.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    base_dir = config.get("base_dir")
    if not base_dir:
        raise ValueError("Missing required 'base_dir' key in configuration.")

    raw_paths: dict[str, str] = config.get("paths", {})
    resolved_paths: dict[str, str] = {
        key: os.path.normpath(os.path.join(base_dir, rel_path))
        for key, rel_path in raw_paths.items()
    }

    if create_dirs:
        os.makedirs(base_dir, exist_ok=True)
        for key, folder_path in resolved_paths.items():
            os.makedirs(folder_path, exist_ok=True)
            logger.debug("Verified directory [%s]: %s", key, folder_path)
        logger.info("Initialized storage directories under: %s", base_dir)

    mode_cfg = config.get("modes", {}).get(mode)
    if mode_cfg is None:
        raise ValueError(f"Mode '{mode}' is missing from 'modes' section in {config_path}.")

    config["active_mode"] = mode
    config["sample_frac"] = float(mode_cfg["sample_frac"])
    config["paths"] = resolved_paths

    return config
