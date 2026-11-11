"""
config_loader.py
================
Loads and validates the detection YAML configuration file.

Design note: keeping configuration in a separate YAML file (rather than
hard-coding values in every script) means you can change model, camera,
thresholds, and class list in one place and all scripts pick up the change
automatically.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Fields that must be present in a valid config file.
REQUIRED_KEYS = [
    "input_source",
    "camera_index",
    "model_path",
    "device",
    "imgsz",
    "conf",
    "iou",
    "target_classes",
    "output_dir",
]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load and validate a YAML detection configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If required keys are missing or values are out of acceptable range.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping at the top level.")

    # --- Required-key check ---
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ValueError(f"Configuration is missing required keys: {missing}")

    # --- Value range checks ---
    if not (0.0 < config["conf"] <= 1.0):
        raise ValueError(f"'conf' must be in (0, 1], got {config['conf']}")
    if not (0.0 < config["iou"] <= 1.0):
        raise ValueError(f"'iou' must be in (0, 1], got {config['iou']}")
    if config["imgsz"] % 32 != 0:
        raise ValueError(f"'imgsz' must be a multiple of 32, got {config['imgsz']}")

    valid_sources = {"webcam", "image", "video"}
    if config["input_source"] not in valid_sources:
        raise ValueError(
            f"'input_source' must be one of {valid_sources}, got '{config['input_source']}'"
        )

    valid_formats = {"csv", "json"}
    if config.get("log_format", "csv") not in valid_formats:
        raise ValueError(f"'log_format' must be 'csv' or 'json', got '{config['log_format']}'")

    logger.info("Configuration loaded from %s", config_path)
    return config


def get_target_class_ids(config: dict[str, Any], model_names: dict[int, str]) -> list[int] | None:
    """
    Resolve target class *names* from the config to their integer class IDs
    as defined by the loaded YOLO model.

    Returns None if target_classes is empty (meaning: detect all classes).
    Logs a warning for any target name that the model does not recognise.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary.
    model_names : dict[int, str]
        YOLO model's {class_id: class_name} mapping.

    Returns
    -------
    list[int] | None
        Sorted list of matching class IDs, or None if no filter is applied.
    """
    targets: list[str] = config.get("target_classes") or []
    if not targets:
        return None  # Detect all COCO classes

    # Build a lookup of lowercase name -> id
    name_to_id = {name.lower(): cid for cid, name in model_names.items()}

    class_ids: list[int] = []
    for name in targets:
        cid = name_to_id.get(name.lower())
        if cid is None:
            logger.warning(
                "Target class '%s' was not found in the model's class list and will be ignored.",
                name,
            )
        else:
            class_ids.append(cid)
            logger.info("Target class resolved: '%s' -> ID %d", name, cid)

    return sorted(class_ids) if class_ids else None

