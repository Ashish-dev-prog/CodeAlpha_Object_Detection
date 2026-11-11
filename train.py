"""
train.py
========
Custom YOLO fine-tuning script.

Starting from pretrained YOLO11s weights, this script fine-tunes the model
on the custom dataset defined in datasets/object_detection/dataset.yaml.

SAFETY RULES
------------
* The original yolo11s.pt is NEVER modified or overwritten.
* Training outputs go to runs/train/<experiment_name>/.
* The best checkpoint is copied to models/weights/<experiment_name>_best.pt
  after training completes.
* Training does NOT start automatically — the user must invoke this script
  explicitly after reviewing and validating the dataset.

Usage
-----
  python train.py --config configs/train_config.yaml

  # Override individual parameters:
  python train.py --epochs 30 --batch 8 --imgsz 640
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

try:
    import yaml
except ImportError:
    logger.error("PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError as exc:
    logger.error("ultralytics not installed. Run: pip install ultralytics")
    raise SystemExit(1) from exc


# ===========================================================================
# Default training configuration (conservative for 8 GB GPU)
# ===========================================================================

DEFAULT_TRAIN_CONFIG = {
    # Pretrained weights — NEVER the fine-tuned output path.
    "pretrained_weights": "yolo11s.pt",
    # Dataset YAML
    "data": "datasets/object_detection/dataset.yaml",
    # Experiment name (subfolder under runs/train/)
    "experiment_name": "yolo11s_custom_v1",
    # Training hyperparameters
    "epochs": 50,
    "imgsz": 640,
    "batch": 4,          # Conservative: 4 images/batch for 8 GB GPU
    "patience": 15,      # Early stop after N epochs with no val improvement
    "device": 0,
    # Augmentation: start conservative; tune after first baseline run.
    "augment": True,
    "mosaic": 1.0,
    "mixup": 0.0,
    # Optimiser
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    # Regularisation
    "weight_decay": 0.0005,
    # Output
    "project": "runs/train",
    "save_period": 5,    # Save checkpoint every N epochs
    "plots": True,
    "verbose": True,
}


def load_train_config(config_path: Path) -> dict:
    """Load training config YAML, merging over defaults."""
    if not config_path.exists():
        logger.warning("Train config not found at %s — using defaults.", config_path)
        return DEFAULT_TRAIN_CONFIG.copy()

    with config_path.open("r", encoding="utf-8") as fh:
        user_config = yaml.safe_load(fh) or {}

    merged = {**DEFAULT_TRAIN_CONFIG, **user_config}
    logger.info("Train config loaded from %s", config_path)
    return merged


def validate_prerequisites(config: dict) -> None:
    """
    Check that required files exist before starting training.

    Raises FileNotFoundError with a clear message if anything is missing.
    Training should never start against an unvalidated dataset.
    """
    weights_path = _PROJECT_ROOT / config["pretrained_weights"]
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Pretrained weights not found: {weights_path}\n"
            "Place yolo11s.pt in the project root."
        )

    data_path = _PROJECT_ROOT / config["data"]
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}\n"
            "Create and validate your dataset first (see dataset_validator.py)."
        )

    # Check that at least one training image exists.
    train_img_dir = _PROJECT_ROOT / "datasets" / "object_detection" / "images" / "train"
    if not any(train_img_dir.iterdir()) if train_img_dir.exists() else True:
        raise FileNotFoundError(
            f"No training images found in {train_img_dir}.\n"
            "Add labelled images before training."
        )

    logger.info("Prerequisites validated.")


def run_training(config: dict) -> Path:
    """
    Fine-tune YOLO11s on the custom dataset.

    Returns the path to the run directory created by Ultralytics.
    """
    validate_prerequisites(config)

    weights_path = str(_PROJECT_ROOT / config["pretrained_weights"])
    data_path = str(_PROJECT_ROOT / config["data"])
    exp_name = config["experiment_name"]

    logger.info(
        "Starting training: weights=%s  data=%s  epochs=%d  imgsz=%d  batch=%d  device=%s",
        config["pretrained_weights"],
        config["data"],
        config["epochs"],
        config["imgsz"],
        config["batch"],
        config["device"],
    )

    model = YOLO(weights_path)

    results = model.train(
        data=data_path,
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        patience=config["patience"],
        device=config["device"],
        augment=config["augment"],
        mosaic=config["mosaic"],
        mixup=config["mixup"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        lrf=config["lrf"],
        weight_decay=config["weight_decay"],
        project=str(_PROJECT_ROOT / config["project"]),
        name=exp_name,
        save_period=config["save_period"],
        plots=config["plots"],
        verbose=config["verbose"],
        exist_ok=True,
    )

    run_dir = Path(results.save_dir)
    logger.info("Training complete.  Run saved to: %s", run_dir)

    # --- Copy best checkpoint to models/weights/ ---
    best_src = run_dir / "weights" / "best.pt"
    if best_src.exists():
        weights_out_dir = _PROJECT_ROOT / "models" / "weights"
        weights_out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = weights_out_dir / f"{exp_name}_best_{ts}.pt"
        shutil.copy2(best_src, dest)
        logger.info("Best checkpoint copied to: %s", dest)

    return run_dir


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11s on a custom dataset.")
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Path to train_config.yaml (default: configs/train_config.yaml).",
    )
    parser.add_argument("--epochs",  type=int, default=None, help="Override epochs.")
    parser.add_argument("--batch",   type=int, default=None, help="Override batch size.")
    parser.add_argument("--imgsz",   type=int, default=None, help="Override image size.")
    parser.add_argument("--device",  default=None,           help="Override device (e.g. 0, cpu).")
    parser.add_argument("--name",    default=None,           help="Override experiment name.")
    args = parser.parse_args()

    config_path = _PROJECT_ROOT / args.config
    config = load_train_config(config_path)

    # Apply CLI overrides
    if args.epochs  is not None: config["epochs"]            = args.epochs
    if args.batch   is not None: config["batch"]             = args.batch
    if args.imgsz   is not None: config["imgsz"]             = args.imgsz
    if args.device  is not None: config["device"]            = args.device
    if args.name    is not None: config["experiment_name"]   = args.name

    run_dir = run_training(config)
    print(f"\nTraining complete.  Results: {run_dir}")


if __name__ == "__main__":
    main()

