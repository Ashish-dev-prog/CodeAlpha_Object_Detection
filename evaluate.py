"""
evaluate.py
===========
Fair evaluation system comparing:

  A. The original pretrained YOLO11s (80 COCO classes)
  B. A custom fine-tuned YOLO11s   (custom classes)

Both models are evaluated on the exact same held-out test images.

Class mapping design
--------------------
The pretrained model uses COCO class IDs (e.g. chair=56, bed=59, book=73),
while the custom dataset labels use custom IDs defined in dataset.yaml
(e.g. 0=book, 1=bed, 2=chair).

Instead of hardcoding IDs, this script:
1. Reads custom class names dynamically from dataset.yaml.
2. Looks up each custom class name in the pretrained model's `model.names`
   (with optional `--class-map custom:coco` overrides).
3. Creates a temporary COCO-mapped copy of the evaluated split's labels so
   Ultralytics `model.val()` evaluates the pretrained model against the exact
   same ground truth boxes, filtered to the target COCO class IDs.
4. Evaluates the fine-tuned model directly on the native dataset.yaml.

Reported metrics
----------------
* Precision, Recall, mAP@50, mAP@50-95 (overall and per-class)
* Confusion matrix (numeric matrix + saved PNG paths)
* Inference speed (ms/image)

Usage
-----
  python evaluate.py \\
      --pretrained yolo11s.pt \\
      --finetuned  models/weights/yolo11s_custom_v1_best.pt \\
      --data       datasets/object_detection/dataset.yaml \\
      --split      test

  # Evaluate only the pretrained baseline:
  python evaluate.py --pretrained yolo11s.pt --data datasets/object_detection/dataset.yaml
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

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
    logger.error("ultralytics not installed.")
    raise SystemExit(1) from exc


# ===========================================================================
# Class mapping helpers
# ===========================================================================

def parse_dataset_names(ds_cfg: dict[str, Any]) -> dict[int, str]:
    """Return {class_id: class_name} from dataset.yaml (supports list or dict)."""
    raw = ds_cfg.get("names", [])
    if isinstance(raw, list):
        return {idx: str(name) for idx, name in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in sorted(raw.items(), key=lambda kv: int(kv[0]))}
    raise ValueError("dataset.yaml 'names' must be a list or mapping.")


def resolve_dataset_root(ds_cfg: dict[str, Any], dataset_yaml_path: Path) -> Path:
    """Resolve the dataset root directory from dataset.yaml."""
    raw_path = ds_cfg.get("path")
    if not raw_path:
        return dataset_yaml_path.parent
    p = Path(raw_path)
    if p.is_absolute():
        return p
    # Check relative to project root first, then relative to dataset.yaml parent
    proj_candidate = (_PROJECT_ROOT / p).resolve()
    if proj_candidate.exists():
        return proj_candidate
    return (dataset_yaml_path.parent / p).resolve()


def build_custom_to_pretrained_map(
    custom_names: dict[int, str],
    pretrained_names: dict[int, str],
    overrides: dict[str, str] | None = None,
) -> dict[int, int]:
    """
    Dynamically map each custom class ID to its matching class ID in the
    pretrained model (case-insensitive name match, with optional overrides).

    Raises ValueError if any custom class has no match in pretrained_names.
    """
    overrides_lower = {k.lower(): v.lower() for k, v in (overrides or {}).items()}
    pretrained_lookup = {name.lower(): int(cid) for cid, name in pretrained_names.items()}

    mapping: dict[int, int] = {}
    for custom_id, custom_name in sorted(custom_names.items()):
        lookup_name = overrides_lower.get(custom_name.lower(), custom_name.lower())
        if lookup_name not in pretrained_lookup:
            raise ValueError(
                f"Custom class '{custom_name}' (ID {custom_id}) was not found in "
                f"pretrained model classes. Use --class-map {custom_name}:<coco_name> "
                "to provide a manual mapping."
            )
        mapping[custom_id] = pretrained_lookup[lookup_name]

    logger.info(
        "Class mapping (custom -> pretrained): %s",
        {
            f"{cid}:{custom_names[cid]}": f"{pid}:{pretrained_names[pid]}"
            for cid, pid in mapping.items()
        },
    )
    return mapping


def create_coco_mapped_dataset(
    dataset_yaml_path: Path,
    split: str,
    custom_to_coco: dict[int, int],
    pretrained_names: dict[int, str],
) -> Path:
    """
    Create a temporary dataset directory where label IDs for *split* are
    remapped from custom IDs to the pretrained model's class IDs.

    Only the evaluated *split* is linked/copied; empty train/val/test folders
    are created for the remaining splits so Ultralytics dataset checks pass
    without duplicating unused images.
    """
    with dataset_yaml_path.open("r", encoding="utf-8") as fh:
        ds_cfg = yaml.safe_load(fh) or {}

    dataset_root = resolve_dataset_root(ds_cfg, dataset_yaml_path)
    tmp_dir = Path(tempfile.mkdtemp(prefix="eval_coco_mapped_"))

    for sp in ("train", "val", "test"):
        (tmp_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (tmp_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)

    # Link or copy only the split being evaluated
    src_img_rel = ds_cfg.get(split, f"images/{split}")
    src_img_dir = (dataset_root / src_img_rel).resolve()
    if not src_img_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise FileNotFoundError(f"Split '{split}' image directory not found: {src_img_dir}")

    dst_img_dir = tmp_dir / "images" / split
    # Replace empty placeholder with symlink or copy
    dst_img_dir.rmdir()
    try:
        dst_img_dir.symlink_to(src_img_dir, target_is_directory=True)
    except OSError:
        shutil.copytree(str(src_img_dir), str(dst_img_dir))

    # Derive corresponding label directory (standard YOLO images/<split> -> labels/<split>)
    src_lbl_dir = dataset_root / "labels" / split
    dst_lbl_dir = tmp_dir / "labels" / split

    if src_lbl_dir.exists():
        for lbl_file in src_lbl_dir.iterdir():
            if not lbl_file.is_file() or lbl_file.suffix.lower() != ".txt":
                continue
            with lbl_file.open("r", encoding="utf-8") as fh:
                lines = fh.readlines()

            remapped_lines: list[str] = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    custom_id = int(parts[0])
                except ValueError:
                    continue
                if custom_id not in custom_to_coco:
                    raise ValueError(
                        f"Label {lbl_file.name} references unknown class_id {custom_id}."
                    )
                parts[0] = str(custom_to_coco[custom_id])
                remapped_lines.append(" ".join(parts) + "\n")

            with (dst_lbl_dir / lbl_file.name).open("w", encoding="utf-8") as fh:
                fh.writelines(remapped_lines)

    tmp_yaml = {
        "path": str(tmp_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(pretrained_names),
        "names": {int(k): str(v) for k, v in pretrained_names.items()},
    }
    tmp_yaml_path = tmp_dir / "dataset.yaml"
    with tmp_yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(tmp_yaml, fh, sort_keys=False)

    logger.info("Created remapped evaluation dataset at: %s", tmp_dir)
    return tmp_yaml_path


# ===========================================================================
# Confusion matrix extraction
# ===========================================================================

def extract_target_confusion_matrix(
    metrics: Any,
    target_class_ids: list[int],
    model_names: dict[int, str],
) -> dict[str, Any] | None:
    """
    Extract a compact confusion matrix covering only *target_class_ids* plus
    the trailing 'background' row/column from Ultralytics ConfusionMatrix.

    Ultralytics ConfusionMatrix stores shape (nc + 1, nc + 1) where index `nc`
    is background, rows = predicted (or true depending on version), let's
    preserve exact Ultralytics matrix orientation and document axes clearly:
    In Ultralytics `ConfusionMatrix`:
      columns = true classes + background (last col)
      rows    = predicted classes + background (last row)
    """
    cm_obj = getattr(metrics, "confusion_matrix", None)
    if cm_obj is None or getattr(cm_obj, "matrix", None) is None:
        return None

    mat = cm_obj.matrix
    nc = len(model_names)
    bg_idx = nc
    if mat.shape[0] <= bg_idx or mat.shape[1] <= bg_idx:
        return None

    indices = [*target_class_ids, bg_idx]
    labels = [model_names[cid] for cid in target_class_ids] + ["background"]
    sub = mat[indices, :][:, indices]

    return {
        "labels": labels,
        "axes_note": "rows = predicted class, columns = ground-truth class (Ultralytics convention)",
        "matrix": [[int(round(float(val))) for val in row] for row in sub],
    }


# ===========================================================================
# Evaluation helper
# ===========================================================================

def evaluate_model(
    model: YOLO,
    weights: Path,
    data_yaml: Path,
    split: str,
    device: int | str,
    label: str,
    target_eval_class_ids: list[int],
    filter_classes: list[int] | None = None,
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.6,
    project_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run Ultralytics val() on *model* against *data_yaml* and extract all metrics.
    """
    logger.info("Evaluating: %s  (split=%s, imgsz=%d)", label, split, imgsz)

    val_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "split": split,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "device": device,
        "verbose": True,
        "plots": True,
        "exist_ok": True,
    }
    if filter_classes is not None:
        val_kwargs["classes"] = filter_classes
    if project_dir is not None:
        run_name = (
            label.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        val_kwargs["project"] = str(project_dir)
        val_kwargs["name"] = run_name

    metrics = model.val(**val_kwargs)
    rd = getattr(metrics, "results_dict", {}) or {}
    speed = getattr(metrics, "speed", {}) or {}

    per_class: dict[str, dict[str, Any]] = {}
    if hasattr(metrics, "summary"):
        for row in metrics.summary(decimals=4):
            cname = str(row["Class"])
            per_class[cname] = {
                "images": int(row["Images"]),
                "instances": int(row["Instances"]),
                "precision": round(float(row["Box-P"]), 4),
                "recall": round(float(row["Box-R"]), 4),
                "f1": round(float(row["Box-F1"]), 4),
                "mAP50": round(float(row["mAP50"]), 4),
                "mAP50_95": round(float(row["mAP50-95"]), 4),
            }

    save_dir = Path(metrics.save_dir) if getattr(metrics, "save_dir", None) else None
    cm_png = None
    cm_norm_png = None
    if save_dir and save_dir.exists():
        cand = save_dir / "confusion_matrix.png"
        cand_norm = save_dir / "confusion_matrix_normalized.png"
        if cand.exists():
            cm_png = str(cand)
        if cand_norm.exists():
            cm_norm_png = str(cand_norm)

    cm_data = extract_target_confusion_matrix(
        metrics, target_eval_class_ids, model.names
    )

    result: dict[str, Any] = {
        "label": label,
        "weights": str(weights),
        "split": split,
        "imgsz": imgsz,
        "num_model_classes": len(model.names),
        "filtered_to_target_classes": filter_classes is not None,
        "precision": round(float(rd.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(rd.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(rd.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(rd.get("metrics/mAP50-95(B)", 0.0)), 4),
        "preprocess_ms_per_image": round(float(speed.get("preprocess", 0.0)), 2),
        "inference_ms_per_image": round(float(speed.get("inference", 0.0)), 2),
        "postprocess_ms_per_image": round(float(speed.get("postprocess", 0.0)), 2),
        "per_class": per_class,
        "confusion_matrix": cm_data,
        "confusion_matrix_png": cm_png,
        "confusion_matrix_normalized_png": cm_norm_png,
        "note": (
            "Metrics are computed against ground-truth test annotations on the exact "
            "same images. Keep yolo11s.pt preserved for general 80-class COCO detection."
        ),
    }

    logger.info(
        "%s -> P=%.4f  R=%.4f  mAP50=%.4f  mAP50-95=%.4f  infer=%.2f ms/img",
        label,
        result["precision"],
        result["recall"],
        result["mAP50"],
        result["mAP50_95"],
        result["inference_ms_per_image"],
    )
    return result


# ===========================================================================
# Comparison printer
# ===========================================================================

def print_comparison(results: list[dict[str, Any]], target_names: list[str]) -> None:
    """Print a side-by-side comparison of overall, per-class, and confusion metrics."""
    print("\n" + "=" * 74)
    print("EVALUATION RESULTS")
    print("=" * 74)

    col_w = 24
    headers = ["Metric"] + [r["label"] for r in results]
    print("  ".join(h.ljust(col_w) for h in headers))
    print("-" * (col_w * len(headers) + 2 * (len(headers) - 1)))

    rows = [
        ("Precision",          [f"{r['precision']:.4f}"               for r in results]),
        ("Recall",             [f"{r['recall']:.4f}"                  for r in results]),
        ("mAP@50",             [f"{r['mAP50']:.4f}"                   for r in results]),
        ("mAP@50-95",          [f"{r['mAP50_95']:.4f}"                for r in results]),
        ("Inference (ms/img)", [f"{r['inference_ms_per_image']:.2f}"  for r in results]),
        ("Model classes",      [f"{r['num_model_classes']}"           for r in results]),
    ]
    for metric_name, values in rows:
        print("  ".join([metric_name.ljust(col_w)] + [v.ljust(col_w) for v in values]))

    print(f"\n{'-'*74}")
    print("Per-class metrics (target classes):")
    target_lower = {t.lower() for t in target_names}
    for r in results:
        print(f"\n  [{r['label']}]")
        matched = {
            k: v for k, v in r["per_class"].items() if k.lower() in target_lower
        }
        if matched:
            print(
                f"    {'Class':<12} {'Imgs':>5} {'Inst':>5} "
                f"{'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}"
            )
            for cls_name, d in matched.items():
                print(
                    f"    {cls_name:<12} {d['images']:>5} {d['instances']:>5} "
                    f"{d['precision']:>8.4f} {d['recall']:>8.4f} "
                    f"{d['mAP50']:>8.4f} {d['mAP50_95']:>10.4f}"
                )
        else:
            print("    (no target-class instances evaluated)")

    print(f"\n{'-'*74}")
    print("Confusion matrices (rows = predicted, cols = true):")
    for r in results:
        print(f"\n  [{r['label']}]")
        cm = r.get("confusion_matrix")
        if cm:
            lbls = cm["labels"]
            header_row = "    " + " ".join(f"{l[:10]:>10}" for l in ["pred\\true", *lbls])
            print(header_row)
            for lbl, row_vals in zip(lbls, cm["matrix"]):
                vals_str = " ".join(f"{v:>10d}" for v in row_vals)
                print(f"    {lbl[:10]:>10} {vals_str}")
        if r.get("confusion_matrix_png"):
            print(f"    Plot: {r['confusion_matrix_png']}")

    print(f"\n{'-'*74}")
    print("FAIRNESS & SCOPE NOTE:")
    print("  * Pretrained model labels were remapped dynamically to COCO class IDs.")
    print("  * Both models were scored on identical held-out test images.")
    print("  * Custom fine-tuned weights apply only to configured custom classes;")
    print("    retain yolo11s.pt for general 80-class COCO detection.")
    if len(results) == 2:
        diff50 = results[1]["mAP50"] - results[0]["mAP50"]
        diff95 = results[1]["mAP50_95"] - results[0]["mAP50_95"]
        print(f"  * Observed test delta (Custom - Pretrained): mAP@50 = {diff50:+.4f}, mAP@50-95 = {diff95:+.4f}")
    print("=" * 74 + "\n")


# ===========================================================================
# Entry point
# ===========================================================================

def parse_class_map_overrides(raw_items: list[str] | None) -> dict[str, str]:
    """Parse ['custom:coco', ...] CLI items into a dict."""
    overrides: dict[str, str] = {}
    for item in raw_items or []:
        if ":" not in item:
            raise ValueError(f"Invalid --class-map '{item}'. Expected format 'custom:coco'.")
        k, v = item.split(":", 1)
        overrides[k.strip()] = v.strip()
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare pretrained vs fine-tuned YOLO models."
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Path to original pretrained weights (e.g. yolo11s.pt).",
    )
    parser.add_argument(
        "--finetuned",
        default=None,
        help="Path to fine-tuned weights.",
    )
    parser.add_argument(
        "--data",
        default="datasets/object_detection/dataset.yaml",
        help="Dataset YAML (default: datasets/object_detection/dataset.yaml).",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Evaluation image size (default: 640).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for mAP/PR curve validation (default: 0.001).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="NMS IoU threshold (default: 0.6).",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device (default: 0 for GPU, use 'cpu' for CPU).",
    )
    parser.add_argument(
        "--class-map",
        action="append",
        default=None,
        help="Optional custom:coco name override, e.g. --class-map novel:book",
    )
    parser.add_argument(
        "--no-filter-coco-classes",
        action="store_true",
        help="Do not restrict pretrained model predictions to target COCO classes.",
    )
    parser.add_argument(
        "--output",
        default="outputs/evaluation",
        help="Directory for evaluation reports (default: outputs/evaluation).",
    )
    args = parser.parse_args()

    if not args.pretrained and not args.finetuned:
        logger.error("Provide at least one of --pretrained or --finetuned.")
        sys.exit(1)

    device = int(args.device) if args.device.isdigit() else args.device
    data_yaml = _PROJECT_ROOT / args.data
    out_dir = _PROJECT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_yaml.exists():
        logger.error("Dataset YAML not found: %s", data_yaml)
        sys.exit(1)

    with data_yaml.open("r", encoding="utf-8") as fh:
        ds_cfg = yaml.safe_load(fh) or {}

    custom_names = parse_dataset_names(ds_cfg)
    overrides = parse_class_map_overrides(args.class_map)

    results: list[dict[str, Any]] = []
    tmp_dirs: list[Path] = []

    try:
        # --- Evaluate pretrained model (dynamically COCO-mapped) ---
        if args.pretrained:
            pretrained_path = _PROJECT_ROOT / args.pretrained
            if not pretrained_path.exists():
                logger.error("Pretrained weights not found: %s", pretrained_path)
                sys.exit(1)

            pretrained_model = YOLO(str(pretrained_path))
            custom_to_coco = build_custom_to_pretrained_map(
                custom_names, pretrained_model.names, overrides=overrides
            )
            coco_target_ids = [custom_to_coco[cid] for cid in sorted(custom_names.keys())]

            coco_yaml = create_coco_mapped_dataset(
                data_yaml,
                split=args.split,
                custom_to_coco=custom_to_coco,
                pretrained_names=pretrained_model.names,
            )
            tmp_dirs.append(coco_yaml.parent)

            results.append(
                evaluate_model(
                    model=pretrained_model,
                    weights=pretrained_path,
                    data_yaml=coco_yaml,
                    split=args.split,
                    device=device,
                    label=f"Pretrained ({pretrained_path.stem})",
                    target_eval_class_ids=coco_target_ids,
                    filter_classes=None if args.no_filter_coco_classes else coco_target_ids,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    project_dir=out_dir,
                )
            )

        # --- Evaluate fine-tuned model (native custom IDs) ---
        if args.finetuned:
            finetuned_path = _PROJECT_ROOT / args.finetuned
            if not finetuned_path.exists():
                logger.error("Fine-tuned weights not found: %s", finetuned_path)
                sys.exit(1)

            finetuned_model = YOLO(str(finetuned_path))
            results.append(
                evaluate_model(
                    model=finetuned_model,
                    weights=finetuned_path,
                    data_yaml=data_yaml,
                    split=args.split,
                    device=device,
                    label=f"Fine-tuned ({finetuned_path.stem})",
                    target_eval_class_ids=sorted(custom_names.keys()),
                    filter_classes=None,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    project_dir=out_dir,
                )
            )

        if not results:
            logger.error("No models were evaluated.")
            sys.exit(1)

        print_comparison(results, list(custom_names.values()))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"evaluation_{ts}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        logger.info("Evaluation report saved: %s", out_path)

    finally:
        for tmp in tmp_dirs:
            shutil.rmtree(str(tmp), ignore_errors=True)


if __name__ == "__main__":
    main()
