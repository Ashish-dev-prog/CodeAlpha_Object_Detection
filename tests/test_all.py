"""
tests/test_all.py
=================
Automated test suite for the CodeAlpha Object Detection project.

Tests are grouped by what they verify:
  T1  Python compilation (syntax check) for all project scripts.
  T2  Import checks (config_loader, detection_logger).
  T3  Configuration loading with valid and invalid inputs.
  T4  Dataset validator with a small controlled fixture.
  T5  CUDA availability check.
  T6  YOLO model loading and class resolution.
  T7  Webcam initialisation and release.

Tests that CANNOT be executed headlessly (e.g. displaying OpenCV windows,
running live webcam inference with 'S' and 'Q' keys) are marked as SKIPPED
with an explanation.  A test is only reported as PASSED if it was actually
run.

Usage
-----
  cd C:\\Users\\Lenovo\\CodeAlpha_Object_Detection
  python tests/test_all.py
  python tests/test_all.py --skip-webcam   # skip the camera init test
"""

import argparse
import hashlib
import importlib
import logging
import os
import py_compile
import shutil
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Make both src/ modules AND root-level scripts importable
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Minimal test harness (no external test framework dependency)
# ---------------------------------------------------------------------------

PASSED: list[str] = []
FAILED: list[str] = []
SKIPPED: list[str] = []


def run_test(name: str, fn, *args, **kwargs) -> None:
    """Execute *fn* and record pass/fail/skip."""
    try:
        result = fn(*args, **kwargs)
        if result == "SKIP":
            SKIPPED.append(name)
            print(f"  SKIP   {name}")
        else:
            PASSED.append(name)
            print(f"  PASS   {name}")
    except Exception as exc:
        FAILED.append(name)
        print(f"  FAIL   {name}")
        print(f"         {type(exc).__name__}: {exc}")


def report() -> int:
    """Print final summary and return exit code (0 = all pass/skip)."""
    total = len(PASSED) + len(FAILED) + len(SKIPPED)
    print(f"\n{'='*55}")
    print(f"Results: {len(PASSED)} passed | {len(FAILED)} failed | {len(SKIPPED)} skipped | {total} total")
    if FAILED:
        print("FAILED tests:")
        for t in FAILED:
            print(f"  - {t}")
    print("="*55)
    return 1 if FAILED else 0


# ===========================================================================
# T1 — Python compilation (syntax)
# ===========================================================================

SCRIPTS_TO_COMPILE = [
    "detector.py",
    "benchmark.py",
    "train.py",
    "evaluate.py",
    "dataset_validator.py",
    "dataset_manifest.py",
    "src/config_loader.py",
    "src/detection_logger.py",
    "tests/test_all.py",
]


def test_compile(script_rel: str) -> None:
    path = _PROJECT_ROOT / script_rel
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")
    py_compile.compile(str(path), doraise=True)


# ===========================================================================
# T2 — Import checks
# ===========================================================================

def test_import_config_loader() -> None:
    from config_loader import load_config, get_target_class_ids  # noqa: F401


def test_import_detection_logger() -> None:
    from detection_logger import DetectionLogger  # noqa: F401


def test_import_cv2() -> None:
    import cv2  # noqa: F401


def test_import_torch() -> None:
    import torch  # noqa: F401


def test_import_ultralytics() -> None:
    from ultralytics import YOLO  # noqa: F401


def test_import_yaml() -> None:
    import yaml  # noqa: F401


def test_import_pandas() -> None:
    import pandas  # noqa: F401


# ===========================================================================
# T3 — Configuration loading
# ===========================================================================

def _write_tmp_config(content: str) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "tmp_config.yaml"
    tmp.write_text(content, encoding="utf-8")
    return tmp


def test_config_valid() -> None:
    from config_loader import load_config
    cfg_text = textwrap.dedent("""\
        input_source: webcam
        camera_index: 1
        input_path: ""
        camera_width: 1280
        camera_height: 720
        model_path: yolo11s.pt
        device: 0
        imgsz: 640
        conf: 0.20
        iou: 0.45
        target_classes:
          - book
          - bed
          - chair
        output_dir: outputs
        save_annotated: true
        save_original: true
        log_format: csv
        enable_logging: true
    """)
    tmp = _write_tmp_config(cfg_text)
    cfg = load_config(tmp)
    assert cfg["conf"] == 0.20, f"Expected conf=0.20, got {cfg['conf']}"
    assert cfg["imgsz"] == 640
    assert "book" in cfg["target_classes"]
    shutil.rmtree(tmp.parent, ignore_errors=True)


def test_config_missing_key() -> None:
    from config_loader import load_config
    cfg_text = "input_source: webcam\n"  # Many required keys are absent
    tmp = _write_tmp_config(cfg_text)
    try:
        load_config(tmp)
        raise AssertionError("Expected ValueError for missing keys — not raised.")
    except ValueError:
        pass  # Expected
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def test_config_invalid_conf() -> None:
    from config_loader import load_config
    cfg_text = textwrap.dedent("""\
        input_source: webcam
        camera_index: 1
        input_path: ""
        camera_width: 1280
        camera_height: 720
        model_path: yolo11s.pt
        device: 0
        imgsz: 640
        conf: 1.5
        iou: 0.45
        target_classes: []
        output_dir: outputs
    """)
    tmp = _write_tmp_config(cfg_text)
    try:
        load_config(tmp)
        raise AssertionError("Expected ValueError for conf=1.5 — not raised.")
    except ValueError:
        pass
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def test_config_imgsz_not_multiple_of_32() -> None:
    from config_loader import load_config
    cfg_text = textwrap.dedent("""\
        input_source: webcam
        camera_index: 1
        input_path: ""
        camera_width: 1280
        camera_height: 720
        model_path: yolo11s.pt
        device: 0
        imgsz: 650
        conf: 0.20
        iou: 0.45
        target_classes: []
        output_dir: outputs
    """)
    tmp = _write_tmp_config(cfg_text)
    try:
        load_config(tmp)
        raise AssertionError("Expected ValueError for imgsz=650 — not raised.")
    except ValueError:
        pass
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def test_config_class_resolution() -> None:
    from config_loader import get_target_class_ids
    # Simulate COCO model names (subset)
    model_names = {56: "chair", 59: "bed", 73: "book", 72: "refrigerator"}
    cfg = {"target_classes": ["book", "bed", "chair"]}
    ids = get_target_class_ids(cfg, model_names)
    assert ids == [56, 59, 73], f"Unexpected class IDs: {ids}"


def test_config_unknown_class_warning() -> None:
    """get_target_class_ids should warn but not crash on unknown class names."""
    from config_loader import get_target_class_ids
    model_names = {56: "chair", 59: "bed"}
    cfg = {"target_classes": ["book", "teleporter"]}
    # 'book' and 'teleporter' are not in model_names — should return None
    ids = get_target_class_ids(cfg, model_names)
    assert ids is None  # Neither resolved


# ===========================================================================
# T4 — Dataset validator fixture
# ===========================================================================

def _build_dataset_fixture(tmp_root: Path) -> Path:
    """
    Create a controlled dataset fixture with distinct images:
      - train: 2 valid images + 1 orphan label
      - val:   1 image missing label + 1 valid image
      - test:  1 intentional background image (empty label file)
    """
    import numpy as np
    import cv2

    for split in ("train", "val", "test"):
        (tmp_root / "images" / split).mkdir(parents=True)
        (tmp_root / "labels" / split).mkdir(parents=True)

    def write_img(path: Path, shade: int) -> None:
        img = np.full((100, 100, 3), shade, dtype=np.uint8)
        cv2.imwrite(str(path), img)

    def write_label(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    # Train: 2 valid images + labels + 1 orphan label
    write_img(tmp_root / "images" / "train" / "session_01_img001.jpg", 10)
    write_label(tmp_root / "labels" / "train" / "session_01_img001.txt", "0 0.5 0.5 0.4 0.4\n")
    write_img(tmp_root / "images" / "train" / "session_01_img002.jpg", 30)
    write_label(tmp_root / "labels" / "train" / "session_01_img002.txt", "1 0.3 0.3 0.2 0.2\n")
    write_label(tmp_root / "labels" / "train" / "orphan_only.txt", "0 0.5 0.5 0.2 0.2\n")

    # Val: 1 image missing its label, 1 valid
    write_img(tmp_root / "images" / "val" / "session_02_img003.jpg", 60)
    write_img(tmp_root / "images" / "val" / "session_02_img004.jpg", 90)
    write_label(tmp_root / "labels" / "val" / "session_02_img004.txt", "0 0.5 0.5 0.3 0.3\n")

    # Test: 1 intentional background image (empty label)
    write_img(tmp_root / "images" / "test" / "session_03_img005.jpg", 120)
    write_label(tmp_root / "labels" / "test" / "session_03_img005.txt", "")

    return tmp_root


def test_dataset_validator_fixture() -> None:
    from dataset_validator import validate_dataset

    tmp_root = Path(tempfile.mkdtemp())
    try:
        _build_dataset_fixture(tmp_root)
        rep = validate_dataset(
            dataset_dir=tmp_root,
            num_classes=2,
            class_names=["classA", "classB"],
        )
        # Train: 2 images, 2 annotations, 0 missing, 1 orphan, 0 background
        assert rep["splits"]["train"]["image_count"] == 2
        assert rep["splits"]["train"]["annotation_count"] == 2
        assert len(rep["splits"]["train"]["missing_labels"]) == 0
        assert len(rep["splits"]["train"]["orphan_labels"]) == 1
        assert len(rep["splits"]["train"]["background_images"]) == 0

        # Val: 2 images, 1 missing label
        assert rep["splits"]["val"]["image_count"] == 2
        assert len(rep["splits"]["val"]["missing_labels"]) == 1

        # Test: 1 background image (empty label, NOT reported as label error)
        assert len(rep["splits"]["test"]["background_images"]) == 1
        assert len(rep["splits"]["test"]["label_errors"]) == 0

        # Summary
        assert rep["summary"]["total_images"] == 5
        assert rep["summary"]["total_annotations"] == 3
        assert rep["summary"]["total_background_images"] == 1
        assert rep["summary"]["total_orphan_labels"] == 1
        assert rep["summary"]["cross_split_duplicates"] == 0
        assert rep["summary"]["session_leakage_count"] == 0

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_dataset_validator_bad_labels_and_corrupt_image() -> None:
    """Out-of-bounds boxes, non-positive dims, bad class IDs, and corrupt images."""
    from dataset_validator import validate_dataset
    import numpy as np
    import cv2

    tmp_root = Path(tempfile.mkdtemp())
    try:
        for split in ("train", "val", "test"):
            (tmp_root / "images" / split).mkdir(parents=True)
            (tmp_root / "labels" / split).mkdir(parents=True)

        img = np.full((100, 100, 3), 50, dtype=np.uint8)
        cv2.imwrite(str(tmp_root / "images" / "train" / "bad_box.jpg"), img)
        # Line 1: y_centre out of [0,1] + box exceeds bounds
        # Line 2: w <= 0
        # Line 3: box extends past right edge (0.9 + 0.4/2 = 1.1)
        # Line 4: class_id out of range
        (tmp_root / "labels" / "train" / "bad_box.txt").write_text(
            "0 0.5 1.5 0.3 0.3\n"
            "0 0.5 0.5 0.0 0.3\n"
            "0 0.9 0.5 0.4 0.2\n"
            "5 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )

        # Corrupt image
        corrupt_path = tmp_root / "images" / "val" / "corrupt.jpg"
        corrupt_path.write_bytes(b"not-a-valid-jpeg")
        (tmp_root / "labels" / "val" / "corrupt.txt").write_text("", encoding="utf-8")

        rep = validate_dataset(tmp_root, num_classes=3, class_names=["book", "bed", "chair"])
        assert len(rep["splits"]["train"]["label_errors"]) >= 4, (
            f"Expected >=4 label errors, got {rep['splits']['train']['label_errors']}"
        )
        assert len(rep["splits"]["val"]["corrupt_images"]) == 1
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_dataset_validator_duplicates_and_session_leakage() -> None:
    """Exact cross-split duplicates and cross-split session prefix overlap."""
    from dataset_validator import validate_dataset
    import numpy as np
    import cv2

    tmp_root = Path(tempfile.mkdtemp())
    try:
        for split in ("train", "val", "test"):
            (tmp_root / "images" / split).mkdir(parents=True)
            (tmp_root / "labels" / split).mkdir(parents=True)

        # Exact duplicate bytes in train and test
        dup_img = np.full((80, 80, 3), 77, dtype=np.uint8)
        cv2.imwrite(str(tmp_root / "images" / "train" / "session_01_a.jpg"), dup_img)
        (tmp_root / "labels" / "train" / "session_01_a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        cv2.imwrite(str(tmp_root / "images" / "test" / "session_02_copy.jpg"), dup_img)
        (tmp_root / "labels" / "test" / "session_02_copy.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        # Same session_01 in val with a different image (session prefix leakage)
        diff_img = np.full((80, 80, 3), 180, dtype=np.uint8)
        cv2.imwrite(str(tmp_root / "images" / "val" / "session_01_b.jpg"), diff_img)
        (tmp_root / "labels" / "val" / "session_01_b.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        rep = validate_dataset(tmp_root, num_classes=3, class_names=["book", "bed", "chair"])
        assert rep["summary"]["cross_split_duplicates"] == 1
        assert rep["summary"]["session_leakage_count"] >= 1
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# ===========================================================================
# T5 — CUDA availability
# ===========================================================================

def test_cuda_available() -> None:
    import torch
    assert torch.cuda.is_available(), (
        "CUDA is not available.  Check driver and PyTorch installation."
    )


def test_cuda_device_name() -> None:
    import torch
    if not torch.cuda.is_available():
        raise AssertionError("CUDA not available — skipping device name check.")
    name = torch.cuda.get_device_name(0)
    assert name, "CUDA device name is empty."
    print(f"         GPU: {name}")


# ===========================================================================
# T6 — YOLO model loading
# ===========================================================================

def test_yolo11s_loads() -> None:
    os.chdir(_PROJECT_ROOT)
    from ultralytics import YOLO
    model = YOLO("yolo11s.pt")
    assert len(model.names) == 80, f"Expected 80 COCO classes, got {len(model.names)}"


def test_yolo11n_loads() -> None:
    os.chdir(_PROJECT_ROOT)
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    assert len(model.names) == 80


def test_yolo_coco_class_ids() -> None:
    """book=73, bed=59, chair=56, refrigerator=72 in COCO."""
    os.chdir(_PROJECT_ROOT)
    from ultralytics import YOLO
    model = YOLO("yolo11s.pt")
    expected = {73: "book", 59: "bed", 56: "chair", 72: "refrigerator"}
    for cid, name in expected.items():
        assert model.names.get(cid) == name, (
            f"Expected class {cid}='{name}', got '{model.names.get(cid)}'"
        )


# ===========================================================================
# T7 — Webcam initialisation and release
# ===========================================================================

def test_webcam_init_and_release(camera_index: int = 1) -> str | None:
    """
    Try to open the webcam, read one frame, and release it.

    If no camera is detected this test is SKIPPED, not FAILED.
    A SKIP result means the hardware is unavailable at test time — it does
    not mean the code is broken.
    """
    import cv2
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        cap.release()
        return "SKIP"  # Camera not attached — not a code error

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise AssertionError("Camera opened but failed to read a frame.")

    h, w = frame.shape[:2]
    print(f"         Frame shape: {w}x{h}")
    return None  # Pass


# ===========================================================================
# T8 — Detection logger
# ===========================================================================

def test_detection_logger_csv() -> None:
    from detection_logger import DetectionLogger
    import csv
    tmp = Path(tempfile.mkdtemp())
    try:
        with DetectionLogger(output_dir=tmp, fmt="csv") as dl:
            dl.log_frame(
                inference_ms=12.5,
                fps=30.0,
                detections=[
                    {"class_id": 56, "class_name": "chair",
                     "confidence": 0.87, "x1": 10, "y1": 20, "x2": 100, "y2": 200},
                ],
            )
            dl.log_frame(inference_ms=11.0, fps=31.0, detections=[])

        # Find the CSV file and read it
        csv_files = list((tmp / "logs").glob("*.csv"))
        assert csv_files, "No CSV log file was created."
        with csv_files[0].open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2  # 1 detection + 1 empty frame
        assert rows[0]["class_name"] == "chair"
        assert rows[0]["confidence"] == "0.87"
        assert rows[1]["class_name"] == ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_detection_logger_json() -> None:
    from detection_logger import DetectionLogger
    import json
    tmp = Path(tempfile.mkdtemp())
    try:
        with DetectionLogger(output_dir=tmp, fmt="json") as dl:
            dl.log_frame(
                inference_ms=15.0,
                fps=25.0,
                detections=[
                    {"class_id": 73, "class_name": "book",
                     "confidence": 0.65, "x1": 5, "y1": 5, "x2": 50, "y2": 80},
                ],
            )
        json_files = list((tmp / "logs").glob("*.json"))
        assert json_files, "No JSON log file was created."
        with json_files[0].open(encoding="utf-8") as fh:
            data = json.load(fh)
        assert len(data) == 1
        assert data[0]["class_name"] == "book"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# T9 — Session-aware manifest & split planner
# ===========================================================================

def test_manifest_build_and_propose_valid() -> None:
    """Four multi-class sessions should yield a leakage-free representative split."""
    from dataset_manifest import build_manifest, propose_splits, verify_manifest, assign_session
    import numpy as np
    import cv2

    tmp_root = Path(tempfile.mkdtemp())
    try:
        (tmp_root / "images" / "staging").mkdir(parents=True)
        (tmp_root / "labels" / "staging").mkdir(parents=True)

        class_names = ["book", "bed", "chair"]
        # Create 4 sessions, each containing examples of all 3 classes
        for s_idx in range(1, 5):
            for f_idx in range(1, 4):
                stem = f"session_0{s_idx}_img0{f_idx}"
                img = np.full((64, 64, 3), s_idx * 30 + f_idx, dtype=np.uint8)
                cv2.imwrite(str(tmp_root / "images" / "staging" / f"{stem}.jpg"), img)
                # Put 1 book, 1 bed, 1 chair in each frame
                (tmp_root / "labels" / "staging" / f"{stem}.txt").write_text(
                    "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.3 0.3\n2 0.5 0.5 0.2 0.3\n",
                    encoding="utf-8",
                )

        manifest = build_manifest(tmp_root, class_names)
        assert len(manifest["entries"]) == 12
        assert len(manifest["sessions"]) == 4

        first = manifest["entries"][0]
        for field in ("filename", "session_id", "capture_date", "class_ids", "class_names", "split"):
            assert field in first, f"Missing manifest field: {field}"

        proposal, problems = propose_splits(
            manifest, class_names, min_per_class_train=3, min_per_class_eval=2
        )
        assert proposal is not None, f"Expected valid proposal, got problems: {problems}"
        assert set(proposal.values()) == {"train", "val", "test"}

        for sid, sp in proposal.items():
            assign_session(manifest, sid, sp)
        warnings = verify_manifest(manifest, class_names, min_per_class_train=3, min_per_class_eval=2)
        assert not warnings, f"Unexpected verify warnings: {warnings}"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_manifest_propose_refuses_unrepresentative_sessions() -> None:
    """If a class only exists in 1 session, propose_splits must refuse and explain."""
    from dataset_manifest import build_manifest, propose_splits
    import numpy as np
    import cv2

    tmp_root = Path(tempfile.mkdtemp())
    try:
        (tmp_root / "images" / "staging").mkdir(parents=True)
        (tmp_root / "labels" / "staging").mkdir(parents=True)
        class_names = ["book", "bed", "chair"]

        # 3 sessions, but 'bed' (class 1) only exists in session_01
        for s_idx in range(1, 4):
            for f_idx in range(1, 4):
                stem = f"session_0{s_idx}_img0{f_idx}"
                img = np.full((64, 64, 3), s_idx * 40 + f_idx, dtype=np.uint8)
                cv2.imwrite(str(tmp_root / "images" / "staging" / f"{stem}.jpg"), img)
                lines = ["0 0.5 0.5 0.2 0.2", "2 0.5 0.5 0.2 0.2"]
                if s_idx == 1:
                    lines.append("1 0.5 0.5 0.4 0.4")
                (tmp_root / "labels" / "staging" / f"{stem}.txt").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )

        manifest = build_manifest(tmp_root, class_names)
        proposal, problems = propose_splits(manifest, class_names)
        assert proposal is None, "Expected proposal to fail when 'bed' appears in only 1 session."
        assert any("bed" in p for p in problems), f"Expected 'bed' problem reported, got: {problems}"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# ===========================================================================
# T10 — Evaluation COCO <-> Custom class mapping
# ===========================================================================

def test_evaluate_coco_class_mapping() -> None:
    """Verify dynamic mapping from custom classes to yolo11s.pt COCO IDs."""
    os.chdir(_PROJECT_ROOT)
    from ultralytics import YOLO
    from evaluate import build_custom_to_pretrained_map, create_coco_mapped_dataset
    import yaml

    model = YOLO("yolo11s.pt")
    custom_names = {0: "book", 1: "bed", 2: "chair"}
    mapping = build_custom_to_pretrained_map(custom_names, model.names)
    assert mapping == {0: 73, 1: 59, 2: 56}, f"Unexpected mapping: {mapping}"

    # Verify temporary COCO-mapped dataset creation remaps labels accurately
    tmp_root = Path(tempfile.mkdtemp())
    coco_yaml_path = None
    try:
        for sp in ("train", "val", "test"):
            (tmp_root / "images" / sp).mkdir(parents=True)
            (tmp_root / "labels" / sp).mkdir(parents=True)

        import numpy as np, cv2
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_root / "images" / "test" / "sample.jpg"), img)
        (tmp_root / "labels" / "test" / "sample.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.3 0.3\n2 0.2 0.2 0.1 0.1\n",
            encoding="utf-8",
        )
        ds_yaml = tmp_root / "dataset.yaml"
        ds_yaml.write_text(
            yaml.safe_dump({
                "path": str(tmp_root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": 3,
                "names": ["book", "bed", "chair"],
            }),
            encoding="utf-8",
        )

        coco_yaml_path = create_coco_mapped_dataset(
            ds_yaml, split="test", custom_to_coco=mapping, pretrained_names=model.names
        )
        remapped_txt = (
            coco_yaml_path.parent / "labels" / "test" / "sample.txt"
        ).read_text(encoding="utf-8").strip().splitlines()
        assert remapped_txt == [
            "73 0.5 0.5 0.2 0.2",
            "59 0.4 0.4 0.3 0.3",
            "56 0.2 0.2 0.1 0.1",
        ], f"Unexpected remapped labels: {remapped_txt}"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        if coco_yaml_path is not None:
            shutil.rmtree(coco_yaml_path.parent, ignore_errors=True)


# ===========================================================================
# Main
# ===========================================================================

def main(skip_webcam: bool = False) -> int:
    print(f"\n{'='*55}")
    print("CodeAlpha Object Detection — Test Suite")
    print(f"Project root: {_PROJECT_ROOT}")
    print(f"{'='*55}\n")

    os.chdir(_PROJECT_ROOT)

    print("[T1] Compilation checks")
    for script in SCRIPTS_TO_COMPILE:
        run_test(f"compile:{script}", test_compile, script)

    print("\n[T2] Import checks")
    run_test("import:config_loader",    test_import_config_loader)
    run_test("import:detection_logger", test_import_detection_logger)
    run_test("import:cv2",              test_import_cv2)
    run_test("import:torch",            test_import_torch)
    run_test("import:ultralytics",      test_import_ultralytics)
    run_test("import:yaml",             test_import_yaml)
    run_test("import:pandas",           test_import_pandas)

    print("\n[T3] Configuration loading")
    run_test("config:valid_config",             test_config_valid)
    run_test("config:missing_key",              test_config_missing_key)
    run_test("config:invalid_conf",             test_config_invalid_conf)
    run_test("config:imgsz_not_multiple_of_32", test_config_imgsz_not_multiple_of_32)
    run_test("config:class_resolution",         test_config_class_resolution)
    run_test("config:unknown_class_warning",    test_config_unknown_class_warning)

    print("\n[T4] Dataset validator")
    run_test("dataset:fixture_validation",             test_dataset_validator_fixture)
    run_test("dataset:bad_labels_and_corrupt_image",   test_dataset_validator_bad_labels_and_corrupt_image)
    run_test("dataset:duplicates_and_session_leakage", test_dataset_validator_duplicates_and_session_leakage)

    print("\n[T5] CUDA environment")
    run_test("cuda:available",    test_cuda_available)
    run_test("cuda:device_name",  test_cuda_device_name)

    print("\n[T6] YOLO model loading")
    run_test("yolo:yolo11s_loads",     test_yolo11s_loads)
    run_test("yolo:yolo11n_loads",     test_yolo11n_loads)
    run_test("yolo:coco_class_ids",    test_yolo_coco_class_ids)

    print("\n[T7] Webcam initialisation")
    if skip_webcam:
        SKIPPED.append("webcam:init_and_release")
        print("  SKIP   webcam:init_and_release  (--skip-webcam flag set)")
    else:
        run_test("webcam:init_and_release", test_webcam_init_and_release, 1)

    print("\n[T8] Detection logger")
    run_test("logger:csv",  test_detection_logger_csv)
    run_test("logger:json", test_detection_logger_json)

    print("\n[T9] Session-aware manifest & split planner")
    run_test("manifest:build_and_propose_valid",            test_manifest_build_and_propose_valid)
    run_test("manifest:refuses_unrepresentative_sessions",  test_manifest_propose_refuses_unrepresentative_sessions)

    print("\n[T10] Evaluation COCO <-> custom class mapping")
    run_test("evaluate:coco_class_mapping", test_evaluate_coco_class_mapping)

    # Tests intentionally NOT automated (require a live display or manual review)
    skippable = [
        "detector:interactive_window   — requires display + keyboard input",
        "detector:save_frame_keypress  — requires 'S' key interaction",
        "benchmark:live_webcam         — requires camera + multi-minute run",
        "evaluate:val_on_dataset       — requires labelled test annotations",
        "train:full_training_run       — must not execute without user consent",
    ]
    print("\n[SKIPPED — requires human interaction or not yet applicable]")
    for s in skippable:
        SKIPPED.append(s)
        print(f"  SKIP   {s}")

    return report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-webcam",
        action="store_true",
        help="Skip the webcam initialisation test (use when no camera is attached).",
    )
    args = parser.parse_args()
    sys.exit(main(skip_webcam=args.skip_webcam))
