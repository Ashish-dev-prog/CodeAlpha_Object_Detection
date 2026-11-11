"""
benchmark.py
============
Offline benchmark: runs YOLO inference on a single image (or webcam snapshot)
across a predefined matrix of model / imgsz / confidence configurations and
writes a structured report.

IMPORTANT — honesty policy
--------------------------
* Inference time and FPS are measured, not estimated.
* Detection count is *not* a proxy for accuracy.  A lower confidence threshold
  always increases detection count; that does not mean detections are correct.
* The report labels this explicitly.  Do not interpret count or speed metrics
  as accuracy measurements without a ground-truth annotation.

Usage
-----
  # Benchmark on a saved image:
  python benchmark.py --input path/to/image.jpg

  # Capture a single webcam snapshot and benchmark on it:
  python benchmark.py --webcam --camera-index 1

  # Use a custom config for model paths and device:
  python benchmark.py --input image.jpg --config configs/detection_config.yaml
"""

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

try:
    from ultralytics import YOLO
except ImportError as exc:
    logger.error("ultralytics not installed. Run: pip install ultralytics")
    raise SystemExit(1) from exc


# ===========================================================================
# Benchmark configuration matrix
# ===========================================================================

@dataclass
class BenchmarkConfig:
    """One row in the benchmark matrix."""
    model_path: str
    imgsz: int
    conf: float


# This is the matrix defined in the project brief.
BENCHMARK_MATRIX: list[BenchmarkConfig] = [
    BenchmarkConfig("yolo11n.pt", 640,  0.20),
    BenchmarkConfig("yolo11s.pt", 640,  0.20),
    BenchmarkConfig("yolo11s.pt", 832,  0.20),
    BenchmarkConfig("yolo11s.pt", 960,  0.20),
    BenchmarkConfig("yolo11s.pt", 960,  0.10),
    BenchmarkConfig("yolo11s.pt", 960,  0.30),
    BenchmarkConfig("yolo11s.pt", 960,  0.40),
]

WARMUP_RUNS = 3    # Discard these runs to let the GPU reach steady state.
TIMED_RUNS  = 10   # Average inference time over this many runs.


# ===========================================================================
# Result dataclass
# ===========================================================================

@dataclass
class BenchmarkResult:
    model_name: str
    imgsz: int
    conf: float
    warmup_runs: int
    timed_runs: int
    mean_inference_ms: float
    std_inference_ms: float
    min_inference_ms: float
    max_inference_ms: float
    mean_fps: float
    detection_count: int              # Count at this conf threshold
    detected_classes: list[str] = field(default_factory=list)
    accuracy_note: str = (
        "Detection count is NOT a measure of accuracy. "
        "A lower conf threshold increases count but does not improve correctness."
    )


# ===========================================================================
# Core benchmark function
# ===========================================================================

def benchmark_one(
    cfg: BenchmarkConfig,
    frame: "cv2.Mat",
    device: int | str,
    warmup: int = WARMUP_RUNS,
    timed: int = TIMED_RUNS,
) -> BenchmarkResult:
    """
    Run inference on *frame* for the given configuration and measure timing.

    Parameters
    ----------
    cfg : BenchmarkConfig
        Model path, image size and confidence threshold.
    frame : np.ndarray
        BGR image to run inference on (the same frame is used for all configs).
    device : int | str
        CUDA device index (0) or "cpu".
    warmup : int
        Number of discarded warm-up runs.
    timed : int
        Number of timed runs whose mean / std are reported.

    Returns
    -------
    BenchmarkResult
        Timing statistics and detection metadata for this configuration.
    """
    model_path = _PROJECT_ROOT / cfg.model_path
    logger.info(
        "Loading %s  |  imgsz=%d  conf=%.2f",
        cfg.model_path, cfg.imgsz, cfg.conf,
    )
    model = YOLO(str(model_path))

    predict_kwargs = dict(
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        device=device,
        verbose=False,
    )

    # Warm-up — GPU pipelines have a one-time compilation cost (TorchCompile /
    # CUDA kernel launch overhead).  Discarding these runs gives a fairer picture
    # of steady-state throughput.
    logger.info("  Warming up (%d runs)...", warmup)
    for _ in range(warmup):
        model.predict(frame, **predict_kwargs)

    # Timed runs
    times_ms: list[float] = []
    logger.info("  Timing (%d runs)...", timed)
    for _ in range(timed):
        t0 = time.perf_counter()
        results = model.predict(frame, **predict_kwargs)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = sum(times_ms) / len(times_ms)
    std_ms  = (sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5

    # Collect detections from the last run
    boxes = results[0].boxes
    if boxes is not None and len(boxes):
        detected_classes = sorted(
            {model.names[int(c)] for c in boxes.cls.tolist()}
        )
        detection_count = len(boxes)
    else:
        detected_classes = []
        detection_count = 0

    logger.info(
        "  -> mean=%.1f ms  FPS=%.1f  detections=%d  classes=%s",
        mean_ms,
        1000.0 / mean_ms if mean_ms > 0 else 0,
        detection_count,
        detected_classes,
    )

    return BenchmarkResult(
        model_name=cfg.model_path,
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        warmup_runs=warmup,
        timed_runs=timed,
        mean_inference_ms=round(mean_ms, 2),
        std_inference_ms=round(std_ms, 2),
        min_inference_ms=round(min(times_ms), 2),
        max_inference_ms=round(max(times_ms), 2),
        mean_fps=round(1000.0 / mean_ms if mean_ms > 0 else 0, 2),
        detection_count=detection_count,
        detected_classes=detected_classes,
    )


# ===========================================================================
# Report writers
# ===========================================================================

def save_csv(results: list[BenchmarkResult], path: Path) -> None:
    fieldnames = [
        "model_name", "imgsz", "conf",
        "mean_inference_ms", "std_inference_ms", "min_inference_ms", "max_inference_ms",
        "mean_fps", "detection_count", "detected_classes", "accuracy_note",
        "warmup_runs", "timed_runs",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = asdict(r)
            row["detected_classes"] = ", ".join(r.detected_classes)
            w.writerow(row)
    logger.info("CSV report saved: %s", path)


def save_json(results: list[BenchmarkResult], path: Path) -> None:
    data = [asdict(r) for r in results]
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("JSON report saved: %s", path)


def print_table(results: list[BenchmarkResult]) -> None:
    """Pretty-print the benchmark table to stdout."""
    header = (
        f"{'Model':<14} {'imgsz':>6} {'conf':>5} "
        f"{'mean ms':>8} {'±ms':>6} {'FPS':>7} "
        f"{'Dets':>5}  Detected classes"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        classes_str = ", ".join(r.detected_classes) if r.detected_classes else "(none)"
        print(
            f"{r.model_name:<14} {r.imgsz:>6} {r.conf:>5.2f} "
            f"{r.mean_inference_ms:>8.1f} {r.std_inference_ms:>6.1f} {r.mean_fps:>7.1f} "
            f"{r.detection_count:>5}  {classes_str}"
        )
    print(sep)
    print(
        "\n[NOTE] Detection count varies with confidence threshold and does NOT measure accuracy.\n"
        "   To assess accuracy, evaluate against a labelled test set using mAP metrics.\n"
    )


# ===========================================================================
# Webcam snapshot helper
# ===========================================================================

def capture_snapshot(camera_index: int) -> "cv2.Mat":
    """
    Open the webcam briefly, capture one frame, and release the camera.

    This avoids keeping the camera open during the entire benchmark, which
    can interfere with other processes.
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {camera_index}. "
            "Connect the camera or use --input to specify an image file."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Discard a few frames to let the camera exposure stabilise.
    for _ in range(5):
        cap.read()

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Could not read a frame from the webcam.")

    logger.info("Webcam snapshot captured (%dx%d).", frame.shape[1], frame.shape[0])
    return frame


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO models across imgsz / confidence configurations."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        metavar="PATH",
        help="Path to an image or video frame to use as the benchmark input.",
    )
    source.add_argument(
        "--webcam",
        action="store_true",
        help="Capture a single snapshot from the webcam and use it as input.",
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=1,
        help="Webcam index to use when --webcam is set (default: 1).",
    )
    parser.add_argument(
        "--config",
        default="configs/detection_config.yaml",
        help="Path to detection_config.yaml (used for device setting).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (e.g. 0 for GPU, 'cpu').  Reads from config if not set.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_RUNS,
        help=f"Number of warm-up runs to discard (default: {WARMUP_RUNS}).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=TIMED_RUNS,
        help=f"Number of timed runs to average (default: {TIMED_RUNS}).",
    )
    args = parser.parse_args()

    # --- Load config for device setting ---
    config_path = _PROJECT_ROOT / args.config
    config: dict[str, Any] = {}
    if config_path.exists():
        config = load_config(config_path)

    device = args.device if args.device is not None else config.get("device", 0)
    # Convert "0", "1" strings from CLI to int for CUDA
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    # --- Acquire benchmark frame ---
    if args.webcam:
        frame = capture_snapshot(args.camera_index)
    else:
        frame = cv2.imread(args.input)
        if frame is None:
            logger.error("Could not read image: %s", args.input)
            sys.exit(1)
        logger.info("Loaded image: %s (%dx%d)", args.input, frame.shape[1], frame.shape[0])

    # --- Run benchmark ---
    results: list[BenchmarkResult] = []
    for cfg in BENCHMARK_MATRIX:
        model_file = _PROJECT_ROOT / cfg.model_path
        if not model_file.exists():
            logger.warning("Model not found, skipping: %s", model_file)
            continue
        try:
            result = benchmark_one(cfg, frame, device, warmup=args.warmup, timed=args.runs)
            results.append(result)
        except Exception as exc:
            logger.error("Error benchmarking %s: %s", cfg.model_path, exc, exc_info=True)

    if not results:
        logger.error("No benchmark results were produced.")
        sys.exit(1)

    # --- Print and save ---
    print_table(results)

    bench_dir = _PROJECT_ROOT / "outputs" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_csv(results, bench_dir / f"benchmark_{ts}.csv")
    save_json(results, bench_dir / f"benchmark_{ts}.json")

    print(f"\nReports saved to: {bench_dir}")


if __name__ == "__main__":
    main()

