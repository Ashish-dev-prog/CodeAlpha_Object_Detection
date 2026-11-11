"""
detector.py
===========
Improved interactive detection tuner.

Features
--------
* Selectable input: webcam (live), image file, or video file.
* Runtime-adjustable confidence threshold via trackbar (webcam/video only).
* Adjustable image size (imgsz) via trackbar (webcam/video only).
* GPU inference via CUDA device.
* Bounding boxes, class labels and confidence scores from YOLO results.plot().
* FPS counter and per-frame inference time overlaid on the display.
* Configurable target-class filter (from YAML config).
* Save annotated frame + original frame with 'S'.
* Detection log in CSV or JSON format.
* Clean camera release and window destruction on exit.

Controls
--------
  Q          Quit
  S          Save original + annotated frame to outputs/frames/
  P          Pause / resume (video and webcam modes)
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2

# ---------------------------------------------------------------------------
# Project root for relative imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config_loader import get_target_class_ids, load_config
from detection_logger import DetectionLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("detector")

# ---------------------------------------------------------------------------
# Lazy YOLO import — give a helpful message if ultralytics is missing
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
except ImportError as exc:
    logger.error("ultralytics is not installed. Run: pip install ultralytics")
    raise SystemExit(1) from exc


# ===========================================================================
# Drawing helpers
# ===========================================================================

def draw_overlay(
    frame: "cv2.Mat",
    fps: float,
    inference_ms: float,
    conf: float,
    imgsz: int,
    detection_count: int,
    model_name: str,
) -> None:
    """
    Overlay telemetry text in the top-left corner of *frame* (in-place).

    Uses a semi-transparent black background rectangle so the text is readable
    regardless of frame content.
    """
    lines = [
        f"Model: {model_name}",
        f"imgsz: {imgsz}  conf: {conf:.2f}",
        f"FPS: {fps:.1f}  infer: {inference_ms:.1f} ms",
        f"Detections: {detection_count}",
        "S=Save  P=Pause  Q=Quit",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 1
    padding = 6
    line_height = 20

    # Measure the widest line to size the background rectangle.
    max_w = max(cv2.getTextSize(ln, font, scale, thickness)[0][0] for ln in lines)
    rect_h = line_height * len(lines) + padding * 2
    rect_w = max_w + padding * 2

    # Semi-transparent black rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (rect_w, rect_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        y = padding + line_height * (i + 1)
        cv2.putText(frame, line, (padding, y), font, scale, (0, 255, 180), thickness, cv2.LINE_AA)


def extract_detections(results: Any, model_names: dict[int, str]) -> list[dict]:
    """
    Convert YOLO results into a plain list of dictionaries — one per box.

    This makes the detection data easy to pass to the logger without coupling
    it to the ultralytics API internals.
    """
    detections: list[dict] = []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "class_id": int(box.cls[0]),
                "class_name": model_names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return detections


# ===========================================================================
# Input handlers
# ===========================================================================

def open_webcam(config: dict) -> cv2.VideoCapture:
    """Open webcam with CAP_DSHOW fallback.  Raises RuntimeError on failure."""
    idx = config["camera_index"]
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.warning("CAP_DSHOW failed for index %d, retrying without backend.", idx)
        cap.release()
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {idx}. "
            "Check that the camera is connected and not in use by another application."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get("camera_width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get("camera_height", 720))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Webcam opened at %dx%d (index %d)", actual_w, actual_h, idx)
    return cap


def open_video(path: str) -> cv2.VideoCapture:
    """Open a video file.  Raises RuntimeError if the file cannot be read."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")
    logger.info("Video file opened: %s", path)
    return cap


# ===========================================================================
# Save helpers
# ===========================================================================

def save_frames(
    output_dir: Path,
    original: "cv2.Mat",
    annotated: "cv2.Mat",
    frame_index: int,
) -> None:
    """
    Save both original and annotated frames to outputs/frames/.

    Filenames include the frame index to avoid collisions across saves.
    """
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{frame_index:06d}"
    orig_path = frames_dir / f"{stem}_original.jpg"
    ann_path = frames_dir / f"{stem}_annotated.jpg"
    cv2.imwrite(str(orig_path), original)
    cv2.imwrite(str(ann_path), annotated)
    logger.info("Saved: %s  |  %s", orig_path.name, ann_path.name)
    print(f"  Saved: {orig_path.name}  +  {ann_path.name}")


# ===========================================================================
# Main loop
# ===========================================================================

def run_detection(config_path: str = "configs/detection_config.yaml") -> None:
    """
    Entry point for the interactive detector.

    Loads config, initialises the model and input source, then runs the
    inference loop until the user quits.
    """
    # --- Configuration ---
    config_path = _PROJECT_ROOT / config_path
    config = load_config(config_path)
    logger.info("Loaded config from %s", config_path)

    output_dir = _PROJECT_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    input_source: str = config["input_source"]

    # --- Model ---
    model_path = _PROJECT_ROOT / config["model_path"]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            "Place yolo11s.pt in the project root or update 'model_path' in the config."
        )

    logger.info("Loading model: %s", model_path)
    model = YOLO(str(model_path))
    model_name = model_path.stem

    # Resolve target class IDs (None = detect all)
    class_filter = get_target_class_ids(config, model.names)
    if class_filter is not None:
        logger.info(
            "Filtering to classes: %s",
            [model.names[c] for c in class_filter],
        )
    else:
        logger.info("No class filter — detecting all COCO classes.")

    # --- Input source ---
    cap: cv2.VideoCapture | None = None
    single_image: "cv2.Mat | None" = None

    if input_source == "webcam":
        cap = open_webcam(config)
    elif input_source == "video":
        cap = open_video(config.get("input_path", ""))
    elif input_source == "image":
        img_path = config.get("input_path", "")
        single_image = cv2.imread(str(img_path))
        if single_image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        logger.info("Image loaded: %s", img_path)
    else:
        raise ValueError(f"Unknown input_source: {input_source}")

    # --- Window and trackbars (not used for single image) ---
    window = "Object Detection — Q=Quit, S=Save, P=Pause"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Initial slider values from config (integers for trackbars)
    init_conf_pct = max(1, min(80, int(config["conf"] * 100)))
    # imgsz trackbar: steps of 32 from 320 to 1280 -> slider value = (imgsz-320)/32
    imgsz_steps = [(320 + i * 32) for i in range(31)]  # 320, 352, ..., 1280
    init_imgsz_idx = min(
        range(len(imgsz_steps)),
        key=lambda i: abs(imgsz_steps[i] - config["imgsz"]),
    )

    if input_source in ("webcam", "video"):
        cv2.createTrackbar("Conf %", window, init_conf_pct, 80, lambda _: None)
        cv2.createTrackbar("imgsz idx", window, init_imgsz_idx, len(imgsz_steps) - 1, lambda _: None)

    # --- Detection logger ---
    det_logger: DetectionLogger | None = None
    if config.get("enable_logging", True):
        det_logger = DetectionLogger(
            output_dir=output_dir,
            fmt=config.get("log_format", "csv"),
        )

    # --- State ---
    paused = False
    frame_index = 0
    fps_values: list[float] = []
    fps_window = 30  # Average FPS over this many frames

    # For single-image mode: pre-compute
    if single_image is not None:
        try:
            imgsz = config["imgsz"]
            conf = config["conf"]
            t0 = time.perf_counter()
            results = model.predict(
                single_image,
                imgsz=imgsz,
                conf=conf,
                device=config["device"],
                classes=class_filter,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - t0) * 1000.0
            annotated = results[0].plot()
            detections = extract_detections(results, model.names)
            draw_overlay(annotated, 0.0, inference_ms, conf, imgsz, len(detections), model_name)

            if det_logger:
                det_logger.log_frame(inference_ms, 0.0, detections)

            cv2.imshow(window, annotated)
            print(f"\nImage inference: {inference_ms:.1f} ms | {len(detections)} detections")
            for d in detections:
                print(f"  {d['class_name']} ({d['confidence']:.2f})")

            key = cv2.waitKey(0)
            if key == ord("s") or key == ord("S"):
                save_frames(output_dir, single_image, annotated, 1)
        finally:
            if det_logger:
                det_logger.close()
            cv2.destroyAllWindows()
        return

    # --- Streaming loop (webcam / video) ---
    try:
        t_last = time.perf_counter()

        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    if input_source == "video":
                        logger.info("End of video file reached.")
                    else:
                        logger.error("Failed to read frame from camera.")
                    break
                frame_index += 1
                current_frame = frame.copy()
            # If paused, we reuse current_frame from the previous iteration.

            # --- Read trackbar positions ---
            conf_pct = cv2.getTrackbarPos("Conf %", window)
            conf = max(0.01, conf_pct / 100.0)
            imgsz_idx = cv2.getTrackbarPos("imgsz idx", window)
            imgsz = imgsz_steps[imgsz_idx]

            # --- Inference ---
            t0 = time.perf_counter()
            results = model.predict(
                current_frame,
                imgsz=imgsz,
                conf=conf,
                device=config["device"],
                classes=class_filter,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - t0) * 1000.0

            # --- FPS calculation (exponential moving average) ---
            t_now = time.perf_counter()
            frame_duration = t_now - t_last
            t_last = t_now
            if frame_duration > 0:
                fps_values.append(1.0 / frame_duration)
                if len(fps_values) > fps_window:
                    fps_values.pop(0)
            fps = sum(fps_values) / len(fps_values) if fps_values else 0.0

            # --- Render ---
            annotated = results[0].plot()
            detections = extract_detections(results, model.names)
            draw_overlay(annotated, fps, inference_ms, conf, imgsz, len(detections), model_name)

            cv2.imshow(window, annotated)

            # --- Logging ---
            if det_logger and not paused:
                det_logger.log_frame(inference_ms, fps, detections)

            # --- Key handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                logger.info("Quit requested.")
                break
            elif key == ord("s") or key == ord("S"):
                if config.get("save_original", True):
                    save_frames(output_dir, current_frame, annotated, frame_index)
                else:
                    frames_dir = output_dir / "frames"
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    ann_path = frames_dir / f"frame_{frame_index:06d}_annotated.jpg"
                    cv2.imwrite(str(ann_path), annotated)
                    print(f"  Saved: {ann_path.name}")
            elif key == ord("p") or key == ord("P"):
                paused = not paused
                logger.info("Paused: %s", paused)

            # Also quit if the window is closed via the X button.
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("Window closed by user.")
                break

    finally:
        if cap is not None:
            cap.release()
            logger.info("Camera/video capture released.")
        if det_logger is not None:
            det_logger.close()
        cv2.destroyAllWindows()
        logger.info("Windows closed. Detector shut down cleanly.")


# ===========================================================================
# Script entry point
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interactive YOLO Object Detector")
    parser.add_argument(
        "--config",
        default="configs/detection_config.yaml",
        help="Path to detection_config.yaml (default: configs/detection_config.yaml)",
    )
    args = parser.parse_args()
    run_detection(config_path=args.config)

