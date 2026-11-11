"""
detection_logger.py
====================
Writes detection events (bounding boxes, classes, confidence scores) to a
structured CSV or JSON file.

Design note: keeping logging separate from the inference loop means you can
swap, disable, or extend the logging backend without touching the main
detection code.
"""

import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


class DetectionLogger:
    """
    Thread-safe (write-lock-free in a single-threaded context) logger that
    appends detection events to a file.

    Parameters
    ----------
    output_dir : str | Path
        Directory where the log file is created.
    fmt : "csv" | "json"
        Output format.
    filename_prefix : str
        Prefix prepended to the timestamped filename.
    """

    _CSV_FIELDS = [
        "timestamp",
        "frame_index",
        "inference_ms",
        "fps",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    def __init__(
        self,
        output_dir: str | Path,
        fmt: Literal["csv", "json"] = "csv",
        filename_prefix: str = "detections",
    ) -> None:
        self._fmt = fmt
        self._frame_index: int = 0
        self._records: list[dict] = []  # Buffer for JSON mode

        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = log_dir / f"{filename_prefix}_{timestamp_str}.{fmt}"

        if fmt == "csv":
            self._csv_file = self._path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._CSV_FIELDS
            )
            self._csv_writer.writeheader()
        else:
            self._csv_file = None  # JSON is flushed on close

        logger.info("Detection log opened: %s", self._path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_frame(
        self,
        inference_ms: float,
        fps: float,
        detections: list[dict],
    ) -> None:
        """
        Record all detections for a single frame.

        Parameters
        ----------
        inference_ms : float
            Inference time for this frame in milliseconds.
        fps : float
            Measured frames per second.
        detections : list[dict]
            Each dict must contain: class_id, class_name, confidence,
            x1, y1, x2, y2 (pixel coordinates).
        """
        ts = datetime.now().isoformat(timespec="milliseconds")
        self._frame_index += 1

        if not detections:
            # Log an empty frame so frame count stays accurate.
            row = {
                "timestamp": ts,
                "frame_index": self._frame_index,
                "inference_ms": round(inference_ms, 2),
                "fps": round(fps, 2),
                "class_id": "",
                "class_name": "",
                "confidence": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
            }
            self._write(row)
            return

        for det in detections:
            row = {
                "timestamp": ts,
                "frame_index": self._frame_index,
                "inference_ms": round(inference_ms, 2),
                "fps": round(fps, 2),
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": round(det["confidence"], 4),
                "x1": int(det["x1"]),
                "y1": int(det["y1"]),
                "x2": int(det["x2"]),
                "y2": int(det["y2"]),
            }
            self._write(row)

    def close(self) -> None:
        """Flush and close the log file."""
        if self._fmt == "csv" and self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            logger.info("CSV log closed: %s", self._path)
        elif self._fmt == "json":
            with self._path.open("w", encoding="utf-8") as fh:
                json.dump(self._records, fh, indent=2)
            logger.info("JSON log written: %s", self._path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, row: dict) -> None:
        if self._fmt == "csv":
            self._csv_writer.writerow(row)
        else:
            self._records.append(row)

    def __enter__(self) -> "DetectionLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

