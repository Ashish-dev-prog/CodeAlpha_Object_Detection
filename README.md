# CodeAlpha Object Detection & Tracking

> YOLO11-based real-time object detection with custom training pipeline.

## Project Structure

```
CodeAlpha_Object_Detection/
├── detector.py              # Interactive live/image/video detector (run this first)
├── benchmark.py             # Offline benchmark across model/imgsz/conf matrix
├── train.py                 # Fine-tuning script (do NOT run until dataset is ready)
├── evaluate.py              # Evaluation: pretrained vs fine-tuned comparison
├── dataset_validator.py     # Dataset integrity checker
├── dataset_manifest.py      # Session-aware dataset manifest & split manager
├── CVAT_ANNOTATION_GUIDE.md # CVAT annotation & YOLO export instructions
├── DATA_COLLECTION_GUIDE.md # Best-practice data collection rules
│
├── configs/
│   ├── detection_config.yaml    # Detector settings (edit here, not in code)
│   └── train_config.yaml        # Training hyperparameters
│
├── src/
│   ├── config_loader.py         # YAML config loader with validation
│   └── detection_logger.py      # CSV/JSON detection event logger
│
├── datasets/object_detection/
│   ├── dataset.yaml             # Class names, split paths (YOLO format)
│   ├── images/{train,val,test}/ # Place images here
│   └── labels/{train,val,test}/ # Place YOLO .txt labels here
│
├── models/weights/          # Fine-tuned checkpoint copies saved here
├── outputs/
│   ├── frames/              # Saved annotated + original frames
│   ├── logs/                # Detection CSV/JSON logs
│   └── benchmarks/          # Benchmark CSV/JSON reports
├── runs/train/              # Ultralytics training runs (auto-created)
└── tests/test_all.py        # Test suite
```

---

## Quick Start

### 1. Run the interactive detector (webcam)

```powershell
cd C:\Users\Lenovo\CodeAlpha_Object_Detection
& "C:\Users\Lenovo\.venv\Scripts\python.exe" detector.py
```

Controls: **S** = save frame | **P** = pause | **Q** = quit

Adjust confidence and imgsz live via the trackbars.

### 2. Change settings without touching code

Edit [`configs/detection_config.yaml`](configs/detection_config.yaml):

```yaml
model_path:    yolo11s.pt
camera_index:  1
imgsz:         960
conf:          0.20
target_classes:
  - book
  - bed
  - chair
```

### 3. Run on a saved image

```yaml
# In configs/detection_config.yaml:
input_source: image
input_path: "C:/path/to/room.jpg"
```

### 4. Run the benchmark

```powershell
# Benchmark on a webcam snapshot:
& "C:\Users\Lenovo\.venv\Scripts\python.exe" benchmark.py --webcam --camera-index 1

# Benchmark on a saved image:
& "C:\Users\Lenovo\.venv\Scripts\python.exe" benchmark.py --input path/to/image.jpg
```

Benchmark report is saved to `outputs/benchmarks/`.

> ⚠ **Detection count is not accuracy.** Lower confidence = more boxes, not better predictions.
> Use mAP metrics on a labelled test set to measure accuracy.

### 5. Validate your dataset

```powershell
& "C:\Users\Lenovo\.venv\Scripts\python.exe" dataset_validator.py
```

### 6. Run the test suite

```powershell
& "C:\Users\Lenovo\.venv\Scripts\python.exe" tests/test_all.py

# Skip webcam test if camera isn't available:
& "C:\Users\Lenovo\.venv\Scripts\python.exe" tests/test_all.py --skip-webcam
```

---

## Custom Dataset Preparation (Phase 3 — Not yet started)

### Directory layout

Place images and YOLO `.txt` labels here:

```
datasets/object_detection/
  images/train/   img001.jpg  img002.jpg  …
  images/val/     img020.jpg  …
  images/test/    img030.jpg  …
  labels/train/   img001.txt  img002.txt  …
  labels/val/     img020.txt  …
  labels/test/    img030.txt  …
  dataset.yaml    (class names already configured)
```

### YOLO label format

One `.txt` file per image, one line per object:

```
<class_id> <x_centre> <y_centre> <width> <height>
```

All coordinates normalised 0–1 relative to image size. Example:

```
0 0.512 0.432 0.340 0.280   # book (class 0)
1 0.500 0.700 0.900 0.450   # bed  (class 1)
```

### Recommended annotation tool

Use **CVAT** ([https://app.cvat.ai](https://app.cvat.ai) or self-hosted) and export in **Ultralytics YOLO Detection 1.0** (or **YOLO 1.1**) format with the exact label order `0: book`, `1: bed`, `2: chair`.

See [CVAT_ANNOTATION_GUIDE.md](CVAT_ANNOTATION_GUIDE.md) for step-by-step setup and [DATA_COLLECTION_GUIDE.md](DATA_COLLECTION_GUIDE.md) for image collection rules.

### Session integrity — avoid data leakage

**Do NOT** shuffle individual frames from the same recording session randomly across train/val/test.
Keep all frames from one recording session in the same split using [`dataset_manifest.py`](dataset_manifest.py).

### After annotating

```powershell
# Build and verify session manifest:
& "C:\Users\Lenovo\.venv\Scripts\python.exe" dataset_manifest.py build
& "C:\Users\Lenovo\.venv\Scripts\python.exe" dataset_manifest.py verify

# Validate dataset integrity:
& "C:\Users\Lenovo\.venv\Scripts\python.exe" dataset_validator.py

# Then train (only after reviewing validation output):
& "C:\Users\Lenovo\.venv\Scripts\python.exe" train.py
```

---

## Model Information

| Model     | Size   | COCO classes | Use case |
|-----------|--------|--------------|---------|
| yolo11n.pt | ~5 MB | 80           | Fastest inference; lower accuracy |
| yolo11s.pt | ~19 MB| 80           | Balanced speed/accuracy baseline |
| custom best.pt | — | 3 (book, bed, chair) | Fine-tuned; loses other COCO classes |

> Keep `yolo11s.pt` for general-purpose detection. The fine-tuned model only detects the 3 custom classes.

### COCO class IDs (relevant)

| Class       | COCO ID |
|-------------|---------|
| chair       | 56      |
| bed         | 59      |
| refrigerator| 72      |
| book        | 73      |

---

## Environment

| Component   | Version |
|-------------|---------|
| Python      | 3.14.6  |
| PyTorch     | 2.15.0+cu132 |
| CUDA        | 13.2    |
| Ultralytics | 8.4.210 |
| OpenCV      | 5.0.0   |
| GPU         | RTX 5050 (8 GB VRAM) |

Virtual environment: `C:\Users\Lenovo\.venv`

---

## Why confidence threshold ≠ accuracy

Lowering the confidence threshold makes the model report **more** boxes but does **not** make predictions more correct. It trades false positives for false negatives. The only way to know whether a prediction is correct is to compare it against a human-verified ground-truth label.

The benchmark table reports detection counts alongside inference speed as observable metrics — not as proxies for correctness.

