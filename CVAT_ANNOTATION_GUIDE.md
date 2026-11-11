# CVAT Annotation Workflow Guide
## CodeAlpha Object Detection — Book, Bed, Chair

This guide documents how to annotate images for the custom YOLO training
pipeline using **CVAT** (Computer Vision Annotation Tool).

> CVAT is an open-source annotation tool that runs in your browser.
> It exports directly to Ultralytics YOLO Detection format.

---

## 1. Access CVAT

**Option A — CVAT.ai cloud (recommended to start)**

Go to [https://app.cvat.ai](https://app.cvat.ai) and create a free account.
No installation required.

**Option B — Self-hosted CVAT**

```bash
# Requires Docker Desktop for Windows
git clone https://github.com/cvat-ai/cvat.git
cd cvat
docker compose up -d
# Open http://localhost:8080 in your browser
```

> Do NOT install CVAT automatically.
> Choose the option that fits your setup and install it manually.

---

## 2. Create an Annotation Project

A CVAT **Project** groups all tasks and ensures every task shares the same
class labels in the same order.

1. Click **Projects** → **+** (Create a new project).
2. **Name**: `CodeAlpha_Object_Detection`
3. Under **Labels**, add classes **in this exact order**:

   | Order | Label name | Type |
   |------:|-----------|------|
   | 1     | `book`    | Rectangle |
   | 2     | `bed`     | Rectangle |
   | 3     | `chair`   | Rectangle |

   > **Why the order matters:** When CVAT exports to YOLO format, it maps
   > the first label to class ID 0, the second to class ID 1, etc.
   > This must match `dataset.yaml`:
   > ```
   > 0: book
   > 1: bed
   > 2: chair
   > ```

4. Click **Submit & Open**.

---

## 3. Create Annotation Tasks

A CVAT **Task** is a batch of images to annotate. Create one task per
recording session to keep session boundaries clear.

1. Inside the project, click **+** (Create a new task).
2. **Name**: Use a descriptive session name, e.g.:
   - `session_01_desk_daylight`
   - `session_02_bedroom_evening`
   - `session_03_living_room_lamp`
3. **Project**: Select `CodeAlpha_Object_Detection` (inherits labels).
4. **Select files**: Upload the images for this session.
5. Click **Submit & Open**.

> **Session integrity**: Each task = one recording session.
> When you later export and split into train/val/test, you assign
> entire tasks (sessions) to a split — never individual frames.

---

## 4. Annotate Bounding Boxes

1. Open the task and click **Job #1** to enter the annotation view.
2. Select the **Rectangle** drawing tool (shortcut: `N`).
3. For each object in the frame:
   a. Select the correct label from the label dropdown (`book`, `bed`, or `chair`).
   b. Draw a tight bounding box around the object.
   c. Press `N` to confirm and start the next box.
4. Navigate frames with `D` (next) and `A` (previous).
5. Save frequently: `Ctrl + S`.

### Annotation Quality Rules

- **Tight boxes**: Draw the box as close to the object boundary as
  possible. Do not include large margins of background.
- **Occluded objects**: If more than ~70% of the object is visible,
  annotate the visible portion. If less than ~30% is visible, skip it.
- **Multiple instances**: Annotate every instance of every target class
  in every frame.
- **Ambiguous objects**: If you are unsure whether something is a book,
  bed, or chair, **do not annotate it**. Only label objects you can
  identify with certainty.
- **Background frames**: Some images may intentionally contain no target
  objects. Leave them unannotated — an empty label file is a valid
  "negative example" that teaches the model what is *not* an object.

---

## 5. Export in Ultralytics YOLO Detection Format

### Per-Task Export

1. Go to the **Task** page (not the Job).
2. Click the **⋮** (three dots) menu → **Export task dataset**.
3. **Export format**:
   - **Preferred (CVAT 2.x / app.cvat.ai)**: Select **Ultralytics YOLO Detection 1.0**.
   - **Fallback (older CVAT)**: Select **YOLO 1.1**.
4. Toggle **Save images** ON if you want images bundled with labels.
5. Click **OK** and download the ZIP file.

### What the Export Contains

**Option A — `Ultralytics YOLO Detection 1.0` export:**
```
task_export.zip
├── data.yaml               # Contains names: 0: book, 1: bed, 2: chair
├── images/
│   └── train/              # Images from this task
│       ├── image_001.jpg
│       └── ...
└── labels/
    └── train/              # Matching YOLO .txt files
        ├── image_001.txt
        └── ...
```

**Option B — `YOLO 1.1` export:**
```
task_export.zip
├── obj.names               # Class names (book, bed, chair)
├── obj.data                # Metadata
├── train.txt               # List of image paths
└── obj_train_data/
    ├── image_001.jpg
    ├── image_001.txt       # YOLO labels
    └── ...
```

Each `.txt` file contains YOLO-format annotations:
```
0 0.512 0.432 0.340 0.280
2 0.300 0.600 0.200 0.150
```

### Verify the Class Mapping

For **Ultralytics YOLO Detection 1.0**, open `data.yaml` and confirm:
```yaml
names:
  0: book
  1: bed
  2: chair
```

For **YOLO 1.1**, open `obj.names` and confirm:
```
book
bed
chair
```

If the order is different, your labels in CVAT were defined in the wrong
order. **Do not proceed** — go back and fix the project label order.

---

## 6. Import into the Project

After downloading the export ZIP for each session/task:

### Step-by-Step

1. **Extract and verify class mapping first** (replace `data.yaml` check with `obj.names` if you used `YOLO 1.1`):

```powershell
# 1. Extract the ZIP to a temporary location
Expand-Archive -Path "task_session01.zip" -DestinationPath "temp_session01"

# 2. Verify class mapping (Ultralytics YOLO Detection 1.0)
Get-Content "temp_session01\data.yaml"
# Must show: 0: book, 1: bed, 2: chair
```

2. **Copy exported images and labels into the project**

   Decide whether to place each session directly into its assigned split (`train`, `val`, or `test`), **or** put all imported sessions into `datasets\object_detection\images\staging\` + `labels\staging\` and let `dataset_manifest.py` plan the session-level split.

```powershell
# Example for Ultralytics YOLO Detection 1.0 export:
# Copy session 01 → train (ensure filenames carry a session prefix like session_01_)
Copy-Item "temp_session01\images\train\*" `
    "datasets\object_detection\images\train\" -Force
Copy-Item "temp_session01\labels\train\*" `
    "datasets\object_detection\labels\train\" -Force

# Example for YOLO 1.1 export:
Copy-Item "temp_session02\obj_train_data\*.jpg" `
    "datasets\object_detection\images\train\" -Force
Copy-Item "temp_session02\obj_train_data\*.txt" `
    "datasets\object_detection\labels\train\" -Force
```

> **Tip for background images:** If an image has no annotated objects, CVAT may omit the `.txt` file or write an empty `.txt`. Ensure every intentional background image has a 0-byte `.txt` file in `labels/<split>/` so `dataset_validator.py` distinguishes intentional background images from missing labels!

### Split Assignment Rules

| Split | Purpose | Fraction | Example sessions |
|-------|---------|----------|------------------|
| train | Model learns from these | ~70-80% | Sessions 1, 2, 5, 6 |
| val   | Monitors training progress | ~10-15% | Session 3 |
| test  | Final evaluation (never seen during training) | ~10-15% | Session 4 |

> **Critical**: Assign entire sessions to one split.
> Never split a session across train and test — that is data leakage.

### Step 4 — Build Manifest & Validate

```powershell
# Build session manifest and verify split coverage
& "C:\Users\Lenovo\.venv\Scripts\python.exe" dataset_manifest.py build
& "C:\Users\Lenovo\.venv\Scripts\python.exe" dataset_manifest.py verify

# Validate image/label integrity
& "C:\Users\Lenovo\.venv\Scripts\python.exe" dataset_validator.py
```

Review both outputs before proceeding to training.

---

## 7. Label File Cross-Check

After import, spot-check a few label files against the images:

```powershell
# View label content
Get-Content "datasets\object_detection\labels\train\image_001.txt"
# Expected: lines like "0 0.5 0.4 0.3 0.2"

# Verify class IDs:
#   0 = book
#   1 = bed
#   2 = chair
```

If you see class IDs ≥ 3, or the mapping looks wrong, the CVAT project
labels were not in the correct order.

---

## 8. Common Mistakes to Avoid

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Labels defined in wrong order in CVAT | Class IDs are swapped (book ↔ chair) | Delete project, recreate with correct order |
| Mixing sessions across splits | Data leakage — inflated test metrics | Use `dataset_manifest.py` to track sessions |
| Annotating uncertain objects | Noisy labels degrade model quality | Only label what you can identify with certainty |
| Very loose bounding boxes | Model learns to include background | Tighten boxes to object boundaries |
| Exporting in wrong format | Labels are not YOLO-compatible | Always export as **Ultralytics YOLO Detection 1.0** (or **YOLO 1.1**) |
| Forgetting to save in CVAT | Annotations lost | `Ctrl + S` frequently |

---

## Dataset Structure After Import

```
datasets/object_detection/
├── dataset.yaml                    # Class config (already created)
├── images/
│   ├── train/   img001.jpg  img002.jpg  ...
│   ├── val/     img020.jpg  ...
│   └── test/    img030.jpg  ...
├── labels/
│   ├── train/   img001.txt  img002.txt  ...
│   ├── val/     img020.txt  ...
│   └── test/    img030.txt  ...
```

Every `.txt` in `labels/` corresponds to a `.jpg` in `images/` with the same
stem name. Images without any objects should have an empty `.txt` file (0 bytes).

