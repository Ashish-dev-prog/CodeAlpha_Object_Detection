# Data Collection Guide
## CodeAlpha Object Detection — Book, Bed, Chair

This guide explains how to collect high-quality training images for the
custom YOLO fine-tuning pipeline. Follow these recommendations to build a
dataset that gives the model the best chance of learning to distinguish
books, beds, and chairs in your actual environment.

---

## Initial Collection Target

| Class | Target images | Minimum |
|-------|-------------:|--------:|
| Book  | 150–200      | 50      |
| Bed   | 100–150      | 50      |
| Chair | 150–200      | 50      |
| **Background (no objects)** | 20–30 | 10 |

> **These numbers are not a guaranteed minimum for good accuracy.**
> They are a practical starting point. A model trained on 50 images per
> class may work acceptably in controlled conditions but will not
> generalise well to new rooms, lighting, or angles. More diverse data
> always helps — but 2000 nearly-identical frames do not.

---

## Recording Sessions

Organise your image collection into distinct **sessions**. A session is
one continuous recording under roughly the same conditions.

### Naming convention

Name your image files with a session prefix so the tools can group them:

```
session_01_img001.jpg
session_01_img002.jpg
session_02_img001.jpg
```

Or use a CVAT task name like `task_01_...` — the manifest tool recognises
both patterns.

### Session assignment to splits

| Split | Purpose | Recommended sessions |
|-------|---------|----------------------|
| train | Model learns from these | 5–8 sessions (variety) |
| val   | Monitor training progress | 1–2 sessions |
| test  | Final evaluation | 1–2 sessions |

**Critical rule:** All frames from one session go to the same split.
Never scatter frames from the same session across train and test.

---

## What Makes Good Training Data

### 1. Multiple physical examples

Do not photograph the same single book from 200 angles. Use **different
books** — paperbacks, hardcovers, textbooks, thin notebooks, thick novels.
Same for beds (different sizes, sheets, pillows) and chairs (office chairs,
dining chairs, armchairs, stools).

Why: the model needs to learn "bookness" as a general concept, not the
appearance of one specific book.

### 2. Different angles

For each object, capture images from:
- Front / facing directly
- 45° angle left and right
- Top-down (looking down at a book on a desk)
- Low angle (looking up at a chair)
- Back / reverse side (back of a chair, underside of a book)

### 3. Different distances

- Close-up (object fills most of the frame)
- Medium distance (object is ¼ to ½ of the frame)
- Far away (object is small, ⅛ of the frame or less)

Small objects are harder to detect. Including small-object examples
teaches the model to find them.

### 4. Different backgrounds

Photograph objects in different locations:
- On a desk
- On the floor
- On a shelf
- Against a wall
- In a hallway
- In front of a window

Why: if every book photo has the same desk in the background, the model
may learn to associate "desk" with "book" instead of learning the book's
visual features.

### 5. Lighting conditions

- Bright daylight (window light)
- Overcast / diffuse daylight
- Artificial indoor lighting (ceiling light, desk lamp)
- Dim / low light
- Mixed lighting (lamp + window)
- Backlit (object in front of a bright window)

Why: a model trained only in daylight may fail under a desk lamp.

### 6. Partial occlusion

Include images where:
- A book is partially hidden behind another object
- A chair is partially behind a desk
- A bed is partially covered by a blanket or pillow
- Objects overlap each other

Only annotate the visible portion if >30% of the object is visible.

### 7. Background / negative images

Include 10–30 images that contain **none** of the three target classes.
Examples:
- Empty desk
- Hallway with no furniture
- Close-up of a wall

These images should have empty label files (0 bytes). They teach the
model to not hallucinate objects where none exist.

---

## What Makes Bad Training Data

### ✗ Nearly-identical video frames

If you extract frames from a video at 30 FPS, most adjacent frames are
almost identical. The model memorises them instead of learning general
features.

**Solution:** Extract at most 1 frame per 2–3 seconds. Or manually
select diverse frames.

### ✗ Repetitive scenes

100 photos of the same book on the same desk from the same angle. This
teaches the model one specific appearance, not the general concept.

### ✗ All-dark or all-blurry images

These waste training capacity. Check images for minimum quality before
annotating.

### ✗ Objects too small to annotate

If the object is less than ~15×15 pixels in the image, it is too small
for reliable YOLO detection. Include some small-object examples but
ensure most of your dataset has objects at a reasonable size.

### ✗ Inconsistent annotations

Labelling the same object as "book" in one image and "chair" in another.
Always double-check labels before exporting from CVAT.

---

## Multi-Class Scenes

Many images will contain more than one target class (e.g., a bedroom
with a bed AND a chair AND books on the nightstand). This is excellent
training data — annotate all visible target objects in every frame.

---

## Webcam Capture vs. Phone Camera

| Source | Pros | Cons |
|--------|------|------|
| Webcam (640×480) | Matches deployment conditions | Low resolution, fixed position |
| Phone camera | Higher resolution, more angles | Different from deployment |

**Recommendation:** Capture from **both** sources. The webcam images match
what the model will see in production. The phone images add variety that
improves generalisation.

---

## Collection Checklist

Before annotating, verify your raw collection:

- [ ] At least 3–5 different physical examples per class
- [ ] At least 3 different rooms / backgrounds
- [ ] At least 2 different lighting conditions
- [ ] Includes close-up, medium, and distant shots
- [ ] Includes some partially occluded objects
- [ ] Includes 10+ background-only images
- [ ] No long runs of near-identical video frames
- [ ] Images are not blurry or completely dark
- [ ] Session naming is consistent (session_01_*, session_02_*, ...)
- [ ] Each session is a distinct recording condition

---

## After Collection

1. **Upload to CVAT** — see `CVAT_ANNOTATION_GUIDE.md`
2. **Annotate bounding boxes** — one CVAT task per session
3. **Export in Ultralytics YOLO format** — export as `Ultralytics YOLO Detection 1.0` (or `YOLO 1.1`) and verify `0: book, 1: bed, 2: chair`
4. **Import into project** — copy images + labels to split dirs (or `staging/` before assigning splits)
5. **Build the manifest** — `python dataset_manifest.py build`
6. **Plan or assign sessions to splits** — `python dataset_manifest.py propose` or `python dataset_manifest.py assign --session <id> --split <split>`
7. **Verify** — `python dataset_manifest.py verify`
8. **Validate** — `python dataset_validator.py`

Do not start training until the validator and manifest verification both report no errors or leakage warnings.

