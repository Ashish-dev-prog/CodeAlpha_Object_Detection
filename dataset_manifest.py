"""
dataset_manifest.py
===================
Session-aware dataset manifest and split manager.

This tool maintains a manifest (JSON) that tracks:
  - Image filename
  - Recording session ID
  - Capture date
  - Class labels (IDs, names, and per-class instance counts)
  - Dataset split (train / val / test / staging)

The manifest ensures that ALL frames from the same recording session stay
in the same split, preventing data leakage. Adjacent frames are never
distributed independently across train, validation, and test.

If the available sessions cannot produce representative splits where every
target class has sufficient examples in train, val, and test, `propose` and
`verify` report the exact problem rather than silently generating an
unreliable split.

Commands
--------
  # Build or rebuild the manifest from current images + labels:
  python dataset_manifest.py build

  # Show the current manifest summary:
  python dataset_manifest.py show

  # Propose a whole-session split (checks feasibility, refuses if impossible):
  python dataset_manifest.py propose
  python dataset_manifest.py propose --save   # Save proposed assignments to manifest

  # Manually assign a session to a split:
  python dataset_manifest.py assign --session session_01 --split train

  # Verify that the manifest is consistent and all splits have
  # sufficient class coverage:
  python dataset_manifest.py verify

  # Apply the manifest: move images+labels into the split directories
  # specified by the manifest (dry-run by default):
  python dataset_manifest.py apply
  python dataset_manifest.py apply --execute

Design notes
------------
* Sessions are inferred from filename prefixes by default
  (e.g. session_01_img001.jpg, 20261111_img001.jpg, task_01_img001.jpg).
* You can place newly exported CVAT batches into
  datasets/object_detection/images/staging/ and labels/staging/ before
  assigning sessions to train/val/test.
* The tool NEVER deletes images or labels.
"""

import argparse
import json
import logging
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dataset_manifest")

try:
    import yaml
except ImportError:
    logger.error("PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

_PROJECT_ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
SCAN_LOCATIONS = ("train", "val", "test", "staging")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ===========================================================================
# Helpers
# ===========================================================================

def parse_class_names(ds_cfg: dict) -> list[str]:
    """Extract ordered class names from dataset.yaml (supports list or dict)."""
    raw_names = ds_cfg.get("names", [])
    if isinstance(raw_names, list):
        return [str(n) for n in raw_names]
    if isinstance(raw_names, dict):
        return [str(raw_names[k]) for k in sorted(raw_names.keys(), key=int)]
    return []


def infer_session_from_filename(filename: str) -> str:
    """
    Infer a session ID from an image filename.

    Strategy:
    1. 'session_XX...' or 'sess_XX...' prefix -> 'session_XX' / 'sess_XX'
    2. Date prefix (YYYYMMDD or YYYY-MM-DD)   -> 'session_YYYYMMDD'
    3. CVAT task prefix ('task_XX...')        -> 'task_XX'
    4. Otherwise                              -> 'unknown_session'
    """
    m = re.match(r"^(sess(?:ion)?[_-]?\d+)", filename, re.IGNORECASE)
    if m:
        return m.group(1).lower().replace("-", "_")

    m = re.match(r"^(\d{4}[-_]?\d{2}[-_]?\d{2})", filename)
    if m:
        return "session_" + m.group(1).replace("-", "").replace("_", "")

    m = re.match(r"^(task[_-]?\d+)", filename, re.IGNORECASE)
    if m:
        return m.group(1).lower().replace("-", "_")

    return "unknown_session"


def extract_class_counts_from_label(label_path: Path) -> tuple[dict[int, int], bool]:
    """
    Read annotation class counts from a YOLO label file.

    Returns (counts_by_class_id, is_background).
    """
    if not label_path.exists():
        return {}, False

    with label_path.open("r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    if not lines:
        return {}, True  # Empty .txt = background/negative image

    counts: dict[int, int] = defaultdict(int)
    for line in lines:
        parts = line.split()
        if parts:
            try:
                cid = int(parts[0])
                counts[cid] += 1
            except ValueError:
                pass
    return dict(counts), False


# ===========================================================================
# Manifest I/O
# ===========================================================================

def load_manifest(manifest_path: Path) -> dict:
    """Load an existing manifest or return an empty structure."""
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"entries": [], "sessions": {}, "created": datetime.now().isoformat()}


def save_manifest(manifest: dict, manifest_path: Path) -> None:
    """Save the manifest to disk."""
    manifest["last_modified"] = datetime.now().isoformat()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Manifest saved: %s", manifest_path)


# ===========================================================================
# Build command
# ===========================================================================

def build_manifest(dataset_dir: Path, class_names: list[str]) -> dict:
    """
    Scan train, val, test, and optional staging directories to build the manifest.
    """
    manifest: dict = {
        "created": datetime.now().isoformat(),
        "class_names": class_names,
        "entries": [],
        "sessions": {},
    }

    seen_files: set[str] = set()

    for location in SCAN_LOCATIONS:
        img_dir = dataset_dir / "images" / location
        lbl_dir = dataset_dir / "labels" / location

        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            filename = img_path.name
            if filename in seen_files:
                logger.warning("Duplicate filename across folders: %s", filename)
                continue
            seen_files.add(filename)

            session_id = infer_session_from_filename(filename)
            label_path = lbl_dir / (img_path.stem + ".txt")
            cid_counts, is_background = extract_class_counts_from_label(label_path)
            class_ids = sorted(cid_counts.keys())

            try:
                mtime = img_path.stat().st_mtime
                capture_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            except OSError:
                capture_date = "unknown"

            named_counts = {
                class_names[cid]: cnt
                for cid, cnt in sorted(cid_counts.items())
                if 0 <= cid < len(class_names)
            }

            entry = {
                "filename": filename,
                "session_id": session_id,
                "capture_date": capture_date,
                "class_ids": class_ids,
                "class_names": [class_names[c] for c in class_ids if 0 <= c < len(class_names)],
                "class_counts": named_counts,
                "is_background": is_background,
                "has_label": label_path.exists(),
                "split": location,
            }
            manifest["entries"].append(entry)

            if session_id not in manifest["sessions"]:
                manifest["sessions"][session_id] = {
                    "split": location,
                    "splits_seen": set(),
                    "image_count": 0,
                    "class_ids": set(),
                    "class_counts": defaultdict(int),
                }
            sess = manifest["sessions"][session_id]
            sess["splits_seen"].add(location)
            sess["image_count"] += 1
            sess["class_ids"].update(class_ids)
            for cname, cnt in named_counts.items():
                sess["class_counts"][cname] += cnt

    for sdata in manifest["sessions"].values():
        splits_seen = sorted(sdata.pop("splits_seen"))
        sdata["splits_seen"] = splits_seen
        if len(splits_seen) > 1:
            sdata["split"] = "MIXED"
        sdata["class_ids"] = sorted(sdata["class_ids"])
        sdata["class_counts"] = dict(sdata["class_counts"])

    logger.info(
        "Manifest built: %d images, %d sessions.",
        len(manifest["entries"]),
        len(manifest["sessions"]),
    )
    return manifest


# ===========================================================================
# Assign command
# ===========================================================================

def assign_session(manifest: dict, session_id: str, split: str) -> None:
    """
    Assign all images in *session_id* to *split*.
    Updates both the session record and every matching entry.
    """
    if split not in SPLITS:
        logger.error("Invalid split '%s'. Must be one of: %s", split, SPLITS)
        return

    if session_id not in manifest["sessions"]:
        logger.error("Session '%s' not found in manifest.", session_id)
        logger.info("Available sessions: %s", list(manifest["sessions"].keys()))
        return

    manifest["sessions"][session_id]["split"] = split
    manifest["sessions"][session_id]["splits_seen"] = [split]
    count = 0
    for entry in manifest["entries"]:
        if entry["session_id"] == session_id:
            entry["split"] = split
            count += 1

    logger.info("Assigned %d images from session '%s' -> %s", count, session_id, split)


# ===========================================================================
# Session-aware split planner
# ===========================================================================

def _get_session_stats(manifest: dict, class_names: list[str]) -> dict[str, dict]:
    """Aggregate image count and per-class instance counts per session_id."""
    stats: dict[str, dict] = {}
    for entry in manifest.get("entries", []):
        sid = entry["session_id"]
        if sid not in stats:
            stats[sid] = {
                "images": 0,
                "class_counts": {name: 0 for name in class_names},
            }
        stats[sid]["images"] += 1
        # Support both newer class_counts dict and older class_names list
        if "class_counts" in entry and entry["class_counts"]:
            for cname, cnt in entry["class_counts"].items():
                if cname in stats[sid]["class_counts"]:
                    stats[sid]["class_counts"][cname] += cnt
        else:
            for cname in entry.get("class_names", []):
                if cname in stats[sid]["class_counts"]:
                    stats[sid]["class_counts"][cname] += 1
    return stats


def propose_splits(
    manifest: dict,
    class_names: list[str],
    min_per_class_train: int = 5,
    min_per_class_eval: int = 3,
    target_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> tuple[dict[str, str] | None, list[str]]:
    """
    Propose a whole-session assignment of sessions -> {train, val, test}.

    Rules enforced strictly:
    1. All frames from one recording session stay in the same split.
    2. Adjacent frames are NEVER split across train/val/test.
    3. Every target class must have >= min_per_class_train instances in train,
       and >= min_per_class_eval instances in BOTH val and test.
    4. If the available sessions cannot produce a representative split,
       returns (None, problems) explaining why instead of generating an
       unreliable split.
    """
    problems: list[str] = []
    entries = manifest.get("entries", [])
    if not entries:
        return None, ["Manifest contains 0 images. Build the manifest first."]

    session_stats = _get_session_stats(manifest, class_names)

    if "unknown_session" in session_stats:
        cnt = session_stats["unknown_session"]["images"]
        problems.append(
            f"{cnt} image(s) have session_id='unknown_session'. "
            "Assign meaningful recording session IDs before splitting so frames "
            "from the same recording are never separated."
        )

    named_sessions = [sid for sid in sorted(session_stats.keys()) if sid != "unknown_session"]
    if len(named_sessions) < 3:
        problems.append(
            f"Only {len(named_sessions)} named recording session(s) found ({named_sessions}). "
            "At least 3 distinct sessions are required to populate train, val, and test "
            "without splitting a session across splits."
        )

    # Check per-class session diversity: each class must appear in at least 3 sessions
    # if it is to appear in all 3 splits without session leakage.
    for cname in class_names:
        sessions_with_cls = [
            sid for sid in named_sessions if session_stats[sid]["class_counts"][cname] > 0
        ]
        total_cls = sum(session_stats[sid]["class_counts"][cname] for sid in named_sessions)
        if len(sessions_with_cls) < 3:
            problems.append(
                f"Class '{cname}' appears in only {len(sessions_with_cls)} session(s) "
                f"({sessions_with_cls}, total {total_cls} instances). "
                "It must appear in at least 3 independent sessions to be present in "
                "train, val, and test without session leakage."
            )
        min_needed = min_per_class_train + 2 * min_per_class_eval
        if total_cls < min_needed:
            problems.append(
                f"Class '{cname}' has only {total_cls} total annotated instances, "
                f"which is fewer than the minimum {min_needed} needed "
                f"({min_per_class_train} train + {min_per_class_eval} val + {min_per_class_eval} test)."
            )

    if problems:
        return None, problems

    total_images = sum(session_stats[s]["images"] for s in named_sessions)
    min_req = {
        "train": min_per_class_train,
        "val": min_per_class_eval,
        "test": min_per_class_eval,
    }

    # --- Search for valid whole-session assignment ---
    # For small session counts (typical: 3-14 sessions), search exhaustively with
    # pruning; for larger counts, use deterministic greedy + local swaps.
    best_assignment: dict[str, str] | None = None
    best_score = float("inf")

    def is_valid(split_cls: dict[str, dict[str, int]], split_imgs: dict[str, int]) -> bool:
        for sp in SPLITS:
            if split_imgs[sp] == 0:
                return False
            for cname in class_names:
                if split_cls[sp][cname] < min_req[sp]:
                    return False
        return True

    def ratio_loss(split_imgs: dict[str, int]) -> float:
        r_tr, r_va, r_te = target_ratios
        f_tr = split_imgs["train"] / total_images
        f_va = split_imgs["val"] / total_images
        f_te = split_imgs["test"] / total_images
        return (f_tr - r_tr) ** 2 + (f_va - r_va) ** 2 + (f_te - r_te) ** 2

    if len(named_sessions) <= 12:
        # Order larger sessions first to find good assignments quickly
        ordered = sorted(named_sessions, key=lambda s: -session_stats[s]["images"])
        # Precompute suffix max available per class for branch-and-bound pruning
        n = len(ordered)
        suffix_cls: list[dict[str, int]] = [
            {c: 0 for c in class_names} for _ in range(n + 1)
        ]
        for i in range(n - 1, -1, -1):
            s = ordered[i]
            for c in class_names:
                suffix_cls[i][c] = suffix_cls[i + 1][c] + session_stats[s]["class_counts"][c]

        cur_assign: dict[str, str] = {}
        cur_imgs = {sp: 0 for sp in SPLITS}
        cur_cls = {sp: {c: 0 for c in class_names} for sp in SPLITS}

        def backtrack(idx: int) -> None:
            nonlocal best_assignment, best_score
            # Prune if any split cannot possibly reach its per-class minimum
            for sp in SPLITS:
                for c in class_names:
                    if cur_cls[sp][c] + suffix_cls[idx][c] < min_req[sp]:
                        return

            if idx == n:
                if is_valid(cur_cls, cur_imgs):
                    score = ratio_loss(cur_imgs)
                    if score < best_score:
                        best_score = score
                        best_assignment = dict(cur_assign)
                return

            sid = ordered[idx]
            s_imgs = session_stats[sid]["images"]
            s_cls = session_stats[sid]["class_counts"]

            for sp in SPLITS:
                cur_assign[sid] = sp
                cur_imgs[sp] += s_imgs
                for c in class_names:
                    cur_cls[sp][c] += s_cls[c]

                backtrack(idx + 1)

                for c in class_names:
                    cur_cls[sp][c] -= s_cls[c]
                cur_imgs[sp] -= s_imgs
                del cur_assign[sid]

        backtrack(0)
    else:
        # Deterministic greedy seed + single-session move search for >12 sessions
        ordered = sorted(named_sessions, key=lambda s: -session_stats[s]["images"])
        assign: dict[str, str] = {}
        cur_imgs = {sp: 0 for sp in SPLITS}
        cur_cls = {sp: {c: 0 for c in class_names} for sp in SPLITS}

        # Seed val and test first until requirements met, then train
        for sid in ordered:
            s_imgs = session_stats[sid]["images"]
            s_cls = session_stats[sid]["class_counts"]

            def deficit(sp: str) -> int:
                return sum(max(0, min_req[sp] - cur_cls[sp][c]) for c in class_names)

            # Pick split with largest deficit reduction, tie-breaking toward ratio target
            target_split = max(
                SPLITS,
                key=lambda sp: (
                    sum(
                        min(s_cls[c], max(0, min_req[sp] - cur_cls[sp][c]))
                        for c in class_names
                    ),
                    (target_ratios[SPLITS.index(sp)] * total_images) - cur_imgs[sp],
                ),
            )
            assign[sid] = target_split
            cur_imgs[target_split] += s_imgs
            for c in class_names:
                cur_cls[target_split][c] += s_cls[c]

        if is_valid(cur_cls, cur_imgs):
            best_assignment = assign

    if best_assignment is None:
        problems.append(
            "No whole-session partition exists where train, val, and test ALL contain "
            f"at least ({min_per_class_train}, {min_per_class_eval}, {min_per_class_eval}) "
            "instances of every target class without splitting a session."
        )
        # Detail per-session class breakdown to help user diagnose
        for sid in named_sessions:
            c_str = ", ".join(
                f"{c}={session_stats[sid]['class_counts'][c]}" for c in class_names
            )
            problems.append(
                f"  Session '{sid}' ({session_stats[sid]['images']} imgs): {c_str}"
            )
        return None, problems

    return best_assignment, []


# ===========================================================================
# Verify command
# ===========================================================================

def verify_manifest(
    manifest: dict,
    class_names: list[str],
    min_per_class_train: int = 5,
    min_per_class_eval: int = 3,
) -> list[str]:
    """
    Verify manifest integrity and representative class coverage:
    1. No session spans multiple splits (session leakage).
    2. No entries remain in 'staging'.
    3. All three splits (train, val, test) are populated.
    4. All target classes have sufficient examples in train, val, and test.
    5. No entries have session_id='unknown_session'.
    """
    warnings: list[str] = []
    entries = manifest.get("entries", [])
    if not entries:
        return ["Manifest is empty. Run 'build' after adding images."]

    # --- Session integrity ---
    session_splits: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        session_splits[entry["session_id"]].add(entry["split"])

    for sid, splits in sorted(session_splits.items()):
        if len(splits) > 1:
            warnings.append(
                f"SESSION LEAKAGE: Session '{sid}' has images in multiple splits: "
                f"{sorted(splits)}. Move all its frames to one split."
            )

    # --- Check staging / unassigned ---
    staging_count = sum(1 for e in entries if e["split"] not in SPLITS)
    if staging_count > 0:
        warnings.append(
            f"{staging_count} image(s) are still unassigned/staging. "
            "Use 'propose --save' or 'assign' + 'apply --execute' before training."
        )

    # --- Per-split image and class instance counts ---
    split_class_counts: dict[str, dict[str, int]] = {
        sp: {name: 0 for name in class_names} for sp in SPLITS
    }
    split_counts: dict[str, int] = {sp: 0 for sp in SPLITS}

    for entry in entries:
        sp = entry["split"]
        if sp not in SPLITS:
            continue
        split_counts[sp] += 1
        if "class_counts" in entry and entry["class_counts"]:
            for cname, cnt in entry["class_counts"].items():
                if cname in split_class_counts[sp]:
                    split_class_counts[sp][cname] += cnt
        else:
            for cname in entry.get("class_names", []):
                if cname in split_class_counts[sp]:
                    split_class_counts[sp][cname] += 1

    thresholds = {
        "train": min_per_class_train,
        "val": min_per_class_eval,
        "test": min_per_class_eval,
    }

    for sp in SPLITS:
        if split_counts[sp] == 0:
            warnings.append(f"Split '{sp}' has NO images assigned.")
            continue

        req = thresholds[sp]
        insufficient = [
            f"{cname} ({split_class_counts[sp][cname]} < {req})"
            for cname in class_names
            if split_class_counts[sp][cname] < req
        ]
        if insufficient:
            warnings.append(
                f"Split '{sp}' has insufficient examples for class(es): "
                f"{', '.join(insufficient)}."
            )

    unknown_count = sum(1 for e in entries if e["session_id"] == "unknown_session")
    if unknown_count > 0:
        warnings.append(
            f"{unknown_count} image(s) have session_id='unknown_session'. "
            "Name files with a session prefix (e.g. session_01_) or edit manifest.json."
        )

    return warnings


# ===========================================================================
# Apply command
# ===========================================================================

def apply_manifest(manifest: dict, dataset_dir: Path, dry_run: bool = True) -> None:
    """
    Move images and labels to the split directories specified in the manifest.
    """
    action = "Would move" if dry_run else "Moving"
    moved = 0

    for entry in manifest["entries"]:
        target_split = entry["split"]
        if target_split not in SPLITS:
            logger.warning(
                "Skipping '%s' because split '%s' is not one of %s.",
                entry["filename"], target_split, SPLITS,
            )
            continue

        filename = entry["filename"]
        stem = Path(filename).stem

        current_loc = None
        for loc in SCAN_LOCATIONS:
            candidate = dataset_dir / "images" / loc / filename
            if candidate.exists():
                current_loc = loc
                break

        if current_loc is None:
            logger.warning("Image not found in any split/staging folder: %s", filename)
            continue

        if current_loc == target_split:
            continue

        src_img = dataset_dir / "images" / current_loc / filename
        dst_img = dataset_dir / "images" / target_split / filename
        src_lbl = dataset_dir / "labels" / current_loc / f"{stem}.txt"
        dst_lbl = dataset_dir / "labels" / target_split / f"{stem}.txt"

        if dry_run:
            print(f"  {action}: {filename}  {current_loc} -> {target_split}")
        else:
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_img), str(dst_img))
            if src_lbl.exists():
                shutil.move(str(src_lbl), str(dst_lbl))
            logger.info("Moved: %s  %s -> %s", filename, current_loc, target_split)
        moved += 1

    if dry_run and moved > 0:
        print(f"\n  {moved} image(s) would be moved. Run with --execute to apply.")
    elif moved == 0:
        print("  All images are already in their assigned splits.")
    else:
        print(f"  {moved} image(s) moved.")


# ===========================================================================
# Show command
# ===========================================================================

def show_manifest(manifest: dict) -> None:
    """Print the manifest summary."""
    print(f"\n{'='*68}")
    print("DATASET MANIFEST")
    print(f"Created:       {manifest.get('created', 'unknown')}")
    print(f"Last modified: {manifest.get('last_modified', 'never')}")
    print(f"Total images:  {len(manifest.get('entries', []))}")
    print(f"Sessions:      {len(manifest.get('sessions', {}))}")
    print(f"{'='*68}")

    sessions = manifest.get("sessions", {})
    if sessions:
        print(f"\n{'Session':<24} {'Split':<10} {'Images':>7}  Class instances")
        print("-" * 68)
        for sid, sdata in sorted(sessions.items()):
            cc = sdata.get("class_counts")
            if cc:
                cls_str = ", ".join(f"{k}:{v}" for k, v in cc.items())
            else:
                cls_str = ", ".join(str(c) for c in sdata.get("class_ids", []))
            print(
                f"{sid:<24} {sdata.get('split', '?'):<10} "
                f"{sdata.get('image_count', 0):>7}  [{cls_str}]"
            )

    split_counts: dict[str, int] = defaultdict(int)
    for entry in manifest.get("entries", []):
        split_counts[entry["split"]] += 1

    print("\nSplit totals:")
    for loc in SCAN_LOCATIONS:
        if split_counts.get(loc, 0) > 0 or loc in SPLITS:
            print(f"  {loc:<10}: {split_counts.get(loc, 0)} images")

    print(f"{'='*68}\n")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Session-aware dataset manifest and split manager."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build/rebuild the manifest from current files.")
    p_build.add_argument("--dataset", default="datasets/object_detection")
    p_build.add_argument("--config", default="datasets/object_detection/dataset.yaml")

    p_show = sub.add_parser("show", help="Show the current manifest summary.")
    p_show.add_argument("--dataset", default="datasets/object_detection")

    p_assign = sub.add_parser("assign", help="Assign a session to a split.")
    p_assign.add_argument("--session", required=True, help="Session ID to reassign.")
    p_assign.add_argument("--split", required=True, choices=SPLITS, help="Target split.")
    p_assign.add_argument("--dataset", default="datasets/object_detection")

    p_propose = sub.add_parser(
        "propose",
        help="Propose representative whole-session splits (refuses if infeasible).",
    )
    p_propose.add_argument("--dataset", default="datasets/object_detection")
    p_propose.add_argument("--config", default="datasets/object_detection/dataset.yaml")
    p_propose.add_argument("--min-train", type=int, default=5, help="Min instances/class in train.")
    p_propose.add_argument("--min-eval", type=int, default=3, help="Min instances/class in val/test.")
    p_propose.add_argument(
        "--save", action="store_true",
        help="Write the proposed session assignments into manifest.json.",
    )

    p_verify = sub.add_parser("verify", help="Verify manifest consistency and class coverage.")
    p_verify.add_argument("--dataset", default="datasets/object_detection")
    p_verify.add_argument("--config", default="datasets/object_detection/dataset.yaml")
    p_verify.add_argument("--min-train", type=int, default=5)
    p_verify.add_argument("--min-eval", type=int, default=3)

    p_apply = sub.add_parser("apply", help="Move images+labels to match manifest splits.")
    p_apply.add_argument("--dataset", default="datasets/object_detection")
    p_apply.add_argument(
        "--execute", action="store_true",
        help="Actually move files (default is dry-run).",
    )

    args = parser.parse_args()
    dataset_dir = _PROJECT_ROOT / args.dataset
    manifest_path = dataset_dir / "manifest.json"

    if args.command == "build":
        config_path = _PROJECT_ROOT / args.config
        if not config_path.exists():
            logger.error("dataset.yaml not found: %s", config_path)
            sys.exit(1)
        with config_path.open("r", encoding="utf-8") as fh:
            ds_cfg = yaml.safe_load(fh) or {}
        class_names = parse_class_names(ds_cfg)
        manifest = build_manifest(dataset_dir, class_names)
        save_manifest(manifest, manifest_path)
        show_manifest(manifest)

    elif args.command == "show":
        manifest = load_manifest(manifest_path)
        show_manifest(manifest)

    elif args.command == "assign":
        manifest = load_manifest(manifest_path)
        if not manifest["entries"]:
            logger.error("Manifest is empty. Run 'build' first.")
            sys.exit(1)
        assign_session(manifest, args.session, args.split)
        save_manifest(manifest, manifest_path)

    elif args.command == "propose":
        manifest = load_manifest(manifest_path)
        config_path = _PROJECT_ROOT / args.config
        if not config_path.exists():
            logger.error("dataset.yaml not found: %s", config_path)
            sys.exit(1)
        with config_path.open("r", encoding="utf-8") as fh:
            ds_cfg = yaml.safe_load(fh) or {}
        class_names = parse_class_names(ds_cfg)

        proposal, problems = propose_splits(
            manifest,
            class_names,
            min_per_class_train=args.min_train,
            min_per_class_eval=args.min_eval,
        )
        if proposal is None:
            print(f"\n{'='*68}")
            print("CANNOT GENERATE REPRESENTATIVE SESSION SPLIT")
            print(f"{'='*68}")
            for prob in problems:
                print(f"  [ERROR] {prob}")
            print(f"{'='*68}\n")
            sys.exit(1)

        print(f"\n{'='*68}")
        print("PROPOSED WHOLE-SESSION SPLIT")
        print(f"{'='*68}")
        for sid, sp in sorted(proposal.items()):
            print(f"  {sid:<26} -> {sp}")
        print(f"{'='*68}\n")

        if args.save:
            for sid, sp in proposal.items():
                assign_session(manifest, sid, sp)
            save_manifest(manifest, manifest_path)
            print("Saved proposed assignments to manifest.json. Run 'apply --execute' to move files.")

    elif args.command == "verify":
        manifest = load_manifest(manifest_path)
        config_path = _PROJECT_ROOT / args.config
        if not config_path.exists():
            logger.error("dataset.yaml not found: %s", config_path)
            sys.exit(1)
        with config_path.open("r", encoding="utf-8") as fh:
            ds_cfg = yaml.safe_load(fh) or {}
        class_names = parse_class_names(ds_cfg)

        warnings = verify_manifest(
            manifest,
            class_names,
            min_per_class_train=args.min_train,
            min_per_class_eval=args.min_eval,
        )
        if warnings:
            print(f"\n{'='*68}")
            print(f"MANIFEST VERIFICATION - {len(warnings)} issue(s)")
            print(f"{'='*68}")
            for w in warnings:
                print(f"  [WARN] {w}")
            print(f"{'='*68}\n")
            sys.exit(1)
        else:
            print("\n[OK] Manifest verification passed - no issues found.\n")

    elif args.command == "apply":
        manifest = load_manifest(manifest_path)
        if not manifest["entries"]:
            logger.error("Manifest is empty. Run 'build' first.")
            sys.exit(1)
        apply_manifest(manifest, dataset_dir, dry_run=not args.execute)


if __name__ == "__main__":
    main()
