"""
DatasetManager — Manages the RDD_Ahmedabad YOLO-format dataset directory.

Handles:
- Directory structure creation (images/, labels/, crops/, videos/, metadata/)
- Saving original frames + YOLO bounding box labels
- Saving cropped detection images organized by class
- Video-aware train/val/test split assignment
- Auto-generating YOLO-compatible data.yaml
- Dataset statistics

Usage:
    from rdd_dedup.dataset.class_config import ClassConfig
    from rdd_dedup.dataset.dataset_manager import DatasetManager

    config = ClassConfig.load("configs/rdd_ahmedabad_classes.yaml")
    dm = DatasetManager("datasets/RDD_Ahmedabad", config)
    dm.ensure_structure()
    dm.add_annotation(frame, bbox, class_id, "POTHOLE", "video_01.mp4", 1245, 17, 0.91)
"""

import os
import json
import cv2
import yaml
import shutil
import datetime
import numpy as np
from typing import Dict, List, Optional, Tuple

from rdd_dedup.dataset.class_config import ClassConfig


class DatasetManager:
    """
    Manages the RDD_Ahmedabad dataset directory structure, YOLO annotations,
    crop storage, video-aware splitting, and data.yaml generation.
    """

    def __init__(self, dataset_root: str, class_config: ClassConfig):
        """
        Args:
            dataset_root: Absolute or relative path to the dataset root directory.
            class_config: ClassConfig instance with class definitions and split ratios.
        """
        self.dataset_root = os.path.abspath(dataset_root)
        self.class_config = class_config

        # Key directory paths
        self.images_dir = os.path.join(self.dataset_root, "images")
        self.labels_dir = os.path.join(self.dataset_root, "labels")
        self.crops_dir = os.path.join(self.dataset_root, "crops")
        self.videos_dir = os.path.join(self.dataset_root, "videos")
        self.metadata_dir = os.path.join(self.dataset_root, "metadata")
        self.data_yaml_path = os.path.join(self.dataset_root, "data.yaml")

        # Video-to-split registry
        self._split_registry_path = os.path.join(
            self.metadata_dir, "video_split_registry.json"
        )
        self._split_registry: Dict[str, str] = {}

    # ══════════════════════════════════════════════════════════════════════
    # Directory Structure
    # ══════════════════════════════════════════════════════════════════════

    def ensure_structure(self):
        """Create the full dataset directory tree if it does not exist."""
        splits = ["train", "val", "test"]

        # images/ and labels/ split directories
        for split in splits:
            os.makedirs(os.path.join(self.images_dir, split), exist_ok=True)
            os.makedirs(os.path.join(self.labels_dir, split), exist_ok=True)

        # crops/ per-class directories
        for class_name in self.class_config.get_class_names():
            os.makedirs(
                os.path.join(self.crops_dir, class_name), exist_ok=True
            )

        # videos/ and metadata/
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        # Load split registry if it exists
        self._load_split_registry()

        # Generate data.yaml
        self.update_data_yaml()

    # ══════════════════════════════════════════════════════════════════════
    # Annotation Saving
    # ══════════════════════════════════════════════════════════════════════

    def add_annotation(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        class_id: int,
        class_name: str,
        source_video: str,
        frame_number: int,
        track_id: int,
        confidence: float,
        crop_image: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """
        Save a single annotation to the dataset.

        Saves:
        1. Original frame → images/{split}/
        2. YOLO label → labels/{split}/
        3. Cropped detection → crops/{CLASS_NAME}/

        Args:
            frame: Original video frame (full resolution).
            bbox: Bounding box as (x1, y1, x2, y2) pixel coordinates.
            class_id: Numeric class ID in RDD_Ahmedabad scheme.
            class_name: Class name string (e.g. "POTHOLE").
            source_video: Original source video filename.
            frame_number: Frame index in the source video.
            track_id: BoT-SORT track ID.
            confidence: Detection confidence score.
            crop_image: Pre-cropped detection image (optional, will be extracted from frame if None).

        Returns:
            Dict with paths: {"image": ..., "label": ..., "crop": ...}
        """
        # Determine split for this source video
        split = self.assign_split(source_video)

        # Generate unique filename
        video_stem = os.path.splitext(os.path.basename(source_video))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"{video_stem}_f{frame_number:06d}_t{track_id}_{timestamp}"

        # ── 1. Save original frame ──
        image_filename = f"{base_name}.jpg"
        image_path = os.path.join(self.images_dir, split, image_filename)
        cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # ── 2. Save YOLO label ──
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        # Normalize to [0, 1]
        center_x = ((x1 + x2) / 2.0) / w
        center_y = ((y1 + y2) / 2.0) / h
        bbox_w = (x2 - x1) / w
        bbox_h = (y2 - y1) / h

        # Clamp to valid range
        center_x = max(0.0, min(1.0, center_x))
        center_y = max(0.0, min(1.0, center_y))
        bbox_w = max(0.0, min(1.0, bbox_w))
        bbox_h = max(0.0, min(1.0, bbox_h))

        label_filename = f"{base_name}.txt"
        label_path = os.path.join(self.labels_dir, split, label_filename)
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

        # ── 3. Save crop ──
        crop_filename = f"{base_name}_crop.jpg"
        crop_path = os.path.join(self.crops_dir, class_name, crop_filename)

        if crop_image is not None and crop_image.size > 0:
            cv2.imwrite(crop_path, crop_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            # Extract crop from frame
            cx1 = max(0, int(x1))
            cy1 = max(0, int(y1))
            cx2 = min(w, int(x2))
            cy2 = min(h, int(y2))
            if cx2 > cx1 and cy2 > cy1:
                extracted_crop = frame[cy1:cy2, cx1:cx2]
                cv2.imwrite(crop_path, extracted_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

        return {
            "image": image_path,
            "label": label_path,
            "crop": crop_path,
            "split": split,
            "image_filename": image_filename,
            "crop_filename": crop_filename,
        }

    # ══════════════════════════════════════════════════════════════════════
    # Video-Aware Split Assignment
    # ══════════════════════════════════════════════════════════════════════

    def assign_split(self, source_video: str) -> str:
        """
        Assign a train/val/test split for a source video.

        All frames from the same video always go to the same split
        to prevent data leakage. Split is assigned on first encounter
        and persisted in the video_split_registry.json.
        """
        video_key = os.path.basename(source_video)

        # Already assigned?
        if video_key in self._split_registry:
            return self._split_registry[video_key]

        # Count current videos per split
        split_counts = {"train": 0, "val": 0, "test": 0}
        for assigned_split in self._split_registry.values():
            if assigned_split in split_counts:
                split_counts[assigned_split] += 1

        total_videos = sum(split_counts.values()) + 1  # Including this new one

        # Assign based on target ratios
        train_target = self.class_config.split.train_ratio
        val_target = self.class_config.split.val_ratio

        train_current = split_counts["train"] / total_videos
        val_current = (split_counts["train"] + split_counts["val"]) / total_videos

        if train_current < train_target:
            assigned = "train"
        elif val_current < (train_target + val_target):
            assigned = "val"
        else:
            assigned = "test"

        self._split_registry[video_key] = assigned
        self._save_split_registry()
        return assigned

    def _load_split_registry(self):
        """Load video-to-split assignments from disk."""
        if os.path.exists(self._split_registry_path):
            try:
                with open(self._split_registry_path, "r", encoding="utf-8") as f:
                    self._split_registry = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._split_registry = {}
        else:
            self._split_registry = {}

    def _save_split_registry(self):
        """Persist video-to-split assignments to disk."""
        os.makedirs(os.path.dirname(self._split_registry_path), exist_ok=True)
        with open(self._split_registry_path, "w", encoding="utf-8") as f:
            json.dump(self._split_registry, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════
    # data.yaml Generation
    # ══════════════════════════════════════════════════════════════════════

    def update_data_yaml(self):
        """Auto-generate a YOLO-compatible data.yaml."""
        data = {
            "path": self.dataset_root,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": self.class_config.get_all_classes(),
        }

        with open(self.data_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # ══════════════════════════════════════════════════════════════════════
    # Dataset Statistics
    # ══════════════════════════════════════════════════════════════════════

    def get_dataset_stats(self) -> Dict:
        """
        Compute current dataset statistics.

        Returns:
            Dict with counts per split, per class, and totals.
        """
        stats = {
            "total_images": 0,
            "splits": {"train": 0, "val": 0, "test": 0},
            "classes": {},
            "videos_registered": len(self._split_registry),
            "video_assignments": dict(self._split_registry),
        }

        # Count images per split
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(self.images_dir, split)
            if os.path.exists(split_dir):
                count = len([
                    f for f in os.listdir(split_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                stats["splits"][split] = count
                stats["total_images"] += count

        # Count crops per class
        for class_name in self.class_config.get_class_names():
            class_dir = os.path.join(self.crops_dir, class_name)
            if os.path.exists(class_dir):
                count = len([
                    f for f in os.listdir(class_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                stats["classes"][class_name] = count
            else:
                stats["classes"][class_name] = 0

        return stats

    def get_stats_markdown(self) -> str:
        """Return a formatted markdown string of dataset statistics."""
        stats = self.get_dataset_stats()

        md = f"""### 📊 RDD_Ahmedabad Dataset Statistics

| Metric | Count |
|---|---|
| **Total Images** | **{stats['total_images']}** |
| Train Split | {stats['splits']['train']} |
| Val Split | {stats['splits']['val']} |
| Test Split | {stats['splits']['test']} |
| Videos Registered | {stats['videos_registered']} |

#### Per-Class Breakdown:
"""
        for class_name, count in sorted(stats["classes"].items()):
            md += f"- **{class_name}**: `{count}` crops\n"

        if stats["video_assignments"]:
            md += "\n#### Video → Split Assignments:\n"
            for video, split in stats["video_assignments"].items():
                md += f"- `{video}` → **{split}**\n"

        return md
