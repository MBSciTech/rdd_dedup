"""
AnnotationManager — Metadata CSV tracking for RDD_Ahmedabad annotations.

Maintains a persistent CSV log (annotations.csv) recording every accepted
annotation with full provenance: source video, frame number, track ID,
bounding box, class, confidence, and timestamp.

Usage:
    am = AnnotationManager("datasets/RDD_Ahmedabad/metadata")
    am.add_record(
        image_name="video01_f001245_t17_20260810.jpg",
        crop_name="video01_f001245_t17_20260810_crop.jpg",
        class_id=0, class_name="POTHOLE",
        confidence=0.91, source_video="Ahmedabad_Road_01.mp4",
        frame_number=1245, track_id=17,
        bbox_x=120, bbox_y=340, bbox_width=85, bbox_height=62,
    )
"""

import os
import csv
import datetime
import threading
from typing import Optional

import pandas as pd


# CSV column definitions
ANNOTATION_CSV_COLUMNS = [
    "image_name",
    "crop_name",
    "class_id",
    "class_name",
    "confidence",
    "source_video",
    "frame_number",
    "track_id",
    "timestamp",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "split",
    "created_at",
]


class AnnotationManager:
    """
    Thread-safe metadata CSV manager for dataset annotations.

    Each accepted annotation is appended as a row to annotations.csv
    with full provenance information for auditing and debugging.
    """

    def __init__(self, metadata_dir: str):
        """
        Args:
            metadata_dir: Path to the metadata/ directory inside the dataset.
        """
        self.metadata_dir = os.path.abspath(metadata_dir)
        self.csv_path = os.path.join(self.metadata_dir, "annotations.csv")
        self._lock = threading.Lock()

        # Ensure directory and CSV header exist
        os.makedirs(self.metadata_dir, exist_ok=True)
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        """Create the CSV file with headers if it does not exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(ANNOTATION_CSV_COLUMNS)

    def add_record(
        self,
        image_name: str,
        crop_name: str,
        class_id: int,
        class_name: str,
        confidence: float,
        source_video: str,
        frame_number: int,
        track_id: int,
        bbox_x: int,
        bbox_y: int,
        bbox_width: int,
        bbox_height: int,
        split: str = "train",
        timestamp: Optional[str] = None,
    ):
        """
        Append a single annotation record to the CSV.

        Thread-safe: uses a lock to prevent concurrent write corruption.

        Args:
            image_name: Filename of the saved original frame image.
            crop_name: Filename of the saved crop image.
            class_id: Numeric class ID.
            class_name: Human-readable class name.
            confidence: Original detection confidence [0, 1].
            source_video: Source video filename.
            frame_number: Frame index in the source video.
            track_id: BoT-SORT track ID.
            bbox_x: Bounding box x1 (pixels).
            bbox_y: Bounding box y1 (pixels).
            bbox_width: Bounding box width (pixels).
            bbox_height: Bounding box height (pixels).
            split: Dataset split ("train", "val", "test").
            timestamp: Video timestamp string (optional).
        """
        created_at = datetime.datetime.now().isoformat()

        row = [
            image_name,
            crop_name,
            class_id,
            class_name,
            round(confidence, 4),
            source_video,
            frame_number,
            track_id,
            timestamp or "",
            bbox_x,
            bbox_y,
            bbox_width,
            bbox_height,
            split,
            created_at,
        ]

        with self._lock:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

    def load_annotations(self) -> pd.DataFrame:
        """Load the full annotations CSV as a pandas DataFrame."""
        if not os.path.exists(self.csv_path):
            return pd.DataFrame(columns=ANNOTATION_CSV_COLUMNS)

        try:
            return pd.read_csv(self.csv_path, encoding="utf-8")
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return pd.DataFrame(columns=ANNOTATION_CSV_COLUMNS)

    def get_annotation_count(self) -> int:
        """Return the total number of annotations recorded."""
        if not os.path.exists(self.csv_path):
            return 0

        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                # Subtract 1 for header
                return max(0, sum(1 for _ in f) - 1)
        except IOError:
            return 0

    def get_recent_annotations(self, n: int = 10) -> pd.DataFrame:
        """Return the last N annotations."""
        df = self.load_annotations()
        return df.tail(n)
