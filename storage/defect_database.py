"""
DefectDatabase — Persistence layer for finalized, deduplicated defects.

Supports both SQLite (for robust querying) and JSON (for simplicity).
Each physical road defect should appear exactly once in the database.
"""

import json
import csv
import os
import logging
import sqlite3
from typing import Dict, List, Optional

from rdd_dedup.tracking.defect_track import FinalizedDefect

logger = logging.getLogger(__name__)

# SQLite schema
DEFECT_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS defects (
    defect_id           TEXT PRIMARY KEY,
    defect_type         TEXT NOT NULL,
    class_id            INTEGER NOT NULL,
    avg_confidence      REAL,
    median_confidence   REAL,
    max_confidence      REAL,
    max_confidence_frame INTEGER,
    first_frame         INTEGER,
    last_frame          INTEGER,
    duration_frames     INTEGER,
    frames_observed     INTEGER,
    avg_center_x        REAL,
    avg_center_y        REAL,
    avg_bbox_area       REAL,
    avg_aspect_ratio    REAL,
    crop_path           TEXT,
    source_track_ids    TEXT,
    created_at          TEXT,
    segmentation_data   TEXT,
    severity_score      REAL,
    measurement_data    TEXT
);
"""


class DefectDatabase:
    """
    Persistence layer for finalized, deduplicated road defects.
    
    Supports:
    - SQLite: Full SQL querying, ACID transactions, production-ready
    - JSON: Simple file-based storage, human-readable
    
    Both backends support:
    - Storing individual defects
    - Bulk storage
    - Querying by type
    - Summary statistics
    - CSV/JSON export
    """

    def __init__(self, db_path: str, db_type: str = "sqlite"):
        """
        Args:
            db_path: Path to the database file
            db_type: "sqlite" or "json"
        """
        self.db_path = db_path
        self.db_type = db_type
        self._connection: Optional[sqlite3.Connection] = None

        if db_type == "sqlite":
            self._init_sqlite()
        elif db_type == "json":
            self._init_json()
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def _init_sqlite(self):
        """Initialize SQLite database with schema."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(DEFECT_TABLE_SCHEMA)
        self._connection.commit()
        logger.info("SQLite database initialized at: %s", self.db_path)

    def _init_json(self):
        """Initialize JSON file storage."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump({"defects": []}, f)
        logger.info("JSON database initialized at: %s", self.db_path)

    def store_defect(self, defect: FinalizedDefect):
        """Insert or update a single finalized defect."""
        if self.db_type == "sqlite":
            self._store_sqlite(defect)
        else:
            self._store_json(defect)

    def store_defects(self, defects: List[FinalizedDefect]):
        """Bulk store multiple finalized defects."""
        for defect in defects:
            self.store_defect(defect)

    def _store_sqlite(self, defect: FinalizedDefect):
        """Insert or replace a defect in SQLite."""
        self._connection.execute(
            """
            INSERT OR REPLACE INTO defects (
                defect_id, defect_type, class_id,
                avg_confidence, median_confidence, max_confidence,
                max_confidence_frame, first_frame, last_frame,
                duration_frames, frames_observed,
                avg_center_x, avg_center_y,
                avg_bbox_area, avg_aspect_ratio,
                crop_path, source_track_ids, created_at,
                segmentation_data, severity_score, measurement_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                defect.defect_id,
                defect.defect_type,
                defect.class_id,
                defect.avg_confidence,
                defect.median_confidence,
                defect.max_confidence,
                defect.max_confidence_frame,
                defect.first_frame,
                defect.last_frame,
                defect.detection_duration_frames,
                defect.frames_observed,
                defect.representative_center[0],
                defect.representative_center[1],
                defect.avg_bbox_area,
                defect.avg_aspect_ratio,
                defect.representative_crop_path,
                json.dumps(defect.source_track_ids),
                defect.timestamp,
                json.dumps(defect.segmentation_mask) if defect.segmentation_mask else None,
                defect.severity_score,
                json.dumps(defect.measurement_data) if defect.measurement_data else None,
            ),
        )
        self._connection.commit()

    def _store_json(self, defect: FinalizedDefect):
        """Append a defect to the JSON file."""
        data = self._read_json()

        # Check if defect already exists (update) or is new (append)
        existing_idx = None
        for i, d in enumerate(data["defects"]):
            if d["defect_id"] == defect.defect_id:
                existing_idx = i
                break

        record = self._defect_to_dict(defect)

        if existing_idx is not None:
            data["defects"][existing_idx] = record
        else:
            data["defects"].append(record)

        self._write_json(data)

    def get_all_defects(self) -> List[Dict]:
        """Retrieve all unique defects as dictionaries."""
        if self.db_type == "sqlite":
            cursor = self._connection.execute(
                "SELECT * FROM defects ORDER BY first_frame"
            )
            return [dict(row) for row in cursor.fetchall()]
        else:
            data = self._read_json()
            return data.get("defects", [])

    def get_defects_by_type(self, defect_type: str) -> List[Dict]:
        """Filter defects by class/type."""
        if self.db_type == "sqlite":
            cursor = self._connection.execute(
                "SELECT * FROM defects WHERE defect_type = ? ORDER BY first_frame",
                (defect_type,),
            )
            return [dict(row) for row in cursor.fetchall()]
        else:
            data = self._read_json()
            return [
                d for d in data.get("defects", [])
                if d.get("defect_type") == defect_type
            ]

    def get_summary(self) -> Dict:
        """Return aggregate statistics across all defects."""
        defects = self.get_all_defects()

        if not defects:
            return {
                "total_unique_defects": 0,
                "defect_counts": {},
                "avg_confidence_overall": 0.0,
            }

        # Count by type
        counts: Dict[str, int] = {}
        total_conf = 0.0
        for d in defects:
            dtype = d.get("defect_type", "Unknown")
            counts[dtype] = counts.get(dtype, 0) + 1
            total_conf += d.get("avg_confidence", 0.0)

        return {
            "total_unique_defects": len(defects),
            "defect_counts": counts,
            "avg_confidence_overall": total_conf / len(defects) if defects else 0.0,
        }

    def export_csv(self, output_path: str):
        """Export all defects to a CSV file."""
        defects = self.get_all_defects()
        if not defects:
            logger.warning("No defects to export.")
            return

        fieldnames = list(defects[0].keys())

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(defects)

        logger.info("Exported %d defects to CSV: %s", len(defects), output_path)

    def export_json(self, output_path: str):
        """Export all defects to a JSON file."""
        defects = self.get_all_defects()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"defects": defects, "summary": self.get_summary()},
                      f, indent=2, default=str)

        logger.info("Exported %d defects to JSON: %s", len(defects), output_path)

    def count(self) -> int:
        """Return total number of unique defects."""
        if self.db_type == "sqlite":
            cursor = self._connection.execute("SELECT COUNT(*) FROM defects")
            return cursor.fetchone()[0]
        else:
            data = self._read_json()
            return len(data.get("defects", []))

    def close(self):
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    # ── Internal Helpers ──

    def _defect_to_dict(self, defect: FinalizedDefect) -> Dict:
        """Convert a FinalizedDefect to a serializable dictionary."""
        return {
            "defect_id": defect.defect_id,
            "defect_type": defect.defect_type,
            "class_id": defect.class_id,
            "avg_confidence": round(defect.avg_confidence, 4),
            "median_confidence": round(defect.median_confidence, 4),
            "max_confidence": round(defect.max_confidence, 4),
            "max_confidence_frame": defect.max_confidence_frame,
            "first_frame": defect.first_frame,
            "last_frame": defect.last_frame,
            "duration_frames": defect.detection_duration_frames,
            "frames_observed": defect.frames_observed,
            "avg_center_x": round(defect.representative_center[0], 2),
            "avg_center_y": round(defect.representative_center[1], 2),
            "avg_bbox_area": round(defect.avg_bbox_area, 2),
            "avg_aspect_ratio": round(defect.avg_aspect_ratio, 4),
            "crop_path": defect.representative_crop_path,
            "source_track_ids": defect.source_track_ids,
            "created_at": defect.timestamp,
            "severity_score": defect.severity_score,
        }

    def _read_json(self) -> Dict:
        """Read the JSON database file."""
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"defects": []}

    def _write_json(self, data: Dict):
        """Write the JSON database file."""
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
