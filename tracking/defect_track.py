"""
DefectTrack — Core data structures for the RDD deduplication pipeline.

Contains:
- TrackStatus: Enum for track lifecycle states
- Detection: Single-frame detection from YOLO + BoT-SORT
- DefectTrack: Full lifecycle state for a tracked road defect
- FinalizedDefect: A unique, deduplicated physical road defect
- PipelineResult: Summary of a complete pipeline run
"""

import uuid
import datetime
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


class TrackStatus(Enum):
    """Lifecycle states for a DefectTrack."""
    ACTIVE = "active"           # Currently being detected in frames
    LOST = "lost"               # Recently disappeared, within buffer window
    FINALIZED = "finalized"     # Confirmed as a real defect or merged


@dataclass
class Detection:
    """
    A single detection from YOLO + BoT-SORT in one frame.
    
    This is the atomic unit of information flowing from the detection layer
    into the track manager.
    """
    track_id: int                           # BoT-SORT assigned track ID
    class_id: int                           # Numeric class ID from YOLO
    class_name: str                         # Human-readable class name
    confidence: float                       # Detection confidence [0, 1]
    bbox: Tuple[int, int, int, int]         # (x1, y1, x2, y2) pixel coordinates
    center: Tuple[float, float]            # (cx, cy) center of bounding box
    frame_idx: int                          # Frame index in the video

    @staticmethod
    def compute_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Compute the center point of a bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def compute_area(bbox: Tuple[int, int, int, int]) -> float:
        """Compute the area of a bounding box."""
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def compute_aspect_ratio(bbox: Tuple[int, int, int, int]) -> float:
        """Compute width/height aspect ratio of a bounding box."""
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        return w / h


@dataclass
class DefectTrack:
    """
    Full lifecycle state for a single tracked road defect.
    
    The TrackManager owns and mutates these objects as detections arrive.
    BoT-SORT provides the track_id; everything else is managed independently.
    """
    track_id: int                           # BoT-SORT assigned ID
    defect_class: str                       # e.g., "D40: Pothole"
    class_id: int                           # Numeric class ID

    # Temporal
    first_frame: int = 0
    last_frame: int = 0
    frames_seen: int = 0                   # Total frames where detection was present

    # Confidence
    confidence_history: List[float] = field(default_factory=list)
    max_confidence: float = 0.0
    max_confidence_frame: int = 0

    # Bounding Box
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    center_history: List[Tuple[float, float]] = field(default_factory=list)

    # Best Crop — the image patch at the highest-confidence frame
    best_crop: Optional[np.ndarray] = field(default=None, repr=False)
    best_crop_bbox: Optional[Tuple[int, int, int, int]] = None
    best_seg_crop: Optional[np.ndarray] = field(default=None, repr=False)
    best_seg_crop_offset: Optional[Tuple[int, int]] = None

    # Motion
    cumulative_homography: Optional[np.ndarray] = field(default=None, repr=False)

    # Status
    status: TrackStatus = TrackStatus.ACTIVE
    lost_since_frame: Optional[int] = None  # Frame when status changed to LOST

    # Cached computed properties (invalidated on update)
    _avg_confidence: Optional[float] = field(default=None, repr=False)
    _median_confidence: Optional[float] = field(default=None, repr=False)
    _avg_area: Optional[float] = field(default=None, repr=False)
    _avg_aspect_ratio: Optional[float] = field(default=None, repr=False)

    def invalidate_cache(self):
        """Clear cached computed properties after an update."""
        self._avg_confidence = None
        self._median_confidence = None
        self._avg_area = None
        self._avg_aspect_ratio = None

    # ── Computed Properties ──

    @property
    def avg_confidence(self) -> float:
        """Mean of all per-frame confidence values."""
        if self._avg_confidence is None:
            if self.confidence_history:
                self._avg_confidence = float(np.mean(self.confidence_history))
            else:
                self._avg_confidence = 0.0
        return self._avg_confidence

    @property
    def median_confidence(self) -> float:
        """Median confidence — robust to outlier frames."""
        if self._median_confidence is None:
            if self.confidence_history:
                self._median_confidence = float(np.median(self.confidence_history))
            else:
                self._median_confidence = 0.0
        return self._median_confidence

    @property
    def trimmed_mean_confidence(self) -> float:
        """Trimmed mean (drop top/bottom 10%) — most robust aggregation."""
        h = self.confidence_history
        if not h:
            return 0.0
        if len(h) < 5:
            return float(np.mean(h))
        trim_count = max(1, len(h) // 10)
        sorted_h = sorted(h)
        return float(np.mean(sorted_h[trim_count:-trim_count]))

    @property
    def avg_area(self) -> float:
        """Average bounding box area across all observed frames."""
        if self._avg_area is None:
            if self.bbox_history:
                areas = [Detection.compute_area(bb) for bb in self.bbox_history]
                self._avg_area = float(np.mean(areas))
            else:
                self._avg_area = 0.0
        return self._avg_area

    @property
    def avg_aspect_ratio(self) -> float:
        """Average width/height ratio across all observed frames."""
        if self._avg_aspect_ratio is None:
            if self.bbox_history:
                ars = [Detection.compute_aspect_ratio(bb) for bb in self.bbox_history]
                self._avg_aspect_ratio = float(np.mean(ars))
            else:
                self._avg_aspect_ratio = 1.0
        return self._avg_aspect_ratio

    @property
    def representative_center(self) -> Tuple[float, float]:
        """Center position at the frame of maximum confidence."""
        if self.best_crop_bbox:
            return Detection.compute_center(self.best_crop_bbox)
        if self.center_history:
            return self.center_history[-1]
        return (0.0, 0.0)

    @property
    def duration_frames(self) -> int:
        """Total frame span from first to last detection."""
        return max(0, self.last_frame - self.first_frame + 1)

    def get_final_confidence(self, method: str = "trimmed_mean") -> float:
        """Get the final aggregated confidence using the specified method."""
        if method == "mean":
            return self.avg_confidence
        elif method == "median":
            return self.median_confidence
        elif method == "trimmed_mean":
            return self.trimmed_mean_confidence
        else:
            return self.avg_confidence


@dataclass
class FinalizedDefect:
    """
    A unique physical road defect, after deduplication.
    
    This is the final output record — each physical defect on the road
    should appear exactly once as a FinalizedDefect.
    """
    defect_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    defect_type: str = ""                       # e.g., "D40: Pothole"
    class_id: int = 0

    # Confidence
    avg_confidence: float = 0.0
    median_confidence: float = 0.0
    max_confidence: float = 0.0
    max_confidence_frame: int = 0

    # Temporal
    first_frame: int = 0
    last_frame: int = 0
    detection_duration_frames: int = 0
    frames_observed: int = 0

    # Spatial
    representative_center: Tuple[float, float] = (0.0, 0.0)
    avg_bbox_area: float = 0.0
    avg_aspect_ratio: float = 0.0
    bbox_stats: Dict = field(default_factory=dict)

    # Media
    representative_crop_path: str = ""         # Path to saved best crop image

    # Motion
    cumulative_homography: Optional[np.ndarray] = field(default=None, repr=False)

    # Metadata
    source_track_ids: List[int] = field(default_factory=list)
    source_track_intervals: List[Tuple[int, int]] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )

    # Future extensions
    segmentation_mask: Optional[Any] = None
    severity_score: Optional[float] = None
    measurement_data: Optional[Dict] = None

    @classmethod
    def from_track(cls, track: DefectTrack, crop_path: str = "",
                   confidence_method: str = "trimmed_mean", segmentation_result: Optional[Dict] = None) -> "FinalizedDefect":
        """
        Create a FinalizedDefect from a completed DefectTrack.
        
        Args:
            track: The finalized DefectTrack
            crop_path: Path where the representative crop was saved
            confidence_method: Aggregation method for final confidence
        """
        # Compute bbox statistics
        areas = [Detection.compute_area(bb) for bb in track.bbox_history] if track.bbox_history else [0]
        ars = [Detection.compute_aspect_ratio(bb) for bb in track.bbox_history] if track.bbox_history else [1.0]

        bbox_stats = {
            "min_area": float(min(areas)) if areas else 0.0,
            "max_area": float(max(areas)) if areas else 0.0,
            "avg_area": float(np.mean(areas)) if areas else 0.0,
            "min_aspect_ratio": float(min(ars)) if ars else 1.0,
            "max_aspect_ratio": float(max(ars)) if ars else 1.0,
            "avg_aspect_ratio": float(np.mean(ars)) if ars else 1.0,
        }

        fd = cls(
            defect_type=track.defect_class,
            class_id=track.class_id,
            cumulative_homography=track.cumulative_homography,
            avg_confidence=track.get_final_confidence(confidence_method),
            median_confidence=track.median_confidence,
            max_confidence=track.max_confidence,
            max_confidence_frame=track.max_confidence_frame,
            first_frame=track.first_frame,
            last_frame=track.last_frame,
            detection_duration_frames=track.duration_frames,
            frames_observed=track.frames_seen,
            representative_center=track.representative_center,
            avg_bbox_area=track.avg_area,
            avg_aspect_ratio=track.avg_aspect_ratio,
            bbox_stats=bbox_stats,
            representative_crop_path=crop_path,
            source_track_ids=[track.track_id],
            source_track_intervals=[(track.first_frame, track.last_frame)],
        )
        
        if segmentation_result:
            fd.segmentation_mask = segmentation_result.get("mask_path")
            fd.measurement_data = {
                "pixel_area": segmentation_result.get("pixel_area"),
                "mask_quality_score": segmentation_result.get("mask_quality_score"),
                "overlay_path": segmentation_result.get("overlay_path")
            }
        return fd

    def merge_from(self, other: "FinalizedDefect"):
        """
        Merge another FinalizedDefect into this one (absorb duplicate).
        
        Updates temporal ranges, observation counts, confidence stats,
        and keeps the best representative crop.
        """
        # Temporal
        self.first_frame = min(self.first_frame, other.first_frame)
        self.last_frame = max(self.last_frame, other.last_frame)
        self.detection_duration_frames = self.last_frame - self.first_frame + 1
        self.frames_observed += other.frames_observed

        # Confidence — keep the better stats
        if other.max_confidence > self.max_confidence:
            self.max_confidence = other.max_confidence
            self.max_confidence_frame = other.max_confidence_frame
            self.representative_center = other.representative_center
            if other.representative_crop_path:
                self.representative_crop_path = other.representative_crop_path
            if other.cumulative_homography is not None:
                self.cumulative_homography = other.cumulative_homography
            if other.segmentation_mask is not None:
                self.segmentation_mask = other.segmentation_mask
            if other.measurement_data is not None:
                self.measurement_data = other.measurement_data

        # Weighted average of confidences
        total_frames = self.frames_observed
        if total_frames > 0:
            w1 = (self.frames_observed - other.frames_observed) / total_frames
            w2 = other.frames_observed / total_frames
            self.avg_confidence = w1 * self.avg_confidence + w2 * other.avg_confidence

        # Source tracks
        self.source_track_ids.extend(other.source_track_ids)
        self.source_track_intervals.extend(other.source_track_intervals)


@dataclass
class PipelineResult:
    """Summary of a complete pipeline run."""
    video_path: str = ""
    total_frames: int = 0
    processed_frames: int = 0
    processing_time_sec: float = 0.0
    fps: float = 0.0

    total_raw_detections: int = 0       # Before dedup
    total_unique_defects: int = 0       # After dedup
    tracks_created: int = 0             # Total BoT-SORT tracks seen
    tracks_merged: int = 0              # Number of merge operations

    defect_counts: Dict[str, int] = field(default_factory=dict)
    finalized_defects: List[FinalizedDefect] = field(default_factory=list)

    output_dir: str = ""
    database_path: str = ""
    annotated_video_path: Optional[str] = None
    report_path: Optional[str] = None

    def to_dict(self) -> Dict:
        """Serialize to a dictionary for JSON export."""
        return {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "processing_time_sec": round(self.processing_time_sec, 2),
            "fps": round(self.fps, 1),
            "total_raw_detections": self.total_raw_detections,
            "total_unique_defects": self.total_unique_defects,
            "tracks_created": self.tracks_created,
            "tracks_merged": self.tracks_merged,
            "defect_counts": self.defect_counts,
            "output_dir": self.output_dir,
            "database_path": self.database_path,
            "annotated_video_path": self.annotated_video_path,
            "report_path": self.report_path,
            "finalized_defects": [
                {
                    "defect_id": d.defect_id,
                    "defect_type": d.defect_type,
                    "avg_confidence": round(d.avg_confidence, 4),
                    "max_confidence": round(d.max_confidence, 4),
                    "max_confidence_frame": d.max_confidence_frame,
                    "frames_observed": d.frames_observed,
                    "first_frame": d.first_frame,
                    "last_frame": d.last_frame,
                    "representative_center": list(d.representative_center),
                    "crop_path": d.representative_crop_path,
                    "source_track_ids": d.source_track_ids,
                    "pixel_area": d.measurement_data.get("pixel_area") if d.measurement_data else None,
                    "mask_path": d.segmentation_mask,
                }
                for d in self.finalized_defects
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PipelineResult":
        """Reconstruct from a dictionary generated by to_dict."""
        res = cls()
        res.video_path = data.get("video_path", "")
        res.total_frames = data.get("total_frames", 0)
        res.processed_frames = data.get("processed_frames", 0)
        res.processing_time_sec = data.get("processing_time_sec", 0.0)
        res.fps = data.get("fps", 0.0)
        res.total_raw_detections = data.get("total_raw_detections", 0)
        res.total_unique_defects = data.get("total_unique_defects", 0)
        res.tracks_created = data.get("tracks_created", 0)
        res.tracks_merged = data.get("tracks_merged", 0)
        res.defect_counts = data.get("defect_counts", {})
        res.output_dir = data.get("output_dir", "")
        res.database_path = data.get("database_path", "")
        res.annotated_video_path = data.get("annotated_video_path")
        res.report_path = data.get("report_path")
        
        finalized = []
        for d in data.get("finalized_defects", []):
            fd = FinalizedDefect(
                defect_id=d.get("defect_id", ""),
                defect_type=d.get("defect_type", ""),
            )
            fd.avg_confidence = d.get("avg_confidence", 0.0)
            fd.max_confidence = d.get("max_confidence", 0.0)
            fd.max_confidence_frame = d.get("max_confidence_frame", 0)
            fd.frames_observed = d.get("frames_observed", 0)
            fd.first_frame = d.get("first_frame", 0)
            fd.last_frame = d.get("last_frame", 0)
            fd.representative_center = tuple(d.get("representative_center", (0.0, 0.0)))
            fd.representative_crop_path = d.get("crop_path", "")
            fd.source_track_ids = d.get("source_track_ids", [])
            finalized.append(fd)
            
        res.finalized_defects = finalized
        return res

