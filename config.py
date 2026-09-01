"""
PipelineConfig — Centralized configuration for the RDD deduplication pipeline.

Loads all tunable thresholds and parameters from a YAML file,
with sensible defaults for Indian road conditions and handheld camera footage.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DuplicateWeights:
    """Feature weights for the duplicate verification similarity score."""
    bbox_area: float = 0.10
    aspect_ratio: float = 0.08
    center_distance: float = 0.20
    color_histogram: float = 0.20
    orb_features: float = 0.22
    confidence_trend: float = 0.05
    temporal_gap: float = 0.15

    def validate(self):
        """Verify weights sum to 1.0 (within floating-point tolerance)."""
        total = (
            self.bbox_area + self.aspect_ratio + self.center_distance
            + self.color_histogram + self.orb_features
            + self.confidence_trend + self.temporal_gap
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Duplicate verification weights must sum to 1.0, got {total:.4f}"
            )


@dataclass
class PipelineConfig:
    """
    Complete configuration for the RDD duplicate-removal pipeline.
    
    All thresholds are tuned for:
    - Indian roads with closely-spaced defects
    - Handheld mobile phone cameras with shake/tilt
    - Walking-speed forward motion (~1-2 m/s)
    - No GPS availability
    """

    # ── SAM2 Segmentation ──
    enable_segmentation: bool = True
    sam2_checkpoint_path: str = "C:/Users/MaharshiJB/sam2_lib/sam2/checkpoints/sam2.1_hiera_small.pt"
    sam2_config_name: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    segmentation_crop_padding: int = 15
    masks_dir_name: str = "masks"

    # ── Model ──
    model_path: str = "models/rdd_yolo.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    frame_skip: int = 1

    # ── BoT-SORT Tracker ──
    botsort_config_path: str = "rdd_dedup/detection/botsort_config.yaml"

    # ── Track Manager ──
    track_buffer_frames: int = 45       # Frames to keep LOST tracks alive (~1.5s @ 30fps)
    min_track_length: int = 3           # Minimum frames to consider a track valid
    min_finalize_confidence: float = 0.4  # Minimum max_confidence to finalize a track

    # ── Motion Estimation ──
    enable_motion_compensation: bool = True
    orb_max_features: int = 500
    min_inlier_ratio: float = 0.3       # Below this, fall back to identity homography
    motion_estimation_downsample: float = 0.5  # Downsample factor for ORB speed

    # ── Duplicate Verification ──
    merge_threshold: float = 0.65       # Weighted similarity score threshold for merge
    temporal_window_frames: int = 150   # Only compare against recently finalized tracks
    weights: DuplicateWeights = field(default_factory=DuplicateWeights)

    # ── Confidence Aggregation ──
    confidence_method: str = "trimmed_mean"  # "mean", "median", "trimmed_mean"

    # ── Storage ──
    output_dir: str = "outputs"
    database_type: str = "sqlite"       # "sqlite" or "json"
    save_annotated_video: bool = True
    save_crops: bool = True
    max_gallery_crops: int = 60

    # ── Performance ──
    enable_gpu: bool = True
    batch_size: int = 1

    # ── Derived paths (set during initialization) ──
    _base_dir: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        """Resolve relative paths and validate configuration."""
        if self._base_dir is None:
            # Default: project root (parent of rdd_dedup/)
            self._base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        self.weights.validate()

    @property
    def base_dir(self) -> str:
        return self._base_dir

    def resolve_path(self, relative_path: str) -> str:
        """Resolve a path relative to the project base directory."""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self._base_dir, relative_path)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """
        Load configuration from a YAML file.
        
        Unspecified keys use the dataclass defaults.
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Extract nested weights if present
        weights_raw = raw.pop("weights", {})
        weights = DuplicateWeights(**weights_raw) if weights_raw else DuplicateWeights()

        # Build config, setting base_dir to the YAML file's parent
        config = cls(
            weights=weights,
            _base_dir=os.path.dirname(os.path.dirname(os.path.abspath(yaml_path))),
            **{k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        )
        return config

    def to_dict(self) -> Dict:
        """Serialize config to a dictionary (for logging/saving)."""
        return {
            "enable_segmentation": self.enable_segmentation,
            "sam2_checkpoint_path": self.sam2_checkpoint_path,
            "sam2_config_name": self.sam2_config_name,
            "segmentation_crop_padding": self.segmentation_crop_padding,
            "masks_dir_name": self.masks_dir_name,
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "frame_skip": self.frame_skip,
            "botsort_config_path": self.botsort_config_path,
            "track_buffer_frames": self.track_buffer_frames,
            "min_track_length": self.min_track_length,
            "min_finalize_confidence": self.min_finalize_confidence,
            "enable_motion_compensation": self.enable_motion_compensation,
            "orb_max_features": self.orb_max_features,
            "min_inlier_ratio": self.min_inlier_ratio,
            "motion_estimation_downsample": self.motion_estimation_downsample,
            "merge_threshold": self.merge_threshold,
            "temporal_window_frames": self.temporal_window_frames,
            "weights": {
                "bbox_area": self.weights.bbox_area,
                "aspect_ratio": self.weights.aspect_ratio,
                "center_distance": self.weights.center_distance,
                "color_histogram": self.weights.color_histogram,
                "orb_features": self.weights.orb_features,
                "confidence_trend": self.weights.confidence_trend,
                "temporal_gap": self.weights.temporal_gap,
            },
            "confidence_method": self.confidence_method,
            "output_dir": self.output_dir,
            "database_type": self.database_type,
            "save_annotated_video": self.save_annotated_video,
            "save_crops": self.save_crops,
            "enable_gpu": self.enable_gpu,
        }
