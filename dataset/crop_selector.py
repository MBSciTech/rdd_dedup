"""
CropSelector — Selects representative detections from BoT-SORT tracks.

Uses a pluggable strategy pattern to choose the best detection(s) from
each finalized track for human review. The default strategy selects
the single highest-confidence detection per track.

The CandidateAnnotation dataclass holds all information needed for the
Gradio annotation review UI: crop image, original frame, bbox, metadata.

Usage:
    selector = CropSelector(strategy="highest_confidence")
    candidates = selector.select_from_pipeline_result(result, video_path)
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from rdd_dedup.tracking.defect_track import FinalizedDefect, PipelineResult


@dataclass
class CandidateAnnotation:
    """
    A single annotation candidate for human review.

    Contains all data needed to display in the UI and to save
    to the dataset if accepted.
    """
    # Display data
    crop_image: Optional[np.ndarray] = field(default=None, repr=False)
    original_frame: Optional[np.ndarray] = field(default=None, repr=False)

    # Annotation data
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1, y1, x2, y2)
    initial_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # Original detection bbox for reset
    class_id: int = 0
    class_name: str = ""
    predicted_class: str = ""       # Model's original prediction
    confidence: float = 0.0

    # Provenance
    frame_number: int = 0
    track_id: int = 0
    source_video: str = ""
    defect_id: str = ""

    # Derived
    crop_path: str = ""             # Path to existing crop file (from pipeline output)

    def get_bbox_dimensions(self) -> Tuple[int, int]:
        """Return (width, height) of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (max(0, x2 - x1), max(0, y2 - y1))


class CropSelector:
    """
    Selects the best representative detection from each finalized track
    for human annotation review.

    Strategy pattern allows future extension with different selection
    criteria (largest area, sharpest frame, spaced sampling, etc.).
    """

    STRATEGIES = ["highest_confidence", "largest_area"]

    def __init__(self, strategy: str = "highest_confidence"):
        """
        Args:
            strategy: Selection strategy name.
                - "highest_confidence": Pick the detection with highest confidence.
                - "largest_area": Pick the detection with largest bounding box area.
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: {self.STRATEGIES}"
            )
        self.strategy = strategy

    def select_from_pipeline_result(
        self,
        result: PipelineResult,
        video_path: str,
    ) -> List[CandidateAnnotation]:
        """
        Extract annotation candidates from a completed pipeline result.

        For each finalized defect, extracts the best frame from the source
        video and creates a CandidateAnnotation with the crop and original frame.

        Args:
            result: Completed PipelineResult with finalized_defects.
            video_path: Path to the original source video.

        Returns:
            List of CandidateAnnotation objects ready for UI review.
        """
        if not result.finalized_defects:
            return []

        candidates = []
        source_video_name = os.path.basename(video_path)

        # Open video for frame extraction
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[CropSelector] Cannot open video: {video_path}")
            return []

        try:
            for defect in result.finalized_defects:
                candidate = self._extract_candidate(
                    cap, defect, source_video_name, result
                )
                if candidate is not None:
                    candidates.append(candidate)
        finally:
            cap.release()

        return candidates

    def _extract_candidate(
        self,
        cap: cv2.VideoCapture,
        defect: FinalizedDefect,
        source_video_name: str,
        result: PipelineResult,
    ) -> Optional[CandidateAnnotation]:
        """
        Extract a single CandidateAnnotation from a FinalizedDefect.

        Uses the best frame (max_confidence_frame) to extract the
        original frame and crop.
        """
        target_frame = defect.max_confidence_frame

        if target_frame <= 0:
            return None

        # Seek to the target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
        ret, frame = cap.read()

        if not ret or frame is None:
            return None

        h, w = frame.shape[:2]

        # Reconstruct approximate bbox from representative_center and avg_bbox_area
        cx, cy = defect.representative_center
        area = defect.avg_bbox_area
        ar = defect.avg_aspect_ratio

        if area > 0 and ar > 0:
            # w_box * h_box = area, w_box / h_box = ar
            # h_box = sqrt(area / ar), w_box = ar * h_box
            h_box = int(np.sqrt(area / max(0.01, ar)))
            w_box = int(ar * h_box)
        else:
            # Fallback: use bbox_stats if available
            w_box = 100
            h_box = 100

        x1 = max(0, int(cx - w_box / 2))
        y1 = max(0, int(cy - h_box / 2))
        x2 = min(w, int(cx + w_box / 2))
        y2 = min(h, int(cy + h_box / 2))

        # Ensure valid bbox
        if x2 <= x1 or y2 <= y1:
            return None

        bbox = (x1, y1, x2, y2)

        # Extract crop
        crop = frame[y1:y2, x1:x2].copy()

        # Load existing crop from pipeline output if available
        existing_crop = None
        if defect.representative_crop_path and os.path.exists(defect.representative_crop_path):
            existing_crop = cv2.imread(defect.representative_crop_path)

        # Use existing crop if available (higher quality), else use extracted
        final_crop = existing_crop if existing_crop is not None else crop

        return CandidateAnnotation(
            crop_image=final_crop,
            original_frame=frame.copy(),
            bbox=bbox,
            initial_bbox=bbox,
            class_id=defect.class_id,
            class_name=defect.defect_type,
            predicted_class=defect.defect_type,
            confidence=defect.max_confidence,
            frame_number=target_frame,
            track_id=defect.source_track_ids[0] if defect.source_track_ids else 0,
            source_video=source_video_name,
            defect_id=defect.defect_id,
            crop_path=defect.representative_crop_path or "",
        )
