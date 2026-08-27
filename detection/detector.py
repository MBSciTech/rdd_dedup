"""
DefectDetector — YOLO detection with BoT-SORT tracking integration.

Wraps the ultralytics YOLO model with BoT-SORT tracking to produce
per-frame detections with persistent track IDs. This module is the
bridge between raw YOLO inference and the TrackManager.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional

from rdd_dedup.config import PipelineConfig
from rdd_dedup.tracking.defect_track import Detection

logger = logging.getLogger(__name__)

# Default RDD class mapping (matches the existing Gradio UI)
DEFAULT_RDD_CLASSES = {
    0: "D00: Longitudinal Crack",
    1: "D10: Transverse Crack",
    2: "D20: Alligator Crack",
    3: "D40: Pothole",
    4: "Repair / Patch",
}


class DefectDetector:
    """
    YOLO + BoT-SORT detector for road defects.
    
    Usage:
        detector = DefectDetector(config)
        for frame in video_frames:
            detections = detector.detect_and_track(frame, frame_idx)
            # Each detection has a persistent track_id from BoT-SORT
    
    Key points:
    - persist=True is required for BoT-SORT to maintain IDs across frames
    - The custom botsort_config.yaml is used instead of the ultralytics default
    - Class names are extracted dynamically from the model
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.class_map: Dict[int, str] = {}
        self._model_loaded = False

    def load_model(self):
        """
        Load the YOLO model and extract class names.
        
        Separated from __init__ to allow lazy loading and better error handling.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            )
            raise

        model_path = self.config.resolve_path(self.config.model_path)

        if not os.path.exists(model_path):
            logger.warning(
                "Model file not found at '%s'. "
                "Attempting to load as a named model.",
                model_path
            )
            model_path = self.config.model_path

        logger.info("Loading YOLO model from: %s", model_path)
        self.model = YOLO(model_path)

        # Extract class names from model
        if hasattr(self.model, "names") and self.model.names:
            self.class_map = {
                int(cid): str(cname).title()
                for cid, cname in self.model.names.items()
            }
            logger.info(
                "Loaded %d classes from model: %s",
                len(self.class_map),
                list(self.class_map.values()),
            )
        else:
            self.class_map = DEFAULT_RDD_CLASSES.copy()
            logger.info("Using default RDD class mapping.")

        self._model_loaded = True

    def detect_and_track(
        self,
        frame: np.ndarray,
        frame_idx: int
    ) -> List[Detection]:
        """
        Run YOLO detection + BoT-SORT tracking on a single frame.
        
        Args:
            frame: BGR video frame
            frame_idx: Current frame index
        
        Returns:
            List of Detection objects, each with a persistent track_id.
            Returns empty list if no detections or model not loaded.
        """
        if not self._model_loaded:
            self.load_model()

        if self.model is None:
            return []

        # Resolve BoT-SORT config path
        tracker_config = self.config.resolve_path(
            self.config.botsort_config_path
        )

        # Run tracking (model.track maintains state with persist=True)
        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker=tracker_config,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                verbose=False,
            )
        except Exception as e:
            logger.warning("Custom tracking config failed on frame %d: %s. Falling back to default 'botsort.yaml'", frame_idx, e)
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker="botsort.yaml",
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    verbose=False,
                )
            except Exception as e2:
                logger.error("Default tracking also failed on frame %d: %s", frame_idx, e2)
                return []

        return self._parse_results(results, frame_idx)

    def _parse_results(
        self,
        results,
        frame_idx: int
    ) -> List[Detection]:
        """
        Parse ultralytics tracking results into Detection objects.
        
        Handles cases where tracking IDs may not be available
        (e.g., first few frames before tracker initialization).
        """
        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i, box in enumerate(boxes):
                # Extract class and confidence
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                # Extract track ID (may be None if tracker hasn't assigned yet)
                if box.id is not None:
                    track_id = int(box.id[0].item())
                else:
                    # Assign a pseudo track ID so detection is not lost
                    if not hasattr(self, "_pseudo_id_counter"):
                        self._pseudo_id_counter = 900000
                    self._pseudo_id_counter += 1
                    track_id = self._pseudo_id_counter

                # Look up class name
                class_name = self.class_map.get(cls_id, f"Class-{cls_id}")

                # Compute center
                center = Detection.compute_center((x1, y1, x2, y2))

                detections.append(Detection(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    center=center,
                    frame_idx=frame_idx,
                ))

        return detections

    def reset_tracker(self):
        """
        Reset the BoT-SORT tracker state.
        
        Call this between videos to ensure track IDs don't carry over.
        """
        if self.model is not None:
            # Reload model to reset internal tracker state
            self.load_model()

    def get_class_name(self, class_id: int) -> str:
        """Look up the human-readable class name for a class ID."""
        return self.class_map.get(class_id, f"Class-{class_id}")

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._model_loaded
