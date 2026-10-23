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
from rdd_dedup.measurement.area_calculator import (
    CameraParams,
    compute_defect_area,
    is_area_eligible_class,
)

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
    YOLO + BoT-SORT detector for road defects with per-frame orientation support.
    
    Usage:
        detector = DefectDetector(config, orientation_provider=provider)
        for frame in video_frames:
            detections = detector.detect_and_track(frame, frame_idx)
    """

    def __init__(self, config: PipelineConfig, orientation_provider=None):
        self.config = config
        self.orientation_provider = orientation_provider
        self.model = None
        self.class_map: Dict[int, str] = {}
        self._model_loaded = False
        self.rejection_counts: Dict[str, int] = {}

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
        frame_idx: int,
        down_vector: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """
        Run YOLO detection + BoT-SORT tracking on a single frame.
        
        Args:
            frame: BGR video frame
            frame_idx: Current frame index
            down_vector: Optional per-frame unit gravity down vector (from accelerometer).
                         If None and orientation_provider is set, auto-retrieved.
        
        Returns:
            List of Detection objects, each with a persistent track_id.
            Returns empty list if no detections or model not loaded.
        """
        if not self._model_loaded:
            self.load_model()

        if self.model is None:
            return []

        # Auto-retrieve down vector from orientation provider if not passed explicitly
        if down_vector is None and self.orientation_provider is not None:
            down_vector = self.orientation_provider.get_down_vector(frame_idx)

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

        return self._parse_results(results, frame_idx, down_vector=down_vector, frame=frame)

    def _parse_results(
        self,
        results,
        frame_idx: int,
        down_vector: Optional[np.ndarray] = None,
        frame: Optional[np.ndarray] = None,
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

            # Build camera parameters for ground projection from frame shape
            img_h = r.orig_shape[0] if hasattr(r, "orig_shape") and r.orig_shape else getattr(self.config, "frame_height", 1080)
            img_w = r.orig_shape[1] if hasattr(r, "orig_shape") and r.orig_shape else getattr(self.config, "frame_width", 1920)
            camera_params = CameraParams(
                yc=getattr(self.config, "camera_height_m", 1.2),
                tilt_deg=getattr(self.config, "camera_tilt_deg", 15.0),
                f=getattr(self.config, "camera_focal_length_px", None),
                fx=getattr(self.config, "camera_fx", None),
                fy=getattr(self.config, "camera_fy", None),
                cx=getattr(self.config, "camera_cx", None),
                cy=getattr(self.config, "camera_cy", None),
                image_width=img_w,
                image_height=img_h,
            )

            # Pre-calculate derived frame tilt and roll angles for logging and QC
            if down_vector is not None:
                n_norm = np.linalg.norm(down_vector)
                if n_norm > 1e-8:
                    n_unit = down_vector / n_norm
                    down_z = float(np.clip(n_unit[2], -1.0, 1.0))
                    frame_tilt_deg = float(np.degrees(np.arcsin(down_z)))
                    frame_roll_deg = float(np.degrees(np.arctan2(-n_unit[0], n_unit[1])))
                else:
                    frame_tilt_deg = camera_params.tilt_deg
                    frame_roll_deg = 0.0
            else:
                frame_tilt_deg = camera_params.tilt_deg
                frame_roll_deg = 0.0

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

                # Extract YOLO segmentation mask polygon if present
                polygon = None
                if hasattr(r, "masks") and r.masks is not None:
                    if hasattr(r.masks, "xy") and len(r.masks.xy) > i:
                        poly_cand = r.masks.xy[i]
                        if poly_cand is not None and len(poly_cand) >= 3:
                            polygon = poly_cand

                # Calculate real-world ground area with Quality Gating
                area_m2 = None
                tilt_used = None
                roll_used = None

                if is_area_eligible_class(class_name):
                    rejection_reasons: List[str] = []
                    crop_patch = None
                    if frame is not None and getattr(self.config, "min_laplacian_var", None) is not None:
                        cy1, cy2 = max(0, y1), min(img_h, y2)
                        cx1, cx2 = max(0, x1), min(img_w, x2)
                        if cy2 > cy1 and cx2 > cx1:
                            crop_patch = frame[cy1:cy2, cx1:cx2]

                    area_m2 = compute_defect_area(
                        bbox=(x1, y1, x2, y2),
                        camera_params=camera_params,
                        class_name=class_name,
                        polygon=polygon,
                        down=down_vector,
                        min_tilt_deg=getattr(self.config, "min_tilt_deg", 3.0),
                        max_tilt_deg=getattr(self.config, "max_tilt_deg", 60.0),
                        max_abs_roll_deg=getattr(self.config, "max_abs_roll_deg", 30.0),
                        min_ground_distance_m=getattr(self.config, "min_ground_distance_m", 0.5),
                        max_ground_distance_m=getattr(self.config, "max_ground_distance_m", 8.0),
                        crop_image=crop_patch,
                        min_laplacian_var=getattr(self.config, "min_laplacian_var", None),
                        rejection_sink=rejection_reasons,
                    )

                    for r_reason in rejection_reasons:
                        self.rejection_counts[r_reason] = self.rejection_counts.get(r_reason, 0) + 1

                    if area_m2 is not None:
                        tilt_used = round(frame_tilt_deg, 2)
                        roll_used = round(frame_roll_deg, 2)

                detections.append(Detection(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    center=center,
                    frame_idx=frame_idx,
                    polygon=polygon,
                    area_m2=area_m2,
                    tilt_deg_used=tilt_used,
                    roll_deg_used=roll_used,
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
