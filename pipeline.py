"""
RDDPipeline — Main orchestrator for the complete deduplication pipeline.

Coordinates all layers:
    Frame Reader → Motion Estimator → YOLO + BoT-SORT Detector
    → Track Manager → Duplicate Verifier → Defect Database

Each video frame flows through the pipeline sequentially:
1. Read frame
2. Estimate camera motion (ORB homography)
3. Run YOLO detection + BoT-SORT tracking
4. Update TrackManager with detections
5. Verify duplicates for any newly finalized tracks
6. Report progress

After all frames:
7. Finalize remaining tracks
8. Run final duplicate verification pass
9. Store to DefectDatabase
10. Generate report
"""

import os
import re
import json
import time
import logging
import datetime
import cv2
import numpy as np
from typing import Callable, Dict, List, Optional

from rdd_dedup.config import PipelineConfig
from rdd_dedup.detection.detector import DefectDetector
from rdd_dedup.stabilization.motion_estimator import MotionEstimator
from rdd_dedup.tracking.track_manager import TrackManager
from rdd_dedup.tracking.defect_track import (
    Detection, DefectTrack, FinalizedDefect, PipelineResult, TrackStatus,
)
from rdd_dedup.verification.duplicate_verifier import DuplicateVerifier
from rdd_dedup.storage.defect_database import DefectDatabase
from rdd_dedup.utils.visualizer import PipelineVisualizer

logger = logging.getLogger(__name__)

# Color palette for annotation (BGR)
ANNOTATION_COLORS = [
    (0, 0, 255),       # Red
    (255, 165, 0),     # Orange
    (255, 255, 0),     # Yellow
    (255, 0, 255),     # Magenta
    (0, 255, 0),       # Green
    (255, 192, 203),   # Pink
    (0, 255, 255),     # Cyan
    (128, 0, 128),     # Purple
]


class RDDPipeline:
    """
    Main orchestrator for the Road Defect Detection deduplication pipeline.
    
    Usage:
        pipeline = RDDPipeline("configs/default_pipeline.yaml")
        result = pipeline.process_video("inputs/road_footage.mp4")
        print(f"Found {result.total_unique_defects} unique defects")
    """

    def __init__(self, config_path: Optional[str] = None,
                 config: Optional[PipelineConfig] = None):
        """
        Args:
            config_path: Path to YAML configuration file
            config: Direct PipelineConfig instance (takes precedence)
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = PipelineConfig.from_yaml(config_path)
        else:
            self.config = PipelineConfig()

        # Initialize pipeline components
        self.detector = DefectDetector(self.config)
        self.motion_estimator = MotionEstimator(self.config)
        self.track_manager = TrackManager(self.config)
        self.duplicate_verifier: Optional[DuplicateVerifier] = None
        self.database: Optional[DefectDatabase] = None

        # State
        self._is_initialized = False

    @staticmethod
    def _build_run_folder_name(video_path: str, model_path: str,
                               output_dir: str) -> str:
        """
        Build a run folder name in the format: videoname_modelname.

        If the folder already exists, appends a counter (_2, _3, etc.).

        Args:
            video_path: Path to the input video file
            model_path: Path to the YOLO model weights
            output_dir: Parent output directory

        Returns:
            Full path to the run output directory.
        """
        # Extract video name without extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        # Sanitize: keep only alphanumeric, underscores, hyphens
        video_name = re.sub(r'[^\w\-]', '_', video_name).strip('_')

        # Extract model name without extension
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        model_name = re.sub(r'[^\w\-]', '_', model_name).strip('_')

        base_folder = f"{video_name}_{model_name}"
        candidate = os.path.join(output_dir, base_folder)

        # Handle collisions with counter suffix
        if os.path.exists(candidate):
            counter = 2
            while os.path.exists(os.path.join(output_dir, f"{base_folder}_{counter}")):
                counter += 1
            candidate = os.path.join(output_dir, f"{base_folder}_{counter}")

        return candidate

    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> PipelineResult:
        """
        Process a complete video through the deduplication pipeline.
        
        Args:
            video_path: Path to the input video file
            output_dir: Override output directory (default: config.output_dir)
            progress_callback: Optional callback(progress_fraction, message_str)
            cancel_check: Optional callback returning True if cancellation is requested
        
        Returns:
            PipelineResult with all statistics and finalized defects.
        """
        start_time = time.time()

        # Validate input
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Set up output directory using videoname_modelname format
        if output_dir is None:
            output_dir = self.config.resolve_path(self.config.output_dir)
        run_output_dir = self._build_run_folder_name(
            video_path, self.config.model_path, output_dir
        )
        run_folder_name = os.path.basename(run_output_dir)
        crops_dir = os.path.join(run_output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 0  # Unknown length

        logger.info(
            "Processing video: %s (%dx%d, %.1f fps, %d frames)",
            video_path, width, height, fps,
            total_frames if total_frames > 0 else -1,
        )

        # Initialize components with video dimensions
        self._initialize_for_video(width, height, run_output_dir)

        # Set up annotated video writer
        video_writer = None
        annotated_video_path = None
        if self.config.save_annotated_video:
            annotated_video_path = os.path.join(
                run_output_dir, f"output_video_{run_folder_name}.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            effective_fps = fps / max(1, self.config.frame_skip)
            video_writer = cv2.VideoWriter(
                annotated_video_path, fourcc, effective_fps, (width, height)
            )

        # ── Main Processing Loop ──
        frame_idx = 0
        processed_count = 0
        total_raw_detections = 0

        if progress_callback:
            progress_callback(0.0, "Loading model...")

        try:
            while cap.isOpened():
                if cancel_check and cancel_check():
                    logger.info("Video processing cancelled by user revocation request.")
                    raise InterruptedError("Video processing revoked by user.")

                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Frame skip
                if (frame_idx - 1) % self.config.frame_skip != 0:
                    continue

                processed_count += 1

                # Process this frame through the pipeline
                detections, newly_finalized = self._process_frame(
                    frame, frame_idx, crops_dir
                )

                total_raw_detections += len(detections)

                # Annotate frame for output video
                if video_writer is not None:
                    annotated = self._annotate_frame(
                        frame.copy(), detections, frame_idx, total_frames
                    )
                    video_writer.write(annotated)

                # Progress reporting
                if progress_callback and total_frames > 0:
                    pct = min(1.0, frame_idx / total_frames)
                    unique_count = len(self.duplicate_verifier.unique_defects)
                    progress_callback(
                        pct,
                        f"Frame {frame_idx}/{total_frames} | "
                        f"{total_raw_detections} detections | "
                        f"{unique_count} unique defects",
                    )

                # Periodic logging
                if frame_idx % 100 == 0:
                    stats = self.track_manager.get_stats()
                    logger.info(
                        "Frame %d/%d: active=%d, lost=%d, finalized=%d, "
                        "unique=%d, raw_dets=%d",
                        frame_idx, total_frames,
                        stats["active_tracks"], stats["lost_tracks"],
                        stats["finalized_tracks"],
                        len(self.duplicate_verifier.unique_defects),
                        total_raw_detections,
                    )
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if video_writer is not None:
                try:
                    video_writer.release()
                except Exception:
                    pass

        # ── Finalization ──
        remaining = self.track_manager.finalize_all()
        for track in remaining:
            recently_finalized = self.track_manager.get_all_finalized()
            self.duplicate_verifier.verify_and_store(
                track, recently_finalized, crops_dir
            )

        # Store all unique defects to database
        unique_defects = self.duplicate_verifier.get_unique_defects()
        self.database.store_defects(unique_defects)

        # Generate reports
        elapsed = time.time() - start_time
        result = self._build_result(
            video_path, frame_idx, processed_count, elapsed,
            total_raw_detections, unique_defects,
            run_output_dir, annotated_video_path,
        )

        # Save summary and reports
        self._save_reports(result, run_output_dir, run_folder_name)

        # Generate visualization charts
        try:
            visualizer = PipelineVisualizer(run_output_dir)
            chart_paths = visualizer.generate_all_charts(result)
            logger.info("Generated %d visualization charts.", len(chart_paths))
        except Exception as e:
            logger.warning("Visualization generation failed: %s", e)
            chart_paths = []

        if progress_callback:
            progress_callback(1.0, "Complete!")

        logger.info(
            "Pipeline complete: %d frames, %d raw detections → %d unique defects "
            "(%d merges) in %.1fs (%.1f FPS)",
            processed_count, total_raw_detections,
            result.total_unique_defects, result.tracks_merged,
            elapsed, result.fps,
        )

        return result

    def _initialize_for_video(
        self, width: int, height: int, output_dir: str
    ):
        """Initialize/reset all components for a new video."""
        # Reset stateful components
        self.motion_estimator.reset()
        self.track_manager.reset()

        # Create verifier with correct frame dimensions
        self.duplicate_verifier = DuplicateVerifier(
            self.config, frame_width=width, frame_height=height
        )

        # Create database
        db_ext = ".db" if self.config.database_type == "sqlite" else ".json"
        db_path = os.path.join(output_dir, f"defects{db_ext}")
        self.database = DefectDatabase(db_path, self.config.database_type)

        # Load detector model (lazy — only loads once)
        if not self.detector.is_loaded:
            self.detector.load_model()

        self._is_initialized = True

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        crops_dir: str,
    ) -> tuple:
        """
        Process a single frame through the pipeline.
        
        Returns:
            Tuple of (detections, newly_finalized_tracks)
        """
        # Step 1: Camera motion estimation
        homography = None
        cumulative_homography = None
        if self.config.enable_motion_compensation:
            homography = self.motion_estimator.estimate_motion(frame)
            cumulative_homography = self.motion_estimator.get_cumulative_homography()

        # Step 2: YOLO detection + BoT-SORT tracking
        detections = self.detector.detect_and_track(frame, frame_idx)

        # Step 3: Update track manager
        newly_finalized = self.track_manager.update(
            frame_idx, detections, frame, homography, cumulative_homography
        )

        # Step 4: Verify duplicates for newly finalized tracks
        for track in newly_finalized:
            recently_finalized = self.track_manager.get_recently_finalized(frame_idx)
            self.duplicate_verifier.verify_and_store(
                track, recently_finalized, crops_dir
            )

        return detections, newly_finalized

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_idx: int,
        total_frames: int,
    ) -> np.ndarray:
        """Draw bounding boxes, track IDs, and HUD overlay on a frame."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = ANNOTATION_COLORS[det.class_id % len(ANNOTATION_COLORS)]

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label with track ID
            label = f"#{det.track_id} {det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            cv2.rectangle(
                frame, (x1, max(0, y1 - 18)), (x1 + tw + 6, y1), color, -1
            )
            cv2.putText(
                frame, label, (x1 + 3, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA,
            )

        # HUD overlay
        stats = self.track_manager.get_stats()
        unique_count = (
            len(self.duplicate_verifier.unique_defects)
            if self.duplicate_verifier else 0
        )
        hud = (
            f"RDD-DEDUP | Frame: {frame_idx}/{total_frames} | "
            f"Active: {stats['active_tracks']} | "
            f"Unique: {unique_count}"
        )
        cv2.rectangle(frame, (8, 8), (560, 40), (0, 0, 0), -1)
        cv2.putText(
            frame, hud, (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
        )

        return frame

    def _build_result(
        self,
        video_path: str,
        total_frames: int,
        processed_frames: int,
        elapsed: float,
        total_raw_detections: int,
        unique_defects: List[FinalizedDefect],
        output_dir: str,
        annotated_video_path: Optional[str],
    ) -> PipelineResult:
        """Build the PipelineResult summary."""
        defect_counts = self.duplicate_verifier.get_defect_counts()

        return PipelineResult(
            video_path=video_path,
            total_frames=total_frames,
            processed_frames=processed_frames,
            processing_time_sec=elapsed,
            fps=processed_frames / max(0.001, elapsed),
            total_raw_detections=total_raw_detections,
            total_unique_defects=len(unique_defects),
            tracks_created=self.track_manager.total_tracks_created,
            tracks_merged=self.duplicate_verifier.total_merges,
            defect_counts=defect_counts,
            finalized_defects=unique_defects,
            output_dir=output_dir,
            database_path=self.database.db_path if self.database else "",
            annotated_video_path=annotated_video_path,
        )

    def _save_reports(
        self,
        result: PipelineResult,
        output_dir: str,
        run_folder_name: str,
    ):
        """Save summary JSON, CSV export, and merge audit log."""
        # Summary JSON
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        result.report_path = summary_path

        # CSV export of unique defects
        csv_path = os.path.join(
            output_dir, f"unique_defects_{run_folder_name}.csv"
        )
        if self.database:
            self.database.export_csv(csv_path)

        # Merge audit log
        if self.duplicate_verifier and self.duplicate_verifier.merge_log:
            audit_path = os.path.join(output_dir, "merge_audit_log.json")
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.duplicate_verifier.merge_log, f, indent=2, default=str
                )
            logger.info(
                "Merge audit log saved: %s (%d entries)",
                audit_path, len(self.duplicate_verifier.merge_log),
            )

        # Tracking statistics
        stats_path = os.path.join(output_dir, "tracking_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "track_manager": self.track_manager.get_stats(),
                    "duplicate_verifier": self.duplicate_verifier.get_stats()
                    if self.duplicate_verifier else {},
                    "motion_estimator": {
                        "last_inlier_ratio": self.motion_estimator.last_inlier_ratio,
                        "last_num_matches": self.motion_estimator.last_num_matches,
                    },
                    "config": self.config.to_dict(),
                },
                f,
                indent=2,
            )

        logger.info("Reports saved to: %s", output_dir)
