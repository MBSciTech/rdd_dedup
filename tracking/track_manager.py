"""
TrackManager — Lifecycle controller for all DefectTracks.

This is the central state manager of the dedup pipeline, independent of BoT-SORT.
BoT-SORT only provides track IDs per frame — the TrackManager independently decides
when tracks are created, updated, lost, recovered, and finalized.

State Transitions:
    [*] → ACTIVE:       New track_id from BoT-SORT
    ACTIVE → ACTIVE:    Same track_id detected again (update state)
    ACTIVE → LOST:      track_id absent this frame
    LOST → ACTIVE:      track_id reappears before timeout (recover)
    LOST → FINALIZED:   Lost for > track_buffer_frames (finalize)
    * → FINALIZED:      Video ends (force-finalize all)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from rdd_dedup.config import PipelineConfig
from rdd_dedup.tracking.defect_track import (
    Detection, DefectTrack, TrackStatus,
)
from rdd_dedup.utils.image_utils import extract_crop, is_crop_valid

logger = logging.getLogger(__name__)


class TrackManager:
    """
    Manages the full lifecycle of DefectTracks.
    
    Responsibilities:
    - Create new tracks when BoT-SORT reports new IDs
    - Update active tracks with new detections (confidence, bbox, crop)
    - Transition tracks to LOST when BoT-SORT stops reporting them
    - Recover LOST tracks if they reappear within the buffer
    - Finalize tracks that exceed the lost buffer timeout
    - Force-finalize all remaining tracks at end of video
    
    The TrackManager returns newly finalized tracks each frame so the
    DuplicateVerifier can process them immediately.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Active tracks: currently being detected (track_id → DefectTrack)
        self.active_tracks: Dict[int, DefectTrack] = {}

        # Lost tracks: recently disappeared, within buffer (track_id → DefectTrack)
        self.lost_tracks: Dict[int, DefectTrack] = {}

        # Finalized tracks: completed lifecycle (ordered list)
        self.finalized_tracks: List[DefectTrack] = []

        # Statistics
        self.total_tracks_created: int = 0
        self.total_tracks_recovered: int = 0

    def update(
        self,
        frame_idx: int,
        detections: List[Detection],
        frame: np.ndarray,
        homography: Optional[np.ndarray] = None,
        cumulative_homography: Optional[np.ndarray] = None,
    ) -> List[DefectTrack]:
        """
        Process one frame's worth of detections and return any newly finalized tracks.
        
        This is the main entry point called every frame by the pipeline.
        
        Args:
            frame_idx: Current frame index
            detections: List of Detection objects from DefectDetector
            frame: Current BGR video frame (for crop extraction)
            homography: Optional motion compensation matrix (unused directly here
                        but stored for verification layer to use)
        
        Returns:
            List of DefectTracks that were finalized this frame
            (either from timeout or from short-track rejection).
        """
        newly_finalized: List[DefectTrack] = []

        # Collect track IDs seen this frame
        seen_track_ids: Set[int] = set()

        # Step 1: Process each detection
        for det in detections:
            seen_track_ids.add(det.track_id)

            if det.track_id in self.active_tracks:
                # Update existing active track
                self._update_active_track(
                    self.active_tracks[det.track_id], det, frame_idx, frame, cumulative_homography
                )

            elif det.track_id in self.lost_tracks:
                # Recover lost track!
                self._recover_lost_track(det.track_id, det, frame_idx, frame, cumulative_homography)

            else:
                # Brand new track
                self._create_new_track(det, frame_idx, frame, cumulative_homography)

        # Step 2: Mark active tracks not seen this frame as LOST
        track_ids_to_lose = [
            tid for tid in self.active_tracks
            if tid not in seen_track_ids
        ]
        for tid in track_ids_to_lose:
            self._mark_lost(tid, frame_idx)

        # Step 3: Check lost tracks for timeout → finalize
        newly_finalized.extend(self._check_lost_timeouts(frame_idx))

        return newly_finalized

    def _create_new_track(
        self,
        det: Detection,
        frame_idx: int,
        frame: np.ndarray,
        cumulative_homography: Optional[np.ndarray],
    ):
        """Create a new DefectTrack from a first-time detection."""
        track = DefectTrack(
            track_id=det.track_id,
            defect_class=det.class_name,
            class_id=det.class_id,
            first_frame=frame_idx,
            last_frame=frame_idx,
            frames_seen=1,
            confidence_history=[det.confidence],
            max_confidence=det.confidence,
            max_confidence_frame=frame_idx,
            bbox_history=[det.bbox],
            center_history=[det.center],
            cumulative_homography=cumulative_homography,
            status=TrackStatus.ACTIVE,
        )

        # Extract initial crop
        crop = extract_crop(frame, det.bbox)
        if is_crop_valid(crop):
            track.best_crop = crop
            track.best_crop_bbox = det.bbox

        seg_crop = extract_crop(frame, det.bbox, padding=self.config.segmentation_crop_padding)
        if is_crop_valid(seg_crop):
            track.best_seg_crop = seg_crop
            track.best_seg_crop_offset = (
                max(0, det.bbox[0] - self.config.segmentation_crop_padding),
                max(0, det.bbox[1] - self.config.segmentation_crop_padding)
            )

        self.active_tracks[det.track_id] = track
        self.total_tracks_created += 1

        logger.debug(
            "Frame %d: New track #%d created (%s, conf=%.2f)",
            frame_idx, det.track_id, det.class_name, det.confidence,
        )

    def _update_active_track(
        self,
        track: DefectTrack,
        det: Detection,
        frame_idx: int,
        frame: np.ndarray,
        cumulative_homography: Optional[np.ndarray],
    ):
        """Update an existing active track with a new detection."""
        track.last_frame = frame_idx
        track.frames_seen += 1
        track.confidence_history.append(det.confidence)
        track.bbox_history.append(det.bbox)
        track.center_history.append(det.center)
        track.invalidate_cache()

        # Update best crop if this frame has higher confidence
        if det.confidence > track.max_confidence:
            track.max_confidence = det.confidence
            track.max_confidence_frame = frame_idx
            track.cumulative_homography = cumulative_homography

            crop = extract_crop(frame, det.bbox)
            if is_crop_valid(crop):
                track.best_crop = crop
                track.best_crop_bbox = det.bbox

            seg_crop = extract_crop(frame, det.bbox, padding=self.config.segmentation_crop_padding)
            if is_crop_valid(seg_crop):
                track.best_seg_crop = seg_crop
                track.best_seg_crop_offset = (
                    max(0, det.bbox[0] - self.config.segmentation_crop_padding),
                    max(0, det.bbox[1] - self.config.segmentation_crop_padding)
                )

    def _mark_lost(self, track_id: int, frame_idx: int):
        """Transition a track from ACTIVE → LOST."""
        if track_id not in self.active_tracks:
            return

        track = self.active_tracks.pop(track_id)
        track.status = TrackStatus.LOST
        track.lost_since_frame = frame_idx
        self.lost_tracks[track_id] = track

        logger.debug(
            "Frame %d: Track #%d marked LOST (was active for %d frames)",
            frame_idx, track_id, track.frames_seen,
        )

    def _recover_lost_track(
        self,
        track_id: int,
        det: Detection,
        frame_idx: int,
        frame: np.ndarray,
        cumulative_homography: Optional[np.ndarray],
    ):
        """Recover a LOST track back to ACTIVE when BoT-SORT re-associates it."""
        if track_id not in self.lost_tracks:
            return

        track = self.lost_tracks.pop(track_id)
        track.status = TrackStatus.ACTIVE
        track.lost_since_frame = None
        self.active_tracks[track_id] = track

        # Update with the new detection
        self._update_active_track(track, det, frame_idx, frame, cumulative_homography)

        self.total_tracks_recovered += 1

        logger.debug(
            "Frame %d: Track #%d RECOVERED from lost state "
            "(total frames now: %d)",
            frame_idx, track_id, track.frames_seen,
        )

    def _check_lost_timeouts(self, frame_idx: int) -> List[DefectTrack]:
        """
        Check all lost tracks for buffer timeout and finalize those that exceed it.
        
        Returns:
            List of tracks that were finalized due to timeout.
        """
        newly_finalized = []
        track_ids_to_finalize = []

        for tid, track in self.lost_tracks.items():
            if track.lost_since_frame is not None:
                frames_lost = frame_idx - track.lost_since_frame
                if frames_lost > self.config.track_buffer_frames:
                    track_ids_to_finalize.append(tid)

        for tid in track_ids_to_finalize:
            track = self.lost_tracks.pop(tid)
            finalized = self._finalize_track(track)
            if finalized is not None:
                newly_finalized.append(finalized)

        return newly_finalized

    def _finalize_track(self, track: DefectTrack) -> Optional[DefectTrack]:
        """
        Finalize a track — validate and add to the finalized list.
        
        Returns:
            The finalized track if it passes validation, None if rejected.
        
        Rejection criteria:
        - Too few frames observed (likely false positive)
        - Maximum confidence below threshold (likely noise)
        """
        # Validate: minimum track length
        if track.frames_seen < self.config.min_track_length:
            logger.debug(
                "Track #%d REJECTED: too few frames (%d < %d)",
                track.track_id, track.frames_seen, self.config.min_track_length,
            )
            return None

        # Validate: minimum confidence
        if track.max_confidence < self.config.min_finalize_confidence:
            logger.debug(
                "Track #%d REJECTED: max confidence too low (%.2f < %.2f)",
                track.track_id, track.max_confidence,
                self.config.min_finalize_confidence,
            )
            return None

        track.status = TrackStatus.FINALIZED
        self.finalized_tracks.append(track)

        logger.info(
            "Track #%d FINALIZED: %s, frames=%d, "
            "avg_conf=%.2f, max_conf=%.2f (frame %d)",
            track.track_id, track.defect_class, track.frames_seen,
            track.avg_confidence, track.max_confidence,
            track.max_confidence_frame,
        )

        return track

    def finalize_all(self) -> List[DefectTrack]:
        """
        Force-finalize all remaining ACTIVE and LOST tracks.
        
        Called at the end of video processing.
        
        Returns:
            List of all newly finalized tracks.
        """
        newly_finalized = []

        # Finalize all active tracks
        for tid in list(self.active_tracks.keys()):
            track = self.active_tracks.pop(tid)
            finalized = self._finalize_track(track)
            if finalized is not None:
                newly_finalized.append(finalized)

        # Finalize all lost tracks
        for tid in list(self.lost_tracks.keys()):
            track = self.lost_tracks.pop(tid)
            finalized = self._finalize_track(track)
            if finalized is not None:
                newly_finalized.append(finalized)

        logger.info(
            "End of video: Force-finalized %d tracks "
            "(total finalized: %d, total created: %d)",
            len(newly_finalized),
            len(self.finalized_tracks),
            self.total_tracks_created,
        )

        return newly_finalized

    def get_all_finalized(self) -> List[DefectTrack]:
        """Return all finalized tracks."""
        return list(self.finalized_tracks)

    def get_recently_finalized(self, frame_idx: int) -> List[DefectTrack]:
        """
        Return tracks finalized within the temporal window.
        
        Used by DuplicateVerifier to limit comparison scope.
        """
        window = self.config.temporal_window_frames
        return [
            t for t in self.finalized_tracks
            if (frame_idx - t.last_frame) <= window
        ]

    def get_stats(self) -> Dict:
        """Return current tracking statistics."""
        return {
            "active_tracks": len(self.active_tracks),
            "lost_tracks": len(self.lost_tracks),
            "finalized_tracks": len(self.finalized_tracks),
            "total_created": self.total_tracks_created,
            "total_recovered": self.total_tracks_recovered,
        }

    def reset(self):
        """Reset all state (e.g., for a new video)."""
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self.finalized_tracks.clear()
        self.total_tracks_created = 0
        self.total_tracks_recovered = 0
