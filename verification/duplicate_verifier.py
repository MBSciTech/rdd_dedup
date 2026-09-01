"""
DuplicateVerifier — Multi-feature similarity engine for track deduplication.

When a track is finalized, the DuplicateVerifier compares it against
recently finalized tracks to detect duplicates. It uses a weighted
combination of multiple features to produce a similarity score.

Key design decisions:
- ONLY compares against recently finalized tracks (temporal window)
- Class match is a hard gate (different classes → score = 0)
- Multiple features provide robustness: even if one feature fails,
  others can still identify duplicates
- Neutral scores (0.5) are used when a feature can't be computed
"""

import logging
import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from rdd_dedup.config import PipelineConfig
from rdd_dedup.tracking.defect_track import (
    DefectTrack, Detection, FinalizedDefect,
)
from rdd_dedup.utils.math_utils import (
    area_similarity,
    aspect_ratio_similarity,
    normalized_center_distance,
    cosine_similarity,
    temporal_gap_similarity,
    euclidean_distance,
)
from rdd_dedup.utils.image_utils import (
    compute_color_histogram,
    compare_histograms,
    compute_orb_features,
    match_orb_features,
    is_crop_valid,
)

logger = logging.getLogger(__name__)


class SimilarityBreakdown:
    """Detailed breakdown of a similarity comparison (for audit logging)."""

    def __init__(self):
        self.area_score: float = 0.0
        self.aspect_ratio_score: float = 0.0
        self.center_distance_score: float = 0.0
        self.color_histogram_score: float = 0.0
        self.orb_features_score: float = 0.0
        self.confidence_trend_score: float = 0.0
        self.temporal_gap_score: float = 0.0
        self.weighted_total: float = 0.0
        self.class_match: bool = False

    def to_dict(self) -> Dict:
        return {
            "class_match": self.class_match,
            "area_score": round(self.area_score, 4),
            "aspect_ratio_score": round(self.aspect_ratio_score, 4),
            "center_distance_score": round(self.center_distance_score, 4),
            "color_histogram_score": round(self.color_histogram_score, 4),
            "orb_features_score": round(self.orb_features_score, 4),
            "confidence_trend_score": round(self.confidence_trend_score, 4),
            "temporal_gap_score": round(self.temporal_gap_score, 4),
            "weighted_total": round(self.weighted_total, 4),
        }

    def __repr__(self):
        return (
            f"Similarity(total={self.weighted_total:.3f}, "
            f"area={self.area_score:.2f}, ar={self.aspect_ratio_score:.2f}, "
            f"center={self.center_distance_score:.2f}, "
            f"color={self.color_histogram_score:.2f}, "
            f"orb={self.orb_features_score:.2f}, "
            f"conf={self.confidence_trend_score:.2f}, "
            f"temporal={self.temporal_gap_score:.2f})"
        )


class DuplicateVerifier:
    """
    Multi-feature similarity engine for post-track duplicate detection.
    
    Architecture:
    1. Candidate Selection: Only compare against recently finalized tracks
       within the temporal window (O(R), not O(N))
    2. Gate Check: Defect class must match (hard reject otherwise)
    3. Feature Scoring: Compute weighted similarity across 7 features
    4. Decision: Merge if score ≥ threshold, otherwise create new defect
    
    The verifier maintains a list of FinalizedDefects (the deduped output).
    """

    def __init__(self, config: PipelineConfig, frame_width: int = 1920,
                 frame_height: int = 1080, segmenter=None):
        self.config = config
        self.weights = config.weights
        self.segmenter = segmenter

        # Frame diagonal for normalizing spatial distances
        self.frame_diagonal = float(np.sqrt(
            frame_width ** 2 + frame_height ** 2
        ))

        # Deduplicated output
        self.unique_defects: List[FinalizedDefect] = []

        # Audit log for merge decisions
        self.merge_log: List[Dict] = []
        self.total_merges: int = 0

    def verify_and_store(
        self,
        track: DefectTrack,
        recently_finalized: List[DefectTrack],
        crops_dir: str = "",
        masks_dir: str = "",
    ) -> FinalizedDefect:
        """
        Verify a newly finalized track against existing unique defects.
        
        If a duplicate is found, merge into the existing defect.
        Otherwise, create a new unique defect entry.
        
        Args:
            track: The newly finalized DefectTrack
            recently_finalized: Tracks finalized within the temporal window
                                (provided by TrackManager)
            crops_dir: Directory to save representative crops
        
        Returns:
            The FinalizedDefect (either existing merged or new).
        """
        # Get merge candidates from unique defects within temporal window
        candidates = self._get_merge_candidates(track)

        best_match: Optional[FinalizedDefect] = None
        best_score: float = 0.0
        best_breakdown: Optional[SimilarityBreakdown] = None

        for candidate in candidates:
            # Find the source track for this candidate (for detailed comparison)
            candidate_track = self._find_source_track(
                candidate, recently_finalized
            )

            breakdown = self._compute_similarity(track, candidate, candidate_track)

            if breakdown.weighted_total > best_score:
                best_score = breakdown.weighted_total
                best_match = candidate
                best_breakdown = breakdown

        # Decision: merge or create new
        if best_match is not None and best_score >= self.config.merge_threshold:
            # MERGE — this track is a duplicate
            crop_path = self._save_crop(track, crops_dir) if crops_dir else ""
            seg_result = self._run_segmentation(track, masks_dir) if masks_dir else None
            new_defect = FinalizedDefect.from_track(
                track, crop_path, self.config.confidence_method, segmentation_result=seg_result
            )
            best_match.merge_from(new_defect)
            self.total_merges += 1

            self.merge_log.append({
                "action": "MERGE",
                "track_id": track.track_id,
                "merged_into": best_match.defect_id,
                "score": best_score,
                "breakdown": best_breakdown.to_dict() if best_breakdown else {},
            })

            logger.info(
                "Track #%d MERGED into defect %s (score=%.3f). %s",
                track.track_id, best_match.defect_id[:8],
                best_score, best_breakdown,
            )

            return best_match

        else:
            # NEW DEFECT — create unique entry
            crop_path = self._save_crop(track, crops_dir) if crops_dir else ""
            seg_result = self._run_segmentation(track, masks_dir) if masks_dir else None
            new_defect = FinalizedDefect.from_track(
                track, crop_path, self.config.confidence_method, segmentation_result=seg_result
            )
            self.unique_defects.append(new_defect)

            if best_match is not None:
                self.merge_log.append({
                    "action": "NEW (below threshold)",
                    "track_id": track.track_id,
                    "defect_id": new_defect.defect_id,
                    "closest_match": best_match.defect_id,
                    "score": best_score,
                    "threshold": self.config.merge_threshold,
                    "breakdown": best_breakdown.to_dict() if best_breakdown else {},
                })
            else:
                self.merge_log.append({
                    "action": "NEW (no candidates)",
                    "track_id": track.track_id,
                    "defect_id": new_defect.defect_id,
                })

            logger.info(
                "Track #%d → NEW defect %s (%s, conf=%.2f, frames=%d)",
                track.track_id, new_defect.defect_id[:8],
                new_defect.defect_type, new_defect.avg_confidence,
                new_defect.frames_observed,
            )

            return new_defect

    def _get_merge_candidates(self, track: DefectTrack) -> List[FinalizedDefect]:
        """
        Return unique defects eligible for merging with the given track.
        
        Filtering:
        1. Same defect class (hard gate)
        2. Within temporal window (last_frame proximity)
        
        This is O(R) where R = number of recent unique defects, NOT O(N).
        """
        candidates = []
        for defect in self.unique_defects:
            # Gate: class must match
            if defect.class_id != track.class_id:
                continue

            # Gate: temporal proximity
            frame_gap = abs(track.first_frame - defect.last_frame)
            if frame_gap > self.config.temporal_window_frames:
                continue

            # Gate: temporal overlap (hard reject simultaneous tracks)
            has_overlap = False
            for start, end in defect.source_track_intervals:
                if max(track.first_frame, start) <= min(track.last_frame, end):
                    has_overlap = True
                    break
            if has_overlap:
                continue

            candidates.append(defect)

        return candidates

    def _find_source_track(
        self,
        defect: FinalizedDefect,
        finalized_tracks: List[DefectTrack],
    ) -> Optional[DefectTrack]:
        """
        Find the original DefectTrack for a FinalizedDefect.
        
        Used to access the best_crop and detailed history for comparison.
        """
        for track in finalized_tracks:
            if track.track_id in defect.source_track_ids:
                return track
        return None

    def _compute_similarity(
        self,
        new_track: DefectTrack,
        existing_defect: FinalizedDefect,
        existing_track: Optional[DefectTrack],
    ) -> SimilarityBreakdown:
        """
        Compute the weighted multi-feature similarity score.
        
        Features:
        1. BBox Area similarity
        2. Aspect Ratio similarity
        3. Center Distance (inverse, normalized)
        4. Color Histogram correlation
        5. ORB Feature matching
        6. Confidence Trend cosine similarity
        7. Temporal Gap proximity
        """
        breakdown = SimilarityBreakdown()

        # Gate check
        if new_track.class_id != existing_defect.class_id:
            breakdown.class_match = False
            breakdown.weighted_total = 0.0
            return breakdown
        breakdown.class_match = True

        w = self.weights

        # 1. BBox Area
        breakdown.area_score = area_similarity(
            new_track.avg_area, existing_defect.avg_bbox_area
        )

        # 2. Aspect Ratio
        breakdown.aspect_ratio_score = aspect_ratio_similarity(
            new_track.avg_aspect_ratio, existing_defect.avg_aspect_ratio
        )

        # 3. Center Distance (inverted: closer = higher score)
        existing_center = existing_defect.representative_center
        if new_track.cumulative_homography is not None and existing_defect.cumulative_homography is not None:
            H_new = new_track.cumulative_homography
            H_old = existing_defect.cumulative_homography
            
            det_old = np.linalg.det(H_old)
            if abs(det_old) > 1e-5:
                try:
                    H_old_inv = np.linalg.inv(H_old)
                    H_proj = H_new @ H_old_inv
                    
                    pt = np.array([existing_center[0], existing_center[1], 1.0])
                    pt_proj = H_proj @ pt
                    if abs(pt_proj[2]) > 1e-6:
                        pt_proj = pt_proj / pt_proj[2]
                        existing_center = (float(pt_proj[0]), float(pt_proj[1]))
                except np.linalg.LinAlgError:
                    pass

        norm_dist = normalized_center_distance(
            new_track.representative_center,
            existing_center,
            self.frame_diagonal,
        )
        breakdown.center_distance_score = 1.0 - norm_dist

        # 4. Color Histogram
        if (existing_track is not None and
                is_crop_valid(new_track.best_crop) and
                is_crop_valid(existing_track.best_crop)):
            hist1 = compute_color_histogram(new_track.best_crop)
            hist2 = compute_color_histogram(existing_track.best_crop)
            breakdown.color_histogram_score = compare_histograms(hist1, hist2)
        else:
            breakdown.color_histogram_score = 0.5  # Neutral

        # 5. ORB Feature Matching
        if (existing_track is not None and
                is_crop_valid(new_track.best_crop) and
                is_crop_valid(existing_track.best_crop)):
            _, desc1 = compute_orb_features(new_track.best_crop)
            _, desc2 = compute_orb_features(existing_track.best_crop)
            breakdown.orb_features_score = match_orb_features(desc1, desc2)
        else:
            breakdown.orb_features_score = 0.5  # Neutral

        # 6. Confidence Trend
        if (len(new_track.confidence_history) >= 2 and
                existing_track is not None and
                len(existing_track.confidence_history) >= 2):
            # Resample both to same length for cosine similarity
            target_len = min(
                len(new_track.confidence_history),
                len(existing_track.confidence_history),
                20,  # Cap at 20 samples
            )
            vec1 = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len(new_track.confidence_history)),
                new_track.confidence_history,
            )
            vec2 = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len(existing_track.confidence_history)),
                existing_track.confidence_history,
            )
            cos_sim = cosine_similarity(vec1, vec2)
            breakdown.confidence_trend_score = (cos_sim + 1.0) / 2.0  # Map to [0,1]
        else:
            breakdown.confidence_trend_score = 0.5  # Neutral

        # 7. Temporal Gap
        frame_gap = abs(new_track.first_frame - existing_defect.last_frame)
        breakdown.temporal_gap_score = temporal_gap_similarity(
            frame_gap, self.config.temporal_window_frames
        )

        # Weighted total
        breakdown.weighted_total = (
            w.bbox_area * breakdown.area_score
            + w.aspect_ratio * breakdown.aspect_ratio_score
            + w.center_distance * breakdown.center_distance_score
            + w.color_histogram * breakdown.color_histogram_score
            + w.orb_features * breakdown.orb_features_score
            + w.confidence_trend * breakdown.confidence_trend_score
            + w.temporal_gap * breakdown.temporal_gap_score
        )

        return breakdown

    def _save_crop(
        self,
        track: DefectTrack,
        crops_dir: str,
    ) -> str:
        """Save the best crop for a track and return the file path."""
        if not is_crop_valid(track.best_crop) or not crops_dir:
            return ""

        os.makedirs(crops_dir, exist_ok=True)

        filename = (
            f"defect_track{track.track_id:04d}_"
            f"frame{track.max_confidence_frame:06d}_"
            f"cls{track.class_id}_"
            f"conf{track.max_confidence:.2f}.jpg"
        )
        filepath = os.path.join(crops_dir, filename)

        try:
            cv2.imwrite(filepath, track.best_crop)
            return filepath
        except Exception as e:
            logger.warning("Failed to save crop for track #%d: %s",
                           track.track_id, e)
            return ""

    def _run_segmentation(self, track: DefectTrack, masks_dir: str) -> Optional[dict]:
        if self.segmenter is None or not self.config.enable_segmentation:
            return None
        if not is_crop_valid(track.best_seg_crop) or track.best_seg_crop_offset is None:
            return None
        if not masks_dir:
            return None
            
        # Compute local box in crop coordinates
        ox, oy = track.best_seg_crop_offset
        bx1, by1, bx2, by2 = track.best_crop_bbox
        local_box = (bx1 - ox, by1 - oy, bx2 - ox, by2 - oy)
        
        result = self.segmenter.segment(track.best_seg_crop, local_box)
        if not result:
            return None
            
        os.makedirs(masks_dir, exist_ok=True)
        filename = (
            f"defect_track{track.track_id:04d}_"
            f"frame{track.max_confidence_frame:06d}_"
            f"cls{track.class_id}_"
            f"conf{track.max_confidence:.2f}_mask.png"
        )
        filepath = os.path.join(masks_dir, filename)
        
        try:
            # Set mask path in result
            result["mask_path"] = filepath
            
            # result["mask"] is a boolean array, map to 0-255 uint8
            mask_img = (result["mask"] * 255).astype(np.uint8)
            cv2.imwrite(filepath, mask_img)
            
            # Get consistent BGR color for this class
            np.random.seed(hash(track.defect_class) % (2**32))
            b, g, r = np.random.randint(50, 255, 3) # Avoid dark colors
            color_bgr = (int(b), int(g), int(r))
            np.random.seed() # Reset seed
            
            # Generate 3-panel verification image
            overlay_filename = filename.replace("_mask.png", "_overlay.png")
            overlay_filepath = os.path.join(masks_dir, overlay_filename)
            
            orig_crop = track.best_seg_crop.copy()
            
            # Colorized mask panel
            colored_mask_panel = np.zeros_like(orig_crop)
            colored_mask_panel[result["mask"]] = color_bgr
            
            # Blended overlay panel
            blended_panel = orig_crop.copy()
            alpha = 0.4
            blended_panel[result["mask"]] = cv2.addWeighted(
                blended_panel, 1 - alpha, colored_mask_panel, alpha, 0
            )[result["mask"]]
            
            # Stitch them together side-by-side
            verification_img = cv2.hconcat([orig_crop, colored_mask_panel, blended_panel])
            
            # Add text
            text = f"{track.defect_class} - {result['pixel_area']}px"
            cv2.putText(verification_img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, color_bgr, 2, cv2.LINE_AA)
            cv2.putText(verification_img, "Crop | Mask | Blended", (10, verification_img.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imwrite(overlay_filepath, verification_img)
            result["overlay_path"] = overlay_filepath
            
        except Exception as e:
            import traceback
            logger.warning("Failed to save mask for track #%d: %s", track.track_id, traceback.format_exc())
            return None
            
        return {
            "mask_path": filepath,
            "pixel_area": result["pixel_area"],
            "mask_quality_score": result["quality_score"]
        }

    def set_frame_dimensions(self, width: int, height: int):
        """Update frame dimensions (for center distance normalization)."""
        self.frame_diagonal = float(np.sqrt(width ** 2 + height ** 2))

    def get_unique_defects(self) -> List[FinalizedDefect]:
        """Return all unique, deduplicated defects."""
        return list(self.unique_defects)

    def get_defect_counts(self) -> Dict[str, int]:
        """Return per-class counts of unique defects."""
        counts: Dict[str, int] = {}
        for defect in self.unique_defects:
            counts[defect.defect_type] = counts.get(defect.defect_type, 0) + 1
        return counts

    def get_stats(self) -> Dict:
        """Return verification statistics."""
        return {
            "unique_defects": len(self.unique_defects),
            "total_merges": self.total_merges,
            "merge_log_entries": len(self.merge_log),
        }

    def reset(self):
        """Reset all state."""
        self.unique_defects.clear()
        self.merge_log.clear()
        self.total_merges = 0
