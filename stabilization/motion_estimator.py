"""
MotionEstimator — Lightweight camera motion compensation using ORB features.

Estimates frame-to-frame homography to stabilize bounding box positions
and improve tracking consistency for handheld mobile camera footage.

This is NOT full SLAM — it only estimates relative 2D motion between
consecutive frames. The road surface is approximately planar, making
homography a suitable model for forward-walking video capture.
"""

import logging
import cv2
import numpy as np
from typing import Optional, Tuple

from rdd_dedup.config import PipelineConfig

logger = logging.getLogger(__name__)


class MotionEstimator:
    """
    ORB-based camera motion estimator.
    
    Workflow per frame:
    1. Convert to grayscale (optionally downsample for speed)
    2. Detect ORB keypoints + descriptors
    3. Match against previous frame using brute-force Hamming
    4. Compute homography via RANSAC
    5. Evaluate quality (inlier ratio) — fallback to identity if poor
    6. Update state for next frame
    
    The resulting homography can be used to:
    - Adjust lost track positions for camera movement during lost period
    - Transform historical centers to a common reference frame
    - Reduce apparent motion noise in duplicate verification
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.orb = cv2.ORB_create(nfeatures=config.orb_max_features)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.downsample_factor = config.motion_estimation_downsample

        # State from previous frame
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_keypoints: Optional[list] = None
        self._prev_descriptors: Optional[np.ndarray] = None

        # Cumulative homography (tracks total camera displacement)
        self._cumulative_homography = np.eye(3, dtype=np.float64)

        # Diagnostics
        self._last_inlier_ratio: float = 0.0
        self._last_num_matches: int = 0
        self._frame_count: int = 0

    def estimate_motion(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate camera motion from previous frame to current frame.
        
        Args:
            frame: Current video frame (BGR, full resolution)
        
        Returns:
            3×3 homography matrix mapping points in the previous frame
            to the current frame. Returns identity matrix for the first
            frame or when estimation fails.
        """
        self._frame_count += 1

        # Convert and optionally downsample
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downsample_factor < 1.0:
            h, w = gray.shape
            new_w = int(w * self.downsample_factor)
            new_h = int(h * self.downsample_factor)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Detect features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        # First frame — store and return identity
        if self._prev_gray is None or self._prev_descriptors is None:
            self._prev_gray = gray
            self._prev_keypoints = keypoints
            self._prev_descriptors = descriptors
            return np.eye(3, dtype=np.float64)

        # Check we have enough features
        if (descriptors is None or len(descriptors) < 4 or
                self._prev_descriptors is None or len(self._prev_descriptors) < 4):
            logger.debug(
                "Frame %d: Insufficient ORB features (curr=%d, prev=%d). "
                "Falling back to identity.",
                self._frame_count,
                0 if descriptors is None else len(descriptors),
                0 if self._prev_descriptors is None else len(self._prev_descriptors),
            )
            self._update_state(gray, keypoints, descriptors)
            return np.eye(3, dtype=np.float64)

        # Match features
        try:
            matches = self.bf_matcher.match(self._prev_descriptors, descriptors)
        except cv2.error as e:
            logger.debug("Frame %d: ORB matching failed: %s", self._frame_count, e)
            self._update_state(gray, keypoints, descriptors)
            return np.eye(3, dtype=np.float64)

        self._last_num_matches = len(matches)

        if len(matches) < 4:
            logger.debug(
                "Frame %d: Too few matches (%d). Falling back to identity.",
                self._frame_count, len(matches)
            )
            self._update_state(gray, keypoints, descriptors)
            return np.eye(3, dtype=np.float64)

        # Sort by distance and take top matches
        matches = sorted(matches, key=lambda m: m.distance)
        max_matches = min(len(matches), self.config.orb_max_features)
        matches = matches[:max_matches]

        # Extract matched point coordinates
        src_pts = np.float32(
            [self._prev_keypoints[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [keypoints[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        # Compute homography with RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None or mask is None:
            logger.debug(
                "Frame %d: Homography estimation failed. "
                "Falling back to identity.",
                self._frame_count
            )
            self._update_state(gray, keypoints, descriptors)
            return np.eye(3, dtype=np.float64)

        # Evaluate quality via inlier ratio
        inlier_count = int(mask.sum())
        total_count = len(mask)
        self._last_inlier_ratio = inlier_count / max(total_count, 1)

        if self._last_inlier_ratio < self.config.min_inlier_ratio:
            logger.debug(
                "Frame %d: Low inlier ratio (%.2f < %.2f). "
                "Falling back to identity.",
                self._frame_count,
                self._last_inlier_ratio,
                self.config.min_inlier_ratio,
            )
            self._update_state(gray, keypoints, descriptors)
            return np.eye(3, dtype=np.float64)

        # Scale homography back to full resolution if downsampled
        if self.downsample_factor < 1.0:
            scale = 1.0 / self.downsample_factor
            S = np.array([
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1]
            ], dtype=np.float64)
            S_inv = np.array([
                [self.downsample_factor, 0, 0],
                [0, self.downsample_factor, 0],
                [0, 0, 1]
            ], dtype=np.float64)
            H = S @ H @ S_inv

        # Update cumulative homography
        self._cumulative_homography = H @ self._cumulative_homography

        # Update state for next frame
        self._update_state(gray, keypoints, descriptors)

        logger.debug(
            "Frame %d: Motion estimated. Matches=%d, Inliers=%d (%.1f%%)",
            self._frame_count, len(matches), inlier_count,
            self._last_inlier_ratio * 100,
        )

        return H

    def _update_state(
        self,
        gray: np.ndarray,
        keypoints: Optional[list],
        descriptors: Optional[np.ndarray]
    ):
        """Update the stored state for the next frame comparison."""
        self._prev_gray = gray
        self._prev_keypoints = keypoints
        self._prev_descriptors = descriptors

    def get_cumulative_homography(self) -> np.ndarray:
        """
        Get the cumulative homography from the first frame to the current frame.
        
        Useful for transforming coordinates from any historical frame
        to the current frame's coordinate system.
        """
        return self._cumulative_homography.copy()

    def reset(self):
        """Reset the estimator state (e.g., for a new video)."""
        self._prev_gray = None
        self._prev_keypoints = None
        self._prev_descriptors = None
        self._cumulative_homography = np.eye(3, dtype=np.float64)
        self._last_inlier_ratio = 0.0
        self._last_num_matches = 0
        self._frame_count = 0

    @property
    def last_inlier_ratio(self) -> float:
        """Inlier ratio from the most recent estimation."""
        return self._last_inlier_ratio

    @property
    def last_num_matches(self) -> int:
        """Number of feature matches from the most recent estimation."""
        return self._last_num_matches

    @property
    def is_motion_reliable(self) -> bool:
        """Whether the last motion estimate was considered reliable."""
        return self._last_inlier_ratio >= self.config.min_inlier_ratio
