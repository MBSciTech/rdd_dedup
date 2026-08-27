"""
image_utils — Image processing utilities for the RDD dedup pipeline.

Provides crop extraction, color histogram computation, ORB feature matching,
and image quality checks used in duplicate verification.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


def extract_crop(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 0
) -> Optional[np.ndarray]:
    """
    Extract a crop from a frame given a bounding box.
    
    Args:
        frame: Full video frame (BGR)
        bbox: (x1, y1, x2, y2) pixel coordinates
        padding: Extra pixels to include around the bbox
    
    Returns:
        Cropped image array, or None if the crop is invalid.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # Apply padding with bounds checking
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


def is_crop_valid(
    crop: Optional[np.ndarray],
    min_size: int = 20
) -> bool:
    """
    Check if a crop is valid for feature comparison.
    
    A crop is invalid if it's too small, mostly black, or heavily blurred.
    
    Args:
        crop: Image array to validate
        min_size: Minimum width and height in pixels
    """
    if crop is None:
        return False

    h, w = crop.shape[:2]
    if h < min_size or w < min_size:
        return False

    # Check if mostly black (mean pixel value < 10)
    if np.mean(crop) < 10:
        return False

    return True


def compute_color_histogram(
    crop: np.ndarray,
    bins: int = 32
) -> Optional[np.ndarray]:
    """
    Compute a normalized HSV color histogram for a crop.
    
    HSV is more robust to lighting changes than BGR — important for
    outdoor handheld recordings with variable illumination.
    
    Args:
        crop: BGR image array
        bins: Number of bins per channel
    
    Returns:
        Flattened, normalized histogram vector, or None if crop is invalid.
    """
    if not is_crop_valid(crop):
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Compute histogram for H and S channels (V is too variable with lighting)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])

    # Concatenate and normalize
    hist = np.concatenate([hist_h, hist_s]).flatten()
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    return hist


def compare_histograms(
    hist1: Optional[np.ndarray],
    hist2: Optional[np.ndarray]
) -> float:
    """
    Compare two color histograms using correlation method.
    
    Returns:
        Similarity score in [0, 1] where 1 = identical histograms.
        Returns 0.5 (neutral) if either histogram is None.
    """
    if hist1 is None or hist2 is None:
        return 0.5  # Neutral score when comparison is not possible

    # cv2.HISTCMP_CORREL returns [-1, 1], map to [0, 1]
    score = cv2.compareHist(
        hist1.astype(np.float32),
        hist2.astype(np.float32),
        cv2.HISTCMP_CORREL
    )
    return float(max(0.0, (score + 1.0) / 2.0))


def compute_orb_features(
    crop: np.ndarray,
    max_features: int = 200
) -> Tuple[Optional[list], Optional[np.ndarray]]:
    """
    Detect ORB keypoints and compute descriptors for a crop.
    
    Args:
        crop: BGR image array
        max_features: Maximum number of features to detect
    
    Returns:
        Tuple of (keypoints, descriptors), or (None, None) if detection fails.
    """
    if not is_crop_valid(crop):
        return None, None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if keypoints is None or descriptors is None or len(keypoints) == 0:
        return None, None

    return keypoints, descriptors


def match_orb_features(
    desc1: Optional[np.ndarray],
    desc2: Optional[np.ndarray],
    ratio_threshold: float = 0.75,
    min_descriptors: int = 15
) -> float:
    """
    Match ORB descriptors between two crops and return a similarity score.
    
    Uses brute-force matching with Hamming distance and Lowe's ratio test.
    
    Args:
        desc1: ORB descriptors from crop 1
        desc2: ORB descriptors from crop 2
        ratio_threshold: Lowe's ratio test threshold
        min_descriptors: Minimum descriptors needed to trust a match score
    
    Returns:
        Similarity score in [0, 1] based on ratio of good matches.
        Returns 0.5 (neutral) if matching is not possible or untrustworthy.
    """
    if desc1 is None or desc2 is None:
        return 0.5  # Neutral score when comparison is not possible

    if len(desc1) < min_descriptors or len(desc2) < min_descriptors:
        return 0.5

    # Use BFMatcher with KNN (k=2) for ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        matches = bf.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return 0.5

    # Apply Lowe's ratio test
    good_matches = 0
    total_matches = 0
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            total_matches += 1
            if m.distance < ratio_threshold * n.distance:
                good_matches += 1

    if total_matches == 0:
        return 0.5

    return float(good_matches / total_matches)


def resize_crop_for_comparison(
    crop: np.ndarray,
    target_size: Tuple[int, int] = (64, 64)
) -> np.ndarray:
    """
    Resize a crop to a standard size for consistent comparison.
    
    Args:
        crop: BGR image array
        target_size: (width, height) target dimensions
    
    Returns:
        Resized image array
    """
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
