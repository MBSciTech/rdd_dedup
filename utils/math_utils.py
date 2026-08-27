"""
math_utils — Mathematical utility functions for the RDD dedup pipeline.

Provides geometric computations used across tracking, verification,
and motion estimation modules.
"""

import numpy as np
from typing import Tuple


def compute_iou(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int]
) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: (x1, y1, x2, y2)
        bbox2: (x1, y1, x2, y2)
    
    Returns:
        IoU score in [0, 1]
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / max(union, 1e-6)


def compute_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Compute the center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def compute_area(bbox: Tuple[int, int, int, int]) -> float:
    """Compute the pixel area of a bounding box."""
    x1, y1, x2, y2 = bbox
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def compute_aspect_ratio(bbox: Tuple[int, int, int, int]) -> float:
    """Compute width/height aspect ratio of a bounding box."""
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return float(w / h)


def euclidean_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float]
) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def normalized_center_distance(
    center1: Tuple[float, float],
    center2: Tuple[float, float],
    frame_diagonal: float
) -> float:
    """
    Normalized distance between two centers, scaled by frame diagonal.
    
    Returns:
        Value in [0, 1] where 0 = same position, 1 = opposite corners.
    """
    dist = euclidean_distance(center1, center2)
    return min(1.0, dist / max(frame_diagonal, 1.0))


def area_similarity(area1: float, area2: float) -> float:
    """
    Area similarity score.
    
    Returns:
        Value in [0, 1] where 1 = identical areas.
    """
    if max(area1, area2) < 1e-6:
        return 1.0
    return min(area1, area2) / max(area1, area2)


def aspect_ratio_similarity(ar1: float, ar2: float) -> float:
    """
    Aspect ratio similarity score.
    
    Returns:
        Value in [0, 1] where 1 = identical aspect ratios.
    """
    max_ar = max(ar1, ar2)
    if max_ar < 1e-6:
        return 1.0
    return 1.0 - abs(ar1 - ar2) / max_ar


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    
    Returns:
        Value in [-1, 1] where 1 = identical direction.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def temporal_gap_similarity(
    frame_gap: int,
    max_window: int
) -> float:
    """
    Temporal proximity score — closer tracks score higher.
    
    Args:
        frame_gap: Absolute frame difference between two track endpoints
        max_window: Maximum temporal window for comparison
    
    Returns:
        Value in [0, 1] where 1 = adjacent frames.
    """
    if max_window <= 0:
        return 0.0
    return max(0.0, 1.0 - frame_gap / max_window)


def transform_point(
    point: Tuple[float, float],
    homography: np.ndarray
) -> Tuple[float, float]:
    """
    Apply a 3×3 homography transformation to a 2D point.
    
    Args:
        point: (x, y) coordinates
        homography: 3×3 transformation matrix
    
    Returns:
        Transformed (x', y') coordinates
    """
    pt = np.array([point[0], point[1], 1.0], dtype=np.float64)
    transformed = homography @ pt
    w = transformed[2]
    if abs(w) < 1e-8:
        return point  # Degenerate case — return original
    return (float(transformed[0] / w), float(transformed[1] / w))


def transform_bbox(
    bbox: Tuple[int, int, int, int],
    homography: np.ndarray
) -> Tuple[int, int, int, int]:
    """
    Apply a homography to a bounding box by transforming its corners
    and computing the axis-aligned bounding box of the result.
    
    Args:
        bbox: (x1, y1, x2, y2)
        homography: 3×3 transformation matrix
    
    Returns:
        Transformed (x1, y1, x2, y2) axis-aligned bounding box
    """
    x1, y1, x2, y2 = bbox
    corners = [
        (float(x1), float(y1)),
        (float(x2), float(y1)),
        (float(x2), float(y2)),
        (float(x1), float(y2)),
    ]
    transformed_corners = [transform_point(c, homography) for c in corners]
    xs = [c[0] for c in transformed_corners]
    ys = [c[1] for c in transformed_corners]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
