"""
rdd_dedup.measurement — Defect physical measurement and camera ground projection.
"""

from rdd_dedup.measurement.area_calculator import (
    CameraParams,
    is_area_eligible_class,
    pixel_to_ground,
    calculate_area,
    compute_defect_area,
    compute_defect_area_from_bbox,
    compute_defect_area_from_points,
)

__all__ = [
    "CameraParams",
    "is_area_eligible_class",
    "pixel_to_ground",
    "calculate_area",
    "compute_defect_area",
    "compute_defect_area_from_bbox",
    "compute_defect_area_from_points",
]
