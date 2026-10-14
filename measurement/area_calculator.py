"""
rdd_dedup/measurement/area_calculator.py — Ground Projection and Area Calculator.

Implements flat-plane camera ground projection and polygon area computation
specifically for Pothole, Manhole, and Patch road defects.

Mathematics:
- Camera at height yc above ground plane, with downward tilt angle theta.
- Ray projection:
    a = (u - cx) / f
    b = (v - cy) / f
    down = sin(tilt) + b * cos(tilt)
    forward = cos(tilt) - b * sin(tilt)
    scale = yc / down
    Z = forward * scale  (longitudinal ground distance in meters)
    X = a * scale        (lateral ground distance in meters)
- Area calculation via Shoelace formula (Gauss's area formula):
    Area = 0.5 * |sum_{i}(X_i * Z_{i+1} - X_{i+1} * Z_i)|
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any


# Only Pothole, Manhole, and Patch are eligible for area computation
AREA_ELIGIBLE_KEYWORDS = {
    "POTHOLE", "D40", "MANHOLE", "PATCH", "REPAIR"
}


def is_area_eligible_class(class_name: Optional[Union[str, int]]) -> bool:
    """
    Check if defect class is eligible for area computation.
    Returns True ONLY for Pothole, Manhole, and Patch defects.
    """
    if class_name is None:
        return False
    
    # Handle numeric class ID or string
    s = str(class_name).strip().upper().replace(" ", "_").replace("-", "_")
    
    # Check for pothole (including D40 standard RDD code)
    if "POTHOLE" in s or s == "D40" or s.startswith("D40:"):
        return True
        
    # Check for manhole
    if "MANHOLE" in s:
        return True
        
    # Check for patch / repair (excluding long crack repairs if marked separately)
    if ("PATCH" in s or "REPAIR" in s) and "LONG_REPAIR" not in s:
        return True
        
    return False


class CameraParams:
    """
    Camera calibration parameters for flat-plane ground projection.
    """
    def __init__(
        self,
        yc: float = 1.2,
        tilt_deg: float = 15.0,
        f: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        image_width: int = 1920,
        image_height: int = 1080,
    ):
        self.yc = float(yc)
        self.tilt_deg = float(tilt_deg)
        self.tilt = float(np.radians(tilt_deg))
        
        # Default focal length: assuming ~60 degree horizontal FOV if not specified
        if f is not None and f > 0:
            self.f = float(f)
        else:
            # f = width / (2 * tan(HFOV / 2)), with HFOV ~ 65 deg -> f ~ 0.95 * width
            self.f = float(image_width * 0.95) if image_width > 0 else 1200.0
            
        self.cx = float(cx) if cx is not None else float(image_width / 2.0)
        self.cy = float(cy) if cy is not None else float(image_height / 2.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "yc": self.yc,
            "tilt_deg": self.tilt_deg,
            "tilt": self.tilt,
            "f": self.f,
            "cx": self.cx,
            "cy": self.cy,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], image_width: int = 1920, image_height: int = 1080) -> "CameraParams":
        tilt_deg = d.get("tilt_deg")
        if tilt_deg is None and "tilt" in d:
            tilt_deg = float(np.degrees(d["tilt"]))
        return cls(
            yc=float(d.get("yc", 1.2)),
            tilt_deg=float(tilt_deg if tilt_deg is not None else 15.0),
            f=float(d.get("f")) if d.get("f") is not None else None,
            cx=float(d.get("cx")) if d.get("cx") is not None else None,
            cy=float(d.get("cy")) if d.get("cy") is not None else None,
            image_width=image_width,
            image_height=image_height,
        )


def pixel_to_ground(
    data: Union[Dict[str, Any], List[Tuple[float, float]], np.ndarray],
    camera_params: Optional[CameraParams] = None,
) -> Tuple[List[float], List[float]]:
    """
    Project pixel coordinates (u, v) onto flat ground plane (X, Z).
    
    Accepts:
    1. Dict matching user's format: {'yc': yc, 'p': [(u, v), ...], 'tilt': tilt_rad, 'f': f, 'cx': cx, 'cy': cy}
    2. Or (points_list, camera_params)
    
    Returns:
        (X, Z) where:
        X: lateral ground coordinates in meters (right positive, left negative)
        Z: longitudinal ground distance in meters (forward distance from camera)
    """
    if isinstance(data, dict):
        yc = float(data["yc"])
        tilt = float(data["tilt"])
        f = float(data["f"])
        cx = float(data["cx"])
        cy = float(data["cy"])
        pts = data["p"]
    else:
        if camera_params is None:
            camera_params = CameraParams()
        yc = camera_params.yc
        tilt = camera_params.tilt
        f = camera_params.f
        cx = camera_params.cx
        cy = camera_params.cy
        pts = data

    X, Z = [], []
    s = float(np.sin(tilt))
    c = float(np.cos(tilt))

    for pt in pts:
        if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2:
            u, v = float(pt[0]), float(pt[1])
        else:
            continue

        a = (u - cx) / f
        b = (v - cy) / f
        down = s + b * c
        forward = c - b * s

        # Safe clamp for points at or above horizon (avoids division by zero / negative depth)
        if down <= 1e-6:
            down = 1e-6

        scale = yc / down
        Z.append(float(forward * scale))
        X.append(float(a * scale))

    return X, Z


def calculate_area(X: List[float], Z: List[float]) -> float:
    """
    Compute 2D polygon area on ground plane using Shoelace formula.
    
    Args:
        X: List of X ground coordinates (meters)
        Z: List of Z ground coordinates (meters)
        
    Returns:
        Area in square meters (m²).
    """
    n = len(X)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += X[i] * Z[j] - X[j] * Z[i]

    return float(abs(area) / 2.0)


def compute_defect_area_from_points(
    points: Union[List[Tuple[float, float]], np.ndarray],
    camera_params: CameraParams,
    class_name: Optional[str] = None,
) -> Optional[float]:
    """
    Compute real-world area (m²) from polygon contour points.
    Returns None if class is not eligible (only Pothole, Manhole, Patch).
    """
    if class_name is not None and not is_area_eligible_class(class_name):
        return None

    if points is None or len(points) < 3:
        return None

    X, Z = pixel_to_ground(points, camera_params)
    area = calculate_area(X, Z)
    return round(area, 4)


def compute_defect_area_from_bbox(
    bbox: Tuple[int, int, int, int],
    camera_params: CameraParams,
    class_name: Optional[str] = None,
) -> Optional[float]:
    """
    Compute real-world area (m²) from bounding box (x1, y1, x2, y2).
    Generates 4 corner polygon points in clockwise order.
    Returns None if class is not eligible (only Pothole, Manhole, Patch).
    """
    if class_name is not None and not is_area_eligible_class(class_name):
        return None

    if not bbox or len(bbox) < 4:
        return None

    x1, y1, x2, y2 = bbox
    # 4 corners of bounding box in clockwise order:
    # Top-Left, Top-Right, Bottom-Right, Bottom-Left
    p = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    X, Z = pixel_to_ground(p, camera_params)
    area = calculate_area(X, Z)
    return round(area, 4)


def compute_defect_area(
    bbox: Tuple[int, int, int, int],
    camera_params: CameraParams,
    class_name: Optional[str] = None,
    polygon: Optional[Union[List[Tuple[float, float]], np.ndarray]] = None,
) -> Optional[float]:
    """
    Main defect area computation dispatcher:
    Uses segmentation polygon if available with >= 3 points;
    otherwise falls back to bbox corners.
    
    Returns area in m² if eligible (Pothole, Manhole, Patch), else None.
    """
    if class_name is not None and not is_area_eligible_class(class_name):
        return None

    if polygon is not None and len(polygon) >= 3:
        return compute_defect_area_from_points(polygon, camera_params, class_name)

    return compute_defect_area_from_bbox(bbox, camera_params, class_name)
