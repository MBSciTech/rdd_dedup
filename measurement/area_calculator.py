"""
rdd_dedup/measurement/area_calculator.py — Ground Projection and Area Calculator.

Implements flat-plane camera ground projection and polygon area computation
specifically for Pothole, Manhole, and Patch road defects.

Upgraded to support:
1. Dynamic per-frame camera orientation (unit 'down' gravity vector from accelerometer).
2. Backward-compatible fixed camera tilt angle fallback.
3. Comprehensive Quality Gating:
   - Horizon rejection (points above horizon)
   - Tilt and roll angle range enforcement
   - Ground distance boundaries (Z_min to Z_max)
   - Frame border / clipped bbox detection
   - Optional Laplacian blur check on crop
4. Area calculation via Gauss's Shoelace formula:
   Area = 0.5 * |sum_{i}(X_i * Z_{i+1} - X_{i+1} * Z_i)|
"""

import math
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

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
    Supports both isotropic focal length f and anisotropic fx, fy.
    """
    def __init__(
        self,
        yc: float = 1.2,
        tilt_deg: float = 15.0,
        f: Optional[float] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        image_width: int = 1920,
        image_height: int = 1080,
    ):
        self.yc = float(yc)
        self.tilt_deg = float(tilt_deg)
        self.tilt = float(np.radians(tilt_deg))
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        
        # Focal lengths
        if fx is not None and fx > 0:
            self.fx = float(fx)
        elif f is not None and f > 0:
            self.fx = float(f)
        else:
            self.fx = float(image_width * 0.95) if image_width > 0 else 1200.0

        if fy is not None and fy > 0:
            self.fy = float(fy)
        elif f is not None and f > 0:
            self.fy = float(f)
        else:
            self.fy = self.fx

        self.f = self.fx  # Backward compatibility alias
            
        self.cx = float(cx) if cx is not None else float(image_width / 2.0)
        self.cy = float(cy) if cy is not None else float(image_height / 2.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "yc": self.yc,
            "tilt_deg": self.tilt_deg,
            "tilt": self.tilt,
            "f": self.f,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "image_width": self.image_width,
            "image_height": self.image_height,
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
            fx=float(d.get("fx")) if d.get("fx") is not None else None,
            fy=float(d.get("fy")) if d.get("fy") is not None else None,
            cx=float(d.get("cx")) if d.get("cx") is not None else None,
            cy=float(d.get("cy")) if d.get("cy") is not None else None,
            image_width=d.get("image_width", image_width),
            image_height=d.get("image_height", image_height),
        )


def pixel_to_ground(
    data: Union[Dict[str, Any], List[Tuple[float, float]], np.ndarray],
    camera_params: Optional[CameraParams] = None,
    down: Optional[Union[np.ndarray, List[float], Tuple[float, float, float]]] = None,
) -> Tuple[List[float], List[float]]:
    """
    Project pixel coordinates (u, v) onto flat ground plane (X, Z).
    
    Supports two calculation methods:
    1. Per-frame gravity orientation vector `down` = (down_x, down_y, down_z):
       Accounts for both pitch and roll on a per-frame basis from accelerometer telemetry.
    2. Fallback fixed camera tilt theta:
       When `down is None`, uses camera_params.tilt_deg with roll = 0.
       (Note: with roll = 0 and down = [0, cos(tilt), sin(tilt)], both methods are identical).
    
    Returns:
        (X, Z) where:
        X: lateral ground coordinates in meters (right positive, left negative)
        Z: longitudinal ground distance in meters (forward distance from camera)
        
    Raises:
        ValueError if any point is at or above the horizon (ray @ down <= 1e-6).
    """
    if isinstance(data, dict):
        yc = float(data["yc"])
        tilt = float(data.get("tilt", 0.0))
        fx = float(data.get("fx", data.get("f", 1200.0)))
        fy = float(data.get("fy", data.get("f", 1200.0)))
        cx = float(data["cx"])
        cy = float(data["cy"])
        pts = data["p"]
        if "down" in data and down is None:
            down = data["down"]
    else:
        if camera_params is None:
            camera_params = CameraParams()
        yc = camera_params.yc
        tilt = camera_params.tilt
        fx = camera_params.fx
        fy = camera_params.fy
        cx = camera_params.cx
        cy = camera_params.cy
        pts = data

    X, Z = [], []

    # ── Method 1: Per-frame accelerometer gravity orientation ──
    if down is not None:
        n = np.asarray(down, dtype=np.float64)
        n_len = np.linalg.norm(n)
        if n_len < 1e-8:
            raise ValueError("Degenerate gravity down vector (length near zero)")
        n = n / n_len

        # Camera optical forward direction (along +z in camera coords)
        z_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        
        # Camera forward direction flattened onto the road surface
        e_f = z_cam - float(np.dot(z_cam, n)) * n
        norm_f = float(np.linalg.norm(e_f))
        if norm_f < 1e-8:
            raise ValueError("Optical axis points straight along gravity vector")
        e_f = e_f / norm_f
        
        # Road-plane lateral "right" axis
        e_r = np.cross(n, e_f)
        norm_r = float(np.linalg.norm(e_r))
        if norm_r > 1e-8:
            e_r = e_r / norm_r

        for pt in pts:
            if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2:
                u, v = float(pt[0]), float(pt[1])
            else:
                continue

            ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
            denom = float(np.dot(ray, n))
            
            if denom <= 1e-6:
                raise ValueError(f"Point ({u}, {v}) is at or above horizon (denom={denom:.2e})")

            # 3D intersection point on the road surface
            P = (yc / denom) * ray
            X.append(float(np.dot(P, e_r)))
            Z.append(float(np.dot(P, e_f)))

        return X, Z

    # ── Method 2: Fixed-tilt fallback ──
    s = float(np.sin(tilt))
    c = float(np.cos(tilt))

    for pt in pts:
        if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2:
            u, v = float(pt[0]), float(pt[1])
        else:
            continue

        a = (u - cx) / fx
        b = (v - cy) / fy
        # Clamp down_proj for legacy callers to avoid zero-division or crash on sky points
        down_proj = max(s + b * c, 1e-6)
        forward = c - b * s

        scale = yc / down_proj
        Z.append(float(forward * scale))
        X.append(float(a * scale))

    return X, Z


def calculate_area(X: Union[List[float], np.ndarray], Z: Union[List[float], np.ndarray]) -> float:
    """
    Compute 2D polygon area on ground plane using Shoelace formula (Gauss's area formula).
    
    Args:
        X: Lateral ground coordinates in meters.
        Z: Longitudinal ground distance in meters.
        
    Returns:
        Area in square meters (m²).
    """
    n = len(X)
    if n < 3:
        return 0.0

    x_arr = np.asarray(X, dtype=np.float64)
    z_arr = np.asarray(Z, dtype=np.float64)
    return float(0.5 * abs(np.dot(x_arr, np.roll(z_arr, -1)) - np.dot(z_arr, np.roll(x_arr, -1))))


def compute_defect_area_from_points(
    points: Union[List[Tuple[float, float]], np.ndarray],
    camera_params: CameraParams,
    class_name: Optional[str] = None,
    down: Optional[Union[np.ndarray, List[float]]] = None,
) -> Optional[float]:
    """
    Compute real-world area (m²) from polygon contour points.
    Returns None if class is not eligible or points are invalid.
    """
    if class_name is not None and not is_area_eligible_class(class_name):
        return None

    if points is None or len(points) < 3:
        return None

    try:
        X, Z = pixel_to_ground(points, camera_params, down=down)
        area = calculate_area(X, Z)
        return round(area, 4)
    except ValueError:
        return None


def compute_defect_area_from_bbox(
    bbox: Tuple[int, int, int, int],
    camera_params: CameraParams,
    class_name: Optional[str] = None,
    down: Optional[Union[np.ndarray, List[float]]] = None,
) -> Optional[float]:
    """
    Compute real-world area (m²) from bounding box (x1, y1, x2, y2).
    Generates 4 corner polygon points in clockwise order.
    Returns None if class is not eligible.
    """
    if class_name is not None and not is_area_eligible_class(class_name):
        return None

    if not bbox or len(bbox) < 4:
        return None

    x1, y1, x2, y2 = bbox
    p = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    try:
        X, Z = pixel_to_ground(p, camera_params, down=down)
        area = calculate_area(X, Z)
        return round(area, 4)
    except ValueError:
        return None


def compute_defect_area(
    bbox: Tuple[int, int, int, int],
    camera_params: CameraParams,
    class_name: Optional[str] = None,
    polygon: Optional[Union[List[Tuple[float, float]], np.ndarray]] = None,
    down: Optional[Union[np.ndarray, List[float], Tuple[float, float, float]]] = None,
    min_tilt_deg: float = 3.0,
    max_tilt_deg: float = 60.0,
    max_abs_roll_deg: float = 30.0,
    min_ground_distance_m: float = 0.5,
    max_ground_distance_m: float = 8.0,
    crop_image: Optional[np.ndarray] = None,
    min_laplacian_var: Optional[float] = None,
    rejection_sink: Optional[List[str]] = None,
) -> Optional[float]:
    """
    Main defect area computation dispatcher with strict quality gating.
    
    Quality Gating Rules (returns None if violated):
    1. Class Eligibility: Only Pothole, Manhole, and Patch are computed.
    2. Clipped Border: Rejects bbox touching frame borders (x1<=0, y1<=0, x2>=W-1, y2>=H-1).
    3. Tilt & Roll Gating: Rejects if tilt or |roll| is outside configured bounds.
    4. Horizon Gating: Rejects if any vertex is at or above the horizon.
    5. Distance Gating: Rejects if mean forward ground distance Z is outside [Z_min, Z_max].
    6. Blur Gating (optional): Rejects if crop variance of Laplacian is below threshold.
    
    Args:
        bbox: (x1, y1, x2, y2) detection bounding box.
        camera_params: Camera intrinsic parameters and dimensions.
        class_name: Defect class name.
        polygon: Optional segmentation mask polygon points (preferred).
        down: Optional per-frame unit gravity down vector (from accelerometer).
        min_tilt_deg: Minimum allowed downward camera tilt in degrees.
        max_tilt_deg: Maximum allowed downward camera tilt in degrees.
        max_abs_roll_deg: Maximum allowed absolute roll angle in degrees.
        min_ground_distance_m: Minimum allowed forward distance in meters.
        max_ground_distance_m: Maximum allowed forward distance in meters.
        crop_image: Optional BGR crop image for blur assessment.
        min_laplacian_var: Optional minimum variance of Laplacian threshold.
        rejection_sink: Optional list to append rejection reason strings for QC stats.
        
    Returns:
        Area in m² if valid and eligible; otherwise None.
    """
    # Gate 1: Class Eligibility
    if class_name is not None and not is_area_eligible_class(class_name):
        return None

    if not bbox or len(bbox) < 4:
        return None

    x1, y1, x2, y2 = bbox
    img_w = camera_params.image_width
    img_h = camera_params.image_height

    # Gate 2: Clipped Bounding Box at Frame Border
    if x1 <= 0 or y1 <= 0 or x2 >= (img_w - 1) or y2 >= (img_h - 1):
        if rejection_sink is not None:
            rejection_sink.append("clipped_border")
        return None

    # Gate 3: Camera Attitude / Tilt & Roll Limits
    if down is not None:
        n = np.asarray(down, dtype=np.float64)
        n_len = np.linalg.norm(n)
        if n_len > 1e-8:
            n = n / n_len
            down_z = float(np.clip(n[2], -1.0, 1.0))
            tilt_deg = float(np.degrees(np.arcsin(down_z)))
            roll_deg = float(np.degrees(np.arctan2(-n[0], n[1])))

            if tilt_deg < min_tilt_deg or tilt_deg > max_tilt_deg:
                if rejection_sink is not None:
                    rejection_sink.append("tilt_out_of_range")
                return None

            if abs(roll_deg) > max_abs_roll_deg:
                if rejection_sink is not None:
                    rejection_sink.append("roll_out_of_range")
                return None

    # Gate 4: Blur Assessment (Optional)
    if crop_image is not None and min_laplacian_var is not None and crop_image.size > 0:
        try:
            import cv2
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY) if crop_image.ndim == 3 else crop_image
            var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if var < min_laplacian_var:
                if rejection_sink is not None:
                    rejection_sink.append("blurry_frame")
                return None
        except Exception:
            pass

    # Select points to project (polygon if valid, else bbox 4 corners)
    if polygon is not None and len(polygon) >= 3:
        pts = polygon
    else:
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    # Gate 5: Horizon Ray Projection
    if down is not None:
        try:
            X, Z = pixel_to_ground(pts, camera_params, down=down)
        except ValueError:
            if rejection_sink is not None:
                rejection_sink.append("above_horizon")
            return None
    else:
        s = float(np.sin(np.radians(camera_params.tilt_deg)))
        c = float(np.cos(np.radians(camera_params.tilt_deg)))
        fy = camera_params.fy
        cy = camera_params.cy
        for p in pts:
            b = (float(p[1]) - cy) / fy
            if (s + b * c) <= 1e-6:
                if rejection_sink is not None:
                    rejection_sink.append("above_horizon")
                return None
        X, Z = pixel_to_ground(pts, camera_params)

    # Gate 6: Longitudinal Distance Boundaries [Z_min, Z_max]
    if len(Z) == 0:
        return None
        
    mean_z = float(np.mean(Z))
    if mean_z < min_ground_distance_m or mean_z > max_ground_distance_m:
        if rejection_sink is not None:
            rejection_sink.append("distance_out_of_range")
        return None

    area = calculate_area(X, Z)
    return round(float(area), 4)
