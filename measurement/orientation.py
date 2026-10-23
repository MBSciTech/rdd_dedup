"""
rdd_dedup/measurement/orientation.py — Accelerometer Gravity Orientation Provider.

Computes per-frame 3D camera orientation (unit 'down' gravity vector) from
mobile phone IMU accelerometer telemetry synced with video frame timestamps.

Coordinate Conventions:
- Android sensor axes: x = right, y = up, z = out of screen (accel reads reaction force in m/s²).
- Camera coordinates: x = right, y = down, z = forward (along lens optical axis).
- For portrait orientation with the rear camera (viewing along -z of device):
      down_camera = normalize(-ax, +ay, +az)
  This single vector encapsulates both pitch (tilt) and roll without gimbal lock or
  wrap-around anomalies.
"""

import os
import logging
from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FrameOrientationProvider:
    """
    Synchronizes IMU accelerometer telemetry with video frames to provide
    per-frame gravity 'down' vectors and attitude angles (tilt and roll).
    
    Usage:
        provider = FrameOrientationProvider("sensor_data.csv", "frame_coordinates.csv")
        down = provider.get_down_vector(frame_index) # (3,) unit vector
        tilt, roll = provider.get_orientation_deg(frame_index)
    """

    def __init__(
        self,
        sensor_csv_path: Optional[str] = None,
        frame_csv_path: Optional[str] = None,
        smooth_sec: float = 0.25,
        smooth_window_sec: Optional[float] = None,
    ):
        """
        Args:
            sensor_csv_path: Path to sensor_data.csv containing accelerometer data.
            frame_csv_path: Path to frame_coordinates.csv with frame_index and elapsed_time_sec.
            smooth_sec: Centered rolling mean window in seconds for filtering vibration.
            smooth_window_sec: Alias for smooth_sec.
        """
        self.sensor_csv_path = sensor_csv_path
        self.frame_csv_path = frame_csv_path
        if smooth_window_sec is not None:
            smooth_sec = smooth_window_sec
        self.smooth_sec = float(smooth_sec)

        self._is_available = False
        self._down_vectors: Dict[int, np.ndarray] = {}
        self._tilt_deg_map: Dict[int, float] = {}
        self._roll_deg_map: Dict[int, float] = {}

        # Quality Control / Survey Statistics
        self.mean_down_vector: Optional[np.ndarray] = None
        self.mean_tilt_deg: Optional[float] = None
        self.std_tilt_deg: Optional[float] = None
        self.mean_roll_deg: Optional[float] = None
        self.std_roll_deg: Optional[float] = None
        self.total_frames: int = 0

        if sensor_csv_path and frame_csv_path:
            self._load_and_interpolate()

    @property
    def is_available(self) -> bool:
        """Whether valid orientation data was loaded."""
        return self._is_available

    def _load_and_interpolate(self):
        """
        Load CSVs, smooth accelerometer signals, and interpolate down vectors to frame times.
        Vectorized across all video frames with zero Python per-frame loops.
        """
        if not os.path.exists(self.sensor_csv_path):
            logger.warning("sensor_data.csv not found at: %s", self.sensor_csv_path)
            return
        if not os.path.exists(self.frame_csv_path):
            logger.warning("frame_coordinates.csv not found at: %s", self.frame_csv_path)
            return

        try:
            s = pd.read_csv(self.sensor_csv_path)
            f = pd.read_csv(self.frame_csv_path)

            required_sensor_cols = {"elapsed_time_sec", "accel_x", "accel_y", "accel_z"}
            required_frame_cols = {"frame_index", "elapsed_time_sec"}

            if not required_sensor_cols.issubset(s.columns):
                logger.warning(
                    "sensor_data.csv missing required columns: %s",
                    required_sensor_cols - set(s.columns)
                )
                return

            if not required_frame_cols.issubset(f.columns):
                logger.warning(
                    "frame_coordinates.csv missing required columns: %s",
                    required_frame_cols - set(f.columns)
                )
                return

            if len(s) < 2 or len(f) == 0:
                logger.warning("Insufficient rows in sensor or frame CSVs.")
                return

            # Sort by elapsed_time_sec to ensure monotonic interpolation
            s = s.sort_values("elapsed_time_sec").reset_index(drop=True)
            f = f.sort_values("frame_index").reset_index(drop=True)

            t_s = s["elapsed_time_sec"].to_numpy(dtype=np.float64)
            dt_s = np.diff(t_s)
            dt_valid = dt_s[dt_s > 0]
            median_dt = float(np.median(dt_valid)) if len(dt_valid) > 0 else 0.024  # ~41 Hz default

            # Centered rolling mean window to smooth human walking / vehicle engine vibration
            win = max(1, int(round(self.smooth_sec / median_dt)))

            acc = s[["accel_x", "accel_y", "accel_z"]].rolling(
                win, center=True, min_periods=1
            ).mean()

            # Synchronize to frame timestamps (np.interp clamps boundaries cleanly)
            t_f = f["elapsed_time_sec"].to_numpy(dtype=np.float64)
            frame_indices = f["frame_index"].to_numpy(dtype=np.int64)

            ax = np.interp(t_f, t_s, acc["accel_x"].to_numpy(dtype=np.float64))
            ay = np.interp(t_f, t_s, acc["accel_y"].to_numpy(dtype=np.float64))
            az = np.interp(t_f, t_s, acc["accel_z"].to_numpy(dtype=np.float64))

            # Transform from phone body to camera coordinates:
            # down_camera = normalize(-ax, +ay, +az)
            g = np.stack([-ax, ay, az], axis=1)
            norms = np.linalg.norm(g, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            down_unit = g / norms

            # Pre-compute tilt and roll angles for all frames:
            # tilt_deg = degrees(asin(down_z))
            # roll_deg = degrees(atan2(-down_x, down_y))
            down_z_clipped = np.clip(down_unit[:, 2], -1.0, 1.0)
            tilts = np.degrees(np.arcsin(down_z_clipped))
            rolls = np.degrees(np.arctan2(-down_unit[:, 0], down_unit[:, 1]))

            # Populate lookup dictionaries
            self._down_vectors = {int(idx): down_unit[i] for i, idx in enumerate(frame_indices)}
            self._tilt_deg_map = {int(idx): float(tilts[i]) for i, idx in enumerate(frame_indices)}
            self._roll_deg_map = {int(idx): float(rolls[i]) for i, idx in enumerate(frame_indices)}

            self.total_frames = len(frame_indices)
            self._is_available = True

            # Calculate survey-wide statistics
            self.mean_down_vector = np.mean(down_unit, axis=0)
            self.mean_tilt_deg = float(np.mean(tilts))
            self.std_tilt_deg = float(np.std(tilts))
            self.mean_roll_deg = float(np.mean(rolls))
            self.std_roll_deg = float(np.std(rolls))

            logger.info(
                "FrameOrientationProvider initialized for %d frames: "
                "mean_down=[%.3f, %.3f, %.3f], tilt=%.1f°±%.1f°, roll=%.1f°±%.1f°",
                self.total_frames,
                self.mean_down_vector[0],
                self.mean_down_vector[1],
                self.mean_down_vector[2],
                self.mean_tilt_deg,
                self.std_tilt_deg,
                self.mean_roll_deg,
                self.std_roll_deg,
            )

            # Orientation Validation Check (Prompt requirement)
            # A phone held upright looking forward-down at the road should have:
            # - Large positive y component (+y ≈ cos(tilt) ~ 0.90 to 0.97)
            # - Small positive z component (+z ≈ sin(tilt) ~ 0.20 to 0.40)
            # - Lateral component near zero (|x| < 0.25)
            if self.mean_down_vector[1] < 0.5:
                logger.warning(
                    "ORIENTATION AUDIT WARNING: Mean down vector y-component is low (%.3f < 0.5). "
                    "The video might be recorded in landscape or inverted orientation. "
                    "Mean vector: [%.3f, %.3f, %.3f]",
                    self.mean_down_vector[1],
                    self.mean_down_vector[0],
                    self.mean_down_vector[1],
                    self.mean_down_vector[2],
                )
            elif self.mean_down_vector[2] < 0:
                logger.warning(
                    "ORIENTATION AUDIT WARNING: Mean down vector z-component is negative (%.3f). "
                    "Camera may be tilted upward toward sky instead of down toward the road surface.",
                    self.mean_down_vector[2],
                )

        except Exception as e:
            logger.error("Error loading frame orientation from %s: %s", self.sensor_csv_path, e, exc_info=True)
            self._is_available = False

    def __len__(self) -> int:
        """Return the number of synchronized frame orientations available."""
        return len(self._down_vectors)

    def get_down_vector(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Get the 3D unit down gravity vector in camera coordinates for the given frame index.
        Returns None if frame index is out of bounds or provider is not available.
        """
        if not self._is_available:
            return None
        return self._down_vectors.get(frame_index)

    def get_orientation_deg(self, frame_index: int) -> Optional[Tuple[float, float]]:
        """
        Get (tilt_deg, roll_deg) for the given frame index.
        Returns None if frame index is not present or provider is not available.
        """
        if not self._is_available:
            return None
        tilt = self._tilt_deg_map.get(frame_index)
        roll = self._roll_deg_map.get(frame_index)
        if tilt is None or roll is None:
            return None
        return (tilt, roll)

    def get_stats(self) -> Dict[str, Any]:
        """Return survey orientation summary statistics."""
        return {
            "is_available": self._is_available,
            "total_frames": self.total_frames,
            "mean_tilt_deg": self.mean_tilt_deg,
            "std_tilt_deg": self.std_tilt_deg,
            "mean_roll_deg": self.mean_roll_deg,
            "std_roll_deg": self.std_roll_deg,
            "mean_down_vector": self.mean_down_vector.tolist() if self.mean_down_vector is not None else None,
        }
