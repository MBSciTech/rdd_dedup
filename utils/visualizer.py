"""
PipelineVisualizer — Generates rich data visualizations from pipeline results.

Creates 6 chart types from the stored pipeline data:
1. Defect Type Distribution (bar chart)
2. Confidence Distribution (histogram)
3. Detection Timeline (scatter plot)
4. Pipeline Funnel (horizontal bar)
5. Spatial Heatmap (2D scatter)
6. Defect Duration Distribution (histogram)

All charts are saved as PNG files and returned as a list of image paths
for display in the Gradio UI.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try importing matplotlib — graceful fallback if not available
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server-side rendering
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed — visualization features disabled.")


# ── Chart Style Constants ──
_DARK_BG = "#0f1117"
_CARD_BG = "#1a1d27"
_GRID_COLOR = "#2a2d3a"
_TEXT_COLOR = "#e2e8f0"
_SUBTEXT_COLOR = "#94a3b8"
_ACCENT_BLUE = "#3b82f6"
_ACCENT_CYAN = "#22d3ee"
_ACCENT_GREEN = "#10b981"
_ACCENT_ORANGE = "#f97316"
_ACCENT_RED = "#ef4444"
_ACCENT_PURPLE = "#a855f7"
_ACCENT_PINK = "#ec4899"
_ACCENT_YELLOW = "#eab308"

# Per-defect-class color mapping
_CLASS_COLORS = {
    "D00": _ACCENT_ORANGE,
    "D10": _ACCENT_YELLOW,
    "D20": _ACCENT_PURPLE,
    "D40": _ACCENT_RED,
    "D00: Longitudinal Crack": _ACCENT_ORANGE,
    "D10: Transverse Crack": _ACCENT_YELLOW,
    "D20: Alligator Crack": _ACCENT_PURPLE,
    "D40: Pothole": _ACCENT_RED,
    "Repair / Patch": _ACCENT_GREEN,
}


def _apply_dark_theme():
    """Apply a consistent dark theme to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": _DARK_BG,
        "axes.facecolor": _CARD_BG,
        "axes.edgecolor": _GRID_COLOR,
        "axes.labelcolor": _TEXT_COLOR,
        "text.color": _TEXT_COLOR,
        "xtick.color": _SUBTEXT_COLOR,
        "ytick.color": _SUBTEXT_COLOR,
        "grid.color": _GRID_COLOR,
        "grid.alpha": 0.4,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
    })


def _get_class_color(class_name: str) -> str:
    """Get the color for a defect class, with fallback."""
    return _CLASS_COLORS.get(class_name, _ACCENT_BLUE)


class PipelineVisualizer:
    """
    Generates visualization charts from a pipeline run's output data.

    Usage:
        visualizer = PipelineVisualizer(output_dir)
        chart_paths = visualizer.generate_all_charts(pipeline_result)
    """

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Path to the pipeline run's output directory
                        (e.g., outputs/road_footage_35_rdd2022_st1_yolo11s/)
        """
        self.output_dir = output_dir
        self.charts_dir = os.path.join(output_dir, "charts")
        os.makedirs(self.charts_dir, exist_ok=True)

    def generate_all_charts(self, result) -> List[Tuple[str, str]]:
        """
        Generate all visualization charts from a PipelineResult.

        Args:
            result: PipelineResult object from the pipeline run

        Returns:
            List of (image_path, caption) tuples for each chart.
            Returns empty list if matplotlib is not available.
        """
        if not HAS_MATPLOTLIB:
            logger.warning("Skipping visualization — matplotlib not available.")
            return []

        _apply_dark_theme()

        charts = []

        # 1. Defect Type Distribution
        try:
            path = self._chart_defect_distribution(result)
            if path:
                charts.append((path, "Defect Type Distribution"))
        except Exception as e:
            logger.warning("Failed to generate defect distribution chart: %s", e)

        # 2. Confidence Distribution
        try:
            path = self._chart_confidence_distribution(result)
            if path:
                charts.append((path, "Confidence Distribution"))
        except Exception as e:
            logger.warning("Failed to generate confidence distribution chart: %s", e)

        # 3. Detection Timeline
        try:
            path = self._chart_detection_timeline(result)
            if path:
                charts.append((path, "Detection Timeline"))
        except Exception as e:
            logger.warning("Failed to generate detection timeline chart: %s", e)

        # 4. Pipeline Funnel
        try:
            path = self._chart_pipeline_funnel(result)
            if path:
                charts.append((path, "Pipeline Reduction Funnel"))
        except Exception as e:
            logger.warning("Failed to generate pipeline funnel chart: %s", e)

        # 5. Spatial Heatmap
        try:
            path = self._chart_spatial_heatmap(result)
            if path:
                charts.append((path, "Defect Spatial Distribution"))
        except Exception as e:
            logger.warning("Failed to generate spatial heatmap chart: %s", e)

        # 6. Duration Distribution
        try:
            path = self._chart_duration_distribution(result)
            if path:
                charts.append((path, "Detection Duration Distribution"))
        except Exception as e:
            logger.warning("Failed to generate duration distribution chart: %s", e)

        logger.info("Generated %d visualization charts in: %s", len(charts), self.charts_dir)
        return charts

    # ── Chart 1: Defect Type Distribution ──

    def _chart_defect_distribution(self, result) -> Optional[str]:
        """Bar chart showing count per defect class."""
        counts = result.defect_counts
        if not counts:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))

        classes = list(counts.keys())
        values = list(counts.values())
        colors = [_get_class_color(c) for c in classes]

        bars = ax.bar(classes, values, color=colors, width=0.6,
                      edgecolor="none", zorder=3)

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom",
                    fontweight="bold", fontsize=13, color=_TEXT_COLOR)

        ax.set_xlabel("Defect Class", fontsize=12, labelpad=10)
        ax.set_ylabel("Unique Defects Count", fontsize=12, labelpad=10)
        ax.set_title("Defect Type Distribution", pad=15)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(self.charts_dir, "defect_distribution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    # ── Chart 2: Confidence Distribution ──

    def _chart_confidence_distribution(self, result) -> Optional[str]:
        """Histogram of average confidence values across unique defects."""
        defects = result.finalized_defects
        if not defects:
            return None

        confidences = [d.avg_confidence for d in defects]
        max_confidences = [d.max_confidence for d in defects]

        fig, ax = plt.subplots(figsize=(8, 5))

        # Avg confidence histogram
        ax.hist(confidences, bins=15, alpha=0.7, color=_ACCENT_BLUE,
                edgecolor=_CARD_BG, linewidth=0.8, label="Avg Confidence", zorder=3)

        # Max confidence histogram (overlay)
        ax.hist(max_confidences, bins=15, alpha=0.4, color=_ACCENT_CYAN,
                edgecolor=_CARD_BG, linewidth=0.8, label="Max Confidence", zorder=2)

        # Mean lines
        avg_mean = np.mean(confidences)
        ax.axvline(avg_mean, color=_ACCENT_ORANGE, linestyle="--", linewidth=1.5,
                   label=f"Mean Avg: {avg_mean:.3f}", zorder=4)

        ax.set_xlabel("Confidence Score", fontsize=12, labelpad=10)
        ax.set_ylabel("Number of Defects", fontsize=12, labelpad=10)
        ax.set_title("Confidence Score Distribution", pad=15)
        ax.legend(loc="upper left", framealpha=0.8,
                  facecolor=_CARD_BG, edgecolor=_GRID_COLOR)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(self.charts_dir, "confidence_distribution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    # ── Chart 3: Detection Timeline ──

    def _chart_detection_timeline(self, result) -> Optional[str]:
        """Scatter plot of defects across the video timeline."""
        defects = result.finalized_defects
        if not defects:
            return None

        fig, ax = plt.subplots(figsize=(12, 5))

        for defect in defects:
            color = _get_class_color(defect.defect_type)
            # Draw a span for each defect's frame range
            ax.barh(
                y=defect.avg_confidence,
                width=defect.last_frame - defect.first_frame,
                left=defect.first_frame,
                height=0.015,
                color=color,
                alpha=0.6,
                zorder=2,
            )
            # Point at max confidence frame
            ax.scatter(
                defect.max_confidence_frame,
                defect.max_confidence,
                color=color,
                s=30,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )

        # Legend — one entry per class
        seen_classes = set()
        legend_handles = []
        for defect in defects:
            if defect.defect_type not in seen_classes:
                seen_classes.add(defect.defect_type)
                handle = ax.scatter([], [], color=_get_class_color(defect.defect_type),
                                    s=40, label=defect.defect_type)
                legend_handles.append(handle)

        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.8,
                  facecolor=_CARD_BG, edgecolor=_GRID_COLOR, fontsize=10)

        ax.set_xlabel("Frame Index", fontsize=12, labelpad=10)
        ax.set_ylabel("Confidence", fontsize=12, labelpad=10)
        ax.set_title("Detection Timeline — Defects Across Video", pad=15)
        ax.grid(alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(self.charts_dir, "detection_timeline.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    # ── Chart 4: Pipeline Reduction Funnel ──

    def _chart_pipeline_funnel(self, result) -> Optional[str]:
        """Horizontal bar chart showing the deduplication funnel."""
        stages = [
            ("Raw Detections", result.total_raw_detections),
            ("BoT-SORT Tracks", result.tracks_created),
            ("Unique Defects", result.total_unique_defects),
        ]

        if all(v == 0 for _, v in stages):
            return None

        fig, ax = plt.subplots(figsize=(8, 4))

        labels = [s[0] for s in stages]
        values = [s[1] for s in stages]
        colors = [_ACCENT_BLUE, _ACCENT_CYAN, _ACCENT_GREEN]

        bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                       height=0.5, edgecolor="none", zorder=3)

        # Value labels
        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + max(values) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", ha="left", va="center",
                    fontweight="bold", fontsize=13, color=_TEXT_COLOR)

        # Reduction annotation
        if result.total_raw_detections > 0:
            reduction = (1 - result.total_unique_defects / result.total_raw_detections) * 100
            ax.text(0.98, 0.05, f"📉 {reduction:.1f}% reduction",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=12, fontweight="bold", color=_ACCENT_GREEN,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=_CARD_BG,
                              edgecolor=_ACCENT_GREEN, alpha=0.8))

        ax.set_xlabel("Count", fontsize=12, labelpad=10)
        ax.set_title("Pipeline Deduplication Funnel", pad=15)
        ax.grid(axis="x", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(self.charts_dir, "pipeline_funnel.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    # ── Chart 5: Spatial Heatmap ──

    def _chart_spatial_heatmap(self, result) -> Optional[str]:
        """2D scatter of defect positions on a frame canvas."""
        defects = result.finalized_defects
        if not defects:
            return None

        # Filter defects with valid centers
        valid = [(d.representative_center[0], d.representative_center[1],
                  d.defect_type, d.avg_confidence)
                 for d in defects
                 if d.representative_center != (0.0, 0.0)]

        if not valid:
            return None

        xs = [v[0] for v in valid]
        ys = [v[1] for v in valid]
        types = [v[2] for v in valid]
        confs = [v[3] for v in valid]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Size proportional to confidence
        sizes = [max(40, c * 200) for c in confs]
        colors = [_get_class_color(t) for t in types]

        ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.7,
                   edgecolors="white", linewidths=0.5, zorder=3)

        # Invert y-axis (image coordinates: 0,0 is top-left)
        ax.invert_yaxis()

        # Legend
        seen = set()
        for t in types:
            if t not in seen:
                seen.add(t)
                ax.scatter([], [], color=_get_class_color(t), s=60, label=t)
        ax.legend(loc="upper right", framealpha=0.8,
                  facecolor=_CARD_BG, edgecolor=_GRID_COLOR, fontsize=10)

        ax.set_xlabel("X Position (pixels)", fontsize=12, labelpad=10)
        ax.set_ylabel("Y Position (pixels)", fontsize=12, labelpad=10)
        ax.set_title("Defect Spatial Distribution", pad=15)
        ax.grid(alpha=0.2, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(self.charts_dir, "spatial_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    # ── Chart 6: Detection Duration Distribution ──

    def _chart_duration_distribution(self, result) -> Optional[str]:
        """Histogram showing how many frames each defect was observed."""
        defects = result.finalized_defects
        if not defects:
            return None

        durations = [d.frames_observed for d in defects]

        fig, ax = plt.subplots(figsize=(8, 5))

        n_bins = min(20, max(5, len(set(durations))))
        ax.hist(durations, bins=n_bins, color=_ACCENT_PURPLE, alpha=0.8,
                edgecolor=_CARD_BG, linewidth=0.8, zorder=3)

        # Mean and median lines
        mean_dur = np.mean(durations)
        median_dur = np.median(durations)
        ax.axvline(mean_dur, color=_ACCENT_ORANGE, linestyle="--", linewidth=1.5,
                   label=f"Mean: {mean_dur:.1f} frames", zorder=4)
        ax.axvline(median_dur, color=_ACCENT_CYAN, linestyle=":", linewidth=1.5,
                   label=f"Median: {median_dur:.0f} frames", zorder=4)

        ax.set_xlabel("Frames Observed", fontsize=12, labelpad=10)
        ax.set_ylabel("Number of Defects", fontsize=12, labelpad=10)
        ax.set_title("Detection Duration Distribution", pad=15)
        ax.legend(loc="upper right", framealpha=0.8,
                  facecolor=_CARD_BG, edgecolor=_GRID_COLOR)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.tight_layout()
        path = os.path.join(self.charts_dir, "duration_distribution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return path


def generate_visualizations_from_directory(output_dir: str) -> List[Tuple[str, str]]:
    """
    Generate visualizations from a previously completed pipeline run directory.

    Loads data from summary.json and reconstructs enough context to
    generate charts. Used for the History page in the Gradio UI.

    Args:
        output_dir: Path to a pipeline run output directory

    Returns:
        List of (image_path, caption) tuples for each chart.
    """
    if not HAS_MATPLOTLIB:
        return []

    charts_dir = os.path.join(output_dir, "charts")

    # If charts already exist, return them directly
    if os.path.exists(charts_dir):
        existing = []
        chart_names = [
            ("defect_distribution.png", "Defect Type Distribution"),
            ("confidence_distribution.png", "Confidence Distribution"),
            ("detection_timeline.png", "Detection Timeline"),
            ("pipeline_funnel.png", "Pipeline Reduction Funnel"),
            ("spatial_heatmap.png", "Defect Spatial Distribution"),
            ("duration_distribution.png", "Detection Duration Distribution"),
        ]
        for filename, caption in chart_names:
            path = os.path.join(charts_dir, filename)
            if os.path.exists(path):
                existing.append((path, caption))
        if existing:
            return existing

    # Otherwise, try to reconstruct from summary.json
    summary_path = os.path.join(output_dir, "summary.json")
    if not os.path.exists(summary_path):
        return []

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    # Build a lightweight mock PipelineResult for chart generation
    from rdd_dedup.tracking.defect_track import FinalizedDefect, PipelineResult

    finalized = []
    for d in data.get("finalized_defects", []):
        defect = FinalizedDefect(
            defect_id=d.get("defect_id", ""),
            defect_type=d.get("defect_type", ""),
            avg_confidence=d.get("avg_confidence", 0.0),
            max_confidence=d.get("max_confidence", 0.0),
            max_confidence_frame=d.get("max_confidence_frame", d.get("first_frame", 0)),
            first_frame=d.get("first_frame", 0),
            last_frame=d.get("last_frame", 0),
            frames_observed=d.get("frames_observed", 0),
            detection_duration_frames=d.get("last_frame", 0) - d.get("first_frame", 0) + 1,
            source_track_ids=d.get("source_track_ids", []),
        )
        finalized.append(defect)

    mock_result = PipelineResult(
        video_path=data.get("video_path", ""),
        total_frames=data.get("total_frames", 0),
        processed_frames=data.get("processed_frames", 0),
        processing_time_sec=data.get("processing_time_sec", 0),
        fps=data.get("fps", 0),
        total_raw_detections=data.get("total_raw_detections", 0),
        total_unique_defects=data.get("total_unique_defects", 0),
        tracks_created=data.get("tracks_created", 0),
        tracks_merged=data.get("tracks_merged", 0),
        defect_counts=data.get("defect_counts", {}),
        finalized_defects=finalized,
        output_dir=output_dir,
    )

    visualizer = PipelineVisualizer(output_dir)
    return visualizer.generate_all_charts(mock_result)
