"""
scripts/benchmark_annotation_style.py
-------------------------------------
Reusable, non-destructive benchmark comparing old vs. new annotation rendering styles.

Features:
- Standalone implementations of both annotation styles (old full-box vs. new HUD brackets + glow + chip label).
- Monkey-patches RDDPipeline._annotate_frame via types.MethodType in memory — NEVER modifies pipeline.py on disk.
- Scans up to 10 videos in --video-dir, sampling 40-100 frames each to display a detection density table.
- Interactive video selection by index (or direct target with --video).
- Back-to-back pipeline benchmark reporting time, FPS, absolute delta, and relative delta.
- Supports --segmentation (default: False).
- Reports hardware device (CUDA device name or 'CPU only').
"""

import os
import sys
import argparse
import types
import time
import cv2
import numpy as np
import torch

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# In case script is in rdd_dedup/scripts, project root might be parent or parent of parent
if not os.path.exists(os.path.join(PROJECT_ROOT, "rdd_dedup")):
    if os.path.exists(os.path.join(os.path.dirname(PROJECT_ROOT), "rdd_dedup")):
        PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rdd_dedup.pipeline import RDDPipeline, ANNOTATION_COLORS
from rdd_dedup.config import PipelineConfig


# ═══════════════════════════════════════════════════════════════════════════
# 1. STANDALONE ANNOTATION STYLES
# ═══════════════════════════════════════════════════════════════════════════

def annotate_frame_old_style(
    self,
    frame: np.ndarray,
    detections: list,
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    """Pre-change annotation style: plain cv2.rectangle bounding box + solid label bar."""
    live_seg_ready = (
        self.config.enable_segmentation
        and self.segmenter is not None
        and self.segmenter.is_loaded
    )
    if live_seg_ready:
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.segmenter.predictor.set_image(img_rgb)
        except Exception:
            live_seg_ready = False

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = ANNOTATION_COLORS[det.class_id % len(ANNOTATION_COLORS)]

        if live_seg_ready:
            try:
                box_np = np.array((x1, y1, x2, y2))
                masks, scores, _ = self.segmenter.predictor.predict(
                    box=box_np, multimask_output=False
                )
                mask = masks[0].squeeze()
                if mask.dtype != bool:
                    mask = mask > 0.0
                colored = np.zeros_like(frame)
                colored[mask] = color
                alpha = 0.35
                frame[mask] = cv2.addWeighted(
                    frame, 1 - alpha, colored, alpha, 0
                )[mask]
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(frame, contours, -1, color, 1, cv2.LINE_AA)
            except Exception:
                pass

        # Plain bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Solid label bar with track ID
        label = f"#{det.track_id} {det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        cv2.rectangle(
            frame, (x1, max(0, y1 - 18)), (x1 + tw + 6, y1), color, -1
        )
        cv2.putText(
            frame, label, (x1 + 3, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
            cv2.LINE_AA,
        )

    # Top-left HUD stats overlay
    stats = self.track_manager.get_stats()
    unique_count = (
        len(self.duplicate_verifier.unique_defects)
        if self.duplicate_verifier else 0
    )
    hud = (
        f"RDD-DEDUP | Frame: {frame_idx}/{total_frames} | "
        f"Active: {stats['active_tracks']} | "
        f"Unique: {unique_count}"
    )
    cv2.rectangle(frame, (8, 8), (560, 40), (0, 0, 0), -1)
    cv2.putText(
        frame, hud, (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
    )

    return frame


def annotate_frame_new_style(
    self,
    frame: np.ndarray,
    detections: list,
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    """Current sci-fi HUD bracket style: corner brackets + modulated fake glow + chip label."""
    live_seg_ready = (
        self.config.enable_segmentation
        and self.segmenter is not None
        and self.segmenter.is_loaded
    )
    if live_seg_ready:
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.segmenter.predictor.set_image(img_rgb)
        except Exception:
            live_seg_ready = False

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = ANNOTATION_COLORS[det.class_id % len(ANNOTATION_COLORS)]

        if live_seg_ready:
            try:
                box_np = np.array((x1, y1, x2, y2))
                masks, scores, _ = self.segmenter.predictor.predict(
                    box=box_np, multimask_output=False
                )
                mask = masks[0].squeeze()
                if mask.dtype != bool:
                    mask = mask > 0.0
                colored = np.zeros_like(frame)
                colored[mask] = color
                alpha = 0.35
                frame[mask] = cv2.addWeighted(
                    frame, 1 - alpha, colored, alpha, 0
                )[mask]
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(frame, contours, -1, color, 1, cv2.LINE_AA)
            except Exception:
                pass

        # Lightweight sci-fi HUD brackets with modulated glow
        frame_h, frame_w = frame.shape[:2]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        corner_len = min(18, max(8, min(bw // 4, bh // 4)))
        corner_len = min(corner_len, bw // 2, bh // 2)
        corner_len = max(2, corner_len)

        base_thickness = 2
        conf = float(np.clip(det.confidence, 0.0, 1.0))
        glow_thickness = base_thickness + (3 if conf >= 0.6 else 2)
        glow_alpha = float(np.clip(0.30 + 0.20 * conf, 0.25, 0.55))
        pad = glow_thickness // 2 + 1

        corners = [
            # Top-left
            ((x1, y1), (x1 + corner_len, y1), (x1, y1), (x1, y1 + corner_len),
             max(0, x1 - pad), max(0, y1 - pad), min(frame_w, x1 + corner_len + pad), min(frame_h, y1 + corner_len + pad)),
            # Top-right
            ((x2, y1), (x2 - corner_len, y1), (x2, y1), (x2, y1 + corner_len),
             max(0, x2 - corner_len - pad), max(0, y1 - pad), min(frame_w, x2 + pad), min(frame_h, y1 + corner_len + pad)),
            # Bottom-left
            ((x1, y2), (x1 + corner_len, y2), (x1, y2), (x1, y2 - corner_len),
             max(0, x1 - pad), max(0, y2 - corner_len - pad), min(frame_w, x1 + corner_len + pad), min(frame_h, y2 + pad)),
            # Bottom-right
            ((x2, y2), (x2 - corner_len, y2), (x2, y2), (x2, y2 - corner_len),
             max(0, x2 - corner_len - pad), max(0, y2 - corner_len - pad), min(frame_w, x2 + pad), min(frame_h, y2 + pad)),
        ]

        # Pass 1: Local ROI glow pass (lower opacity, wider line)
        for p1, p2, p3, p4, rx1, ry1, rx2, ry2 in corners:
            if rx2 > rx1 and ry2 > ry1:
                roi = frame[ry1:ry2, rx1:rx2]
                glow = roi.copy()
                cv2.line(glow, (p1[0] - rx1, p1[1] - ry1), (p2[0] - rx1, p2[1] - ry1), color, glow_thickness, cv2.LINE_AA)
                cv2.line(glow, (p3[0] - rx1, p3[1] - ry1), (p4[0] - rx1, p4[1] - ry1), color, glow_thickness, cv2.LINE_AA)
                cv2.addWeighted(glow, glow_alpha, roi, 1.0 - glow_alpha, 0, dst=roi)

        # Pass 2: Crisp normal-thickness bracket lines
        for p1, p2, p3, p4, _, _, _, _ in corners:
            cv2.line(frame, p1, p2, color, base_thickness, cv2.LINE_AA)
            cv2.line(frame, p3, p4, color, base_thickness, cv2.LINE_AA)

        # HUD chip label (tight padding, dark fill, 1px accent border)
        label = f"#{det.track_id} {det.class_name} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        pad_x = 3
        pad_y = 2
        chip_w = tw + 2 * pad_x
        chip_h = th + baseline + 2 * pad_y

        chip_x1 = max(0, x1)
        chip_x2 = min(frame_w - 1, chip_x1 + chip_w)
        if y1 - chip_h - 2 >= 0:
            chip_y2 = y1 - 2
            chip_y1 = chip_y2 - chip_h
        else:
            chip_y1 = min(frame_h - 1, y1 + 3)
            chip_y2 = min(frame_h - 1, chip_y1 + chip_h)

        cv2.rectangle(frame, (chip_x1, chip_y1), (chip_x2, chip_y2), (15, 15, 15), -1)
        cv2.rectangle(frame, (chip_x1, chip_y1), (chip_x2, chip_y2), color, 1)
        text_x = chip_x1 + pad_x
        text_y = min(frame_h - 1, chip_y1 + pad_y + th)
        cv2.putText(
            frame, label, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
            cv2.LINE_AA,
        )

    # Top-left HUD stats overlay
    stats = self.track_manager.get_stats()
    unique_count = (
        len(self.duplicate_verifier.unique_defects)
        if self.duplicate_verifier else 0
    )
    hud = (
        f"RDD-DEDUP | Frame: {frame_idx}/{total_frames} | "
        f"Active: {stats['active_tracks']} | "
        f"Unique: {unique_count}"
    )
    cv2.rectangle(frame, (8, 8), (560, 40), (0, 0, 0), -1)
    cv2.putText(
        frame, hud, (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
    )

    return frame


# ═══════════════════════════════════════════════════════════════════════════
# 2. SCANNING & DENSITY TABLE
# ═══════════════════════════════════════════════════════════════════════════

def scan_videos(video_paths: list, sample_frames: int, model_path: str) -> list:
    """Scan candidate videos with YOLO to compute detection density."""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
    except Exception as e:
        print(f"[Warning] Could not load YOLO detector for scanning: {e}")
        model = None

    results = []
    print(f"\nScanning {len(video_paths)} videos (sampling up to {sample_frames} frames each)...")
    for idx, path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        det_counts = []
        frames_scanned = 0
        for _ in range(sample_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames_scanned += 1
            if model is not None:
                preds = model.predict(frame, conf=0.25, verbose=False)[0]
                det_counts.append(len(preds.boxes))
            else:
                det_counts.append(0)
        cap.release()

        avg_d = sum(det_counts) / max(1, frames_scanned) if det_counts else 0.0
        max_d = max(det_counts) if det_counts else 0
        total_d = sum(det_counts)

        results.append({
            "index": idx,
            "path": path,
            "filename": os.path.basename(path),
            "resolution": f"{w}x{h}",
            "fps": fps,
            "total_frames": total_f,
            "scanned_frames": frames_scanned,
            "avg_detections": avg_d,
            "max_detections": max_d,
            "total_detections": total_d,
        })

    return results


def print_density_table(scan_results: list):
    """Print a clean formatted Markdown / text table of scanned video densities."""
    print("\n" + "=" * 90)
    print("  VIDEO DETECTION DENSITY SCAN (First 10 candidates)")
    print("=" * 90)
    header = f"{'Idx':<4} | {'Video Filename':<32} | {'Resolution':<10} | {'Frames':<8} | {'Avg Det/Fr':<10} | {'Max Det':<8}"
    print(header)
    print("-" * 90)
    for r in scan_results:
        print(
            f"{r['index']:<4} | {r['filename']:<32} | {r['resolution']:<10} | {r['total_frames']:<8} | "
            f"{r['avg_detections']:<10.2f} | {r['max_detections']:<8}"
        )
    print("=" * 90)


# ═══════════════════════════════════════════════════════════════════════════
# 3. BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_single_benchmark(
    video_path: str,
    enable_segmentation: bool,
    style_name: str,
    annotate_fn,
    model_path: str,
) -> dict:
    """Run pipeline on video using in-memory monkey patched annotation function."""
    config_path = os.path.join(PROJECT_ROOT, "configs", "default_pipeline.yaml")
    if os.path.exists(config_path):
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig()

    config.enable_segmentation = enable_segmentation
    config.model_path = model_path
    config.save_annotated_video = True

    # Output directory
    safe_video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(PROJECT_ROOT, "outputs", f"bench_{safe_video_name}_{style_name}")
    os.makedirs(out_dir, exist_ok=True)

    pipeline = RDDPipeline(config=config)
    # Monkey-patch _annotate_frame strictly in memory
    pipeline._annotate_frame = types.MethodType(annotate_fn, pipeline)

    t0 = time.time()
    result = pipeline.process_video(video_path, output_dir=out_dir)
    wall_time = time.time() - t0

    return {
        "style_name": style_name,
        "processed_frames": result.processed_frames,
        "total_frames": result.total_frames,
        "total_raw_detections": result.total_raw_detections,
        "unique_defects": result.total_unique_defects,
        "processing_time_sec": result.processing_time_sec,
        "pipeline_fps": result.fps,
        "wall_time_sec": wall_time,
    }


def parse_bool(value) -> bool:
    """Parse flexible boolean string for CLI."""
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1", "yes", "on", "t"):
        return True
    if str(value).lower() in ("false", "0", "no", "off", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Old vs New HUD Bracket Annotation Styles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video-dir", type=str, default=None,
        help="Path to folder containing .mp4 video files to scan",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to a specific .mp4 file to benchmark directly",
    )
    parser.add_argument(
        "--segmentation", type=parse_bool, default=False,
        help="Enable SAM2 segmentation (default: False)",
    )
    parser.add_argument(
        "--sample-frames", type=int, default=50,
        help="Frames to sample per video during density scan",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to YOLO model file (default: models/rdd_yolo.pt or models/general/rdd_yolo.pt)",
    )

    args = parser.parse_args()

    # Hardware reporting
    if torch.cuda.is_available():
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device_name = "CPU only"

    # Resolve model path
    model_path = args.model
    if not model_path:
        cand1 = os.path.join(PROJECT_ROOT, "models", "rdd_yolo.pt")
        cand2 = os.path.join(PROJECT_ROOT, "models", "general", "rdd_yolo.pt")
        model_path = cand1 if os.path.exists(cand1) else cand2

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    target_video = None

    if args.video:
        target_video = args.video if os.path.isabs(args.video) else os.path.join(PROJECT_ROOT, args.video)
        if not os.path.exists(target_video):
            print(f"Error: Video file not found: {target_video}")
            sys.exit(1)
    elif args.video_dir:
        vdir = args.video_dir if os.path.isabs(args.video_dir) else os.path.join(PROJECT_ROOT, args.video_dir)
        if not os.path.exists(vdir):
            print(f"Error: Video directory not found: {vdir}")
            sys.exit(1)

        candidates = [
            os.path.join(vdir, f) for f in sorted(os.listdir(vdir))
            if f.lower().endswith(".mp4")
        ][:10]

        if not candidates:
            print(f"Error: No .mp4 videos found in {vdir}")
            sys.exit(1)

        scan_table = scan_videos(candidates, args.sample_frames, model_path)
        print_density_table(scan_table)

        # Interactive selection or default
        if sys.stdin.isatty():
            try:
                raw_choice = input(f"\nSelect video index to benchmark [0-{len(scan_table)-1}] (default 0): ").strip()
                choice_idx = int(raw_choice) if raw_choice else 0
            except Exception:
                choice_idx = 0
        else:
            # Pick the video with highest average detections if non-interactive
            sorted_by_density = sorted(scan_table, key=lambda x: x["avg_detections"], reverse=True)
            choice_idx = sorted_by_density[0]["index"]
            print(f"\nNon-interactive mode: Auto-selected video with highest density [index {choice_idx}]: {scan_table[choice_idx]['filename']}")

        target_video = scan_table[choice_idx]["path"]
    else:
        # Default to inputs/ directory
        default_dir = os.path.join(PROJECT_ROOT, "inputs")
        if os.path.exists(default_dir):
            print(f"No --video or --video-dir specified. Defaulting to: {default_dir}")
            candidates = [
                os.path.join(default_dir, f) for f in sorted(os.listdir(default_dir))
                if f.lower().endswith(".mp4") and not f.startswith("sample")
            ][:10]
            scan_table = scan_videos(candidates, args.sample_frames, model_path)
            print_density_table(scan_table)
            sorted_by_density = sorted(scan_table, key=lambda x: x["avg_detections"], reverse=True)
            choice_idx = sorted_by_density[0]["index"]
            print(f"\nSelected video [index {choice_idx}]: {scan_table[choice_idx]['filename']}")
            target_video = scan_table[choice_idx]["path"]
        else:
            print("Error: Please provide either --video <file> or --video-dir <folder>.")
            sys.exit(1)

    # Get video metadata
    cap = cv2.VideoCapture(target_video)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print("\n" + "=" * 70)
    print("  RUNNING COMPARATIVE BENCHMARK (BACK-TO-BACK IN SAME PROCESS)")
    print("=" * 70)
    print(f"  Target Video:       {os.path.basename(target_video)}")
    print(f"  Resolution:         {vw}x{vh}")
    print(f"  Total Frames:       {total_f}")
    print(f"  Device:             {device_name}")
    print(f"  Segmentation:       {'ON (SAM2 Enabled)' if args.segmentation else 'OFF (Pure Detection + Track)'}")
    print("=" * 70)

    # Run 1: Baseline (Old Style)
    print("\n[Run 1/2] Benchmarking Baseline (Old Box + Solid Bar)...")
    res_old = run_single_benchmark(
        video_path=target_video,
        enable_segmentation=args.segmentation,
        style_name="old_style",
        annotate_fn=annotate_frame_old_style,
        model_path=model_path,
    )
    print(f"  -> Done in {res_old['processing_time_sec']:.2f}s ({res_old['pipeline_fps']:.2f} FPS)")

    # Run 2: New HUD (Corner Brackets + Fake Glow + Chip Label)
    print("\n[Run 2/2] Benchmarking New HUD Brackets (Corner 'L' + Glow + Chip)...")
    res_new = run_single_benchmark(
        video_path=target_video,
        enable_segmentation=args.segmentation,
        style_name="new_hud_style",
        annotate_fn=annotate_frame_new_style,
        model_path=model_path,
    )
    print(f"  -> Done in {res_new['processing_time_sec']:.2f}s ({res_new['pipeline_fps']:.2f} FPS)")

    # Calculations
    fps_old = res_old["pipeline_fps"]
    fps_new = res_new["pipeline_fps"]
    delta_fps = fps_new - fps_old
    rel_change_pct = (delta_fps / max(1e-6, fps_old)) * 100.0

    time_old = res_old["processing_time_sec"]
    time_new = res_new["processing_time_sec"]
    delta_time = time_new - time_old

    # Output Results
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Video:                    {os.path.basename(target_video)} ({vw}x{vh})")
    print(f"  Device:                   {device_name}")
    print(f"  Segmentation Mode:        {'ON' if args.segmentation else 'OFF'}")
    print(f"  Frames Processed:         {res_new['processed_frames']}")
    print(f"  Raw Detections Rendered:  {res_new['total_raw_detections']}")
    print("-" * 70)
    print(f"  Baseline (Old Style):     {time_old:.2f}s | {fps_old:.2f} FPS")
    print(f"  New HUD (Bracket Style):  {time_new:.2f}s | {fps_new:.2f} FPS")
    print("-" * 70)
    print(f"  Absolute Delta:           {delta_fps:+.2f} FPS ({delta_time:+.2f}s total)")
    print(f"  Relative Speed Change:    {rel_change_pct:+.1f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
