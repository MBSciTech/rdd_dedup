# RDD Dedup Pipeline

A modular Road Defect Detection tracking and deduplication pipeline.

## Pipeline Flow

Video Frame
→ Motion Estimation
→ YOLO + BoT-SORT Detection
→ Track Management
→ Duplicate Verification
→ Unique Defect Storage

## Project Structure

- `rdd_dedup/pipeline.py` - Main pipeline orchestrator
- `rdd_dedup/config.py` - Pipeline configuration
- `rdd_dedup/detection/` - YOLO and BoT-SORT detection
- `rdd_dedup/tracking/` - Defect track lifecycle management
- `rdd_dedup/stabilization/` - Camera motion estimation
- `rdd_dedup/verification/` - Duplicate detection and merging
- `rdd_dedup/storage/` - Defect database layer
- `rdd_dedup/utils/` - Helper utilities