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

PROPRIETARY AND CONFIDENTIAL SOURCE CODE LICENSE

Copyright (c) 2026 [Your Full Name or Organization]. All rights reserved.

This source code and related documentation ("Software") are proprietary to and the exclusive property of [Your Full Name or Organization].

1. GRANT OF EVALUATION ACCESS ONLY
Access to this repository is provided solely for inspection and review purposes authorized in writing by the copyright holder. No license or right to execute, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software is granted.

2. STRICT PROHIBITIONS
You explicitly agree NOT to:
- Copy, reproduce, or duplicate any part of the codebase, either manually or programmatically.
- Modify, adapt, translate, reverse engineer, decompile, disassemble, or create derivative works from the Software.
- Share, transmit, leak, or disclose the repository contents, architecture, or logic to any unauthorized third party.
- Use any portion of this codebase for training machine learning models or artificial intelligence systems without prior written consent.

3. OWNERSHIP
All rights, title, and interest in and to the Software—including all associated intellectual property rights—remain exclusively with [Your Full Name or Organization].

4. TERMINATION OF ACCESS
The copyright holder reserves the right to revoke repository access at any time without prior notice. Upon revocation, you must immediately delete any locally cloned, downloaded, or cached copies of the source code.

5. DISCLAIMER
THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.
