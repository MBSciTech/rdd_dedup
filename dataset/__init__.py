"""
rdd_dedup.dataset — RDD_Ahmedabad Dataset Collection & Annotation System.

Provides semi-automatic dataset building from the detection pipeline output
with human-in-the-loop annotation review.

Public API:
    ClassConfig       — Centralized class name/ID management from YAML
    DatasetManager    — Directory structure, YOLO labels, data.yaml
    AnnotationManager — Metadata CSV tracking
    CropSelector      — Best-crop selection from BoT-SORT tracks
"""

from rdd_dedup.dataset.class_config import ClassConfig
from rdd_dedup.dataset.dataset_manager import DatasetManager
from rdd_dedup.dataset.annotation_manager import AnnotationManager
from rdd_dedup.dataset.crop_selector import CropSelector, CandidateAnnotation

__all__ = [
    "ClassConfig",
    "DatasetManager",
    "AnnotationManager",
    "CropSelector",
    "CandidateAnnotation",
]
