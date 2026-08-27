"""
ClassConfig — Centralized class name/ID management for RDD_Ahmedabad.

Loads from a YAML configuration file so that adding a new defect class
requires changing only one file (configs/rdd_ahmedabad_classes.yaml),
with zero code modifications anywhere else.

Usage:
    config = ClassConfig.load("configs/rdd_ahmedabad_classes.yaml")
    print(config.get_all_classes())       # {0: 'POTHOLE', 1: 'CRACK_LONG', ...}
    print(config.get_class_name(0))       # 'POTHOLE'
    print(config.get_class_id("POTHOLE")) # 0
    print(config.map_model_class("D40"))  # 'POTHOLE'
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SplitConfig:
    """Dataset split ratios configuration."""
    train_ratio: float = 0.70
    val_ratio: float = 0.20
    test_ratio: float = 0.10
    split_by: str = "video"  # "video" or "random"

    def validate(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.2f} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )


@dataclass
class ClassConfig:
    """
    Centralized class configuration for RDD_Ahmedabad.

    Loads all class names, IDs, model mappings, and split settings
    from a single YAML file. This is the single source of truth —
    adding a new class requires only editing the YAML.
    """
    dataset_name: str = "RDD_Ahmedabad"
    dataset_version: str = "1.0"
    classes: Dict[int, str] = field(default_factory=dict)
    model_class_mapping: Dict[str, str] = field(default_factory=dict)
    split: SplitConfig = field(default_factory=SplitConfig)

    # Internal reverse lookup (name -> id), built on load
    _name_to_id: Dict[str, int] = field(default_factory=dict, repr=False)
    _yaml_path: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        """Build reverse lookup table."""
        self._rebuild_reverse_lookup()

    def _rebuild_reverse_lookup(self):
        """Rebuild the name-to-id reverse mapping."""
        self._name_to_id = {name: cid for cid, name in self.classes.items()}

    # ── Class Lookups ──

    def get_class_name(self, class_id: int) -> str:
        """Get class name by numeric ID. Returns 'UNKNOWN' if not found."""
        return self.classes.get(class_id, "UNKNOWN")

    def get_class_id(self, class_name: str) -> int:
        """Get numeric ID by class name. Returns -1 if not found."""
        return self._name_to_id.get(class_name.upper(), -1)

    def get_all_classes(self) -> Dict[int, str]:
        """Return the full {id: name} class dictionary."""
        return dict(self.classes)

    def get_class_names(self) -> List[str]:
        """Return ordered list of class names."""
        return [self.classes[k] for k in sorted(self.classes.keys())]

    def get_num_classes(self) -> int:
        """Return the total number of classes."""
        return len(self.classes)

    # ── Model Class Mapping ──

    def map_model_class(self, model_class_name: str) -> str:
        """
        Map a model's predicted class name to an RDD_Ahmedabad class.

        Tries exact match first, then case-insensitive partial matching.
        Returns the original name if no mapping is found.
        """
        # Exact match
        if model_class_name in self.model_class_mapping:
            return self.model_class_mapping[model_class_name]

        # Case-insensitive match
        lower_name = model_class_name.lower().strip()
        for key, value in self.model_class_mapping.items():
            if key.lower().strip() == lower_name:
                return value

        # Partial match — check if any mapping key is contained in the name
        for key, value in self.model_class_mapping.items():
            if key.lower() in lower_name or lower_name in key.lower():
                return value

        return model_class_name

    def get_suggested_class_id(self, model_class_name: str) -> int:
        """Map model class name to RDD_Ahmedabad class ID. Returns 0 if unmapped."""
        mapped_name = self.map_model_class(model_class_name)
        cid = self.get_class_id(mapped_name)
        return cid if cid >= 0 else 0

    # ── Persistence ──

    @classmethod
    def load(cls, yaml_path: str) -> "ClassConfig":
        """
        Load class configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            Populated ClassConfig instance.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Class config not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Parse classes (ensure int keys)
        classes_raw = raw.get("classes", {})
        classes = {int(k): str(v) for k, v in classes_raw.items()}

        # Parse model mapping
        model_mapping = raw.get("model_class_mapping", {})

        # Parse split config
        split_raw = raw.get("split", {})
        split = SplitConfig(
            train_ratio=split_raw.get("train_ratio", 0.70),
            val_ratio=split_raw.get("val_ratio", 0.20),
            test_ratio=split_raw.get("test_ratio", 0.10),
            split_by=split_raw.get("split_by", "video"),
        )
        split.validate()

        config = cls(
            dataset_name=raw.get("dataset_name", "RDD_Ahmedabad"),
            dataset_version=raw.get("dataset_version", "1.0"),
            classes=classes,
            model_class_mapping=model_mapping,
            split=split,
        )
        config._yaml_path = os.path.abspath(yaml_path)
        return config

    def save(self, yaml_path: Optional[str] = None):
        """Save current configuration back to YAML."""
        path = yaml_path or self._yaml_path
        if not path:
            raise ValueError("No YAML path specified for saving.")

        data = {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "classes": self.classes,
            "model_class_mapping": self.model_class_mapping,
            "split": {
                "train_ratio": self.split.train_ratio,
                "val_ratio": self.split.val_ratio,
                "test_ratio": self.split.test_ratio,
                "split_by": self.split.split_by,
            },
        }

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # ── Utility ──

    def add_class(self, class_name: str) -> int:
        """
        Add a new class dynamically. Returns the assigned ID.
        Does nothing if the class already exists.
        """
        class_name = class_name.upper()
        existing_id = self.get_class_id(class_name)
        if existing_id >= 0:
            return existing_id

        new_id = max(self.classes.keys()) + 1 if self.classes else 0
        self.classes[new_id] = class_name
        self._rebuild_reverse_lookup()
        return new_id

    def __repr__(self) -> str:
        return (
            f"ClassConfig(dataset='{self.dataset_name}', "
            f"version='{self.dataset_version}', "
            f"classes={len(self.classes)})"
        )
