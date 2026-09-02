import os
import logging
import torch
import numpy as np
import cv2
from typing import Tuple, Optional, Dict
from rdd_dedup.config import PipelineConfig

logger = logging.getLogger(__name__)

class DefectSegmenter:
    """SAM2-based image segmentation wrapper."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._is_loaded = False
        self.predictor = None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_model(self):
        if self._is_loaded:
            return
            
        checkpoint_path = self.config.resolve_path(self.config.sam2_checkpoint_path)
        if not os.path.exists(checkpoint_path):
            logger.error(f"SAM2 checkpoint not found at: {checkpoint_path}")
            return
            
        try:
            # pyrefly: ignore [missing-import]
            from sam2.build_sam import build_sam2
            # pyrefly: ignore [missing-import]
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            
            sam2_model = build_sam2(self.config.sam2_config_name, checkpoint_path, device=device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            self._is_loaded = True
            logger.info("SAM2 Segmenter loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SAM2: {e}")

    def segment(self, crop: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[Dict]:
        if not self._is_loaded or self.predictor is None:
            return None
            
        try:
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(img_rgb)
            
            box_np = np.array(box)
            masks, scores, _ = self.predictor.predict(
                box=box_np,
                multimask_output=False
            )
            
            mask = masks[0].squeeze()
            if mask.dtype != bool:
                mask = mask > 0.0
            
            score = float(scores[0])
            pixel_area = int(mask.sum())
            
            return {
                "mask": mask,
                "pixel_area": pixel_area,
                "quality_score": score
            }
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return None
